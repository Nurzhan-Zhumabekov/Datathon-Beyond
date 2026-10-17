"""python -m signalhunter DATA --target COL [options]"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import drift, impact, leakage, model, profile, report, segments, story
from .util import (
    fix_decimal_commas,
    infer_kinds,
    prepare_target,
    read_table,
    time_array,
    to_datetime_smart,
)


def _split(s: str | None) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="signalhunter", description="От сырой таблицы к заготовке истории за одну команду.")
    p.add_argument("data", help="CSV / parquet / xlsx с целевой колонкой")
    p.add_argument("--target", required=True, help="имя целевой колонки")
    p.add_argument("--test", help="таблица без цели: даст проверку сдвига и baseline-предсказания")
    p.add_argument("--time-col", help="колонка времени: валидация обучается на прошлом, проверяется на будущем")
    p.add_argument("--group-col", help="колонка группы (клиент, магазин): одна группа не делится между train и valid")
    p.add_argument("--id-cols", help="колонки-идентификаторы через запятую (не идут в модель)")
    p.add_argument("--drop", help="колонки, которые надо игнорировать, через запятую")
    p.add_argument("--positive", help="какой класс считать целевым (для бинарной цели и мультикласса)")
    p.add_argument("--gain-tp", type=float, default=10.0, help="ПРЕДПОЛОЖЕНИЕ: ценность найденного случая (default 10)")
    p.add_argument("--cost-fp", type=float, default=1.0, help="ПРЕДПОЛОЖЕНИЕ: цена лишнего действия (default 1)")
    p.add_argument("--keep-leaky", action="store_true", help="не исключать колонки с критичной утечкой из модели")
    p.add_argument("--story", action="store_true", help="LLM пишет черновик питча по отчёту (нужен ANTHROPIC_API_KEY; уходит только отчёт, без строк)")
    p.add_argument("--context", help="одна-две фразы о задаче и бизнесе для промпта LLM")
    p.add_argument("--fast", action="store_true", help="меньше деревьев и строк: для первой быстрой прогонки")
    p.add_argument("--max-rows", type=int, default=300000)
    p.add_argument("--out", default="out", help="папка результата (default out)")
    return p


def run(a) -> dict:
    t0 = time.time()

    def log(msg: str):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    out = Path(a.out)
    (out / "charts").mkdir(parents=True, exist_ok=True)

    log(f"Читаю {a.data}")
    df = fix_decimal_commas(read_table(a.data))
    if a.target not in df.columns:
        raise SystemExit(f"Колонки {a.target!r} нет. Есть: {', '.join(map(str, df.columns))}")
    y_all, task, tinfo = prepare_target(df[a.target], a.positive)
    ok = y_all.notna().to_numpy()
    df = df.loc[ok].reset_index(drop=True)
    y = y_all.loc[ok].to_numpy(float)
    max_rows = 100000 if a.fast else a.max_rows
    if len(df) > max_rows:
        keep = np.sort(np.random.default_rng(0).choice(len(df), max_rows, replace=False))
        df, y = df.iloc[keep].reset_index(drop=True), y[keep]
        log(f"Строк слишком много, работаю на случайной выборке {max_rows}")
    binary = task == "binary"
    base = float(y.mean())

    feats = df.drop(columns=[a.target])
    id_cols = _split(a.id_cols) + _split(a.drop)
    if a.group_col:
        id_cols.append(a.group_col)
    kinds = infer_kinds(feats, id_cols=id_cols)
    for c, k in kinds.items():
        if k == "datetime":
            df[c] = to_datetime_smart(df[c])
    tv = time_array(df[a.time_col]) if a.time_col else None
    if a.time_col and a.time_col in kinds and kinds[a.time_col] != "datetime":
        kinds[a.time_col] = "datetime"
        df[a.time_col] = to_datetime_smart(df[a.time_col])
    groups = df[a.group_col].astype(str).to_numpy() if a.group_col else None
    log(f"Строк: {len(df)}, колонок-признаков: {feats.shape[1]}, задача: {'бинарная' if binary else 'регрессия'}")

    prof = profile.column_profile(feats, kinds)
    warns = profile.data_warnings(df, kinds, y, task, tinfo, bool(a.time_col))
    prof.to_csv(out / "profile.csv", index=False, encoding="utf-8-sig")

    log("Аудит утечек")
    lk = leakage.univariate_power(df, y, kinds, task)
    lk.to_csv(out / "leakage.csv", index=False, encoding="utf-8-sig")
    leaks = [] if a.keep_leaky else lk.loc[lk["level"] == "critical", "column"].tolist()
    kinds_model = {c: ("leak" if c in leaks else k) for c, k in kinds.items()}
    for _, r in lk[lk["level"] != "ok"].head(5).iterrows():
        log(f"  [{r['level']}] {r['column']}: {r['metric']}={r['score']:.3f}")

    test = None
    if a.test:
        test = fix_decimal_commas(read_table(a.test))
        for c, k in kinds.items():
            if k == "datetime" and c in test.columns:
                test[c] = to_datetime_smart(test[c])

    fb = model.Features(kinds_model).fit(pd.concat([df[feats.columns], test[[c for c in feats.columns if c in test.columns]]]) if test is not None else df[feats.columns])
    X = fb.transform(df)
    splits, scheme = model.make_splits(y, task, tv, groups)
    log(f"Валидация: {scheme}")
    log("Обучаю LightGBM и линейный baseline")
    cv = model.cross_validate(X, y, task, splits, fast=a.fast)
    key = "roc_auc" if binary else "r2"
    for name in ("lightgbm", "linear"):
        s = cv["summary"][name][key]
        log(f"  {name}: {key} = {s['mean']:.3f} +- {s['std']:.3f}")

    log("Важность признаков")
    mdl, imp = model.fit_full(X, y, task, fast=a.fast)
    cov = cv["covered"]
    oof = cv["oof"]
    effects = [(fb.display.get(f, f), model.feature_effect(X[f], y, oof)) for f in imp.index[:3]]

    log("Поиск сегментов")
    use_kinds = {c: k for c, k in kinds_model.items() if k in ("numeric", "categorical")}
    min_support = max(30, int(0.01 * len(df)))
    si = np.arange(len(df)) if len(df) <= 150000 else np.sort(np.random.default_rng(1).choice(len(df), 150000, replace=False))
    dfs, ys, tvs = df.iloc[si].reset_index(drop=True), y[si], (tv[si] if tv is not None else None)
    items = segments.build_items(dfs, use_kinds, min_support)
    seg = segments.find_segments(
        items, ys, min_support=min_support, halves=segments.make_halves(len(dfs), tvs),
        effect="rel_mean" if binary else "std", min_effect=0.25 if binary else 0.2,
    )
    seg.to_csv(out / "segments.csv", index=False, encoding="utf-8-sig")

    log("Слепые зоны модели (где факт систематически не совпадает с предсказанием)")
    ci = np.flatnonzero(cov)
    if len(ci) > 150000:
        ci = np.sort(np.random.default_rng(2).choice(ci, 150000, replace=False))
    dfc = df.iloc[ci].reset_index(drop=True)
    resid = y[ci] - oof[ci]
    tvc = tv[ci] if tv is not None else None
    items_c = segments.build_items(dfc, use_kinds, min_support)
    err_seg = segments.find_segments(
        items_c, resid, min_support=min_support, halves=segments.make_halves(len(dfc), tvc),
        effect="std", min_effect=0.2, z_min=4.0, top=6, top_int=0,
    )
    err_seg.to_csv(out / "blind_spots.csv", index=False, encoding="utf-8-sig")

    dec = lift = None
    if binary:
        dec = impact.decision_analysis(y[cov], oof[cov], a.gain_tp, a.cost_fp)
        lift = impact.lift_table(y[cov], oof[cov])
        lift.to_csv(out / "lift.csv", index=False, encoding="utf-8-sig")

    dr = None
    if test is not None:
        log("Сравниваю train и test")
        Xt = fb.transform(test)
        dr = drift.adversarial_validation(X, Xt)
        log(f"  adversarial AUC = {dr['auc']:.3f}")
        pt = model.predict_model(mdl, Xt, task)
        idc = next((c for c in id_cols if c in test.columns), None)
        sub = pd.DataFrame({idc or "row": test[idc] if idc else np.arange(len(test)), "prediction": pt})
        sub.to_csv(out / "predictions_test.csv", index=False, encoding="utf-8-sig")
        log("  baseline-предсказания: predictions_test.csv")

    log("Графики и отчёт")
    scheme_note = None
    if tv is not None:
        scheme_note = "Первый блок времени используется только для обучения, поэтому оценка идёт по более поздним данным."
    ctx = dict(
        data_name=Path(a.data).name, task=task, tinfo=tinfo, target=a.target, n_rows=len(df), n_cols=feats.shape[1],
        base=base, scheme=scheme, scheme_note=scheme_note, summary=cv["summary"], naive=cv["naive"], imp=imp,
        display=fb.display, effects=effects, seg=seg, err_seg=err_seg, dec=dec, lift=lift,
        lk=lk, warnings=warns, excluded=leaks, drift=dr, y_cov=y[cov], oof_cov=oof[cov],
    )
    path = report.build(ctx, out)

    prompt = story.build_prompt(path.read_text(encoding="utf-8"), getattr(a, "context", None))
    (out / "story_prompt.md").write_text(prompt, encoding="utf-8")
    if getattr(a, "story", False):
        log("LLM: отправляю только агрегированный отчёт (без строк данных). Убедитесь, что правила это разрешают")
        try:
            (out / "story.md").write_text(story.generate(prompt), encoding="utf-8")
            log("  черновик питча: story.md")
        except Exception as e:  # the analysis itself must never fail because of the LLM
            log(f"  LLM не сработал ({e}). Промпт сохранён: story_prompt.md")

    summary = {
        "task": task,
        "rows": len(df),
        "base_rate_or_mean": base,
        "scheme": scheme,
        "metrics": {k: {m: v["mean"] for m, v in s.items()} for k, s in cv["summary"].items()},
        "excluded_leaky": leaks,
        "stable_segments": seg.loc[seg["stable"], "rule"].tolist() if len(seg) else [],
        "blind_spots": err_seg["rule"].tolist() if len(err_seg) else [],
        "adversarial_auc": dr["auc"] if dr else None,
        "seconds": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Готово: {path}")
    return summary


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
