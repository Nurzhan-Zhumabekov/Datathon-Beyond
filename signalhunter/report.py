"""Slide-ready charts and the markdown story brief."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .util import fmt  # noqa: E402

BLUE, AMBER, RED, GRAY, INK = "#2563eb", "#f59e0b", "#dc2626", "#94a3b8", "#0f172a"


def _style():
    plt.rcParams.update(
        {
            "figure.figsize": (11, 6.2),
            "figure.dpi": 150,
            "font.size": 12,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#cbd5e1",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.8,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _pct_or_num(binary: bool):
    return (lambda v: f"{v:.1%}") if binary else fmt


def chart_drivers(imp: pd.Series, display: dict, path: Path):
    top = imp.head(12)[::-1]
    fig, ax = plt.subplots()
    ax.barh([display.get(i, i) for i in top.index], top.values, color=BLUE)
    ax.set_xlabel("средний вклад признака в предсказание (|SHAP|)")
    first = display.get(imp.index[0], imp.index[0])
    ax.set_title(f"Главный драйвер: {first}")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def chart_effect(eff: pd.DataFrame, name: str, base: float, binary: bool, path: Path):
    f = _pct_or_num(binary)
    fig, ax = plt.subplots()
    x = np.arange(len(eff))
    ax.bar(x, eff["actual"], color=BLUE, label="факт")
    ax.plot(x, eff["pred"], color=AMBER, marker="o", lw=2.5, label="предсказание модели")
    ax.axhline(base, color=GRAY, ls="--", lw=1.5)
    ax.text(len(eff) - 0.5, base, f" в среднем {f(base)}", color=GRAY, va="bottom", ha="right")
    for xi, (a, n) in enumerate(zip(eff["actual"], eff["n"])):
        ax.text(xi, a, f"{f(a)}\nn={int(n)}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(eff["label"], rotation=25, ha="right")
    ax.set_ylabel("доля цели" if binary else "среднее значение цели")
    ax.set_title(f"Как «{name}» связан с результатом")
    ax.legend(frameon=False)
    ax.margins(y=0.18)
    _save(fig, path)


def chart_segments(seg: pd.DataFrame, base: float, binary: bool, title: str, path: Path, color=BLUE):
    """Bars = observed value in the segment. Black diamond = what the parts alone would predict."""
    f = _pct_or_num(binary)
    d = seg.head(8)[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 0.8 * len(d) + 2.6))
    cols = [color if s else GRAY for s in d["stable"]]
    ax.barh(range(len(d)), d["mean"], color=cols)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([r if len(r) < 58 else r[:55] + "..." for r in d["rule"]], fontsize=10)
    ax.axvline(base, color=INK, ls="--", lw=1.4)
    ax.set_ylim(-0.6, len(d) + 0.2)
    ax.text(base, len(d) - 0.3, f"  в среднем {f(base)}", color=INK, va="bottom", fontsize=10)
    for i, r in d.iterrows():
        label = f"  {f(r['mean'])}  (n={int(r['n'])})"
        if pd.notna(r["expected"]):
            ax.scatter([r["expected"]], [i], marker="D", color=INK, s=45, zorder=5)
            label = f"  {f(r['mean'])}  (по отдельности ждали {f(r['expected'])}, n={int(r['n'])})"
        ax.text(max(r["mean"], r["expected"]) if pd.notna(r["expected"]) else r["mean"], i, label, va="center", fontsize=10)
    ax.set_title(title)
    ax.margins(x=0.45)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("доля цели" if binary else "среднее значение цели")
    _save(fig, path)


def chart_blind(err: pd.DataFrame, binary: bool, path: Path):
    d = err.head(8)[::-1].reset_index(drop=True)
    scale = 100 if binary else 1
    unit = "п.п." if binary else ""
    fig, ax = plt.subplots(figsize=(11, 0.75 * len(d) + 2.6))
    vals = (d["mean"] - d["base"]) * scale
    ax.barh(range(len(d)), vals, color=[(RED if x > 0 else AMBER) if s else GRAY for x, s in zip(vals, d["stable"])])
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([r if len(r) < 58 else r[:55] + "..." for r in d["rule"]], fontsize=10)
    ax.axvline(0, color=INK, lw=1.2)
    for i, (x, n) in enumerate(zip(vals, d["n"])):
        ax.text(x, i, f"  {x:+.1f} {unit} (n={int(n)})", va="center", ha="left" if x >= 0 else "right", fontsize=10)
    ax.set_title("Где модель систематически ошибается")
    ax.set_xlabel("факт минус предсказание (красный: модель занижает, жёлтый: завышает)")
    ax.margins(x=0.3)
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def chart_gain(lift: pd.DataFrame, path: Path):
    x = np.r_[0, lift["cum_share"].to_numpy()]
    y = np.r_[0, lift["cum_capture"].to_numpy()]
    fig, ax = plt.subplots()
    ax.plot(x, y, color=BLUE, lw=3, marker="o", label="модель")
    ax.plot([0, 1], [0, 1], color=GRAY, ls="--", label="случайно")
    k = min(1, len(lift) - 1)
    ax.annotate(
        f"верхние {lift['cum_share'].iloc[k]:.0%} по скору\nловят {lift['cum_capture'].iloc[k]:.0%} всех случаев",
        (lift["cum_share"].iloc[k], lift["cum_capture"].iloc[k]),
        xytext=(0.38, 0.35),
        arrowprops=dict(arrowstyle="->", color=INK),
        fontsize=12,
    )
    ax.set_xlabel("доля клиентов, отобранных по скору")
    ax.set_ylabel("доля найденных случаев цели")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Если работать только с верхом списка")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, path)


def chart_profit(dec: dict, path: Path):
    fig, ax = plt.subplots()
    ax.plot(dec["thresholds"], dec["values"], color=BLUE, lw=3)
    ax.axhline(0, color=GRAY, lw=1)
    ax.axhline(dec["value_all"], color=RED, ls="--", lw=1.5)
    ax.text(0.99, dec["value_all"], "действовать на всех ", color=RED, ha="right", va="bottom", fontsize=10)
    ax.scatter([dec["best_threshold"]], [dec["best_value"]], color=AMBER, s=120, zorder=5)
    ax.annotate(
        f"порог {dec['best_threshold']:.2f}\nдействуем на {dec['share_acted']:.0%} клиентов\nточность {dec['precision']:.0%}, охват {dec['recall']:.0%}",
        (dec["best_threshold"], dec["best_value"]),
        xytext=(0.55, dec["best_value"] * 0.55),
        arrowprops=dict(arrowstyle="->", color=INK),
        fontsize=11,
    )
    ax.set_xlabel("порог скора: действуем, если скор выше")
    ax.set_ylabel(f"чистая ценность (выигрыш {fmt(dec['gain_tp'])}, цена ложного {fmt(dec['cost_fp'])})")
    ax.set_title("Сколько стоит правильный порог")
    _save(fig, path)


def chart_calibration(y: np.ndarray, p: np.ndarray, path: Path):
    d = pd.DataFrame({"y": y, "p": p})
    d["bin"] = pd.qcut(d["p"], q=10, duplicates="drop")
    g = d.groupby("bin", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    lim = max(g["p"].max(), g["y"].max()) * 1.1
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.plot([0, lim], [0, lim], color=GRAY, ls="--", label="идеально")
    ax.plot(g["p"], g["y"], color=BLUE, marker="o", lw=2.5, label="модель")
    ax.set_xlabel("предсказанная вероятность")
    ax.set_ylabel("фактическая доля")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Можно ли верить вероятностям")
    ax.legend(frameon=False)
    _save(fig, path)


def chart_pred_actual(y: np.ndarray, p: np.ndarray, path: Path):
    idx = np.random.default_rng(0).choice(len(y), min(len(y), 4000), replace=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(p[idx], y[idx], s=8, alpha=0.35, color=BLUE)
    lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
    ax.plot([lo, hi], [lo, hi], color=AMBER, lw=2)
    ax.set_xlabel("предсказание")
    ax.set_ylabel("факт")
    ax.set_title("Предсказание против факта")
    _save(fig, path)


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_нет_\n"
    head = "| " + " | ".join(map(str, df.columns)) + " |\n"
    sep = "|" + "|".join("---" for _ in df.columns) + "|\n"
    body = "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in df.itertuples(index=False))
    return head + sep + body


def build(ctx: dict, out: Path) -> Path:
    _style()
    charts = out / "charts"
    binary = ctx["task"] == "binary"
    f = _pct_or_num(binary)
    base = ctx["base"]
    disp = ctx["display"]
    made: list[tuple[str, str]] = []

    chart_drivers(ctx["imp"], disp, charts / "01_drivers.png")
    made.append(("01_drivers.png", "Что сильнее всего двигает результат"))
    for i, (name, eff) in enumerate(ctx["effects"], start=1):
        if len(eff):
            fn = f"02_effect_{i}.png"
            chart_effect(eff, name, base, binary, charts / fn)
            made.append((fn, f"Эффект признака «{name}»"))
    seg, err = ctx["seg"], ctx["err_seg"]
    groups = seg[seg["kind"] == "группа"] if len(seg) else seg
    inter = seg[seg["kind"] == "взаимодействие"] if len(seg) else seg
    if len(groups):
        title = f"Где результат заметно отличается от среднего ({int(groups['stable'].sum())} подтверждено на обеих половинах)"
        chart_segments(groups.sort_values("stable", ascending=False, kind="stable"), base, binary, title, charts / "03_segments.png")
        made.append(("03_segments.png", "Сегменты (серые не подтвердились на обеих половинах)"))
    if len(inter):
        chart_segments(inter.sort_values("stable", ascending=False, kind="stable"), base, binary, "Сочетания сильнее, чем сумма частей", charts / "03b_interactions.png", color=AMBER)
        made.append(("03b_interactions.png", "Взаимодействия признаков"))
    if len(err):
        chart_blind(err, binary, charts / "04_blind_spots.png")
        made.append(("04_blind_spots.png", "Слепые зоны модели"))
    if binary:
        chart_gain(ctx["lift"], charts / "05_gain.png")
        made.append(("05_gain.png", "Кривая охвата"))
        chart_calibration(ctx["y_cov"], ctx["oof_cov"], charts / "07_calibration.png")
        made.append(("07_calibration.png", "Калибровка вероятностей"))
        chart_profit(ctx["dec"], charts / "06_profit.png")
        made.append(("06_profit.png", "Ценность решения по порогу"))
    else:
        chart_pred_actual(ctx["y_cov"], ctx["oof_cov"], charts / "05_pred_actual.png")
        made.append(("05_pred_actual.png", "Предсказание против факта"))

    L: list[str] = []
    L.append(f"# Signal Hunter: {ctx['data_name']}\n")
    L.append(f"_{datetime.now():%Y-%m-%d %H:%M}_ · строк: **{ctx['n_rows']}** · колонок: **{ctx['n_cols']}** · цель: **{ctx['target']}** ({'бинарная' if binary else 'число'})\n")
    if binary:
        L.append(f"Положительный класс: «{ctx['tinfo']['positive']}», доля **{base:.2%}**.\n")
    L.append(f"Схема валидации: **{ctx['scheme']}**.\n")

    L.append("## 1. Что нужно знать о данных\n")
    L += [f"- {w}" for w in ctx["warnings"]] or ["- Явных проблем не найдено."]
    L.append("")

    L.append("## 2. Подозрение на утечки\n")
    lk = ctx["lk"]
    bad = lk[lk["level"] != "ok"][["column", "score", "metric", "level", "comment"]] if len(lk) else lk
    if len(bad):
        L.append(_md_table(bad))
        if ctx["excluded"]:
            L.append(f"**Исключены из модели:** {', '.join(ctx['excluded'])}. (вернуть: `--keep-leaky`)\n")
        L.append("Прежде чем гордиться метрикой, выясните у организаторов, когда заполняется каждая такая колонка.\n")
    else:
        L.append("Колонок, которые в одиночку подозрительно хорошо предсказывают цель, не найдено.\n")

    L.append("## 3. Baseline и честная оценка\n")
    rows = []
    for name, label in (("lightgbm", "LightGBM"), ("linear", "Линейная модель")):
        s = ctx["summary"][name]
        rows.append({"модель": label, **{k: f"{v['mean']:.3f} ± {v['std']:.3f}" for k, v in s.items()}})
    rows.append({"модель": "Наивная (без знаний)", **{k: f"{v:.3f}" for k, v in ctx["naive"].items()}})
    L.append(_md_table(pd.DataFrame(rows).fillna("")))
    key = "roc_auc" if binary else "r2"
    gap = ctx["summary"]["lightgbm"][key]["mean"] - ctx["summary"]["linear"][key]["mean"]
    if gap < 0.01:
        L.append("Линейная модель не уступает бустингу. Сигнал в основном простой, нет смысла усложнять модель, лучше потратить время на объяснение.\n")
    elif gap > 0.05:
        L.append("Бустинг заметно лучше линейной модели. В данных есть нелинейности или взаимодействия, ищите их в сегментах ниже.\n")
    else:
        L.append("Бустинг немного лучше линейной модели.\n")
    if ctx.get("scheme_note"):
        L.append(ctx["scheme_note"] + "\n")

    L.append("## 4. Что двигает результат\n")
    L.append("![drivers](charts/01_drivers.png)\n")
    top = [f"{disp.get(i, i)}" for i in ctx["imp"].head(5).index]
    L.append("Топ-5: " + ", ".join(top) + ".\n")

    L.append("## 5. Сегменты: кандидаты в историю\n")
    if len(seg):
        def _view(d: pd.DataFrame) -> pd.DataFrame:
            v = d.copy()
            v["support"] = v["support"].map(lambda x: f"{x:.1%}")
            for c in ("mean", "expected", "mean_A", "mean_B"):
                v[c] = v[c].map(lambda x: f(x) if pd.notna(x) else "-")
            v["lift"] = v["lift"].map(lambda x: f"x{x:.2f}")
            v["z"] = v["z"].map(lambda x: f"{x:.1f}")
            v["stable"] = v["stable"].map({True: "да", False: "нет"})
            return v

        if len(groups):
            L.append("### Группы с сильным отличием\n")
            L.append(_md_table(_view(groups)[["rule", "n", "support", "mean", "lift", "z", "stable", "mean_A", "mean_B"]]))
            L.append("![segments](charts/03_segments.png)\n")
        if len(inter):
            L.append("### Взаимодействия: вместе сильнее, чем по отдельности\n")
            L.append(_md_table(_view(inter)[["rule", "n", "mean", "expected", "z", "stable", "mean_A", "mean_B"]]))
            L.append("`expected` — что получилось бы, если бы эффекты условий просто складывались. Разрыв между `mean` и `expected` и есть неочевидная находка.\n")
            L.append("![interactions](charts/03b_interactions.png)\n")
        L.append("`stable = да`: эффект того же знака и значим в обеих половинах данных; `mean_A`/`mean_B` — значение в каждой половине.\n")
        L.append("**Готовые формулировки (проверьте смысл руками):**\n")
        for r in seg[seg["stable"]].head(6).itertuples():
            if r.kind == "взаимодействие":
                L.append(
                    f"- «{r.rule}» ({r.n} строк, {r.support:.1%}): показатель {f(r.mean)}, хотя по отдельности условия дали бы около {f(r.expected)}. "
                    f"Половины данных: {f(r.mean_A)} и {f(r.mean_B)}."
                )
            else:
                word = "выше" if r.mean > r.base else "ниже"
                L.append(
                    f"- «{r.rule}» ({r.n} строк, {r.support:.1%}): показатель {f(r.mean)} против {f(r.base)} в среднем, "
                    f"{word} в x{max(r.lift, 1 / r.lift) if r.lift else float('nan'):.1f}. Половины данных: {f(r.mean_A)} и {f(r.mean_B)}."
                )
        L.append("")
    else:
        L.append("Сегментов с сильным и значимым отклонением не найдено. Это тоже результат: значит, различия размыты по многим признакам.\n")

    L.append("## 6. Слепые зоны модели\n")
    if len(err):
        e = err.copy()
        sc = 100 if binary else 1
        e["support"] = e["support"].map(lambda v: f"{v:.1%}")
        e["смещение"] = [f"{(a - b) * sc:+.1f}{' п.п.' if binary else ''}" for a, b in zip(e["mean"], e["base"])]
        e["модель"] = ["занижает" if a > b else "завышает" for a, b in zip(e["mean"], e["base"])]
        e["stable"] = e["stable"].map({True: "да", False: "нет"})
        L.append(_md_table(e[["rule", "n", "support", "смещение", "модель", "stable"]]))
        L.append("Смещение = средний (факт − предсказание) в сегменте. Это честный материал для слайда «ограничения и что делать дальше».\n")
        L.append("![blind](charts/04_blind_spots.png)\n")
    else:
        L.append("Систематических смещений по сегментам не найдено: модель откалибрована ровно.\n")

    if binary:
        d = ctx["dec"]
        L.append("## 7. Ценность решения\n")
        L.append(
            f"**Предположения (замените реальными):** поймать настоящий случай = **+{fmt(d['gain_tp'])}**, лишнее действие = **-{fmt(d['cost_fp'])}**.\n"
        )
        L.append(
            f"- Лучший порог: **{d['best_threshold']:.2f}**, действуем на **{d['share_acted']:.0%}** клиентов; точность {d['precision']:.0%}, охват {d['recall']:.0%}.\n"
            f"- Чистая ценность: **{fmt(d['best_value'])}** на {d['n']} клиентов (**{fmt(d['per_1000'])}** на 1000).\n"
            f"- Для сравнения: действовать на всех = {fmt(d['value_all'])}; случайно на ту же долю = {fmt(d['value_random_same_budget'])}.\n"
        )
        lt = ctx["lift"]
        L.append(f"- Верхние 10% по скору содержат **{lt['rate'].iloc[0]:.1%}** целевых (lift x{lt['lift'].iloc[0]:.1f}), верхние 20% ловят **{lt['cum_capture'].iloc[min(1, len(lt) - 1)]:.0%}** всех случаев.\n")
        L.append("![gain](charts/05_gain.png)\n\n![profit](charts/06_profit.png)\n")

    if ctx.get("drift"):
        dr = ctx["drift"]
        L.append("## 8. Совпадают ли train и test\n")
        verdict = "почти не отличаются" if dr["auc"] < 0.6 else ("заметно отличаются" if dr["auc"] < 0.75 else "сильно отличаются")
        L.append(f"Adversarial AUC = **{dr['auc']:.3f}**: таблицы {verdict}.\n")
        if dr["auc"] >= 0.6:
            L.append("Больше всего различаются: " + ", ".join(f"{disp.get(k, k)} ({v:.0%})" for k, v in dr["top_features"].items()) + ".\n")
            L.append("Метрики на train будут оптимистичными. Рассмотрите валидацию, похожую на test, и исключение сдвинутых признаков.\n")

    L.append("## Что делать дальше (руками)\n")
    L.append(
        "1. Выберите 1-2 сегмента с `stable = да`, которые можно объяснить бизнесу, и проверьте их на здравый смысл.\n"
        "2. Проверьте подозрительные колонки у организаторов.\n"
        "3. Улучшайте признаки там, где мало данных или модель слепа.\n"
        "4. Графики из `charts/` уже готовы для слайдов. Подправьте заголовки под ваш вывод.\n"
    )
    path = out / "report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
