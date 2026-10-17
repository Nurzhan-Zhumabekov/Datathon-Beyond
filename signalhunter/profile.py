"""Column profile and plain-language warnings about the table."""

from __future__ import annotations

import numpy as np
import pandas as pd


def column_profile(df: pd.DataFrame, kinds: dict[str, str]) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        ex = s.dropna().iloc[0] if s.notna().any() else ""
        rows.append(
            {
                "column": c,
                "kind": kinds.get(c, "?"),
                "missing_%": round(100 * s.isna().mean(), 1),
                "n_unique": int(s.nunique()),
                "example": str(ex)[:30],
            }
        )
    return pd.DataFrame(rows)


def data_warnings(df, kinds, y, task, info, has_time_col) -> list[str]:
    w: list[str] = []
    dup = int(df.duplicated().sum())
    if dup:
        w.append(f"Дубликаты строк: {dup} ({dup / len(df):.1%}). Решите, это ошибка данных или повторные события.")
    ids = [c for c, k in kinds.items() if k == "id"]
    if ids:
        w.append(f"Похоже на идентификаторы, в модель не идут: {', '.join(ids[:8])}.")
    const = [c for c, k in kinds.items() if k == "constant"]
    if const:
        w.append(f"Константные колонки (без информации): {', '.join(const[:8])}.")
    text = [c for c, k in kinds.items() if k == "text"]
    if text:
        w.append(f"Текстовые колонки не используются в baseline: {', '.join(text[:8])}. Это шанс для TF-IDF или эмбеддингов.")
    hi_miss = [c for c in df.columns if c in kinds and df[c].isna().mean() > 0.4]
    if hi_miss:
        w.append(f"Больше 40% пропусков: {', '.join(hi_miss[:8])}. Пропуск может сам быть сигналом.")
    if task == "binary":
        rate = float(np.nanmean(y))
        if rate < 0.05 or rate > 0.95:
            w.append(f"Сильный дисбаланс: доля цели {rate:.2%}. Смотрите PR-AUC и порог решения, не accuracy.")
    if info.get("collapsed"):
        w.append(
            f"В цели {info['n_classes']} класса. Анализ сделан для класса «{info['positive']}» против остальных "
            f"(поменять: --positive)."
        )
    dts = [c for c, k in kinds.items() if k == "datetime"]
    if dts and not has_time_col:
        w.append(
            f"Есть даты ({', '.join(dts[:3])}), но валидация случайная. Если тест относится к будущему, "
            f"запустите с --time-col, иначе метрики будут оптимистичными."
        )
    return w
