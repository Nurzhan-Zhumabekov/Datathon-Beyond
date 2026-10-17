"""Leakage audit: how much can a single column predict the target on its own, out of fold?

A column that alone gives AUC ~1.0 is almost never a "great feature". It is a column that was
filled in after the outcome happened (closure reason, status, resolution date...).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .util import LEAK_HINT, to_datetime_smart

COLUMNS = ["column", "kind", "score", "metric", "level", "comment"]


def _to_num(s: pd.Series, kind: str) -> np.ndarray:
    if kind == "datetime":
        d = to_datetime_smart(s)
        return ((d - d.min()).dt.total_seconds() / 86400.0).to_numpy(float)
    return pd.to_numeric(s, errors="coerce").to_numpy(float)


def _tree_score(x: np.ndarray, y: np.ndarray, cv, task: str) -> float:
    isn = np.isnan(x)
    med = np.nanmedian(x) if (~isn).any() else 0.0
    z = np.column_stack([np.where(isn, med, x), isn.astype(float)])
    leaf = max(20, int(0.005 * len(y)))
    if task == "binary":
        m = DecisionTreeClassifier(max_depth=4, min_samples_leaf=leaf, random_state=0)
        p = cross_val_predict(m, z, y.astype(int), cv=cv, method="predict_proba")[:, 1]
        return float(roc_auc_score(y, p))
    m = DecisionTreeRegressor(max_depth=4, min_samples_leaf=leaf, random_state=0)
    p = cross_val_predict(m, z, y, cv=cv)
    return float(max(0.0, r2_score(y, p)))


def _te_score(s: pd.Series, y: np.ndarray, cv, task: str) -> float:
    codes = s.astype(str).where(s.notna(), "__nan__").to_numpy()
    enc = np.zeros(len(y))
    prior_all = y.mean()
    strat = y.astype(int) if task == "binary" else y
    for tr, va in cv.split(codes, strat):
        prior = y[tr].mean() if len(tr) else prior_all
        g = pd.Series(y[tr]).groupby(codes[tr]).agg(["sum", "count"])
        smooth = (g["sum"] + 10 * prior) / (g["count"] + 10)
        enc[va] = pd.Series(codes[va]).map(smooth).fillna(prior).to_numpy()
    if task == "binary":
        return float(roc_auc_score(y, enc))
    c = np.corrcoef(enc, y)[0, 1]
    return float(0 if np.isnan(c) else c**2)


def univariate_power(df: pd.DataFrame, y: np.ndarray, kinds: dict[str, str], task: str, seed=0, max_rows=50000):
    n = len(df)
    if n > max_rows:
        idx = np.random.default_rng(seed).choice(n, max_rows, replace=False)
        df, y = df.iloc[idx], y[idx]
    if task == "binary":
        minority = int(min((y == 1).sum(), (y == 0).sum()))
        if minority < 6:
            return pd.DataFrame(columns=COLUMNS)
        cv = StratifiedKFold(max(2, min(3, minority // 2)), shuffle=True, random_state=seed)
        hi, mid, metric = 0.97, 0.85, "AUC"
    else:
        cv = KFold(3, shuffle=True, random_state=seed)
        hi, mid, metric = 0.95, 0.70, "R2"

    rows = []
    for c, kind in kinds.items():
        if kind not in ("numeric", "categorical", "datetime"):
            continue
        s = df[c]
        try:
            score = _te_score(s, y, cv, task) if kind == "categorical" else _tree_score(_to_num(s, kind), y, cv, task)
        except Exception:
            continue
        name_hint = bool(LEAK_HINT.search(str(c)))
        if score >= hi:
            level, comment = "critical", "почти один предсказывает цель: скорее всего утечка (колонка заполняется после события)"
        elif score >= mid:
            level, comment = "suspect", "слишком сильный одиночный признак, проверьте происхождение колонки"
        elif name_hint and score >= (0.6 if task == "binary" else 0.3):
            level, comment = "suspect", "название намекает на результат, и признак заметно предсказывает цель"
        else:
            level, comment = "ok", ""
        rows.append({"column": c, "kind": kind, "score": round(score, 4), "metric": metric, "level": level, "comment": comment})
    out = pd.DataFrame(rows, columns=COLUMNS)
    return out.sort_values("score", ascending=False).reset_index(drop=True)
