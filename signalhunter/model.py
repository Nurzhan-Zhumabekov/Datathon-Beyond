"""Honest baselines: features, validation scheme, LightGBM vs a simple linear model."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .util import fmt, to_datetime_smart


def _as_str(s: pd.Series) -> pd.Series:
    return s.astype(str).where(s.notna())


class Features:
    """Turns raw columns into model-ready ones. Uses no target, so fitting on all rows is leak-free."""

    def __init__(self, kinds: dict[str, str], max_cat: int = 40):
        self.kinds = {c: k for c, k in kinds.items() if k in ("numeric", "categorical", "datetime")}
        self.max_cat = max_cat
        self.cats: dict[str, list] = {}
        self.freq: dict[str, dict] = {}
        self.safe: dict[str, str] = {}
        self.display: dict[str, str] = {}

    def fit(self, df: pd.DataFrame):
        used: set[str] = set()
        for c, k in self.kinds.items():
            base = re.sub(r"[^\w]+", "_", str(c)).strip("_") or "col"
            name, i = base, 1
            while name in used:
                i += 1
                name = f"{base}_{i}"
            used.add(name)
            self.safe[c] = name
            if k == "categorical":
                vc = _as_str(df[c]).value_counts()
                if len(vc) <= self.max_cat:
                    self.cats[c] = list(vc.index)
                else:
                    self.freq[c] = (vc / vc.sum()).to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out: dict[str, object] = {}
        for c, k in self.kinds.items():
            nm = self.safe[c]
            s = df[c] if c in df.columns else pd.Series(np.nan, index=df.index)
            if k == "numeric":
                out[nm] = pd.to_numeric(s, errors="coerce").astype(float)
                self.display[nm] = str(c)
            elif k == "categorical":
                sv = _as_str(s)
                if c in self.cats:
                    out[nm] = pd.Categorical(sv, categories=self.cats[c])
                    self.display[nm] = str(c)
                else:
                    out[nm + "_freq"] = sv.map(self.freq[c]).astype(float)
                    self.display[nm + "_freq"] = f"{c} (частота)"
            else:
                d = to_datetime_smart(s)
                for part, label in (("month", "месяц"), ("dayofweek", "день недели"), ("day", "число")):
                    out[f"{nm}_{part}"] = getattr(d.dt, part).astype(float)
                    self.display[f"{nm}_{part}"] = f"{c} ({label})"
                if d.dt.hour.nunique() > 1:
                    out[nm + "_hour"] = d.dt.hour.astype(float)
                    self.display[nm + "_hour"] = f"{c} (час)"
        return pd.DataFrame(out, index=df.index)


def lgbm_params(fast: bool) -> dict:
    return dict(
        n_estimators=200 if fast else 400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def make_lgbm(task: str, fast: bool):
    return LGBMClassifier(**lgbm_params(fast)) if task == "binary" else LGBMRegressor(**lgbm_params(fast))


def _predict(m, X, task: str) -> np.ndarray:
    return m.predict_proba(X)[:, 1] if task == "binary" else m.predict(X)


def linear_frame(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].astype(object).where(out[c].notna(), "__nan__").astype(str)
    return out


def make_linear(task: str, X: pd.DataFrame):
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    parts = []
    if num:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), num))
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01), cat))
    est = LogisticRegression(max_iter=500, C=0.5) if task == "binary" else Ridge(alpha=10.0)
    return Pipeline([("pre", ColumnTransformer(parts)), ("est", est)])


def score(task: str, y: np.ndarray, p: np.ndarray) -> dict:
    if task == "binary":
        return {
            "roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "logloss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
            "brier": float(brier_score_loss(y, p)),
        }
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)),
    }


def make_splits(y, task, time_values=None, groups=None, n_splits=5, seed=42):
    n = len(y)
    if time_values is not None:
        order = np.argsort(time_values, kind="stable")
        sp = TimeSeriesSplit(n_splits=n_splits)
        return [(order[a], order[b]) for a, b in sp.split(order)], "по времени (обучение на прошлом, проверка на будущем)"
    if groups is not None:
        return list(GroupKFold(n_splits).split(np.zeros(n), y, groups)), "по группам (одна группа не попадает и в train, и в valid)"
    if task == "binary":
        return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(n), y.astype(int))), "стратифицированный K-Fold"
    return list(KFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(n))), "K-Fold"


def cross_validate(X: pd.DataFrame, y: np.ndarray, task: str, splits, fast=False) -> dict:
    n = len(y)
    oof = np.full(n, np.nan)
    oof_lin = np.full(n, np.nan)
    Xl = linear_frame(X)
    folds = {"lightgbm": [], "linear": []}
    for tr, va in splits:
        m = make_lgbm(task, fast).fit(X.iloc[tr], y[tr])
        p = _predict(m, X.iloc[va], task)
        oof[va] = p
        lin = make_linear(task, Xl).fit(Xl.iloc[tr], y[tr])
        pl = _predict(lin, Xl.iloc[va], task)
        oof_lin[va] = pl
        for name, pred in (("lightgbm", p), ("linear", pl)):
            try:
                folds[name].append(score(task, y[va], pred))
            except ValueError:
                pass  # a fold with one class only

    cov = ~np.isnan(oof)
    res = {"oof": oof, "oof_linear": oof_lin, "covered": cov, "summary": {}}
    for name, arr in (("lightgbm", oof), ("linear", oof_lin)):
        fs = pd.DataFrame(folds[name])
        pooled = score(task, y[cov], arr[cov])
        res["summary"][name] = {
            k: {"mean": float(fs[k].mean()) if len(fs) else float("nan"), "std": float(fs[k].std()) if len(fs) > 1 else 0.0, "pooled": v}
            for k, v in pooled.items()
        }
    # what a model with no knowledge would get
    if task == "binary":
        res["naive"] = {"roc_auc": 0.5, "pr_auc": float(y[cov].mean())}
    else:
        res["naive"] = {"mae": float(np.mean(np.abs(y[cov] - np.median(y[cov])))), "r2": 0.0}
    return res


def fit_full(X: pd.DataFrame, y: np.ndarray, task: str, fast=False):
    m = make_lgbm(task, fast).fit(X, y)
    samp = X.sample(min(len(X), 5000), random_state=0)
    contrib = m.booster_.predict(samp, pred_contrib=True)
    contrib = np.asarray(contrib)[:, : X.shape[1]]
    imp = pd.Series(np.abs(contrib).mean(axis=0), index=X.columns).sort_values(ascending=False)
    return m, imp


def predict_model(m, X, task):
    return _predict(m, X, task)


def feature_effect(x: pd.Series, y: np.ndarray, oof: np.ndarray, max_bins: int = 8) -> pd.DataFrame:
    d = pd.DataFrame({"x": x.reset_index(drop=True), "y": y, "p": oof})
    d = d[d["p"].notna()]
    if isinstance(x.dtype, pd.CategoricalDtype):
        key = d["x"].astype(object).where(d["x"].notna(), "пропуск").astype(str)
        top = key.value_counts().head(10).index
        d, key = d[key.isin(top)], key[key.isin(top)]
        g = d.groupby(key).agg(n=("y", "size"), actual=("y", "mean"), pred=("p", "mean"))
        out = g.sort_values("actual", ascending=False).reset_index()
        out.columns = ["label", "n", "actual", "pred"]
        return out
    xv = pd.to_numeric(d["x"], errors="coerce")
    miss = xv.isna()
    nun = xv.nunique()
    if nun <= max_bins:
        key = xv.map(lambda v: fmt(v) if pd.notna(v) else "пропуск")
        order = xv.groupby(key).min()
    else:
        try:
            b = pd.qcut(xv, q=max_bins, duplicates="drop")
        except ValueError:
            return pd.DataFrame(columns=["label", "n", "actual", "pred"])
        key = b.map(lambda iv: f"{fmt(iv.left)}…{fmt(iv.right)}" if pd.notna(iv) else "пропуск").astype(str)
        key = key.where(~miss, "пропуск")
        order = xv.groupby(key).min()
    g = d.groupby(key).agg(n=("y", "size"), actual=("y", "mean"), pred=("p", "mean"))
    g["_o"] = order.reindex(g.index).fillna(np.inf)
    g = g.sort_values("_o").drop(columns="_o")
    out = g.reset_index()
    out.columns = ["label", "n", "actual", "pred"]
    return out
