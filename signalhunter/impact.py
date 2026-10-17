"""From a score to money: who to act on, and what it is worth.

Value model (deliberately simple, so the jury can follow it):
    act on a client with score >= t
    caught a real positive  -> + gain_tp   (avoided loss / retained revenue)
    acted on a negative     -> - cost_fp   (cost of the call, offer, friction)
    not acting              -> 0
gain_tp and cost_fp are ASSUMPTIONS. Replace them with real numbers from the organisers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def decision_analysis(y: np.ndarray, p: np.ndarray, gain_tp: float, cost_fp: float) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    P = int(y.sum())
    Nn = int(len(y) - P)
    ths = np.linspace(0.005, 0.995, 199)
    vals = np.empty_like(ths)
    for i, t in enumerate(ths):
        act = p >= t
        tp = int((act & (y == 1)).sum())
        fp = int((act & (y == 0)).sum())
        vals[i] = tp * gain_tp - fp * cost_fp
    j = int(vals.argmax())
    t = float(ths[j])
    act = p >= t
    tp = int((act & (y == 1)).sum())
    fp = int((act & (y == 0)).sum())
    n = len(y)
    all_value = P * gain_tp - Nn * cost_fp
    frac = float(act.mean())
    random_value = frac * all_value
    return {
        "thresholds": ths,
        "values": vals,
        "best_threshold": t,
        "best_value": float(vals[j]),
        "value_all": float(all_value),
        "value_random_same_budget": float(random_value),
        "share_acted": frac,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(P, 1),
        "per_1000": float(vals[j] / n * 1000),
        "n": n,
        "gain_tp": gain_tp,
        "cost_fp": cost_fp,
    }


def lift_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(p)}).sort_values("p", ascending=False).reset_index(drop=True)
    d["bin"] = (np.arange(len(d)) * bins // len(d)) + 1
    base = d["y"].mean()
    g = d.groupby("bin").agg(n=("y", "size"), positives=("y", "sum"), rate=("y", "mean"))
    g["lift"] = g["rate"] / base if base else np.nan
    g["cum_capture"] = g["positives"].cumsum() / max(d["y"].sum(), 1)
    g["cum_share"] = g["n"].cumsum() / len(d)
    return g.reset_index()
