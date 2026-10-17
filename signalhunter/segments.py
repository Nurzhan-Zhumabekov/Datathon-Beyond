"""Subgroup discovery with a replication check.

Two kinds of findings:
  * group        - a condition or a pair of conditions where the target differs a lot from average
  * interaction  - a pair or triple whose joint effect is much stronger than the sum of its parts
                   ("region = Almaty" and "channel = online" alone: +2 pts each; together: +7 pts)

A segment is marked `stable` only if the effect has the same sign and is still significant in BOTH
halves of the data (random halves, or older/newer halves when a time column is given). That keeps
us from pitching noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .util import fmt

COLUMNS = ["kind", "rule", "n", "support", "mean", "base", "lift", "z", "expected", "stable", "mean_A", "mean_B"]
GROUP, INTERACTION = "группа", "взаимодействие"


@dataclass
class Item:
    col: str
    label: str
    mask: np.ndarray


def _interval_label(col: str, iv: pd.Interval, first: bool, last: bool) -> str:
    if first:
        return f"{col} <= {fmt(iv.right)}"
    if last:
        return f"{col} > {fmt(iv.left)}"
    return f"{col} in ({fmt(iv.left)}, {fmt(iv.right)}]"


def build_items(df: pd.DataFrame, kinds: dict[str, str], min_support: int, n_bins: int = 5, max_card: int = 25) -> list[Item]:
    items: list[Item] = []
    for c, k in kinds.items():
        if k not in ("categorical", "numeric") or c not in df.columns:
            continue
        s = df[c]
        miss = s.isna().to_numpy()
        if k == "categorical":
            sv = s.astype(str).where(s.notna())
            for v, cnt in sv.value_counts().head(max_card).items():
                if cnt >= min_support:
                    items.append(Item(c, f"{c} = {v}", (sv == v).to_numpy()))
        else:
            x = pd.to_numeric(s, errors="coerce")
            if x.nunique() <= 6:
                for v, cnt in x.value_counts().items():
                    if cnt >= min_support:
                        items.append(Item(c, f"{c} = {fmt(v)}", (x == v).to_numpy()))
            else:
                try:
                    b = pd.qcut(x, q=n_bins, duplicates="drop")
                except ValueError:
                    continue
                cats = list(b.cat.categories)
                for i, iv in enumerate(cats):
                    m = (b == iv).to_numpy()
                    if m.sum() >= min_support:
                        items.append(Item(c, _interval_label(c, iv, i == 0, i == len(cats) - 1), m))
        if miss.sum() >= min_support:
            items.append(Item(c, f"{c} пропущен", miss))
    return items


def _z(mean, ref, s, n, N):
    se = s * np.sqrt(np.maximum(1.0 / np.maximum(n, 1) - 1.0 / N, 1e-12))
    return (mean - ref) / se


def find_segments(
    items: list[Item],
    values: np.ndarray,
    *,
    min_support: int,
    halves: tuple[np.ndarray, np.ndarray],
    effect: str = "rel_mean",  # 'rel_mean': |dev| >= min_effect * |mean| ; 'std': |dev| >= min_effect * std
    min_effect: float = 0.25,
    direction: str = "both",  # 'both' | 'up'
    z_min: float = 4.5,
    top: int = 8,
    top_int: int = 6,
    pool: int = 300,
) -> pd.DataFrame:
    v = np.asarray(values, dtype=np.float32)
    N = len(v)
    m, s = float(v.mean()), float(v.std())
    if s == 0 or not items:
        return pd.DataFrame(columns=COLUMNS)
    min_abs = min_effect * (abs(m) if effect == "rel_mean" else s)

    # keep the most promising single conditions so the matrices stay small
    cnt1 = np.array([it.mask.sum() for it in items], dtype=float)
    mean1 = np.array([v[it.mask].mean() if it.mask.any() else m for it in items])
    z1 = _z(mean1, m, s, cnt1, N)
    keep = [i for i in np.argsort(-np.abs(z1))[:pool] if cnt1[i] >= min_support]
    if not keep:
        return pd.DataFrame(columns=COLUMNS)
    items = [items[i] for i in keep]
    cnt1, mean1, z1 = cnt1[keep], mean1[keep], z1[keep]
    K = len(items)
    M = np.column_stack([it.mask for it in items]).astype(np.float32)
    cols = np.array([it.col for it in items])
    dev1 = mean1 - m

    cnt2 = M.T @ M
    mean2 = ((M * v[:, None]).T @ M) / np.maximum(cnt2, 1)
    dev2 = mean2 - m
    z2 = _z(mean2, m, s, cnt2, N)
    exp2 = m + dev1[:, None] + dev1[None, :]  # what we'd expect if the two effects just added up
    int2 = mean2 - exp2
    zi2 = _z(mean2, exp2, s, cnt2, N)
    valid2 = (cnt2 >= min_support) & (cols[:, None] != cols[None, :]) & np.triu(np.ones((K, K), dtype=bool), 1)

    groups: list[tuple] = []
    for i in range(K):
        if abs(z1[i]) >= z_min and abs(dev1[i]) >= min_abs and (direction == "both" or dev1[i] > 0):
            groups.append((abs(z1[i]), (i,), None))
    parent_max = np.maximum(np.abs(dev1)[:, None], np.abs(dev1)[None, :])
    okg = valid2 & (np.abs(z2) >= z_min) & (np.abs(dev2) >= min_abs) & (np.abs(dev2) >= 1.25 * parent_max)
    if direction == "up":
        okg &= dev2 > 0
    for a, b in np.argwhere(okg):
        groups.append((abs(z2[a, b]), (int(a), int(b)), None))

    inter: list[tuple] = []
    oki = (
        valid2
        & (np.abs(zi2) >= z_min)
        & (np.abs(z2) >= 3.0)
        & (np.abs(dev2) >= min_abs)
        & (np.abs(int2) >= 0.35 * np.abs(dev2))
        & (np.sign(int2) == np.sign(dev2))
    )
    if direction == "up":
        oki &= dev2 > 0
    for a, b in np.argwhere(oki):
        inter.append((abs(zi2[a, b]), (int(a), int(b)), float(exp2[a, b])))

    # triples: extend the 25 most "surprising" pairs by one more condition
    pairs = np.argwhere(valid2)
    if len(pairs) and top_int > 0:
        seeds = pairs[np.argsort(-np.abs(zi2[valid2]))[:25]]
        for a, b in seeds:
            sd = M[:, a] * M[:, b]
            c3 = sd @ M
            mean3 = ((sd * v) @ M) / np.maximum(c3, 1)
            exp3 = mean2[a, b] + dev1
            zi3 = _z(mean3, exp3, s, c3, N)
            dev3 = mean3 - m
            ok3 = (
                (c3 >= min_support)
                & (cols != cols[a])
                & (cols != cols[b])
                & (np.abs(zi3) >= z_min)
                & (np.abs(_z(mean3, m, s, c3, N)) >= 3.0)
                & (np.abs(dev3) >= min_abs)
                & (np.abs(mean3 - exp3) >= 0.5 * np.abs(dev3))
                & (np.sign(mean3 - exp3) == np.sign(dev3))
            )
            if direction == "up":
                ok3 &= dev3 > 0
            for k in np.flatnonzero(ok3):
                inter.append((abs(zi3[k]), (int(a), int(b), int(k)), float(exp3[k])))

    def pick(cands, limit):
        kept = []
        for c in sorted(cands, key=lambda t: -t[0])[:3000]:
            mask = M[:, c[1][0]] > 0
            for j in c[1][1:]:
                mask = mask & (M[:, j] > 0)
            dup = False
            for km, _ in kept:
                union = np.logical_or(mask, km).sum()
                if union and np.logical_and(mask, km).sum() / union > 0.6:
                    dup = True
                    break
            if not dup:
                kept.append((mask, c))
            if len(kept) >= limit:
                break
        return kept

    hA, hB = halves

    def row(mask, idx, kind, expected):
        n = int(mask.sum())
        mean = float(v[mask].mean())
        dev = mean - m
        hi = []
        for h in (hA, hB):
            nh, Nh = int((mask & h).sum()), int(h.sum())
            if nh < 10 or Nh < 20:
                hi.append((float("nan"), 0.0))
                continue
            mh, sh = float(v[h].mean()), float(v[h].std())
            meanh = float(v[mask & h].mean())
            zh = float(_z(meanh, mh, sh if sh > 0 else 1.0, nh, Nh))
            hi.append((meanh, zh if np.sign(meanh - mh) == np.sign(dev) else 0.0))
        return {
            "kind": kind,
            "rule": " & ".join(items[i].label for i in idx),
            "n": n,
            "support": n / N,
            "mean": mean,
            "base": m,
            "lift": mean / m if m else float("nan"),
            "z": float(_z(mean, m, s, n, N)),
            "expected": expected if expected is not None else float("nan"),
            "stable": bool(all(abs(z) >= 2.0 for _, z in hi)),
            "mean_A": hi[0][0],
            "mean_B": hi[1][0],
        }

    rows = [row(mk, c[1], GROUP, None) for mk, c in pick(groups, top)]
    seen = {r["rule"] for r in rows}
    for mk, c in pick(inter, top_int):
        r = row(mk, c[1], INTERACTION, c[2])
        if r["rule"] not in seen:
            rows.append(r)
    return pd.DataFrame(rows, columns=COLUMNS)


def make_halves(n: int, time_values: np.ndarray | None, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if time_values is not None:
        a = time_values <= np.median(time_values)
        return a, ~a
    a = np.random.default_rng(seed).random(n) < 0.5
    return a, ~a
