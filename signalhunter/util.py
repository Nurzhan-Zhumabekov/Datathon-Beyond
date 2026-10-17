"""Loading data, guessing column kinds, preparing the target."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ID_HINT = re.compile(r"(^id$|_id$|^id_|uuid|guid|^index$|^idx$)", re.I)
LEAK_HINT = re.compile(
    r"(target|label|outcome|result|churn|default|fraud|status|closed|closure|cancel|resolved|future)", re.I
)
_DATE_RE = re.compile(r"^\s*\d{1,4}[-./]\d{1,2}[-./]\d{1,4}([ T]\d{1,2}:\d{2}(:\d{2})?)?")
_DAYFIRST_RE = re.compile(r"^\s*\d{1,2}[./]\d{1,2}[./]\d{4}")
_DECIMAL_COMMA_RE = re.compile(r"^\s*-?\d+(,\d+)?\s*$")


def _sniff_sep(path: Path) -> str:
    with open(path, "rb") as f:
        head = f.read(65536)
    try:
        text = head.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = head.decode("cp1251", errors="replace")
    first = text.splitlines()[0] if text else ""
    cands = {",": first.count(","), ";": first.count(";"), "\t": first.count("\t"), "|": first.count("|")}
    return max(cands, key=cands.get)


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(p)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(p)
    sep = _sniff_sep(p)
    try:
        return pd.read_csv(p, sep=sep, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(p, sep=sep, encoding="cp1251", low_memory=False)


def fix_decimal_commas(df: pd.DataFrame) -> pd.DataFrame:
    """'1,5' -> 1.5: a classic trap in regional CSV exports."""
    for c in df.columns:
        s = df[c]
        if s.dtype != object:
            continue
        samp = s.dropna().astype(str).head(300)
        if len(samp) == 0:
            continue
        ok = samp.map(lambda v: bool(_DECIMAL_COMMA_RE.match(v)))
        if ok.mean() > 0.97 and samp.str.contains(",").any():
            fixed = pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False).str.strip(), errors="coerce")
            df[c] = fixed.where(s.notna())
    return df


def to_datetime_smart(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    samp = s.dropna().astype(str).head(200)
    dayfirst = len(samp) > 0 and samp.map(lambda v: bool(_DAYFIRST_RE.match(v))).mean() > 0.9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=dayfirst)


def _looks_datetime(s: pd.Series) -> bool:
    samp = s.dropna().astype(str).head(200)
    if len(samp) == 0 or samp.map(lambda v: bool(_DATE_RE.match(v))).mean() < 0.9:
        return False
    return to_datetime_smart(samp).notna().mean() > 0.95


def infer_kinds(df: pd.DataFrame, id_cols=()) -> dict[str, str]:
    """numeric | categorical | datetime | id | text | constant"""
    kinds: dict[str, str] = {}
    n = len(df)
    for c in df.columns:
        s = df[c]
        nun = s.nunique(dropna=True)
        if c in id_cols:
            kinds[c] = "id"
        elif nun <= 1:
            kinds[c] = "constant"
        elif pd.api.types.is_datetime64_any_dtype(s):
            kinds[c] = "datetime"
        elif pd.api.types.is_bool_dtype(s):
            kinds[c] = "categorical"
        elif pd.api.types.is_numeric_dtype(s):
            if ID_HINT.search(str(c)) and nun > 0.5 * n:
                kinds[c] = "id"
            elif pd.api.types.is_integer_dtype(s) and nun > 0.98 * n:
                kinds[c] = "id"
            else:
                kinds[c] = "numeric"
        elif _looks_datetime(s):
            kinds[c] = "datetime"
        else:
            avg_len = s.dropna().astype(str).str.len().mean()
            if ID_HINT.search(str(c)) and nun > 0.5 * n:
                kinds[c] = "id"
            elif nun > 0.9 * n:
                kinds[c] = "id" if avg_len < 40 else "text"
            elif avg_len > 40 and nun > 0.5 * n:
                kinds[c] = "text"
            else:
                kinds[c] = "categorical"
    return kinds


def _guess_positive(labels: list[str]) -> str | None:
    pos_like = {"1", "1.0", "true", "yes", "y", "да"}
    hits = [l for l in labels if l.strip().lower() in pos_like]
    return hits[0] if len(hits) == 1 else None


def prepare_target(y: pd.Series, positive=None):
    """Return (numeric target with NaN for missing, task, info). Task is 'binary' or 'regression'."""
    if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y) and y.nunique() > 10:
        return y.astype(float), "regression", {}
    ys = y.astype(str).where(y.notna())
    vc = ys.value_counts()
    if len(vc) < 2:
        raise SystemExit("В целевой колонке меньше двух разных значений.")
    if positive is not None:
        pos = str(positive)
        if pos not in vc.index:
            raise SystemExit(f"--positive {pos!r} нет среди значений цели: {list(vc.index)[:10]}")
    else:
        pos = _guess_positive(list(vc.index)) or vc.idxmin()
    info = {"positive": pos, "classes": list(vc.index), "n_classes": len(vc), "collapsed": len(vc) > 2}
    yb = (ys == pos).astype(float).where(ys.notna())
    return yb, "binary", info


def time_array(s: pd.Series) -> np.ndarray:
    d = to_datetime_smart(s)
    v = d.astype("int64").to_numpy().astype(float)
    bad = d.isna().to_numpy()
    if bad.all():
        raise SystemExit("Не получилось распознать даты в колонке времени.")
    v[bad] = v[~bad].min()
    return v


def fmt(v) -> str:
    return f"{v:.4g}"
