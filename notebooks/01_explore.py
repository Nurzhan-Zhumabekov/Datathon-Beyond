# %% [markdown]
# # Ручное исследование
# Открывайте как ноутбук в VS Code (ячейки `# %%`). Файл .py, а не .ipynb: так в git нет конфликтов.
# У каждого свой файл: скопируйте этот в `notebooks/<имя>_explore.py`.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

import numpy as np
import pandas as pd

from signalhunter import leakage, model, profile, segments
from signalhunter.util import fix_decimal_commas, infer_kinds, prepare_target, read_table, to_datetime_smart

DATA = "examples/demo_train.csv"  # <- путь к данным
TARGET = "churn"  # <- целевая колонка

df = fix_decimal_commas(read_table(DATA))
y_s, task, info = prepare_target(df[TARGET])
df = df.loc[y_s.notna()].reset_index(drop=True)
y = y_s.loc[y_s.notna()].to_numpy(float)
kinds = infer_kinds(df.drop(columns=[TARGET]))
for c, k in kinds.items():
    if k == "datetime":
        df[c] = to_datetime_smart(df[c])
print(task, info)
profile.column_profile(df.drop(columns=[TARGET]), kinds)

# %% [markdown]
# ## Утечки: какая колонка в одиночку подозрительно хорошо предсказывает цель?

# %%
leakage.univariate_power(df, y, kinds, task).head(10)

# %% [markdown]
# ## Свой baseline: меняйте признаки и смотрите метрику

# %%
use = {c: ("leak" if c in ("closure_reason",) else k) for c, k in kinds.items()}  # <- что исключить
fb = model.Features(use).fit(df)
X = fb.transform(df)
splits, scheme = model.make_splits(y, task)
cv = model.cross_validate(X, y, task, splits, fast=True)
print(scheme)
pd.DataFrame({k: {m: v["mean"] for m, v in s.items()} for k, s in cv["summary"].items()})

# %% [markdown]
# ## CatBoost как альтернатива LightGBM (хорош на категориях)

# %%
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

Xc = df[[c for c, k in use.items() if k in ("numeric", "categorical")]].copy()
cat_cols = [c for c in Xc.columns if use[c] == "categorical"]
for c in cat_cols:
    Xc[c] = Xc[c].astype(str)
Xtr, Xva, ytr, yva = train_test_split(Xc, y, test_size=0.25, random_state=0, stratify=y)
cb = CatBoostClassifier(iterations=300, learning_rate=0.08, verbose=0, cat_features=cat_cols, random_seed=0)
cb.fit(Xtr, ytr)
print("CatBoost AUC:", round(roc_auc_score(yva, cb.predict_proba(Xva)[:, 1]), 3))

# %% [markdown]
# ## Свои сегменты: гипотеза в одну строку

# %%
mask = (df["region"] == "Almaty") & (df["channel"] == "online")  # <- ваша гипотеза
print(f"n={mask.sum()}  доля цели в сегменте={y[mask].mean():.1%}  в среднем={y.mean():.1%}")
