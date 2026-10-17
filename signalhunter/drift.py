"""Adversarial validation: can a model tell the train table from the test table?

AUC ~0.5 means they look alike. AUC well above 0.5 means the test set is shifted, and a
validation score measured on train will be too optimistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def adversarial_validation(X_train: pd.DataFrame, X_test: pd.DataFrame, max_rows: int = 40000, seed: int = 0) -> dict:
    a = X_train.sample(min(len(X_train), max_rows), random_state=seed)
    b = X_test.sample(min(len(X_test), max_rows), random_state=seed)
    X = pd.concat([a, b], ignore_index=True)
    for c in X.columns:  # categories must be consistent after concat
        if isinstance(a[c].dtype, pd.CategoricalDtype):
            X[c] = pd.Categorical(X[c], categories=a[c].cat.categories)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))].astype(int)
    m = LGBMClassifier(n_estimators=120, learning_rate=0.1, num_leaves=15, min_child_samples=40, verbose=-1, random_state=seed, n_jobs=-1)
    p = cross_val_predict(m, X, y, cv=StratifiedKFold(3, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, p))
    m.fit(X, y)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    imp = imp / imp.sum() if imp.sum() else imp
    return {"auc": auc, "top_features": imp.head(5)}
