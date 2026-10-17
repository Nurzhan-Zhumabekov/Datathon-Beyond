"""Synthetic bank-churn table with PLANTED findings, to check that Signal Hunter finds them.

Planted:
  * segment: region = Almaty & channel = online & age < 30 churns far more
  * protective: salary_project = True churns much less
  * leaky column `closure_reason`: only filled for churned clients (a post-outcome field)
  * shifted test set: more online clients, no target
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make(n: int, seed: int, shift: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    region = rng.choice(["Astana", "Almaty", "Shymkent", "Karaganda", "Aktobe", "Other"], n, p=[0.22, 0.25, 0.12, 0.1, 0.08, 0.23])
    ch_p = [0.25, 0.45, 0.2, 0.1] if shift else [0.4, 0.25, 0.25, 0.1]
    channel = rng.choice(["mobile", "online", "branch", "call"], n, p=ch_p)
    age = np.clip(rng.normal(38, 11, n), 18, 80).round()
    income = np.exp(rng.normal(12.3, 0.5, n)).round(-3)
    tenure = rng.integers(1, 120, n)
    n_products = rng.integers(1, 6, n)
    last_login = rng.exponential(25, n).round()
    salary_project = rng.random(n) < 0.35
    complaints = rng.poisson(0.3, n)
    signup = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 1000, n), unit="D")

    logit = (
        -2.3
        + 0.012 * (last_login - 25)
        - 0.30 * (n_products - 3)
        + 0.55 * (complaints > 0)
        - 1.1 * salary_project
        - 0.008 * (tenure - 60)
        + 1.9 * ((region == "Almaty") & (channel == "online") & (age < 30))
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame(
        {
            "client_id": [f"C{i:06d}" for i in range(n)],
            "signup_date": signup.strftime("%d.%m.%Y"),
            "age": age,
            "income": income,
            "region": region,
            "channel": channel,
            "tenure_months": tenure,
            "n_products": n_products,
            "last_login_days": last_login,
            "salary_project": salary_project,
            "complaints_90d": complaints,
            "churn": churn.astype(int),
        }
    )
    df["closure_reason"] = np.where(churn, rng.choice(["moved", "price", "service"], n), None)
    df.loc[rng.random(n) < 0.05, "income"] = np.nan
    df.loc[rng.random(n) < 0.02, "age"] = np.nan
    return df


if __name__ == "__main__":
    out = Path(__file__).parent
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    make(n, 1).to_csv(out / "demo_train.csv", index=False)
    test = make(max(n // 6, 500), 2, shift=True).drop(columns=["churn", "closure_reason"])
    test.to_csv(out / "demo_test.csv", index=False)
    print(f"written: {out / 'demo_train.csv'} ({n} rows), demo_test.csv ({len(test)} rows)")
