"""End-to-end check on data with planted findings. Run: uv run pytest -q"""

import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from make_demo_data import make  # noqa: E402

from signalhunter import cli  # noqa: E402


def _args(data, out, **kw):
    base = dict(
        data=str(data), target="churn", test=None, time_col=None, group_col=None, id_cols="client_id", drop=None,
        positive=None, gain_tp=10.0, cost_fp=1.0, keep_leaky=False, fast=True, max_rows=300000, out=str(out),
    )
    base.update(kw)
    return Namespace(**base)


def test_finds_planted_signals(tmp_path):
    csv = tmp_path / "train.csv"
    make(12000, 1).to_csv(csv, index=False)
    s = cli.run(_args(csv, tmp_path / "out"))

    assert "closure_reason" in s["excluded_leaky"]  # post-outcome column is caught
    assert s["metrics"]["lightgbm"]["roc_auc"] > 0.6
    rules = " | ".join(s["stable_segments"])
    assert "salary_project" in rules  # main protective effect
    seg = (tmp_path / "out" / "segments.csv").read_text(encoding="utf-8-sig")
    assert "region = Almaty" in seg and "channel = online" in seg  # planted interaction
    assert (tmp_path / "out" / "report.md").exists()
    prompt = (tmp_path / "out" / "story_prompt.md").read_text(encoding="utf-8")
    assert "ОТЧЁТ" in prompt and "![" not in prompt and "C0000" not in prompt  # aggregates only, no raw ids
    assert (tmp_path / "out" / "charts" / "06_profit.png").exists()


def test_time_split_and_test_set(tmp_path):
    csv, tcsv = tmp_path / "train.csv", tmp_path / "test.csv"
    make(8000, 3).to_csv(csv, index=False)
    make(1500, 4, shift=True).drop(columns=["churn", "closure_reason"]).to_csv(tcsv, index=False)
    s = cli.run(_args(csv, tmp_path / "out", time_col="signup_date", test=str(tcsv)))
    assert "времени" in s["scheme"]
    assert s["adversarial_auc"] > 0.55  # shifted channel mix is detected
    assert (tmp_path / "out" / "predictions_test.csv").exists()
