"""Job files must be UTF-8 on every OS: the analysis log contains characters such as the arrow, which the
Windows default encoding (cp1251) cannot store. That used to fail every job on Windows. Run: python -m pytest -q tests"""
import importlib
import os
import sys
import tempfile
from pathlib import Path

os.environ["API_KEY"] = "test"
os.environ["DATA_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

config = importlib.import_module("app.config")
jobs = importlib.import_module("app.jobs")


def _job(job_id="enc1"):
    return {"job_id": job_id, "status": "processing", "source": "api", "created_at": 1.0, "stage": "в очереди",
            "log": [{"t": 1.0, "src": "8B", "text": "Эскалация → эксперт 8B: подтверждено ≥ 50%"}]}


def test_job_file_is_utf_8_with_arrow_and_cyrillic():
    jobs._save(_job())
    text = (config.JOB_DIR / "enc1.json").read_bytes().decode("utf-8")
    assert "→" in text and "Эскалация" in text


def test_job_survives_reload_from_disk():
    jobs._save(_job("enc2"))
    jobs._jobs.pop("enc2", None)
    assert jobs.get_job("enc2")["log"][0]["text"].startswith("Эскалация →")


def test_disk_error_does_not_kill_the_worker(monkeypatch):
    jobs._jobs["enc3"] = _job("enc3")

    def boom(job):
        raise OSError("disk full")

    monkeypatch.setattr(jobs, "_save", boom)
    jobs.update_job("enc3", status="failed", error="x")  # must not raise
    assert jobs._jobs["enc3"]["status"] == "failed"


def test_job_id_with_path_characters_is_rejected():
    # The id comes from the URL of unauthenticated routes: backslashes and drive or UNC paths must never reach the filesystem.
    bs = chr(92)  # a backslash, built here so that no escape sequence can be mangled in the source
    for bad in (".." + bs + "secret", "../secret", "a/b", bs * 2 + "host" + bs + "share" + bs + "x", "C:" + bs + "Windows", "", "x" * 65):
        assert jobs.get_job(bad) is None
        try:
            jobs._job_path(bad)
        except ValueError:
            continue
        raise AssertionError(f"_job_path accepted {bad!r}")


def test_normal_job_ids_still_work():
    assert jobs._job_path("3f2a9c1e0b11").name == "3f2a9c1e0b11.json"
    assert jobs._job_path("enc_2-x").name == "enc_2-x.json"
