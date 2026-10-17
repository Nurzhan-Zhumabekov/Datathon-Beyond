import os
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
JOB_DIR = DATA_DIR / "jobs"
ENV_FILE = ROOT / ".env"


def _load_env_file() -> None:
    """Minimal .env loader (KEY=VALUE lines) so we don't need python-dotenv."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _ensure_api_key() -> str:
    """Read API_KEY from env/.env, or generate one and persist it to .env."""
    key = os.environ.get("API_KEY")
    if key:
        return key
    key = "fb-" + secrets.token_urlsafe(24)
    with ENV_FILE.open("a") as f:
        f.write(f"API_KEY={key}\n")
    os.environ["API_KEY"] = key
    return key


_load_env_file()
API_KEY = _ensure_api_key()

# Model runs locally through MLX (Apple Silicon) from weights stored in ./models — no internet needed.
VLM_MODEL = os.environ.get("VLM_MODEL", str(ROOT / "models" / "Qwen3-VL-4B-Instruct-4bit"))
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", str(ROOT / "models" / "Qwen3-VL-8B-Instruct-4bit"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Jury limit is 5 minutes per video including queueing; keep our own budget well below it.
TIME_BUDGET_SEC = float(os.environ.get("TIME_BUDGET_SEC", "150"))
JURY_LIMIT_SEC = 270.0  # 5 min minus a safety margin for upload and polling

# Delete videos uploaded through the jury API right after processing (test videos must not be kept).
KEEP_UI_VIDEOS_SEC = 3600

for d in (UPLOAD_DIR, JOB_DIR):
    d.mkdir(parents=True, exist_ok=True)
