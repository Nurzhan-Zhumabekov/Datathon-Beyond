import os
import sys
import tempfile
from pathlib import Path

# Isolated data folder and key for every test run; never touches the live service's data/.
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="beyond-test-"))
os.environ.setdefault("API_KEY", "test-key")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
