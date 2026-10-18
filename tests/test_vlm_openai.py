"""The 'openai' backend (llama.cpp / Ollama on this machine): payload, availability, loopback guard.
No model is used: a tiny fake server answers. Needs: pyyaml, pillow, numpy, httpx. Run: python -m pytest -q tests"""
import importlib
import json
import os
import sys
import tempfile
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

os.environ["API_KEY"] = "test"
os.environ["DATA_DIR"] = tempfile.mkdtemp()
sys.modules.setdefault("av", types.ModuleType("av"))  # decoding is not needed here
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

# app.pipeline re-exports the function `analyze`, which hides the submodule of the same name.
vlm = importlib.import_module("app.pipeline.vlm")
config = importlib.import_module("app.config")


class _Fake(BaseHTTPRequestHandler):
    seen: list[dict] = []
    reply = {"choices": [{"message": {"content": '{"ok": true}'}}]}
    status = 200
    statuses: list[int] = []  # consumed first, one per request, before falling back to `status`

    def log_message(self, *args):  # silence
        pass

    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200, {"data": [{"id": "qwen3-vl-4b"}]} if self.path == "/v1/models" else {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append(body)
        code = type(self).statuses.pop(0) if type(self).statuses else type(self).status
        self._send(code, type(self).reply if code == 200 else {"error": "boom"})


@pytest.fixture
def server(monkeypatch):
    _Fake.seen, _Fake.status, _Fake.statuses = [], 200, []
    _Fake.reply = {"choices": [{"message": {"content": '{"ok": true}'}}]}
    monkeypatch.setattr(vlm, "_RETRY_PAUSE_SEC", 0)
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    monkeypatch.setattr(config, "VLM_BACKEND", "openai")
    monkeypatch.setattr(config, "VLM_URL", url)
    monkeypatch.setattr(config, "JUDGE_URL", "")
    monkeypatch.setattr(vlm, "_reach", {})
    yield url
    srv.shutdown()


def _img(color=(10, 20, 30)):
    return Image.new("RGB", (32, 32), color)


def test_ask_sends_images_then_text_with_temperature_zero(server):
    out = vlm.ask([_img(), _img((200, 0, 0))], "how many people?", max_tokens=33)
    assert out == '{"ok": true}'
    body = _Fake.seen[0]
    parts = body["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["image_url", "image_url", "text"]
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert parts[2]["text"] == "how many people?"
    assert body["temperature"] == 0.0 and body["max_tokens"] == 33 and body["model"] == "qwen3-vl-4b"


def test_list_content_is_joined(server):
    _Fake.reply = {"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}]}
    assert vlm.ask([_img()], "q") == "ab"


def test_http_error_becomes_runtime_error(server):
    _Fake.status, _Fake.reply = 500, {"error": "boom"}
    with pytest.raises(RuntimeError, match="500"):
        vlm.ask([_img()], "q")


def test_scanner_is_available_and_judge_is_not_without_url(server):
    assert vlm.available("scan") is True
    assert vlm.available("judge") is False
    assert vlm.load("scan") is None


def test_judge_is_available_when_its_url_is_set(server, monkeypatch):
    monkeypatch.setattr(config, "JUDGE_URL", server)
    assert vlm.available("judge") is True


def test_unreachable_server_is_not_available_and_load_fails(monkeypatch):
    monkeypatch.setattr(config, "VLM_BACKEND", "openai")
    monkeypatch.setattr(config, "VLM_URL", "http://127.0.0.1:9")  # nothing listens on the discard port
    monkeypatch.setattr(vlm, "_reach", {})
    assert vlm.available("scan") is False
    with pytest.raises(RuntimeError, match="not reachable"):
        vlm.load("scan")


def test_non_local_url_is_refused_and_nothing_is_sent(monkeypatch):
    monkeypatch.setattr(config, "VLM_BACKEND", "openai")
    monkeypatch.setattr(config, "VLM_URL", "https://api.example.com")
    monkeypatch.setattr(vlm, "_reach", {})
    assert vlm.available("scan") is False
    with pytest.raises(RuntimeError, match="local"):
        vlm.ask([_img()], "q")


def test_one_server_error_is_retried(server):
    _Fake.statuses = [500]
    assert vlm.ask([_img()], "q") == '{"ok": true}'
    assert len(_Fake.seen) == 2


def test_two_server_errors_are_reported(server):
    _Fake.statuses = [500, 503]
    with pytest.raises(RuntimeError, match="503"):
        vlm.ask([_img()], "q")
    assert len(_Fake.seen) == 2


def test_refused_connection_is_retried_once_then_reported(monkeypatch):
    monkeypatch.setattr(config, "VLM_BACKEND", "openai")
    monkeypatch.setattr(config, "VLM_URL", "http://127.0.0.1:9")  # nothing listens on the discard port
    monkeypatch.setattr(vlm, "_RETRY_PAUSE_SEC", 0)
    with pytest.raises(RuntimeError, match="not answering"):
        vlm.ask([_img()], "q")


def test_compose_service_name_is_allowed_only_when_listed(monkeypatch):
    monkeypatch.setattr(config, "VLM_BACKEND", "openai")
    monkeypatch.setattr(config, "VLM_URL", "http://llama:8081")
    monkeypatch.delenv("VLM_ALLOWED_HOSTS", raising=False)
    with pytest.raises(RuntimeError, match="local"):
        vlm._local_url("scan")
    monkeypatch.setenv("VLM_ALLOWED_HOSTS", "llama")
    assert vlm._local_url("scan") == "http://llama:8081"


def test_dotted_hosts_are_never_allowed(monkeypatch):
    monkeypatch.setattr(config, "VLM_BACKEND", "openai")
    monkeypatch.setattr(config, "VLM_URL", "https://api.example.com")
    monkeypatch.setenv("VLM_ALLOWED_HOSTS", "api.example.com, llama")
    with pytest.raises(RuntimeError, match="local"):
        vlm._local_url("scan")


@pytest.mark.parametrize("entry", ["127.0.0.1", "8.8.8.8", "[::1]", "*", "llama.", "ллама", "llama.example.com"])
def test_odd_allowlist_entries_are_ignored(monkeypatch, entry):
    monkeypatch.setenv("VLM_ALLOWED_HOSTS", entry)
    assert vlm._allowed_hosts() == vlm._LOOPBACK


@pytest.mark.parametrize("url", ["http://localhost:8081", "http://127.0.0.1:8081", "http://[::1]:8081"])
def test_loopback_urls_are_allowed(monkeypatch, url):
    monkeypatch.setattr(config, "VLM_URL", url)
    assert vlm._local_url("scan") == url


@pytest.mark.parametrize("url", ["https://127.0.0.1:8081", "ftp://localhost", "localhost:8081", "//localhost"])
def test_non_http_schemes_are_refused(monkeypatch, url):
    monkeypatch.setattr(config, "VLM_URL", url)
    with pytest.raises(RuntimeError):
        vlm._local_url("scan")


def test_model_requests_ignore_proxy_environment(monkeypatch):
    import httpx
    seen = []
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setattr(config, "VLM_BACKEND", "openai")
    monkeypatch.setattr(config, "VLM_URL", "http://127.0.0.1:8081")
    monkeypatch.setattr(httpx, "get", lambda *a, **k: seen.append(k.get("trust_env")) or type("R", (), {"status_code": 200})())
    vlm._reach.clear()
    assert vlm._reachable("http://127.0.0.1:8081")
    assert seen == [False]
