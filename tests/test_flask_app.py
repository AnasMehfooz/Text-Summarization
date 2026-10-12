import pytest

import app as flask_module
from conftest import FakeSummarizer
from summarizer import ModelNotAvailableError


@pytest.fixture
def client(monkeypatch, fake_summarizer):
    fake_summarizer.load()
    monkeypatch.setattr(flask_module, "summarizer", fake_summarizer)
    return flask_module.app.test_client()


def test_health(client):
    assert client.get("/health").get_json() == {"status": "healthy", "model_loaded": True, "device": "cpu"}


def test_summarize(client):
    resp = client.post("/summarize", json={"text": "Tom: See you at 5?\nAnna: Sure."})
    assert resp.status_code == 200
    assert resp.get_json() == {"summary": "hannah and amanda will meet at 4pm."}


@pytest.mark.parametrize("payload", [None, {}, {"text": ""}, {"text": "  "}, {"text": 5}, {"text": "x" * 20_001}])
def test_bad_input_returns_400(client, payload):
    resp = client.post("/summarize", json=payload) if payload is not None else client.post("/summarize")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


@pytest.mark.parametrize(
    "error, code",
    [(ModelNotAvailableError("missing"), 503), (RuntimeError("boom"), 500), (ValueError("empty"), 400)],
)
def test_backend_errors_map_to_status_codes(monkeypatch, error, code):
    monkeypatch.setattr(flask_module, "summarizer", FakeSummarizer(error=error))
    resp = flask_module.app.test_client().post("/summarize", json={"text": "Tom: hi"})
    assert resp.status_code == code
