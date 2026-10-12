import pytest
from fastapi.testclient import TestClient

from api import create_app
from conftest import FakeSummarizer
from summarizer import ModelNotAvailableError

DIALOGUE = "Hannah: Are you free this afternoon?\nAmanda: Yes, why?\nHannah: Let's meet at 4pm."


@pytest.fixture
def client(fake_summarizer):
    with TestClient(create_app(fake_summarizer)) as c:
        yield c


def test_health_reports_loaded_model(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy", "model_loaded": True, "device": "cpu"}


def test_health_before_model_load():
    with TestClient(create_app(FakeSummarizer(), load_on_startup=False)) as c:
        assert c.get("/health").json()["status"] == "loading"


def test_summarize_returns_summary_and_word_counts(client, fake_summarizer):
    resp = client.post("/summarize", json={"text": DIALOGUE})
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"] == "hannah and amanda will meet at 4pm."
    assert body["input_words"] == len(DIALOGUE.split())
    assert body["summary_words"] == 7
    assert fake_summarizer.calls[0] == {"text": DIALOGUE, "max_length": 150, "num_beams": 4}


def test_generation_params_are_forwarded(client, fake_summarizer):
    resp = client.post("/summarize", json={"text": DIALOGUE, "max_length": 60, "num_beams": 2})
    assert resp.status_code == 200
    assert fake_summarizer.calls[0]["max_length"] == 60
    assert fake_summarizer.calls[0]["num_beams"] == 2


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": ""},
        {"text": "   "},
        {"text": 123},
        {"text": "x" * 20_001},
        {"text": DIALOGUE, "num_beams": 0},
        {"text": DIALOGUE, "max_length": 1000},
    ],
)
def test_invalid_payloads_are_rejected(client, fake_summarizer, payload):
    resp = client.post("/summarize", json=payload)
    assert resp.status_code == 422
    assert fake_summarizer.calls == []


def test_text_empty_after_cleaning_returns_400():
    app = create_app(FakeSummarizer(error=ValueError("Input text is empty after cleaning.")))
    with TestClient(app) as c:
        resp = c.post("/summarize", json={"text": "<br>"})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Input text is empty after cleaning."


def test_missing_model_returns_503():
    with TestClient(create_app(FakeSummarizer(error=ModelNotAvailableError("lfs pointer")))) as c:
        resp = c.post("/summarize", json={"text": DIALOGUE})
    assert resp.status_code == 503


def test_inference_failure_returns_500_without_leaking_details():
    with TestClient(create_app(FakeSummarizer(error=RuntimeError("CUDA out of memory")))) as c:
        resp = c.post("/summarize", json={"text": DIALOGUE})
    assert resp.status_code == 500
    assert "CUDA" not in resp.text


def test_index_serves_web_ui(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "SummarAI" in resp.text
    assert client.get("/static/script.js").status_code == 200
