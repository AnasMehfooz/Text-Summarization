import pytest


class FakeSummarizer:
    """Stands in for the T5 model so the web layers can be tested without torch or weights."""

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.is_loaded = False
        self.device = "not loaded"
        self.calls: list[dict] = []

    def load(self) -> None:
        self.is_loaded = True
        self.device = "cpu"

    def summarize(self, text: str, *, max_length: int = 150, num_beams: int = 4) -> str:
        self.calls.append({"text": text, "max_length": max_length, "num_beams": num_beams})
        if self.error:
            raise self.error
        return "hannah and amanda will meet at 4pm."


@pytest.fixture
def fake_summarizer() -> FakeSummarizer:
    return FakeSummarizer()
