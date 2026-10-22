"""Model loading and inference shared by the Flask (app.py) and FastAPI (api.py) servers.

torch and transformers are imported lazily inside ``Summarizer.load`` so the web layers
and the test suite can be imported without pulling in the full ML stack.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", str(Path(__file__).resolve().parent / "saved_summary_model"))
MAX_INPUT_TOKENS = 512
DEFAULT_SUMMARY_TOKENS = 150
DEFAULT_NUM_BEAMS = 4

_WHITESPACE = re.compile(r"\s+")
_HTML_TAG = re.compile(r"<.*?>")


def clean_text(text: str) -> str:
    """Normalise input exactly like train.py does, so inference sees the training distribution."""
    text = text.replace("\r\n", " ")
    text = _WHITESPACE.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    return text.strip().lower()


class ModelNotAvailableError(RuntimeError):
    """Raised when the fine-tuned weights are missing or are only a Git LFS pointer."""


def _check_weights(model_dir: Path) -> None:
    weights = model_dir / "model.safetensors"
    if not weights.exists():
        raise ModelNotAvailableError(f"No model weights found at {weights}. Run train.py first.")
    # A Git LFS pointer is a ~130-byte text file; the real T5-small weights are ~240 MB.
    if weights.stat().st_size < 1024 and weights.read_bytes().startswith(b"version https://git-lfs"):
        raise ModelNotAvailableError(
            f"{weights} is a Git LFS pointer, not the actual weights. "
            "Install Git LFS and run `git lfs pull` in the repository root."
        )


class Summarizer:
    """Thread-safe, lazily loaded wrapper around the fine-tuned T5 model."""

    def __init__(self, model_path: str = MODEL_PATH) -> None:
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
        self._device = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return str(self._device) if self._device is not None else "not loaded"

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            _check_weights(Path(self.model_path))

            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("Loading model from %s on %s", self.model_path, device)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(device)
            model.eval()

            self._tokenizer, self._device = tokenizer, device
            self._model = model  # assigned last: is_loaded flips only when fully ready
            logger.info("Model ready on %s", device)

    def summarize(
        self,
        text: str,
        *,
        max_length: int = DEFAULT_SUMMARY_TOKENS,
        num_beams: int = DEFAULT_NUM_BEAMS,
    ) -> str:
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError("Input text is empty after cleaning.")

        self.load()
        import torch

        # No padding: a single sequence padded to 512 tokens only wastes encoder compute.
        inputs = self._tokenizer(
            cleaned,
            return_tensors="pt",
            max_length=MAX_INPUT_TOKENS,
            truncation=True,
        ).to(self._device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
