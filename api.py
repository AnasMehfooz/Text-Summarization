"""FastAPI server for the fine-tuned T5 dialogue summarizer.

Run locally:   uvicorn api:app --reload
Interactive docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Protocol

from fastapi import FastAPI, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from summarizer import (
    DEFAULT_NUM_BEAMS,
    DEFAULT_SUMMARY_TOKENS,
    ModelNotAvailableError,
    Summarizer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MAX_INPUT_CHARS = 20_000


class SummarizerBackend(Protocol):
    is_loaded: bool
    device: str

    def load(self) -> None: ...
    def summarize(self, text: str, *, max_length: int, num_beams: int) -> str: ...


# ── Schemas ──────────────────────────────────────────────────────────────────
class SummarizeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_INPUT_CHARS,
        description="Dialogue to summarize, one speaker turn per line.",
        examples=["Hannah: Are you free this afternoon?\nAmanda: Yes, why?\nHannah: Let's meet at the cafe at 4pm."],
    )
    max_length: int = Field(DEFAULT_SUMMARY_TOKENS, ge=10, le=300, description="Max summary length in tokens.")
    num_beams: int = Field(DEFAULT_NUM_BEAMS, ge=1, le=8, description="Beam search width.")

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class SummarizeResponse(BaseModel):
    summary: str
    input_words: int
    summary_words: int


class HealthResponse(BaseModel):
    status: Literal["healthy", "loading"]
    model_loaded: bool
    device: str


# ── App factory ──────────────────────────────────────────────────────────────
def create_app(summarizer: SummarizerBackend | None = None, *, load_on_startup: bool = True) -> FastAPI:
    backend: SummarizerBackend = summarizer or Summarizer()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if load_on_startup:
            # Load in a worker thread so startup does not block the event loop.
            await run_in_threadpool(backend.load)
        yield

    app = FastAPI(
        title="T5 Dialogue Summarizer",
        description="Abstractive dialogue summarization with T5-small fine-tuned on SAMSum.",
        version="1.1.0",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(BASE_DIR / "templates" / "index.html")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="healthy" if backend.is_loaded else "loading",
            model_loaded=backend.is_loaded,
            device=backend.device,
        )

    # Sync handler on purpose: FastAPI runs it in its threadpool, so CPU/GPU-bound
    # generation never blocks the event loop serving other requests.
    @app.post("/summarize", response_model=SummarizeResponse)
    def summarize(req: SummarizeRequest) -> SummarizeResponse:
        try:
            summary = backend.summarize(req.text, max_length=req.max_length, num_beams=req.num_beams)
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        except ModelNotAvailableError as exc:
            logger.error("Model unavailable: %s", exc)
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model is not available.") from exc
        except Exception as exc:
            logger.exception("Inference failed")
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal server error during summarization."
            ) from exc

        return SummarizeResponse(
            summary=summary,
            input_words=len(req.text.split()),
            summary_words=len(summary.split()),
        )

    return app


app = create_app()
