# CPU image for the FastAPI server.
# Requires the real model weights: run `git lfs pull` before building.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/saved_summary_model

WORKDIR /app

# CPU-only torch keeps the image ~1.5 GB smaller than the default CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install "transformers>=4.40.0" "sentencepiece>=0.1.99" \
    "fastapi>=0.110.0,<1.0" "uvicorn[standard]>=0.29.0" "pydantic>=2.6,<3"

COPY summarizer.py api.py ./
COPY templates/ templates/
COPY static/ static/
COPY saved_summary_model/ saved_summary_model/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if b'healthy' in urllib.request.urlopen('http://127.0.0.1:8000/health').read() else 1)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
