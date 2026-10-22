# Dialogue Summarization with Fine-Tuned T5

[![tests](https://github.com/AnasMehfooz/Text-Summarization/actions/workflows/tests.yml/badge.svg)](https://github.com/AnasMehfooz/Text-Summarization/actions/workflows/tests.yml)

Abstractive summarization of chat dialogues using **T5-small fine-tuned on SAMSum**, served through a **FastAPI** REST API (Pydantic validation, OpenAPI docs) or an equivalent **Flask + Waitress** server, with a small web UI. Runs on CUDA when available and falls back to CPU, where a summary takes about 0.5 s.

```text
Hannah: Hey, are you free this afternoon?
Amanda: Yes, why?
Hannah: Let's meet at the cafe around 4pm to work on our presentation.
Amanda: Sounds great! See you then.

→ hannah and amanda will meet at the cafe around 4 pm to work on their presentation.
```

## Architecture

```text
             ┌──────────── api.py  (FastAPI + Pydantic, uvicorn) ──┐
web UI ──►   │                                                     ├──► summarizer.py ──► T5-small (saved_summary_model/)
REST   ──►   └──────────── app.py  (Flask, Waitress)  ─────────────┘     clean_text · lazy, thread-safe load · beam search
```

- **`summarizer.py`** – one place for preprocessing and inference. Applies the same `clean_text` as training, loads the model once (thread-safe, lazy), detects CUDA, and runs beam search (4 beams, no repeated 3-grams). torch/transformers are imported lazily so the web layers and tests start without them.
- **`api.py`** – FastAPI app factory with typed request/response models, input limits, a lifespan hook that loads the model off the event loop, and a sync inference handler so generation runs in FastAPI's threadpool.
- **`app.py`** – the original Flask server, now using the shared module.

## Model and training

| | |
| --- | --- |
| Base model | `t5-small` (60M parameters) |
| Data | [SAMSum](https://huggingface.co/datasets/samsum): ~16k chat dialogues with human-written summaries |
| Training subset | 4,000 dialogues (train), 500 (validation), sampled with `random_state=42` |
| Setup | Hugging Face `Trainer`, 6 epochs, batch size 8, 500 warmup steps, weight decay 0.01 |
| Input / output length | 512 / 150 tokens |

Retrain with `python train.py`; weights are written to `saved_summary_model/`. `text_summarizer.ipynb` walks through the same steps interactively.

## Results

Evaluated on the full SAMSum **test set (819 dialogues)** with the same generation settings the server uses (4 beams, no repeated 3-grams). Scores are ROUGE F1 (×100) with stemming.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| `t5-small`, no fine-tuning (`summarize:` prefix) | 28.37 | 7.60 | 21.98 |
| **`t5-small` fine-tuned on SAMSum (this repo)** | **43.39** | **18.60** | **35.16** |

Fine-tuning on 4,000 training dialogues improves ROUGE-1 by **+15.0** and more than doubles ROUGE-2 over the base model.

**Limitations:** the model is small and was trained on about a quarter of the training set. It sometimes attributes an action to the wrong speaker, and it tends to copy sentences when the input is not a chat. Every prediction is in [`eval/test_predictions.csv`](eval/test_predictions.csv) for error analysis.

Reproduce with `python evaluate_model.py` (needs the real weights and a few minutes on a GPU).

## Quick start

The model weights (~240 MB) are stored with **Git LFS**. Without `git lfs pull` you only get a pointer file, and the server returns a clear 503 error.

```bash
git clone https://github.com/AnasMehfooz/Text-Summarization.git
cd Text-Summarization
git lfs install && git lfs pull

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**FastAPI** (recommended)

```bash
uvicorn api:app --reload
# UI:   http://127.0.0.1:8000
# Docs: http://127.0.0.1:8000/docs
```

**Flask**

```bash
python app.py        # http://127.0.0.1:5000  (HOST / PORT env vars override)
```

**Docker** (CPU)

```bash
docker build -t t5-summarizer .
docker run -p 8000:8000 t5-summarizer
```

## API

### `POST /summarize`

```bash
curl -X POST http://127.0.0.1:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Tom: Are we still on for 5?\nAnna: Yes, see you at the gym.", "num_beams": 4}'
```

| Field | Type | Default | Constraints |
| --- | --- | --- | --- |
| `text` | string | required | 1–20,000 chars, not blank |
| `max_length` | int | 150 | 10–300 tokens (FastAPI only) |
| `num_beams` | int | 4 | 1–8 (FastAPI only) |

FastAPI response:

```json
{ "summary": "tom and anna are still on for 5 at the gym.", "input_words": 14, "summary_words": 11 }
```

The Flask server returns `{"summary": "..."}`.

| Status | Meaning |
| --- | --- |
| 200 | Summary generated |
| 400 | Text is empty after cleaning (e.g. only HTML tags) |
| 422 | Request failed validation (FastAPI); Flask returns 400 |
| 503 | Model weights missing |
| 500 | Inference error (details are logged, not returned) |

### `GET /health`

```json
{ "status": "healthy", "model_loaded": true, "device": "cuda" }
```

## Tests

The suite mocks the model, so it runs in about a second without torch or the weights. CI runs it on Python 3.11 and 3.12.

```bash
pip install -r requirements-dev.txt
pytest
```

## Project structure

```text
├── api.py                  FastAPI server
├── app.py                  Flask server (Waitress)
├── summarizer.py           Shared preprocessing and inference
├── train.py                Fine-tuning script (he.py: original version)
├── evaluate_model.py       ROUGE evaluation vs. the base t5-small model
├── eval/                   Test-set scores and per-dialogue predictions
├── text_summarizer.ipynb   Notebook walkthrough
├── saved_summary_model/    Fine-tuned weights (Git LFS) and tokenizer
├── templates/, static/     Web UI
├── tests/                  pytest suite
├── Dockerfile
└── samsum-*.csv            Dataset splits
```

## Author

**Anas Mehfooz** · [GitHub](https://github.com/AnasMehfooz) · [LinkedIn](https://www.linkedin.com/in/anas-mehfooz-670501302/)
