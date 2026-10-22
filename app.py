"""Flask server for the fine-tuned T5 dialogue summarizer (served with Waitress).

The FastAPI version with Pydantic validation and OpenAPI docs lives in api.py.
"""
import logging
import os

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from summarizer import ModelNotAvailableError, Summarizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_INPUT_CHARS = 20_000

app = Flask(__name__)
CORS(app)

summarizer = Summarizer()


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy" if summarizer.is_loaded else "loading",
            "model_loaded": summarizer.is_loaded,
            "device": summarizer.device,
        }
    )


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("text"), str):
        return jsonify({"error": "Please provide a 'text' string in the request body."}), 400

    raw_text = data["text"].strip()
    if not raw_text:
        return jsonify({"error": "Input text cannot be empty."}), 400
    if len(raw_text) > MAX_INPUT_CHARS:
        return jsonify({"error": f"Input text exceeds {MAX_INPUT_CHARS} characters."}), 400

    try:
        summary = summarizer.summarize(raw_text)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except ModelNotAvailableError as exc:
        logger.error("Model unavailable: %s", exc)
        return jsonify({"error": "Model is not available."}), 503
    except Exception:
        logger.exception("Inference failed")
        return jsonify({"error": "Internal server error during summarization."}), 500

    return jsonify({"summary": summary})


if __name__ == "__main__":
    from waitress import serve

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    summarizer.load()
    logger.info("Serving with Waitress on http://%s:%d", host, port)
    serve(app, host=host, port=port)
