import re
import torch
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import T5Tokenizer, T5ForConditionalGeneration
from waitress import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Load model once at startup ──────────────────────────────────────────────
MODEL_PATH = "./saved_summary_model"

print("Loading model…")
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model     = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Model ready on {device}")


# ── Helper ───────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+",  " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip().lower()


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "device": str(device)})


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or "text" not in data:
        return jsonify({"error": "Please provide a 'text' field in the request body."}), 400

    if not isinstance(data["text"], str):
        return jsonify({"error": "The 'text' field must be a string."}), 400

    raw_text = data["text"].strip()
    if not raw_text:
        return jsonify({"error": "Input text cannot be empty."}), 400

    cleaned = clean_text(raw_text)

    try:
        inputs = tokenizer(
            cleaned,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=150,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return jsonify({"summary": summary})
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return jsonify({"error": "Internal server error during summarization."}), 500


if __name__ == "__main__":
    print("Serving with Waitress on http://127.0.0.1:5000")
    serve(app, host="127.0.0.1", port=5000)
