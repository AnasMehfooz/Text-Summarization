import os
import re
import json
import time
import argparse
import torch
import pandas as pd
from rouge_score import rouge_scorer
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ── Settings ─────────────────────────────────────────────────────────────────
# Same generation settings as app.py, so the scores reflect what users see.
GEN_KWARGS = dict(max_length=150, num_beams=4, early_stopping=True, no_repeat_ngram_size=3)

MODELS = [
    # (label, model path, input prefix, lowercase input like training did)
    ("t5-small (no fine-tuning)", "t5-small", "summarize: ", False),
    ("t5-small fine-tuned on SAMSum (this repo)", "./saved_summary_model", "", True),
]

OUT_DIR = "eval"


# ── Helpers ──────────────────────────────────────────────────────────────────
def clean_text(text: str, lower: bool) -> str:
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+",  " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip()
    return text.lower() if lower else text


def summarize_all(model_path, dialogues, prefix, lower, batch_size, device):
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model     = T5ForConditionalGeneration.from_pretrained(model_path).to(device).eval()

    inputs = [prefix + clean_text(d, lower) for d in dialogues]
    preds  = []
    for i in range(0, len(inputs), batch_size):
        batch = tokenizer(
            inputs[i:i + batch_size],
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(**batch, **GEN_KWARGS)
        preds += tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        print(f"  {min(i + batch_size, len(inputs))}/{len(inputs)}", end="\r")
    print()
    return preds


def rouge(preds, refs):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, ref in zip(preds, refs):
        for key, score in scorer.score(ref, pred).items():
            totals[key] += score.fmeasure
    return {key: round(100 * total / len(refs), 2) for key, total in totals.items()}


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ROUGE evaluation on the SAMSum test set.")
    parser.add_argument("--data", default="samsum-test.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Only score the first N dialogues.")
    args = parser.parse_args()

    data = pd.read_csv(args.data).dropna(subset=["dialogue", "summary"])
    if args.limit:
        data = data.head(args.limit)
    dialogues = data["dialogue"].tolist()
    refs      = data["summary"].tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating {len(dialogues)} dialogues on {device}\n")

    results     = {"dataset": args.data, "num_dialogues": len(dialogues), "generation": GEN_KWARGS, "models": {}}
    predictions = data[["id", "dialogue", "summary"]].rename(columns={"summary": "reference"})

    for label, path, prefix, lower in MODELS:
        print(f"{label}")
        start  = time.time()
        preds  = summarize_all(path, dialogues, prefix, lower, args.batch_size, device)
        scores = rouge(preds, refs)
        scores["seconds"] = round(time.time() - start, 1)
        results["models"][label] = scores
        predictions[label] = preds
        print(f"  ROUGE-1 {scores['rouge1']}  ROUGE-2 {scores['rouge2']}  ROUGE-L {scores['rougeL']}\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "rouge_scores.json"), "w") as f:
        json.dump(results, f, indent=2)
    predictions.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)
    print(f"Saved scores and predictions to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
