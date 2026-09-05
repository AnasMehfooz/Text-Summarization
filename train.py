import re
import torch
import pandas as pd
from transformers import T5Tokenizer, Trainer, TrainingArguments, T5ForConditionalGeneration

# 1. Load Data
print("Loading SAMSum dataset...")
train_data = pd.read_csv("samsum-train.csv")
val_data = pd.read_csv("samsum-validation.csv")

# 2. Random Sampling (for efficient training and fast convergence)
train_data = train_data.sample(n=4000, random_state=42).reset_index(drop=True)
val_data = val_data.sample(n=500, random_state=42).reset_index(drop=True)

# 3. Data Pre-processing
def clean_data(text: str) -> str:
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip().lower()

train_data["dialogue"] = train_data["dialogue"].apply(clean_data)
train_data["summary"] = train_data["summary"].apply(clean_data)
val_data["dialogue"] = val_data["dialogue"].apply(clean_data)
val_data["summary"] = val_data["summary"].apply(clean_data)

# 4. Tokenization Setup
tokenizer = T5Tokenizer.from_pretrained("t5-small")

def tokenize(batch):
    inputs = tokenizer(batch["dialogue"], padding="max_length", max_length=512, truncation=True)
    targets = tokenizer(batch["summary"], padding="max_length", max_length=150, truncation=True)
    inputs["labels"] = targets["input_ids"]
    return inputs

print("Tokenizing datasets...")
train_dataset = train_data.apply(tokenize, axis=1).tolist()
val_dataset = val_data.apply(tokenize, axis=1).tolist()

# 5. Initialize Model & Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.to(device)

# 6. Training Configuration
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=6,
    weight_decay=0.01,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    warmup_steps=500,
    logging_steps=50,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# 7. Main Execution
if __name__ == "__main__":
    print("Starting fine-tuning...")
    trainer.train()

    print("Saving fine-tuned model and tokenizer...")
    model.save_pretrained("./saved_summary_model")
    tokenizer.save_pretrained("./saved_summary_model")
    print("Model saved to ./saved_summary_model")
