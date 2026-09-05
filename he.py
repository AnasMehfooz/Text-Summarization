import re
import torch
import pandas as pd
from transformers import T5Tokenizer, Trainer, TrainingArguments, T5ForConditionalGeneration

# 1. Load Data
train_data = pd.read_csv("samsum-train.csv")
val_data = pd.read_csv("samsum-validation.csv")

# 2. Random Sampling
train_data = train_data.sample(n=4000, random_state=42).reset_index(drop=True)
val_data = val_data.sample(n=500, random_state=42).reset_index(drop=True)

# 3. Data Pre-processing
def clean_data(text):
    text = re.sub(r"\r\n", " ", text)  # lines
    text = re.sub(r"\s+", " ", text)   # spaces
    text = re.sub(r"<.*?>", " ", text) # html tags 
    text = text.strip().lower()
    return text

train_data["dialogue"] = train_data["dialogue"].apply(clean_data)
train_data["summary"] = train_data["summary"].apply(clean_data)

val_data["dialogue"] = val_data["dialogue"].apply(clean_data)
val_data["summary"] = val_data["summary"].apply(clean_data)

# 4. Tokenization Setup
tokenizer = T5Tokenizer.from_pretrained("t5-small")

def tokenize(data):
    inputs = tokenizer(data["dialogue"], padding="max_length", max_length=512, truncation=True)
    targets = tokenizer(data["summary"], padding="max_length", max_length=150, truncation=True)

    inputs["labels"] = targets["input_ids"]  # token ids => add to input as label
    return inputs

train_dataset = train_data.apply(tokenize, axis=1).tolist()
val_dataset = val_data.apply(tokenize, axis=1).tolist()

# 5. Initialize Model and Device Configuration
model = T5ForConditionalGeneration.from_pretrained("t5-small")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"device: {device}")
model.to(device)

# 6. Training Arguments & Trainer Setup
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=6,
    weight_decay=0.01,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    warmup_steps=500
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# 7. Train Model
trainer.train()

# 8. Save Model and Tokenizer
model.save_pretrained("./saved_summary_model")
tokenizer.save_pretrained("./saved_summary_model")

# 9. Reload Model and Tokenizer for Testing
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")