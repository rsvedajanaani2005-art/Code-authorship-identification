# ============================================================
# SemEval Code Attribution - Subtask B
# Fine-tuning CodeBERT for 11-class Attribution
# Fully trained on Google Colab T4 GPU
# ============================================================

# =========================
# 1. Install Dependencies
# =========================
# !pip install -q transformers accelerate

import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import f1_score


# =========================
# 2. Mount Google Drive
# =========================
from google.colab import drive
drive.mount('/content/drive')

# =========================
# 3. Load Dataset
# =========================

BASE_DIR = "/content/drive/MyDrive/"

train_path = os.path.join(BASE_DIR, "train.parquet")
val_path   = os.path.join(BASE_DIR, "validation.parquet")
test_path  = os.path.join(BASE_DIR, "test.parquet")

train_df = pd.read_parquet(train_path)
val_df   = pd.read_parquet(val_path)
test_df  = pd.read_parquet(test_path)

print("Train:", train_df.shape, train_df.columns.tolist())
print("Val:", val_df.shape, val_df.columns.tolist())
print("Test:", test_df.shape, test_df.columns.tolist())

print("\nTrain label distribution:")
print(train_df["label"].value_counts().sort_index())


# =========================
# 4. Handle Class Imbalance
# =========================

df_majority = train_df[train_df["label"] == 0]
df_minority = train_df[train_df["label"] != 0]

print("Majority (0) count:", len(df_majority))
print("Minority (!=0) count:", len(df_minority))

KEEP_MAJ = 80000

rng = np.random.default_rng(42)
idx_majority = rng.choice(df_majority.index.values, size=KEEP_MAJ, replace=False)
df_majority_sampled = df_majority.loc[idx_majority]

train_balanced_df = pd.concat(
    [df_majority_sampled, df_minority],
    ignore_index=True
).sample(frac=1.0, random_state=42).reset_index(drop=True)

print("Balanced train size:", len(train_balanced_df))
print("Balanced label distribution:")
print(train_balanced_df["label"].value_counts().sort_index())


# =========================
# 5. Text Construction
# =========================

MAX_CHARS_TRANS = 800

def make_text_for_transformer(df):
    if "language" in df.columns:
        lang = df["language"].fillna("")
    else:
        lang = pd.Series([""] * len(df), index=df.index)

    code = df["code"].fillna("").str.slice(0, MAX_CHARS_TRANS)
    return (lang + " " + code).astype(str).tolist()


train_texts = make_text_for_transformer(train_balanced_df)
train_labels = train_balanced_df["label"].to_numpy().astype(int)

val_texts = make_text_for_transformer(val_df)
val_labels = val_df["label"].to_numpy().astype(int)

num_labels = len(np.unique(train_labels))
print("Num labels:", num_labels)
print("Label set:", np.unique(train_labels))


# =========================
# 6. Model Setup
# =========================

MODEL_NAME = "microsoft/codebert-base"
MAX_TOKENS = 256
BATCH_SIZE = 8
EPOCHS = 3

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class CodeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


train_dataset = CodeDataset(train_texts, train_labels, tokenizer, max_length=MAX_TOKENS)
val_dataset   = CodeDataset(val_texts,   val_labels,   tokenizer, max_length=MAX_TOKENS)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)


# =========================
# 7. Training Setup
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)

model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

total_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

scaler = torch.cuda.amp.GradScaler()


# =========================
# 8. Evaluation Function
# =========================

def evaluate(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]

            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=labels
                )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return macro_f1


# =========================
# 9. Training Loop
# =========================

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    print(f"\n===== Transformer Epoch {epoch+1}/{EPOCHS} =====")

    for step, batch in enumerate(tqdm(train_loader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=labels
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

        if (step + 1) % 200 == 0:
            avg_loss = total_loss / (step + 1)
            print(f"Step {step+1}/{len(train_loader)}, loss={avg_loss:.4f}")

    avg_train_loss = total_loss / len(train_loader)
    val_macro_f1 = evaluate(model, val_loader)

    print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_macro_f1={val_macro_f1:.4f}")


# =========================
# 10. Save Model
# =========================

save_dir = os.path.join(BASE_DIR, "codebert_attempt_final")

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print("Saved model to:", save_dir)


# =========================
# 11. Inference on Test
# =========================

test_texts = make_text_for_transformer(test_df)
dummy_labels = np.zeros(len(test_texts), dtype=int)

test_dataset = CodeDataset(test_texts, dummy_labels, tokenizer, max_length=MAX_TOKENS)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model.eval()
all_test_preds = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}

        with torch.cuda.amp.autocast():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        all_test_preds.append(preds.cpu().numpy())

all_test_preds = np.concatenate(all_test_preds)


# =========================
# 12. Submission File
# =========================

submission = pd.DataFrame({
    "ID": test_df["ID"],
    "label": all_test_preds.astype(int)
})

sub_path = os.path.join(BASE_DIR, "submission_attempt_final.csv")
submission.to_csv(sub_path, index=False)

print("Submission saved at:", sub_path)


# =========================
# 13. Evaluate on test_sample
# =========================

test_sample_path = os.path.join(BASE_DIR, "test_sample.parquet")
test_sample_df = pd.read_parquet(test_sample_path)

print("test_sample shape:", test_sample_df.shape)
print(test_sample_df["label"].value_counts().sort_index())

test_sample_texts = make_text_for_transformer(test_sample_df)
test_sample_labels = test_sample_df["label"].to_numpy().astype(int)

test_sample_dataset = CodeDataset(
    test_sample_texts,
    test_sample_labels,
    tokenizer,
    max_length=MAX_TOKENS
)

test_sample_loader = DataLoader(
    test_sample_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_sample_macro_f1 = evaluate(model, test_sample_loader)
print("Macro F1 on test_sample:", test_sample_macro_f1)
