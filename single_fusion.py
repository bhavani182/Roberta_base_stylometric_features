import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
    brier_score_loss
)

from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

for var in ["LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
    os.environ.pop(var, None)

BASE_PATH = "/mnt/ceph/storage/data-tmp/current/tawe7542/obfuscated-and-normal-val-test-roberta"
RESULTS_DIR = f"{BASE_PATH}/results"
LOGS_DIR = f"{BASE_PATH}/logs"
FINAL_MODEL_DIR = f"{BASE_PATH}/final_fusion_model"
METRICS_FILE = f"{RESULTS_DIR}/eval_results.json"
PREDICTIONS_FILE = f"{RESULTS_DIR}/val_predictions.jsonl"

TRAIN_FILE = "/app/obfuscated_data/train_pan_obfuscated_augmented.jsonl"
VAL_FILE = "/app/obfuscated_data/val.jsonl"

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
Path(FINAL_MODEL_DIR).mkdir(parents=True, exist_ok=True)

train_df = pd.read_json(TRAIN_FILE, lines=True)
val_df = pd.read_json(VAL_FILE, lines=True)

print(train_df.head())
print(train_df["label"].value_counts())
print(val_df["label"].value_counts())


def extract_features(text):
    words = text.split()
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]

    num_words = len(words)
    num_chars = len(text)
    num_sentences = max(len(sentences), 1)

    avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
    avg_sentence_len = num_words / num_sentences

    punctuation_count = len(re.findall(r"[.,!?;:]", text))
    uppercase_ratio = sum(1 for c in text if c.isupper()) / max(num_chars, 1)
    digit_ratio = sum(1 for c in text if c.isdigit()) / max(num_chars, 1)
    space_ratio = sum(1 for c in text if c.isspace()) / max(num_chars, 1)

    unique_words = len(set(w.lower() for w in words)) if words else 0
    lexical_diversity = unique_words / max(num_words, 1)

    return [
        num_words,
        num_chars,
        avg_word_len,
        avg_sentence_len,
        punctuation_count,
        uppercase_ratio,
        digit_ratio,
        space_ratio,
        lexical_diversity,
    ]


X_train_feat = np.array([extract_features(t) for t in train_df["text"]], dtype=np.float32)
X_val_feat = np.array([extract_features(t) for t in val_df["text"]], dtype=np.float32)

scaler = StandardScaler()
X_train_feat = scaler.fit_transform(X_train_feat)
X_val_feat = scaler.transform(X_val_feat)

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(
    train_df["text"].tolist(),
    truncation=True,
    max_length=512
)

val_encodings = tokenizer(
    val_df["text"].tolist(),
    truncation=True,
    max_length=512
)


class FusionDataset(Dataset):
    def __init__(self, encodings, features, labels, ids=None):
        self.encodings = encodings
        self.features = features
        self.labels = labels.astype(np.int64)
        self.ids = ids.tolist() if ids is not None else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: self.encodings[k][idx] for k in self.encodings}
        item["features"] = torch.tensor(self.features[idx], dtype=torch.float)
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.ids is not None:
            item["sample_id"] = self.ids[idx]
        return item


train_dataset = FusionDataset(
    train_encodings,
    X_train_feat,
    train_df["label"].values,
    train_df["id"].values if "id" in train_df.columns else None
)

val_dataset = FusionDataset(
    val_encodings,
    X_val_feat,
    val_df["label"].values,
    val_df["id"].values if "id" in val_df.columns else None
)


class FusionDataCollator:
    def __init__(self, tokenizer):
        self.base_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features):
        text_features = []
        extra_features = []
        labels = []
        sample_ids = []

        for f in features:
            text_features.append({
                "input_ids": f["input_ids"],
                "attention_mask": f["attention_mask"]
            })
            extra_features.append(f["features"])
            labels.append(f["labels"])
            if "sample_id" in f:
                sample_ids.append(f["sample_id"])

        batch = self.base_collator(text_features)
        batch["features"] = torch.stack(extra_features)
        batch["labels"] = torch.stack(labels)

        if sample_ids:
            batch["sample_id"] = sample_ids

        return batch


data_collator = FusionDataCollator(tokenizer)


class FusionModel(nn.Module):
    def __init__(self, model_name, num_features, num_labels=2, dropout=0.3):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.text_dropout = nn.Dropout(dropout)

        self.feature_net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 32, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids, attention_mask, features, labels=None, **kwargs):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        text_repr = outputs.last_hidden_state[:, 0, :]
        text_repr = self.text_dropout(text_repr)

        feat_repr = self.feature_net(features)

        combined = torch.cat([text_repr, feat_repr], dim=1)
        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


model = FusionModel(
    model_name=model_name,
    num_features=X_train_feat.shape[1],
    num_labels=2,
    dropout=0.3
)


def compute_pan_metrics(eval_pred):
    logits, y_true = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    y_pred = (probs >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_true, probs)
    brier = 1 - brier_score_loss(y_true, probs)

    n = len(y_true)
    non_answers = np.sum(probs == 0.5)
    correct = np.sum(y_pred == y_true)

    if n - non_answers == 0:
        c_at_1 = 0.0
    else:
        c_at_1 = (correct + non_answers * (correct / (n - non_answers))) / n

    f1 = f1_score(y_true, y_pred, zero_division=0)

    beta = 0.5
    adjusted_preds = y_pred.copy()
    adjusted_preds[probs == 0.5] = 0

    precision = precision_score(y_true, adjusted_preds, zero_division=0)
    recall = recall_score(y_true, adjusted_preds, zero_division=0)

    if precision + recall == 0:
        f05u = 0.0
    else:
        f05u = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall)

    cm = confusion_matrix(y_true, y_pred)
    mean_score = np.mean([roc_auc, brier, c_at_1, f1, f05u])

    return {
        "roc_auc": float(roc_auc),
        "brier": float(brier),
        "c@1": float(c_at_1),
        "f1": float(f1),
        "f05u": float(f05u),
        "mean": float(mean_score),
        "tn": int(cm[0][0]),
        "fp": int(cm[0][1]),
        "fn": int(cm[1][0]),
        "tp": int(cm[1][1]),
    }


training_args = TrainingArguments(
    output_dir=RESULTS_DIR,
    logging_dir=LOGS_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    load_best_model_at_end=True,
    metric_for_best_model="mean",
    greater_is_better=True,
    save_total_limit=2,
    logging_steps=50,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_pan_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

results = trainer.evaluate()
print(results)

with open(METRICS_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

pred_output = trainer.predict(val_dataset)
logits = pred_output.predictions
probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
preds = (probs >= 0.5).astype(int)

with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
    for idx, pred, prob in zip(val_df["id"].tolist(), preds, probs):
        f.write(json.dumps({
            "id": idx,
            "label": int(pred),
            "score": float(prob)
        }) + "\n")

print(f"Saved metrics to: {METRICS_FILE}")
print(f"Saved validation predictions to: {PREDICTIONS_FILE}")
print(f"Saved final model to: {FINAL_MODEL_DIR}")