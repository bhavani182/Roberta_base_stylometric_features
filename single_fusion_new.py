import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding


MODEL_DIR = "/app/final_fusion_model"


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


class FusionDataset(Dataset):
    def __init__(self, texts, ids, tokenizer, features):
        self.ids = ids
        self.features = features
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=512
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item = {k: self.encodings[k][idx] for k in self.encodings}
        item["features"] = torch.tensor(self.features[idx], dtype=torch.float)
        item["id"] = self.ids[idx]
        return item


class FusionDataCollator:
    def __init__(self, tokenizer):
        self.base_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, batch):
        text_features = []
        extra_features = []
        ids = []

        for item in batch:
            text_features.append({
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"]
            })
            extra_features.append(item["features"])
            ids.append(item["id"])

        output = self.base_collator(text_features)
        output["features"] = torch.stack(extra_features)
        output["ids"] = ids

        return output


class FusionModel(nn.Module):
    def __init__(self, model_name, num_features=9, num_labels=2, dropout=0.3):
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

    def forward(self, input_ids, attention_mask, features):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        text_repr = outputs.last_hidden_state[:, 0, :]
        text_repr = self.text_dropout(text_repr)

        feat_repr = self.feature_net(features)

        combined = torch.cat([text_repr, feat_repr], dim=1)
        logits = self.classifier(combined)

        return logits


def load_jsonl(input_path):
    ids = []
    texts = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                ids.append(obj["id"])
                texts.append(obj["text"])

    return ids, texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input dataset.jsonl")
    parser.add_argument("--output", required=True, help="Path to output predictions.jsonl")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="Path to trained model directory")
    parser.add_argument("--batch-size", type=int, default=8)

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    model_dir = args.model_dir

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ids, texts = load_jsonl(input_path)

    raw_features = np.array(
        [extract_features(text) for text in texts],
        dtype=np.float32
    )

    scaler_path = Path(model_dir) / "scaler.joblib"
    scaler = joblib.load(scaler_path)
    scaled_features = scaler.transform(raw_features)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model = FusionModel(
        model_name=model_dir,
        num_features=scaled_features.shape[1],
        num_labels=2,
        dropout=0.3
    )

    state_dict_path = Path(model_dir) / "pytorch_model.bin"

    if state_dict_path.exists():
        state_dict = torch.load(state_dict_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    dataset = FusionDataset(
        texts=texts,
        ids=ids,
        tokenizer=tokenizer,
        features=scaled_features
    )

    collator = FusionDataCollator(tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator
    )

    with open(output_path, "w", encoding="utf-8") as out:
        with torch.no_grad():
            for batch in dataloader:
                batch_ids = batch.pop("ids")

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                features = batch["features"].to(device)

                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    features=features
                )

                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

                for sample_id, prob in zip(batch_ids, probs):
                    out.write(json.dumps({
                        "id": sample_id,
                        "label": float(prob)
                    }) + "\n")


if __name__ == "__main__":
    main()