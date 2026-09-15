"""Small, transparent fine-tuning workflow for the two real transformer models."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.emotion import MODEL_SPECS, create_model, save_model
from src.emotion_labels import EMOTION_LABELS, labels_to_target


def load_training_csv(path) -> list[dict]:
    """Read `text` plus either six label columns or a `labels` JSON/semicolon field."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "text" not in rows[0]:
        raise ValueError("Training CSV needs a `text` column.")
    samples = []
    normalized_columns = {column.casefold(): column for column in rows[0]}
    for line, row in enumerate(rows, start=2):
        text = (row.get("text") or "").strip()
        if not text:
            raise ValueError(f"Training CSV line {line} has empty text.")
        if "labels" in normalized_columns:
            raw = row[normalized_columns["labels"]].strip()
            try:
                labels = json.loads(raw) if raw.startswith("[") else [item.strip() for item in raw.split(";")]
            except json.JSONDecodeError as exc:
                raise ValueError(f"Training CSV line {line} has invalid labels JSON.") from exc
            target = labels_to_target(labels)
        else:
            missing = [label for label in EMOTION_LABELS if label.casefold() not in normalized_columns]
            if missing:
                raise ValueError("Training CSV needs `labels` or all six columns: " + ", ".join(EMOTION_LABELS))
            target = [float(row[normalized_columns[label.casefold()]]) for label in EMOTION_LABELS]
            if any(value not in (0.0, 1.0) for value in target):
                raise ValueError(f"Training CSV line {line} labels must be 0 or 1.")
        samples.append({"text": text, "labels": target})
    return samples


def fine_tune(model_kind: str, training_csv, output_dir, epochs: int = 2, batch_size: int = 4,
              learning_rate: float = 2e-5, max_length: int = 256) -> Path:
    """Fine-tune BERT or DistilBERT with BCE-with-logits multi-label loss and save it."""
    if model_kind not in MODEL_SPECS:
        raise ValueError(f"model_kind must be one of {sorted(MODEL_SPECS)}")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive.")
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError("Fine-tuning requires torch and transformers.") from exc
    samples = load_training_csv(training_csv)
    model, tokenizer = create_model(model_kind)

    class TextDataset(Dataset):
        def __len__(self): return len(samples)
        def __getitem__(self, index):
            encoded = tokenizer(samples[index]["text"], truncation=True, max_length=max_length, padding="max_length")
            encoded["labels"] = samples[index]["labels"]
            return encoded

    def collate(batch):
        return {key: torch.tensor([item[key] for item in batch], dtype=torch.float32 if key == "labels" else torch.long)
                for key in batch[0]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        for batch in DataLoader(TextDataset(), batch_size=batch_size, shuffle=True, collate_fn=collate):
            batch = {name: value.to(device) for name, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
    return save_model(model, tokenizer, Path(output_dir) / model_kind)
