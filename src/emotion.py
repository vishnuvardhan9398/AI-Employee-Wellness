"""Hugging Face BERT/DistilBERT multi-label emotion inference helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from src.emotion_labels import EMOTION_LABELS, ID_TO_LABEL, LABEL_TO_ID

MODEL_SPECS = {
    "bert": "google-bert/bert-base-uncased",
    "distilbert": "distilbert/distilbert-base-uncased",
}


class ModelUnavailableError(RuntimeError):
    """Raised when transformers/torch or a requested local model is unavailable."""


def _dependencies():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # Keeps Milestone 1 usable without ML packages.
        raise ModelUnavailableError(
            "Emotion analysis needs the Milestone 2 dependencies. Run "
            "`py -3.13 -m pip install -r requirements.txt`."
        ) from exc
    return torch, AutoModelForSequenceClassification, AutoTokenizer


def _token():
    """Optional public-Hub token; never log or expose its value."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        # Environment variables still work if optional local dotenv support is absent.
        pass
    return os.getenv("HUGGINGFACE_TOKEN") or None


def _model_kwargs() -> dict:
    token = _token()
    return {"token": token} if token else {}


def create_model(model_kind: str, source: str | Path | None = None):
    """Load a real pretrained encoder and configure its six sigmoid labels."""
    if model_kind not in MODEL_SPECS:
        raise ValueError(f"model_kind must be one of {sorted(MODEL_SPECS)}")
    _, AutoModelForSequenceClassification, AutoTokenizer = _dependencies()
    model_source = str(source or MODEL_SPECS[model_kind])
    kwargs = _model_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(model_source, **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        num_labels=len(EMOTION_LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
        **kwargs,
    )
    return model, tokenizer


def save_model(model, tokenizer, destination) -> Path:
    """Save model configuration/weights and its matching tokenizer locally."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    return destination


def load_local_model(model_path):
    """Load a previously saved model/tokenizer pair without downloading or retraining."""
    path = Path(model_path)
    if not path.is_dir():
        raise ModelUnavailableError(f"Saved model directory was not found: {path}")
    _, AutoModelForSequenceClassification, AutoTokenizer = _dependencies()
    try:
        return (
            AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True),
            AutoTokenizer.from_pretrained(path, local_files_only=True),
        )
    except OSError as exc:
        raise ModelUnavailableError(f"Could not load a complete saved model from: {path}") from exc


def format_probabilities(probabilities: Iterable[float], threshold: float = 0.5) -> dict:
    """Turn six sigmoid outputs into the public, threshold-based response schema."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    values = [float(value) for value in probabilities]
    if len(values) != len(EMOTION_LABELS):
        raise ValueError(f"Expected {len(EMOTION_LABELS)} probabilities, got {len(values)}.")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("Probabilities must be between 0 and 1.")
    all_probabilities = dict(zip(EMOTION_LABELS, values))
    primary_emotion = max(all_probabilities, key=all_probabilities.get)
    predicted = [
        {"emotion": label, "confidence": probability}
        for label, probability in all_probabilities.items() if probability >= threshold
    ]
    return {
        "primary_emotion": primary_emotion,
        "primary_confidence": all_probabilities[primary_emotion],
        "predicted_emotions": predicted,
        "all_probabilities": all_probabilities,
        "threshold": threshold,
        "threshold_met": bool(predicted),
    }


def predict_text(model, tokenizer, text: str, threshold: float = 0.5, max_length: int = 256) -> dict:
    """Run one real forward pass and apply sigmoid to independent label logits."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Emotion prediction requires non-empty text.")
    torch, _, _ = _dependencies()
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    device = next(model.parameters()).device
    encoded = {name: value.to(device) for name, value in encoded.items()}
    model.eval()
    with torch.no_grad():
        logits = model(**encoded).logits[0]
        probabilities = torch.sigmoid(logits).detach().cpu().tolist()
    return format_probabilities(probabilities, threshold)
