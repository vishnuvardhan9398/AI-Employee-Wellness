"""Dynamic multi-label model evaluation and comparison."""

from __future__ import annotations

from src.emotion_labels import EMOTION_LABELS


def calculate_metrics(y_true, probabilities, threshold: float = 0.5) -> dict:
    """Calculate subset accuracy plus macro precision/recall/F1 from sigmoid outputs.

    Accuracy is *subset accuracy*: a sample is correct only when all six
    thresholded labels exactly match its six expected labels.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    try:
        import numpy as np
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    except ImportError as exc:
        raise RuntimeError("Evaluation requires scikit-learn and numpy.") from exc
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if truth.ndim != 2 or scores.shape != truth.shape or truth.shape[1] != len(EMOTION_LABELS):
        raise ValueError("Expected matching two-dimensional arrays with six label columns.")
    predicted = (scores >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        truth, predicted, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(macro_precision), "recall": float(macro_recall), "macro_f1": float(macro_f1),
        "emotion_wise": {
            label: {"precision": float(precision[index]), "recall": float(recall[index]),
                    "f1": float(f1[index]), "support": int(support[index])}
            for index, label in enumerate(EMOTION_LABELS)
        },
        "threshold": threshold, "sample_count": int(truth.shape[0]),
    }


def compare_models(metrics_by_model: dict) -> dict:
    """Rank models by macro F1, then subset accuracy, then precision."""
    if not metrics_by_model:
        raise ValueError("No model metrics were supplied.")
    ranking = sorted(metrics_by_model, key=lambda name: (
        metrics_by_model[name]["macro_f1"], metrics_by_model[name]["accuracy"], metrics_by_model[name]["precision"]
    ), reverse=True)
    return {"best_model": ranking[0], "selection_rule": "macro_f1, then subset accuracy, then precision", "ranking": ranking}


def comparison_rows(metrics_by_model: dict) -> list[dict]:
    """Return a report-friendly dynamic comparison table for BERT and DistilBERT."""
    return [{"model": name, "accuracy": metric["accuracy"], "precision": metric["precision"],
             "recall": metric["recall"], "macro_f1": metric["macro_f1"]}
            for name, metric in metrics_by_model.items()]


def evaluate_saved_models(heldout_csv, model_root="models", threshold: float = 0.5) -> dict:
    """Run BERT and DistilBERT over the same held-out labelled CSV dynamically."""
    from pathlib import Path
    from src.emotion import load_local_model, predict_text
    from src.training import load_training_csv
    samples = load_training_csv(heldout_csv)
    metrics = {}
    for model_name in ("bert", "distilbert"):
        model, tokenizer = load_local_model(Path(model_root) / model_name)
        scores = [[predict_text(model, tokenizer, sample["text"], threshold)["all_probabilities"][label]
                   for label in EMOTION_LABELS] for sample in samples]
        metrics[model_name] = calculate_metrics([sample["labels"] for sample in samples], scores, threshold)
    return {"metrics": metrics, "comparison": comparison_rows(metrics), "selection": compare_models(metrics)}
