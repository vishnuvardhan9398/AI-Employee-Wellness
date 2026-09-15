"""Local, held-out ISEAR benchmark loading and reporting."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from src.emotion_labels import EMOTION_LABELS, labels_to_target
from src.evaluation import calculate_metrics


ISEAR_LABEL_MAP = {
    "joy": "Joy",
    "sadness": "Sadness",
    "anger": "Anger",
    "fear": "Fear",
    "surprise": "Surprise",
    "disgust": "Disgust",
    "guilt": None,
    "shame": None,
}


def load_isear_csv(path) -> list[dict]:
    """Load a local ISEAR CSV with text and emotion columns."""

    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows or "emotion" not in rows[0] or "text" not in rows[0]:
        raise ValueError(
            "ISEAR CSV needs non-empty `text` and `emotion` columns."
        )

    samples = []

    for row in rows:
        emotion_name = (row.get("emotion") or "").strip().casefold()
        mapped = ISEAR_LABEL_MAP.get(emotion_name)
        text = (row.get("text") or "").strip()

        if mapped and text:
            samples.append({
                "text": text,
                "expected_emotion": mapped
            })

    if not samples:
        raise ValueError("No supported ISEAR samples were found.")

    return samples


def validate_isear(model, tokenizer, csv_path, threshold: float = 0.5) -> dict:
    """Evaluate one saved model against a separate local ISEAR subset."""

    from src.emotion import predict_text

    samples = load_isear_csv(csv_path)

    predictions = [
        predict_text(model, tokenizer, sample["text"], threshold)
        for sample in samples
    ]

    expected_targets = [
        labels_to_target([sample["expected_emotion"]])
        for sample in samples
    ]

    probabilities = [
        [
            prediction["all_probabilities"][label]
            for label in EMOTION_LABELS
        ]
        for prediction in predictions
    ]

    # Existing multi-label metrics
    metrics = calculate_metrics(
        expected_targets,
        probabilities,
        threshold
    )

    # Primary emotion accuracy for ISEAR
    correct_primary = sum(
        prediction["primary_emotion"] == sample["expected_emotion"]
        for sample, prediction in zip(samples, predictions)
    )

    primary_accuracy = correct_primary / len(samples)

    incorrect = [
        {
            "text": sample["text"],
            "expected_emotion": sample["expected_emotion"],
            "predicted_primary": prediction["primary_emotion"],
            "primary_confidence": prediction["primary_confidence"],
        }
        for sample, prediction in zip(samples, predictions)
        if prediction["primary_emotion"] != sample["expected_emotion"]
    ]

    return {
        "metrics": metrics,
        "primary_accuracy": primary_accuracy,
        "correct_primary_predictions": correct_primary,
        "incorrect_predictions": incorrect,
        "sample_count": len(samples),
        "consistency": (
            "All samples were evaluated once with the same "
            "saved model and threshold."
        ),
        "mapping": ISEAR_LABEL_MAP,
    }


def save_isear_report(result: dict, output_directory) -> dict:
    """Write benchmark results as JSON and an inspectable HTML report."""

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)

    json_path = directory / "isear_validation.json"
    html_path = directory / "isear_validation.html"

    json_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )

    metrics = result["metrics"]

    mistakes = "".join(
        f"<tr>"
        f"<td>{html.escape(item['expected_emotion'])}</td>"
        f"<td>{html.escape(item['predicted_primary'])}</td>"
        f"<td>{item['primary_confidence']:.4f}</td>"
        f"<td>{html.escape(item['text'])}</td>"
        f"</tr>"
        for item in result["incorrect_predictions"]
    )

    if not mistakes:
        mistakes = (
            "<tr><td colspan='4'>"
            "No incorrect primary-emotion predictions."
            "</td></tr>"
        )

    html_content = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>ISEAR Validation Report</title>

<style>
body {{
    font-family: Arial, sans-serif;
    margin: 30px;
    color: #222;
}}

h1 {{
    color: #333;
}}

.summary {{
    padding: 15px;
    border: 1px solid #ccc;
    margin-bottom: 25px;
}}

table {{
    border-collapse: collapse;
    width: 100%;
}}

th, td {{
    border: 1px solid #ccc;
    padding: 8px;
    text-align: left;
}}

th {{
    background-color: #eeeeee;
}}

.note {{
    margin-top: 25px;
    color: #555;
}}
</style>

</head>

<body>

<h1>ISEAR Held-Out Validation Report</h1>

<div class="summary">

<p><strong>Total Samples:</strong> {result['sample_count']}</p>

<p>
<strong>Correct Primary Predictions:</strong>
{result['correct_primary_predictions']}
</p>

<p>
<strong>Primary Emotion Accuracy:</strong>
{result['primary_accuracy']:.4f}
</p>

<p>
<strong>Multi-label Subset Accuracy:</strong>
{metrics['accuracy']:.4f}
</p>

<p>
<strong>Macro Precision:</strong>
{metrics['precision']:.4f}
</p>

<p>
<strong>Macro Recall:</strong>
{metrics['recall']:.4f}
</p>

<p>
<strong>Macro F1:</strong>
{metrics['macro_f1']:.4f}
</p>

<p>
<strong>Threshold:</strong>
{metrics['threshold']}
</p>

</div>

<p class="note">
{html.escape(result['consistency'])}
</p>

<h2>Incorrect Primary Predictions</h2>

<table>
<tr>
    <th>Expected</th>
    <th>Predicted</th>
    <th>Confidence</th>
    <th>Text</th>
</tr>

{mistakes}

</table>

</body>
</html>
"""

    html_path.write_text(html_content, encoding="utf-8")

    return {
        "json_path": json_path,
        "html_path": html_path
    }