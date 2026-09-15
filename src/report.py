"""
report.py
---------
Builds and saves analysis reports for the AI-Based Employee Wellness Platform.

Produces:
    - A machine-readable CSV report.
    - A human-readable HTML report.
    - A summary dict containing sentiment counts.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# Milestone 1 columns
REPORT_COLUMNS = [
    "employee_id",
    "original_text",
    "processed_text",
    "classification",
    "compound",
    "positive",
    "negative",
    "neutral",
]


# Milestone 2 emotion columns
EMOTION_COLUMNS = [
    "emotion_model",
    "primary_emotion",
    "primary_confidence",
    "emotion_threshold",
    "predicted_emotions",
    "probability_joy",
    "probability_sadness",
    "probability_anger",
    "probability_fear",
    "probability_surprise",
    "probability_disgust",
]


def build_report_rows(records: list) -> list:
    """
    Convert fully processed records into flat report rows.

    Each record may contain:
        - employee_id
        - original_text
        - processed_text
        - sentiment
        - emotion
    """
    rows = []

    for record in records:
        sentiment = record.get("sentiment", {})

        row = {
            "employee_id": record.get("employee_id"),
            "original_text": record.get("original_text", ""),
            "processed_text": record.get("processed_text", ""),
            "classification": sentiment.get("classification"),
            "compound": sentiment.get("compound"),
            "positive": sentiment.get("positive"),
            "negative": sentiment.get("negative"),
            "neutral": sentiment.get("neutral"),
        }

        emotion = record.get("emotion")

        if emotion:
            probabilities = emotion.get("all_probabilities", {})

            predicted_emotions = emotion.get("predicted_emotions", [])

            row.update({
                "emotion_model": record.get("emotion_model", ""),
                "primary_emotion": emotion.get("primary_emotion", ""),
                "primary_confidence": emotion.get("primary_confidence", ""),
                "emotion_threshold": emotion.get("threshold", ""),
                "predicted_emotions": "; ".join(
                    item.get("emotion", "")
                    for item in predicted_emotions
                ),
                "probability_joy": probabilities.get("Joy", ""),
                "probability_sadness": probabilities.get("Sadness", ""),
                "probability_anger": probabilities.get("Anger", ""),
                "probability_fear": probabilities.get("Fear", ""),
                "probability_surprise": probabilities.get("Surprise", ""),
                "probability_disgust": probabilities.get("Disgust", ""),
            })

        rows.append(row)

    return rows


def compute_summary(rows: list) -> dict:
    """Compute total, positive, negative and neutral counts."""

    return {
        "total_samples": len(rows),
        "positive_count": sum(
            1 for row in rows
            if row.get("classification") == "Positive"
        ),
        "negative_count": sum(
            1 for row in rows
            if row.get("classification") == "Negative"
        ),
        "neutral_count": sum(
            1 for row in rows
            if row.get("classification") == "Neutral"
        ),
    }


def get_report_columns(rows: list) -> list:
    """
    Return Milestone 1 columns and, when emotion analysis exists,
    Milestone 2 columns.
    """

    has_emotion = any(
        "primary_emotion" in row
        for row in rows
    )

    if has_emotion:
        return REPORT_COLUMNS + EMOTION_COLUMNS

    return REPORT_COLUMNS


def save_csv_report(rows: list, output_path) -> Path:
    """Save report rows as a CSV file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = get_report_columns(rows)

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_path, index=False)

    logger.info(
        "CSV report saved: %s (%d row(s)).",
        output_path,
        len(rows),
    )

    return output_path


def save_html_report(rows: list, summary: dict, output_path) -> Path:
    """
    Save a human-readable HTML report.

    The report table is placed inside a horizontally scrollable
    container so Milestone 2 emotion columns remain readable.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = get_report_columns(rows)

    df = pd.DataFrame(rows, columns=columns)

    # Make the HTML table easier to read
    table_html = df.to_html(
        index=False,
        na_rep="",
        classes="report-table",
        border=0,
        justify="left",
    )

    html = f"""<!DOCTYPE html>
<html lang="en">

<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>Employee Wellness Analysis Report</title>

<style>

body {{
    font-family: Arial, sans-serif;
    margin: 2rem;
    color: #222;
    background: #ffffff;
}}

h1 {{
    font-size: 1.6rem;
    margin-bottom: 1rem;
}}

.summary {{
    margin-top: 1rem;
    margin-bottom: 1rem;
}}

.summary p {{
    margin: 0.5rem 0;
}}

.table-container {{
    width: 100%;
    overflow-x: auto;
    overflow-y: hidden;
    margin-top: 1.5rem;
    border: 1px solid #ddd;
}}

.report-table {{
    border-collapse: collapse;
    width: max-content;
    min-width: 100%;
}}

.report-table th,
.report-table td {{
    border: 1px solid #ccc;
    padding: 8px 12px;
    text-align: left;
    font-size: 0.9rem;
    vertical-align: top;
}}

.report-table th {{
    background: #f2f2f2;
    white-space: nowrap;
    position: sticky;
    top: 0;
}}

.report-table td {{
    white-space: normal;
    max-width: 350px;
    word-wrap: break-word;
}}

.note {{
    margin-top: 2rem;
    font-size: 0.85rem;
    color: #666;
}}

</style>
</head>

<body>

<h1>AI-Based Employee Wellness Platform — Analysis Report</h1>

<div class="summary">

<p>
<strong>Total Samples:</strong>
{summary["total_samples"]}
</p>

<p>
<strong>Positive:</strong>
{summary["positive_count"]}
&nbsp;&nbsp;

<strong>Negative:</strong>
{summary["negative_count"]}
&nbsp;&nbsp;

<strong>Neutral:</strong>
{summary["neutral_count"]}
</p>

</div>

<div class="table-container">
{table_html}
</div>

<p class="note">
Sentiment scores are generated by VADER and are a textual signal only.
They do not constitute a medical or psychological diagnosis.
</p>

</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")

    logger.info(
        "HTML report saved: %s.",
        output_path,
    )

    return output_path


def generate_report(records: list, csv_path, html_path=None) -> dict:
    """
    Build report rows, calculate summary, and save CSV and HTML reports.

    Returns:
        {
            "rows": [...],
            "summary": {...},
            "csv_path": Path,
            "html_path": Path | None
        }
    """

    rows = build_report_rows(records)

    summary = compute_summary(rows)

    saved_csv = save_csv_report(
        rows,
        csv_path,
    )

    saved_html = None

    if html_path:
        saved_html = save_html_report(
            rows,
            summary,
            html_path,
        )

    return {
        "rows": rows,
        "summary": summary,
        "csv_path": saved_csv,
        "html_path": saved_html,
    }