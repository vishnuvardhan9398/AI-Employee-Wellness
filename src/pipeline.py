"""
pipeline.py
-----------
Orchestrates the full Milestone 1 workflow:

    Input Source (manual / TXT / CSV)
        -> Ingestion
        -> Validation (performed inside ingestion)
        -> Standard Records
        -> Preprocessing
        -> Sentiment Analysis
        -> Classification
        -> Report Generation

This module contains no business logic of its own — it wires the other
modules together and is the single place that defines the end-to-end
control flow, so each stage stays independently testable.
"""

import logging

from src import ingestion, preprocessing, sentiment, report

logger = logging.getLogger(__name__)


def _enrich_record(record: dict) -> dict:
    """
    Take a standardized {"employee_id", "text"} record through
    preprocessing and sentiment analysis.

    Sentiment is computed on the ORIGINAL text (see preprocessing.py for
    the documented reasoning), while processed_text is retained for the
    report.
    """
    pre = preprocessing.preprocess_text(record["text"])
    sent = sentiment.analyze_sentiment(pre["original_text"])

    return {
        "employee_id": record.get("employee_id"),
        "original_text": pre["original_text"],
        "processed_text": pre["processed_text"],
        "sentiment": sent,
    }


def run_from_records(records: list, csv_path, html_path=None) -> dict:
    """Run preprocessing -> sentiment -> report for a list of standardized records."""
    if not records:
        raise ValueError("Pipeline received no records to process.")

    enriched = [_enrich_record(r) for r in records]
    logger.info("Preprocessing and sentiment analysis completed for %d record(s).", len(enriched))

    result = report.generate_report(enriched, csv_path, html_path)
    logger.info("Report generated. Summary: %s", result["summary"])
    return result


def run_manual(text: str, csv_path, html_path=None) -> dict:
    """End-to-end pipeline: manual text -> report."""
    record = ingestion.ingest_manual(text)
    return run_from_records([record], csv_path, html_path)


def run_txt(path, csv_path, html_path=None) -> dict:
    """End-to-end pipeline: TXT file -> report."""
    records = ingestion.ingest_txt(path)
    return run_from_records(records, csv_path, html_path)


def run_csv(path, csv_path, html_path=None) -> dict:
    """End-to-end pipeline: CSV file -> report."""
    records = ingestion.ingest_csv(path)
    return run_from_records(records, csv_path, html_path)
