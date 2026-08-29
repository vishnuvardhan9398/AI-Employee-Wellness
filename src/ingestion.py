# .\venv\Scripts\Activate.ps1


"""
ingestion.py
------------
Reads input from manual text, TXT files, and CSV files, and converts each
into the platform's standard internal record format:

    {"employee_id": <str or None>, "text": <str>}

This module is deliberately the ONLY place that knows where feedback comes
from. Everything downstream (preprocessing, sentiment, reporting) works
against standardized records and does not care about the original source.
"""

import csv
import logging
from pathlib import Path

import pandas as pd

from src.validation import (
    validate_manual_text,
    validate_txt_file,
    validate_csv_file,
    validate_txt_content,
    find_feedback_column,
    find_employee_id_column,
    is_blank,
    ValidationError,
    SchemaValidationError,
)

logger = logging.getLogger(__name__)


def ingest_manual(text) -> dict:
    """
    Ingest a single piece of manually entered text.

    Returns:
        A standardized record: {"employee_id": None, "text": <cleaned text>}

    Raises:
        ValidationError: if the text is empty or whitespace-only.
    """
    cleaned = validate_manual_text(text)
    logger.info("Manual input accepted.")
    return {"employee_id": None, "text": cleaned}


def ingest_txt(path) -> list:
    """
    Ingest feedback records from a .txt file — one feedback item per
    non-blank line.

    Returns:
        A list of standardized records: [{"employee_id": None, "text": ...}, ...]

    Raises:
        FileValidationError: file missing or wrong extension.
        ValidationError: file has no usable (non-blank) content.
    """
    file_path = validate_txt_file(path)

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"TXT file '{file_path}' could not be decoded as UTF-8: {exc}")

    validate_txt_content(content)

    records = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if stripped == "":
            # Blank lines are ignored, not treated as data loss.
            continue
        records.append({"employee_id": None, "text": stripped})

    if not records:
        raise ValidationError(
            f"TXT file '{file_path}' rejected: contains only blank lines, no usable text."
        )

    logger.info("TXT file loaded: %s (%d record(s)).", file_path, len(records))
    return records


def ingest_csv(path) -> list:
    """
    Ingest feedback records from a .csv file.

    Expects a feedback/text column (see validation.FEEDBACK_COLUMN_CANDIDATES)
    and optionally an employee ID column (see validation.EMPLOYEE_ID_COLUMN_CANDIDATES).

    Rows with blank feedback are skipped (not silently merged into other rows)
    and counted, so callers can see how many were dropped.

    Returns:
        A list of standardized records: [{"employee_id": ..., "text": ...}, ...]

    Raises:
        FileValidationError: file missing or wrong extension.
        SchemaValidationError: no feedback column found.
        ValidationError: file is empty, malformed, or has no usable rows.
    """
    file_path = validate_csv_file(path)

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise ValidationError(f"CSV file '{file_path}' rejected: file is empty.")
    except (pd.errors.ParserError, UnicodeDecodeError, csv.Error) as exc:
        raise ValidationError(f"CSV file '{file_path}' rejected: malformed CSV ({exc}).")

    if df.empty and len(df.columns) == 0:
        raise ValidationError(f"CSV file '{file_path}' rejected: file is empty.")

    feedback_col = find_feedback_column(df.columns)
    employee_id_col = find_employee_id_column(df.columns)

    records = []
    blank_count = 0
    for _, row in df.iterrows():
        raw_text = row[feedback_col]
        if is_blank(raw_text):
            blank_count += 1
            continue

        employee_id = None
        if employee_id_col is not None and not is_blank(row[employee_id_col]):
            employee_id = str(row[employee_id_col]).strip()

        records.append({"employee_id": employee_id, "text": str(raw_text).strip()})

    if blank_count:
        logger.info("Skipped %d row(s) with blank feedback in '%s'.", blank_count, file_path)

    if not records:
        raise ValidationError(
            f"CSV file '{file_path}' rejected: no usable rows "
            f"(all rows had blank feedback, or the file had no data rows)."
        )

    logger.info("CSV file loaded: %s (%d record(s), %d blank skipped).",
                file_path, len(records), blank_count)
    return records
