"""
validation.py
--------------
Centralized input validation rules for the AI-Based Employee Wellness
Management Platform.

This module defines custom exception types and reusable validation
functions used by the ingestion layer. It contains no I/O logic — it only
decides whether a given piece of input is acceptable, and if not, WHY.
"""

from pathlib import Path

SUPPORTED_TEXT_EXTENSIONS = {".txt"}
SUPPORTED_CSV_EXTENSIONS = {".csv"}
FEEDBACK_COLUMN_CANDIDATES = ["feedback", "text", "comment", "comments", "response"]
EMPLOYEE_ID_COLUMN_CANDIDATES = ["employee_id", "emp_id", "id", "employeeid"]


class ValidationError(Exception):
    """Raised when input fails a validation rule (empty, malformed, etc.)."""


class FileValidationError(ValidationError):
    """Raised for file-related validation problems (missing file, wrong type)."""


class SchemaValidationError(ValidationError):
    """Raised when a CSV file does not contain the required columns."""


def validate_manual_text(text) -> str:
    """
    Validate manual text input.

    Rules:
        - Must be a string.
        - Must not be empty after stripping surrounding whitespace.

    Returns:
        The cleaned (stripped) text.

    Raises:
        ValidationError: if text is None, not a string, empty, or whitespace-only.
    """
    if text is None:
        raise ValidationError("Manual input rejected: text is None.")
    if not isinstance(text, str):
        raise ValidationError(f"Manual input rejected: expected str, got {type(text).__name__}.")

    cleaned = text.strip()
    if cleaned == "":
        if text == "":
            raise ValidationError("Manual input rejected: text is empty.")
        raise ValidationError("Manual input rejected: text is whitespace-only.")

    return cleaned


def validate_file_exists(path) -> Path:
    """Validate that a file path exists and is a regular file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileValidationError(f"File not found: '{file_path}'.")
    if not file_path.is_file():
        raise FileValidationError(f"Path is not a file: '{file_path}'.")
    return file_path


def validate_extension(path, allowed_extensions) -> Path:
    """Validate that a file has one of the allowed extensions."""
    file_path = Path(path)
    if file_path.suffix.lower() not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise FileValidationError(
            f"Unsupported file extension '{file_path.suffix}' for '{file_path.name}'. "
            f"Allowed extension(s): {allowed}."
        )
    return file_path


def validate_txt_file(path) -> Path:
    """Validate a .txt file exists and has the correct extension."""
    file_path = validate_file_exists(path)
    return validate_extension(file_path, SUPPORTED_TEXT_EXTENSIONS)


def validate_csv_file(path) -> Path:
    """Validate a .csv file exists and has the correct extension."""
    file_path = validate_file_exists(path)
    return validate_extension(file_path, SUPPORTED_CSV_EXTENSIONS)


def validate_txt_content(content: str) -> str:
    """
    Validate that TXT file content is not empty/whitespace-only overall.

    Individual blank lines inside an otherwise non-empty file are allowed
    (they are skipped during record extraction) — this only rejects a file
    that has NO usable content at all.
    """
    if content is None or content.strip() == "":
        raise ValidationError("TXT file rejected: file is empty or contains only whitespace.")
    return content


def find_feedback_column(columns) -> str:
    """
    Identify the feedback/text column from a CSV header.

    Raises:
        SchemaValidationError: if no recognizable feedback column is found.
    """
    normalized = {c.strip().lower(): c for c in columns}
    for candidate in FEEDBACK_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    raise SchemaValidationError(
        f"CSV rejected: no feedback/text column found. "
        f"Expected one of {FEEDBACK_COLUMN_CANDIDATES}, got columns {list(columns)}."
    )


def find_employee_id_column(columns):
    """
    Identify the employee ID column from a CSV header, if present.

    Returns:
        The original column name, or None if not present (employee_id is optional).
    """
    normalized = {c.strip().lower(): c for c in columns}
    for candidate in EMPLOYEE_ID_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    return None


def is_blank(value) -> bool:
    """Return True if a value is None, NaN-like, or a whitespace-only string."""
    if value is None:
        return True
    try:
        if isinstance(value, float) and value != value:  # NaN check without importing pandas/math
            return True
    except Exception:
        pass
    if isinstance(value, str) and value.strip() == "":
        return True
    return False
