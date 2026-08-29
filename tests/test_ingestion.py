import pytest

from src import ingestion
from src.validation import ValidationError, FileValidationError, SchemaValidationError

DATA = "data"
TEST_DATA = "data/test_data"


# ---------- Manual ----------

def test_manual_valid_text():
    record = ingestion.ingest_manual("I am happy with my work environment.")
    assert record == {"employee_id": None, "text": "I am happy with my work environment."}


def test_manual_empty_text_rejected():
    with pytest.raises(ValidationError):
        ingestion.ingest_manual("")


def test_manual_whitespace_only_rejected():
    with pytest.raises(ValidationError):
        ingestion.ingest_manual("     ")


def test_manual_strips_surrounding_whitespace():
    record = ingestion.ingest_manual("   I enjoy my job.   ")
    assert record["text"] == "I enjoy my job."


# ---------- TXT ----------

def test_txt_valid_file():
    records = ingestion.ingest_txt(f"{DATA}/sample_feedback.txt")
    assert len(records) == 3
    assert records[0]["text"] == "I enjoy working with my team."
    assert all(r["employee_id"] is None for r in records)


def test_txt_blank_lines_are_skipped_not_lost():
    records = ingestion.ingest_txt(f"{TEST_DATA}/blank_lines.txt")
    assert len(records) == 3
    texts = [r["text"] for r in records]
    assert "I enjoy working with my team." in texts
    assert "The workplace environment is good." in texts


def test_txt_empty_file_rejected():
    with pytest.raises(ValidationError):
        ingestion.ingest_txt(f"{TEST_DATA}/empty.txt")


def test_txt_whitespace_only_file_rejected():
    with pytest.raises(ValidationError):
        ingestion.ingest_txt(f"{TEST_DATA}/whitespace_only.txt")


def test_txt_missing_file_raises():
    with pytest.raises(FileValidationError):
        ingestion.ingest_txt(f"{TEST_DATA}/does_not_exist.txt")


def test_txt_unsupported_extension_rejected():
    with pytest.raises(FileValidationError):
        ingestion.ingest_txt(f"{TEST_DATA}/unsupported.docx")


# ---------- CSV ----------

def test_csv_valid_file():
    records = ingestion.ingest_csv(f"{DATA}/sample_feedback.csv")
    assert records == [
        {"employee_id": "E001", "text": "I enjoy working with my team."},
        {"employee_id": "E002", "text": "My workload is becoming stressful."},
        {"employee_id": "E003", "text": "The workplace environment is good."},
    ]


def test_csv_missing_feedback_column_rejected():
    with pytest.raises(SchemaValidationError):
        ingestion.ingest_csv(f"{TEST_DATA}/missing_column.csv")


def test_csv_blank_feedback_rows_skipped():
    records = ingestion.ingest_csv(f"{TEST_DATA}/blank_feedback.csv")
    assert len(records) == 1
    assert records[0]["employee_id"] == "E001"


def test_csv_malformed_raises_validation_error():
    with pytest.raises(ValidationError):
        ingestion.ingest_csv(f"{TEST_DATA}/malformed.csv")


def test_csv_missing_file_raises():
    with pytest.raises(FileValidationError):
        ingestion.ingest_csv(f"{TEST_DATA}/does_not_exist.csv")


def test_csv_unsupported_extension_rejected():
    with pytest.raises(FileValidationError):
        ingestion.ingest_csv(f"{DATA}/sample_feedback.txt")
