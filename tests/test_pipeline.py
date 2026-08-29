import pytest

from src import pipeline
from src.validation import ValidationError, SchemaValidationError

DATA = "data"
TEST_DATA = "data/test_data"


def test_manual_end_to_end(tmp_path):
    csv_path = tmp_path / "manual_report.csv"
    result = pipeline.run_manual(
        "I really enjoy my work and feel happy with my team.", csv_path
    )
    assert result["summary"]["total_samples"] == 1
    assert result["rows"][0]["employee_id"] is None
    assert result["rows"][0]["classification"] == "Positive"
    assert csv_path.exists()


def test_txt_end_to_end(tmp_path):
    csv_path = tmp_path / "txt_report.csv"
    result = pipeline.run_txt(f"{DATA}/sample_feedback.txt", csv_path)

    assert result["summary"]["total_samples"] == 3
    assert csv_path.exists()
    # original text preserved
    originals = [r["original_text"] for r in result["rows"]]
    assert "I enjoy working with my team." in originals
    # processed text generated (not identical to original)
    assert all(r["processed_text"] != "" for r in result["rows"])
    # sentiment scores present for every row
    assert all(r["compound"] is not None for r in result["rows"])


def test_csv_end_to_end(tmp_path):
    csv_path = tmp_path / "csv_report.csv"
    result = pipeline.run_csv(f"{DATA}/sample_feedback.csv", csv_path)

    assert result["summary"]["total_samples"] == 3
    employee_ids = [r["employee_id"] for r in result["rows"]]
    assert employee_ids == ["E001", "E002", "E003"]
    assert csv_path.exists()


def test_invalid_manual_input_does_not_reach_pipeline(tmp_path):
    csv_path = tmp_path / "should_not_exist.csv"
    with pytest.raises(ValidationError):
        pipeline.run_manual("   ", csv_path)
    assert not csv_path.exists()


def test_csv_missing_column_does_not_reach_pipeline(tmp_path):
    csv_path = tmp_path / "should_not_exist.csv"
    with pytest.raises(SchemaValidationError):
        pipeline.run_csv(f"{TEST_DATA}/missing_column.csv", csv_path)
    assert not csv_path.exists()


def test_multiple_records_report_count_matches_input(tmp_path):
    csv_path = tmp_path / "multi.csv"
    result = pipeline.run_txt(f"{TEST_DATA}/blank_lines.txt", csv_path)
    assert result["summary"]["total_samples"] == 3
    assert len(result["rows"]) == 3
