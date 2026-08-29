import pytest

from src import validation


def test_validate_manual_text_accepts_valid():
    assert validation.validate_manual_text("Hello team") == "Hello team"


def test_validate_manual_text_strips_whitespace():
    assert validation.validate_manual_text("  Hello team  ") == "Hello team"


def test_validate_manual_text_rejects_empty():
    with pytest.raises(validation.ValidationError):
        validation.validate_manual_text("")


def test_validate_manual_text_rejects_whitespace_only():
    with pytest.raises(validation.ValidationError):
        validation.validate_manual_text("     ")


def test_validate_manual_text_rejects_none():
    with pytest.raises(validation.ValidationError):
        validation.validate_manual_text(None)


def test_validate_file_exists_raises_for_missing_file(tmp_path):
    missing = tmp_path / "nope.txt"
    with pytest.raises(validation.FileValidationError):
        validation.validate_file_exists(missing)


def test_validate_extension_accepts_allowed(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("hi")
    assert validation.validate_extension(f, {".txt"}) == f


def test_validate_extension_rejects_disallowed(tmp_path):
    f = tmp_path / "sample.docx"
    f.write_text("hi")
    with pytest.raises(validation.FileValidationError):
        validation.validate_extension(f, {".txt"})


def test_find_feedback_column_found():
    assert validation.find_feedback_column(["employee_id", "feedback"]) == "feedback"


def test_find_feedback_column_case_insensitive():
    assert validation.find_feedback_column(["Employee_ID", "Feedback"]) == "Feedback"


def test_find_feedback_column_missing_raises():
    with pytest.raises(validation.SchemaValidationError):
        validation.find_feedback_column(["employee_id", "comments_only_field"])


def test_find_employee_id_column_optional():
    assert validation.find_employee_id_column(["feedback"]) is None
    assert validation.find_employee_id_column(["employee_id", "feedback"]) == "employee_id"


def test_is_blank():
    assert validation.is_blank(None) is True
    assert validation.is_blank("") is True
    assert validation.is_blank("   ") is True
    assert validation.is_blank(float("nan")) is True
    assert validation.is_blank("hello") is False
