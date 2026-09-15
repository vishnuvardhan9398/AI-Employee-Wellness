import pytest

from src.isear import load_isear_csv


def test_isear_guilt_and_shame_are_excluded_not_invented(tmp_path):
    source = tmp_path / "isear.csv"
    source.write_text("text,emotion\nHappy news,joy\nI feel guilty,guilt\n", encoding="utf-8")
    assert load_isear_csv(source) == [{"text": "Happy news", "expected_emotion": "Joy"}]


def test_isear_requires_expected_schema(tmp_path):
    source = tmp_path / "bad.csv"
    source.write_text("sentence,label\ntext,joy\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_isear_csv(source)
