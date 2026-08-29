import pandas as pd

from src import report

SAMPLE_RECORDS = [
    {
        "employee_id": "E001",
        "original_text": "I enjoy working with my team.",
        "processed_text": "enjoy working team",
        "sentiment": {"positive": 0.5, "negative": 0.0, "neutral": 0.5,
                       "compound": 0.6, "classification": "Positive"},
    },
    {
        "employee_id": "E002",
        "original_text": "My workload is becoming stressful.",
        "processed_text": "workload becoming stressful",
        "sentiment": {"positive": 0.0, "negative": 0.4, "neutral": 0.6,
                       "compound": -0.5, "classification": "Negative"},
    },
    {
        "employee_id": "E003",
        "original_text": "The workplace environment is good.",
        "processed_text": "workplace environment good",
        "sentiment": {"positive": 0.3, "negative": 0.0, "neutral": 0.7,
                       "compound": 0.2, "classification": "Positive"},
    },
]


def test_build_report_rows_shape():
    rows = report.build_report_rows(SAMPLE_RECORDS)
    assert len(rows) == 3
    assert set(rows[0].keys()) == set(report.REPORT_COLUMNS)


def test_compute_summary_counts():
    rows = report.build_report_rows(SAMPLE_RECORDS)
    summary = report.compute_summary(rows)
    assert summary == {
        "total_samples": 3,
        "positive_count": 2,
        "negative_count": 1,
        "neutral_count": 0,
    }


def test_save_csv_report_creates_file_with_correct_columns(tmp_path):
    rows = report.build_report_rows(SAMPLE_RECORDS)
    csv_path = tmp_path / "reports" / "out.csv"
    saved_path = report.save_csv_report(rows, csv_path)

    assert saved_path.exists()
    df = pd.read_csv(saved_path)
    assert list(df.columns) == report.REPORT_COLUMNS
    assert len(df) == 3


def test_save_html_report_creates_file(tmp_path):
    rows = report.build_report_rows(SAMPLE_RECORDS)
    summary = report.compute_summary(rows)
    html_path = tmp_path / "reports" / "out.html"
    saved_path = report.save_html_report(rows, summary, html_path)

    assert saved_path.exists()
    content = saved_path.read_text(encoding="utf-8")
    assert "Total Samples" in content
    assert "3" in content


def test_generate_report_end_to_end(tmp_path):
    csv_path = tmp_path / "reports" / "report.csv"
    html_path = tmp_path / "reports" / "report.html"
    result = report.generate_report(SAMPLE_RECORDS, csv_path, html_path)

    assert result["summary"]["total_samples"] == 3
    assert result["csv_path"].exists()
    assert result["html_path"].exists()
