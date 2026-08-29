
# Milestone 1 — Test Matrix

| Test ID | Category    | Test Case                   | Expected Result   | Automated Test |
|---------|-------------|------------------------------|--------------------|-----------------|
| T01 | Manual | Valid text | Accepted | `test_ingestion.py::test_manual_valid_text` |
| T02 | Manual | Empty text | Rejected | `test_ingestion.py::test_manual_empty_text_rejected` |
| T03 | Manual | Whitespace text | Rejected | `test_ingestion.py::test_manual_whitespace_only_rejected` |
| T04 | TXT | Valid TXT | Read successfully | `test_ingestion.py::test_txt_valid_file` |
| T05 | TXT | Empty TXT | Rejected | `test_ingestion.py::test_txt_empty_file_rejected` |
| T06 | TXT | Missing file | Error handled | `test_ingestion.py::test_txt_missing_file_raises` |
| T07 | CSV | Valid CSV | Read successfully | `test_ingestion.py::test_csv_valid_file` |
| T08 | CSV | Missing feedback column | Rejected | `test_ingestion.py::test_csv_missing_feedback_column_rejected` |
| T09 | CSV | Blank feedback | Handled (skipped, counted) | `test_ingestion.py::test_csv_blank_feedback_rows_skipped` |
| T10 | CSV | Malformed CSV | Error handled | `test_ingestion.py::test_csv_malformed_raises_validation_error` |
| T11 | File | Unsupported extension | Rejected | `test_ingestion.py::test_txt_unsupported_extension_rejected` |
| T12 | Integration | Valid records passed onward | Successful | `test_pipeline.py::test_txt_end_to_end`, `test_csv_end_to_end`, `test_manual_end_to_end` |

Additional coverage beyond the base matrix:

- Preprocessing: tokenization, stop-word removal, lemmatization, noise
  filtering, punctuation/emoji handling, repeated spaces, empty input,
  short vs. long text (`tests/test_preprocessing.py`).
- Sentiment: required score fields, numeric types, positive/negative/neutral
  classification, dynamic (non-hardcoded) scoring across multiple additional
  examples (`tests/test_sentiment.py`).
- Report: correct columns, correct row/summary counts, CSV + HTML output
  (`tests/test_report.py`).
- Integration: invalid input never reaches the report stage; record counts
  match input counts; employee IDs and original text preserved
  (`tests/test_pipeline.py`).
