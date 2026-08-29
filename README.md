# AI-Based Employee Wellness Management Platform

## Overview

This platform ingests employee feedback text from multiple sources (manual
entry, `.txt` files, `.csv` files), validates and preprocesses it, and runs
baseline sentiment analysis using **VADER** to produce a sentiment-oriented
report. It is designed as a modular foundation that later milestones can
extend toward richer employee wellness analysis.

**Sentiment analysis is a textual signal, not a medical diagnosis.**
Negative sentiment in feedback text does not mean an employee has a
mental-health condition. VADER classifies the *tone of the text*, nothing
more.

## Current Milestone

**Milestone 1 — Text Ingestion & Baseline Sentiment**

## Features

- Manual text ingestion
- `.txt` file ingestion
- `.csv` file ingestion
- Input validation (empty/whitespace input, missing files, unsupported
  extensions, missing CSV columns, blank CSV rows, malformed CSV)
- Text preprocessing (normalize, tokenize, noise filtering, stop-word
  removal, lemmatization) via NLTK
- VADER sentiment analysis (dynamic, non-hardcoded scores)
- CSV + HTML sentiment reporting with summary counts
- Full pipeline integration (ingestion → preprocessing → sentiment → report)
- Automated pytest test suite

## Project Structure

```text
AI_Employee_Wellness/
├── data/
│   ├── sample_feedback.txt
│   ├── sample_feedback.csv
│   └── test_data/            # edge-case fixtures used by the tests
├── src/
│   ├── ingestion.py           # manual / TXT / CSV -> standardized records
│   ├── validation.py          # validation rules + custom exceptions
│   ├── preprocessing.py       # NLTK normalize/tokenize/stopwords/lemmatize
│   ├── sentiment.py           # VADER scoring + classification
│   ├── report.py              # CSV/HTML report + summary generation
│   └── pipeline.py            # orchestrates the end-to-end workflow
├── tests/                     # pytest suite (unit + integration)
├── output/reports/            # generated reports land here
├── main.py                    # CLI entry point
└── requirements.txt
```

## Installation

```bash
cd AI_Employee_Wellness
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dependencies

- **pandas** — CSV reading/writing
- **nltk** — tokenization, stop-words, WordNet lemmatization
- **vaderSentiment** — baseline sentiment scoring
- **pytest** — automated testing

### One-time NLTK data download

The first run automatically downloads the required NLTK resources
(`punkt`, `punkt_tab`, `stopwords`, `wordnet`, `omw-1.4`) via
`preprocessing.ensure_nltk_resources()`. This requires internet access
once. To pre-download manually instead:

```bash
python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','stopwords','wordnet','omw-1.4']]"
```

## Running the Project

```bash
# Manual text
python main.py manual "I really enjoy my work and feel happy with my team."

# TXT file
python main.py txt data/sample_feedback.txt

# CSV file
python main.py csv data/sample_feedback.csv
```

Each run prints a summary and writes:

- `output/reports/sentiment_report.csv`
- `output/reports/sentiment_report.html`

## Running Tests

```bash
pytest
```

Run with detail:

```bash
pytest -v
```

## Sample Input

`data/sample_feedback.txt`
```text
I enjoy working with my team.
My workload is becoming stressful.
The workplace environment is good.
```

`data/sample_feedback.csv`
```csv
employee_id,feedback
E001,I enjoy working with my team.
E002,My workload is becoming stressful.
E003,The workplace environment is good.
```

## Sample Output

```text
--- Milestone 1 Run Summary ---
Total Samples : 3
Positive      : 2
Negative      : 1
Neutral       : 0
CSV Report    : output/reports/sentiment_report.csv
HTML Report   : output/reports/sentiment_report.html
```

`sentiment_report.csv` columns: `employee_id, original_text, processed_text,
classification, compound, positive, negative, neutral`.

## Milestone 1 Validation

| Task | Description | Status |
|------|-------------|--------|
| 1 | Manual / TXT / CSV ingestion + validation (empty, whitespace, missing file, unsupported extension, missing column, blank rows, malformed CSV) | Implemented, covered by `tests/test_ingestion.py`, `tests/test_validation.py` |
| 2 | Preprocessing pipeline (normalize, tokenize, noise filter, stop-words, lemmatize); `original_text` preserved alongside `processed_text` | Implemented, covered by `tests/test_preprocessing.py` |
| 3 | VADER sentiment scoring + classification, no hardcoded values | Implemented, covered by `tests/test_sentiment.py` |
| 4 | CSV + HTML report generation with summary counts | Implemented, covered by `tests/test_report.py` |
| 5 | End-to-end pipeline integration (manual/TXT/CSV → report) | Implemented, covered by `tests/test_pipeline.py` |

See `TEST_MATRIX.md` for the full manual test matrix (T01–T12).

## Limitations

VADER is a lexicon/rule-based **baseline** sentiment system. It is not a
clinical or psychological diagnostic tool, and its output should never be
used to infer an individual's mental-health status. Future milestones may
add more advanced wellness-oriented analysis on top of this foundation.
