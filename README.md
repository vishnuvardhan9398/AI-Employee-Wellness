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

**Milestone 2 — Transformer multi-label emotion analysis.** Milestone 1
remains available unchanged as the default CLI workflow.

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
- Real Hugging Face `bert-base-uncased` and `distilbert-base-uncased` model
  creation for six-label, sigmoid-based emotion classification
- Local fine-tuning, model/tokenizer saving, and offline saved-model loading
- Configurable threshold, primary emotion, and all six dynamic probabilities
- Dynamic subset accuracy, macro precision, recall, macro F1, and model ranking
- Held-out local ISEAR CSV support with explicit mapping and result reports

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
│   ├── emotion.py             # BERT/DistilBERT loading, sigmoid inference, save/load
│   ├── emotion_labels.py      # one canonical six-label ordering
│   ├── training.py            # CSV preparation + BCE-with-logits fine-tuning
│   ├── evaluation.py          # dynamic multi-label metrics and comparison
│   └── isear.py               # separate local ISEAR validation/reporting
├── models/                    # generated local models (ignored by Git)
├── tests/                     # pytest suite (unit + integration)
├── output/reports/            # generated reports land here
├── main.py                    # CLI entry point
└── requirements.txt
```

## Installation

In Windows PowerShell, with 64-bit Python 3.13 installed:

```powershell
cd AI_Employee_Wellness
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.13 / Windows compatibility was checked before selecting the
dependencies: current PyTorch publishes CPython 3.13 Windows x86-64 wheels,
and Transformers supports Python 3.9+. The `torch>=2.6,<2.14` range avoids
older releases that predate practical Python 3.13 support. If `pip` reports no
matching Torch wheel, confirm that Python and Windows are both 64-bit, then use
the CPU selector at pytorch.org for the command appropriate to that machine.

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

### Train and use a real emotion model

The included six-row CSV is only a schema/demo smoke dataset, not a valid
research training set. Supply a larger, ethically sourced labelled dataset
before making claims about model quality. Labels may be six `0`/`1` columns
(`Joy`, `Sadness`, `Anger`, `Fear`, `Surprise`, `Disgust`) or a `labels` column
containing JSON (for example `["Joy", "Fear"]`) or semicolon-separated labels.

```powershell
# Downloads the public pretrained encoder on first use, fine-tunes it, then saves
# both model and matching tokenizer under models\bert\.
python main.py train bert data\training\emotion_training_example.csv --epochs 2 --batch-size 2
python main.py train distilbert data\training\emotion_training_example.csv --epochs 2 --batch-size 2

# Full Milestone 1 -> saved DistilBERT workflow.
python main.py manual "I am excited but nervous about the outcome." --emotion-model distilbert --threshold 0.50
python main.py csv data\sample_feedback.csv --emotion-model bert --threshold 0.50
```

Emotion probabilities are computed with `torch.sigmoid(model(...).logits)`;
they are never constants. Every output contains all six probabilities, the
highest-probability primary emotion, and every label at or above the threshold.
When no label crosses it, `predicted_emotions` is an empty list while the
primary emotion is still returned so the caller can present a low-confidence
result safely. The emotion model receives the original validated text: this
preserves transformer context and reuses (rather than duplicates) Milestone 1
ingestion/validation/preprocessing/VADER processing.

### Evaluation

Use `src.evaluation.calculate_metrics(y_true, probabilities, threshold)` after
running each saved model on the same held-out labelled CSV. It computes exact
match **subset accuracy** (all six thresholded labels must match for a sample),
plus macro precision, macro recall and macro F1 with `zero_division=0`.
`compare_models` chooses the model by macro F1, then subset accuracy, then
precision. Values are calculated from the supplied labels and predictions; no
example metric values are stored in this repository.

For a ready-made terminal comparison, use a labelled CSV kept separate from
training (same schema as the training CSV):

```powershell
python main.py evaluate data\held_out_emotions.csv --threshold 0.50
```

### ISEAR held-out validation

Place a legally obtained ISEAR subset outside the training CSV (for example
`data/isear/held_out.csv`, ignored by Git) using this format:

```csv
text,emotion
I was thrilled when I received the good news.,joy
```

`src.isear.load_isear_csv` supports the source labels `joy`, `sadness`,
`anger`, `fear`, `surprise`, and `disgust`, mapping them directly to the six
project labels. ISEAR `guilt` and `shame` have no valid project-label
equivalent, so they are deliberately excluded rather than invented or merged.
Call `validate_isear(model, tokenizer, csv_path)` with the chosen saved model,
then `save_isear_report(result, "output/reports")` to create actual JSON and
HTML results containing metrics, emotion-wise performance, confidence scores,
incorrect predictions and the applied mapping. Do not train on that file; it is
held out by design.

The equivalent CLI command is:

```powershell
python main.py isear distilbert data\isear\held_out.csv --threshold 0.50
```

### Security

Public model downloads work without credentials. If a private/gated resource is
used, copy `.env.example` to `.env` and set `HUGGINGFACE_TOKEN` locally. The
application reads it only from the environment; `.env`, generated models, and
ISEAR data are ignored by Git. Tokens are never printed, logged, reported, or
hardcoded.

## Running Tests

```powershell
python -m pytest -q
```

Run with detail:

```powershell
python -m pytest -v

# Optional real BERT + DistilBERT download/save/load/inference smoke test.
$env:RUN_TRANSFORMER_INTEGRATION = "1"
python -m pytest -m integration -v
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

## Validation scope

| Task | Description | Status |
|------|-------------|--------|
| 1 | Manual / TXT / CSV ingestion + validation (empty, whitespace, missing file, unsupported extension, missing column, blank rows, malformed CSV) | Implemented, covered by `tests/test_ingestion.py`, `tests/test_validation.py` |
| 2 | Preprocessing pipeline (normalize, tokenize, noise filter, stop-words, lemmatize); `original_text` preserved alongside `processed_text` | Implemented, covered by `tests/test_preprocessing.py` |
| 3 | VADER sentiment scoring + classification, no hardcoded values | Implemented, covered by `tests/test_sentiment.py` |
| 4 | CSV + HTML report generation with summary counts | Implemented, covered by `tests/test_report.py` |
| 5 | End-to-end pipeline integration (manual/TXT/CSV → report) | Implemented, covered by `tests/test_pipeline.py` |

Milestone 2 adds fast tests for label ordering, threshold output, invalid inputs,
metrics, ISEAR mapping, plus an explicitly opt-in real-model integration test.
The opt-in test does not fine-tune and makes no accuracy claim; it verifies each
real model can be downloaded, saved, loaded locally, and returns six sigmoid
probabilities. Fine-tuning and ISEAR reports must be run against the user's
actual datasets before reporting performance.

See `TEST_MATRIX.md` for the full manual test matrix (T01–T12).

## Limitations

VADER is a lexicon/rule-based **baseline** sentiment system. It is not a
clinical or psychological diagnostic tool, and its output should never be
used to infer an individual's mental-health status. Future milestones may
add more advanced wellness-oriented analysis on top of this foundation.
