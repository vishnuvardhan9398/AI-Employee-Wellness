# AI-Based Employee Wellness Management Platform

## Overview

The AI-Based Employee Wellness Management Platform analyzes employee feedback text and produces structured sentiment and emotion insights.

**Milestone 1** provides feedback ingestion, validation, preprocessing, VADER sentiment analysis, and reporting. **Milestone 2** extends this foundation with transformer-based six-emotion detection using fine-tuned BERT and DistilBERT models.

> **Important:** Sentiment and emotion outputs are textual signals only. They are not medical or psychological diagnoses.

## Milestone 2 — Transformer-Based Emotion Analysis

M2 adds AI-based emotion detection for six target emotions:

- 😊 Joy
- 😢 Sadness
- 😡 Anger
- 😨 Fear
- 😮 Surprise
- 🤢 Disgust

The system combines VADER sentiment analysis with a fine-tuned transformer emotion model.

## Key Features

### Milestone 1
- Manual, TXT, and CSV employee feedback ingestion
- Input validation
- NLTK text preprocessing
- VADER sentiment analysis
- Positive / Negative / Neutral classification
- CSV and HTML reports
- End-to-end pipeline
- Automated pytest suite

### Milestone 2
- Six-emotion classification
- BERT and DistilBERT fine-tuning
- GoEmotions-based training dataset
- Sigmoid-based emotion probabilities
- Primary emotion and confidence
- Configurable threshold
- BERT vs DistilBERT evaluation
- Accuracy, Precision, Recall and Macro F1
- External ISEAR validation subset
- JSON and HTML validation reports
- Automated M2 tests

## M2 Architecture

```text
Employee Feedback
       |
       v
   Ingestion
       |
       v
  Validation
       |
       v
 Preprocessing
       |
       +--------------------+
       |                    |
       v                    v
VADER Sentiment          BERT Model
       |                    |
       v                    v
Positive/Negative      Six Emotions
/Neutral                    |
                            v
                  Primary Emotion
                  + Confidence
                  + Probabilities
       |                    |
       +---------+----------+
                 |
                 v
          CSV / HTML Report
```

## Project Structure

```text
AI_Employee_Wellness/
├── data/
│   ├── sample_feedback.txt
│   ├── sample_feedback.csv
│   ├── test_data/
│   ├── training_v2/
│   │   ├── emotion_train.csv
│   │   ├── emotion_validation.csv
│   │   └── emotion_test.csv
│   └── isear/
├── src/
│   ├── emotion.py
│   ├── emotion_labels.py
│   ├── evaluation.py
│   ├── ingestion.py
│   ├── isear.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   ├── report.py
│   ├── sentiment.py
│   ├── training.py
│   └── validation.py
├── tests/
├── models_v2/                 # Local trained models
├── output/reports/
├── main.py
├── prepare_goemotions.py
├── requirements.txt
├── pytest.ini
├── TEST_MATRIX.md
├── conftest.py
└── README.md
```

## Installation

```powershell
cd AI_Employee_Wellness
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### NLTK Resources

The first run may download:

- `punkt`
- `punkt_tab`
- `stopwords`
- `wordnet`
- `omw-1.4`

Manual download:

```powershell
python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','stopwords','wordnet','omw-1.4']]"
```

## Running the Project

### Milestone 1 — Manual Sentiment

```powershell
python main.py manual "I really enjoy my work and feel happy with my team."
```

### TXT

```powershell
python main.py txt data\sample_feedback.txt
```

### CSV

```powershell
python main.py csv data\sample_feedback.csv
```

Reports are written to:

```text
output/reports/sentiment_report.csv
output/reports/sentiment_report.html
```

## Milestone 2 — Training

The final M2 training data is in:

```text
data/training_v2/
├── emotion_train.csv
├── emotion_validation.csv
└── emotion_test.csv
```

Two transformer models were trained:

```text
BERT
DistilBERT
```

The selected M2 model is **BERT**.

Trained models are stored locally under:

```text
models_v2/
├── bert/
└── distilbert/
```

The trained `.safetensors` files are large and are not uploaded through the normal GitHub web uploader.

## M2 Evaluation Results

| Model | Accuracy | Precision | Recall | Macro F1 |
|---|---:|---:|---:|---:|
| **BERT** | **79.91%** | **77.35%** | **65.61%** | **69.78%** |
| DistilBERT | 78.50% | 75.04% | 64.52% | 68.83% |

**Selected model: BERT**

The selection rule prioritizes Macro F1, then subset accuracy, then precision.

## External Validation

A separate 12-sample validation subset was used.

```text
Samples:              12
Correct predictions:  11
Subset accuracy:      91.67%
```

This result applies specifically to the 12-sample validation subset and is not an estimate of overall real-world accuracy.

Reports:

```text
output/reports/isear_validation.json
output/reports/isear_validation.html
```

The validation data is kept separate from training data.

## Live M2 Demonstration

Use the final BERT model:

```powershell
python main.py manual "I am feeling very happy and motivated with my work. My team is supportive and I enjoy working on this project." --emotion-model bert --model-root models_v2 --threshold 0.5
```

The report contains:

```text
classification
emotion_model
primary_emotion
primary_confidence
emotion_threshold
predicted_emotions
probability_joy
probability_sadness
probability_anger
probability_fear
probability_surprise
probability_disgust
```

For the demonstrated feedback:

```text
Sentiment:       Positive
Primary Emotion: Joy
Confidence:      approximately 99.68%
```

Report:

```text
output/reports/sentiment_report.html
output/reports/sentiment_report.csv
```

## Testing

Run:

```powershell
python -m pytest -v
```

Current result:

```text
69 passed
2 skipped
71 total tests
```

Tests cover emotion logic, labels, evaluation, ingestion, ISEAR validation, pipelines, preprocessing, reports, sentiment, and input validation.

## Security and Privacy

- Do not commit `.env` or private tokens.
- Do not commit `.venv`, Python caches, or pytest caches.
- Employee feedback should be treated as sensitive data.
- Large trained model files should be handled separately from normal GitHub web uploads.

## Limitations

- VADER is a rule/lexicon-based sentiment baseline.
- Emotion predictions depend on training-data quality and distribution.
- The six-emotion training data is imbalanced.
- The 12-sample external validation set is too small to represent general real-world performance.
- Emotion detection is not a medical or psychological diagnosis.

## M2 Completion Summary

```text
Six-emotion detection       ✅
BERT training               ✅
DistilBERT training         ✅
Model comparison            ✅
BERT selected               ✅
Internal evaluation         ✅
External validation         ✅
CSV/HTML/JSON reporting     ✅
End-to-end pipeline         ✅
Automated testing           ✅
69 tests passed             ✅
```

# Milestone 2 — COMPLETED
