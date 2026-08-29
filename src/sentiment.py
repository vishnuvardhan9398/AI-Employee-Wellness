"""
sentiment.py
------------
Baseline sentiment analysis using VADER (Valence Aware Dictionary and
sEntiment Reasoner).

All scores are generated dynamically by VaderSentiment's
SentimentIntensityAnalyzer — nothing here is hardcoded. Classification
uses VADER's own documented compound-score thresholds:

    compound >=  0.05  -> Positive
    compound <= -0.05  -> Negative
    otherwise           -> Neutral

IMPORTANT (per project scope): sentiment scores are a textual signal only.
They must never be presented or interpreted as a medical, psychological,
or clinical diagnosis of an employee's mental health.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

_analyzer = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def classify_compound(compound: float) -> str:
    """Classify a compound score into Positive / Negative / Neutral."""
    if compound >= POSITIVE_THRESHOLD:
        return "Positive"
    if compound <= NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def analyze_sentiment(text: str) -> dict:
    """
    Run VADER sentiment analysis on a piece of text.

    Args:
        text: The text to analyze. Per the preprocessing design decision,
              callers should pass `original_text` (VADER is tuned for
              natural text, not aggressively stripped/lemmatized text).

    Returns:
        {
            "positive": float,
            "negative": float,
            "neutral": float,
            "compound": float,
            "classification": "Positive" | "Negative" | "Neutral",
        }
    """
    analyzer = _get_analyzer()
    scores = analyzer.polarity_scores(text or "")

    return {
        "positive": scores["pos"],
        "negative": scores["neg"],
        "neutral": scores["neu"],
        "compound": scores["compound"],
        "classification": classify_compound(scores["compound"]),
    }
