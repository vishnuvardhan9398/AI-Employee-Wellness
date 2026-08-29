"""
preprocessing.py
-----------------
Text preprocessing pipeline built on NLTK:

    normalize -> tokenize -> noise filtering -> stop-word removal -> lemmatize

Design decision (documented per project requirement):
VADER's lexicon and heuristics (e.g. capitalization, punctuation emphasis
like "!!!", degree modifiers) are tuned for natural, relatively raw text.
Aggressively stripping stop-words/punctuation and lemmatizing BEFORE
sentiment analysis measurably changes VADER's scores because it removes
negations ("not"), intensifiers, and casing cues.

To avoid destroying signal VADER needs, this module returns BOTH:
    - original_text  : untouched input, used for VADER sentiment scoring
    - processed_text  : normalized/tokenized/stop-word-removed/lemmatized
                         text, kept for reporting/inspection and for any
                         future milestone that needs a cleaned bag-of-words
                         representation (e.g. keyword/topic extraction).

Sentiment analysis (Task 3) uses `original_text`; the report (Task 4)
stores both.
"""

import logging
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

_REQUIRED_NLTK_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]


def ensure_nltk_resources():
    """
    Download required NLTK resources if not already present.

    Safe to call multiple times; only downloads what's missing. Requires
    internet access on first run. See README for offline setup notes.
    """
    for resource_path, package_name in _REQUIRED_NLTK_RESOURCES:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.info("Downloading missing NLTK resource: %s", package_name)
            nltk.download(package_name, quiet=True)


_lemmatizer = None


def _get_lemmatizer():
    global _lemmatizer
    if _lemmatizer is None:
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer


_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)


def normalize(text: str) -> str:
    """
    Normalize raw text: lowercase, collapse repeated whitespace, strip.
    Punctuation/emoji are NOT removed here (that happens during noise
    filtering) so this stage stays a pure "shape" cleanup step.
    """
    if text is None:
        return ""
    collapsed = re.sub(r"\s+", " ", text)
    return collapsed.strip().lower()


def tokenize(text: str) -> list:
    """Tokenize normalized text into words using NLTK's word_tokenize."""
    if not text:
        return []
    return word_tokenize(text)


def remove_noise(tokens: list) -> list:
    """
    Remove punctuation-only tokens and emoji-only tokens from a token list.
    Alphanumeric tokens (including ones with internal hyphens/apostrophes,
    e.g. "well-being", "it's") are preserved.
    """
    cleaned = []
    for token in tokens:
        if token in string.punctuation:
            continue
        stripped_emoji = _EMOJI_PATTERN.sub("", token)
        if stripped_emoji == "":
            continue
        # Drop tokens that are punctuation once emoji are stripped (e.g. "!!!")
        if all(ch in string.punctuation for ch in stripped_emoji):
            continue
        cleaned.append(stripped_emoji)
    return cleaned


def remove_stopwords(tokens: list) -> list:
    """Remove English stop-words from a token list."""
    stop_words = set(stopwords.words("english"))
    return [t for t in tokens if t.lower() not in stop_words]


def lemmatize(tokens: list) -> list:
    """Lemmatize each token to its dictionary/base form."""
    lemmatizer = _get_lemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess_text(text: str) -> dict:
    """
    Run the full preprocessing pipeline on a single piece of text.

    Returns:
        {
            "original_text": <untouched input, whitespace-trimmed>,
            "processed_text": <normalized, tokenized, noise/stopword-filtered,
                                lemmatized text, re-joined into a string>,
        }

    Empty input returns "" for processed_text rather than raising — an
    already-validated record should never reach here with empty text, but
    this keeps the function safe to call in isolation (e.g. from tests).
    """
    original_text = text if text is not None else ""

    normalized = normalize(original_text)
    tokens = tokenize(normalized)
    tokens = remove_noise(tokens)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)

    processed_text = " ".join(tokens)

    return {
        "original_text": original_text.strip(),
        "processed_text": processed_text,
    }
