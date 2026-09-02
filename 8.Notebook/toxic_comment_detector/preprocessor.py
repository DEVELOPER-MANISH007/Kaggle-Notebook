"""
preprocessor.py
----------------
Text preprocessing pipeline for the Toxic Comment Detector.

This module is a faithful, importable version of the cleaning steps
performed inside `Notebook/main.ipynb` (Phase 3 - Text Cleaning), in the
EXACT same order used to build the training data:

    1. clean_basic_text   -> collapse whitespace / newlines / tabs
    2. remove_urls         -> strip http(s)/www/ftp links
    3. lowercase            -> str.lower()
    4. remove_stopwords    -> remove English stopwords, EXCEPT a small
                               "negation" allow-list (not, no, never, you,
                               your, i, me, my) that the notebook keeps
                               on purpose because they carry meaning for
                               toxicity detection
    5. replace_ip_addresses -> replace raw IPv4 addresses with the
                               literal token "IPADDRESS"

`preprocess_text()` runs all five steps in this exact order and is what
train.py and predict.py both call, so the SAME text transformation is
applied at training time and at inference time.
"""

import re

import nltk
from nltk.corpus import stopwords

# ---------------------------------------------------------------------------
# Make sure the NLTK stopwords corpus is available. This mirrors the
# `nltk.download("stopwords")` call in the notebook, but only downloads
# if it isn't already present so repeated app runs don't re-download.
# ---------------------------------------------------------------------------
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

_STOP_WORDS = set(stopwords.words("english"))

# Same negation allow-list used in the notebook (Step 3.7 - Custom
# Stopword Cleaning). These words are removed from the stopword set so
# they are KEPT in the cleaned text.
_CUSTOM_STOPWORDS = _STOP_WORDS - {
    "not",
    "no",
    "never",
    "you",
    "your",
    "i",
    "me",
    "my",
}

_URL_PATTERN = re.compile(
    r"(https?://\S+|www\.\S+|ftp://\S+)", flags=re.IGNORECASE
)
_IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_NEWLINE_TAB_PATTERN = re.compile(r"[\r\n\t]+")


def clean_basic_text(text: str) -> str:
    """Collapse newlines/tabs/extra whitespace and strip the string.

    Matches notebook cell (Step 3.2 - Text Cleaning Function).
    """
    text = str(text)
    text = _NEWLINE_TAB_PATTERN.sub(" ", text)
    text = _WHITESPACE_PATTERN.sub(" ", text)
    text = text.strip()
    return text


def remove_urls(text: str) -> str:
    """Strip http(s)/www/ftp links from the text.

    Matches notebook cell (Step 3.3 - URL & HTML Cleaning).
    """
    return _URL_PATTERN.sub(" ", text)


def remove_stopwords(text: str) -> str:
    """Remove English stopwords, keeping the negation allow-list.

    Matches notebook cell (Step 3.7 - Custom Stopword Cleaning).
    """
    return " ".join(
        word for word in text.split() if word not in _CUSTOM_STOPWORDS
    )


def replace_ip_addresses(text: str) -> str:
    """Replace raw IPv4 addresses with the literal token 'IPADDRESS'.

    Matches notebook cell (Step 3.8 - Numbers / IP addresses).
    """
    return _IP_PATTERN.sub(" IPADDRESS ", text)


def preprocess_text(text: str) -> str:
    """Run the full cleaning pipeline in the exact order used for training.

    1. clean_basic_text
    2. remove_urls
    3. lowercase
    4. remove_stopwords
    5. replace_ip_addresses
    """
    cleaned = clean_basic_text(text)
    cleaned = remove_urls(cleaned)
    cleaned = cleaned.lower()
    cleaned = remove_stopwords(cleaned)
    cleaned = replace_ip_addresses(cleaned)
    return cleaned
