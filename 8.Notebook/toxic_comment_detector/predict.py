"""
predict.py
----------
Loads the trained TF-IDF vectorizer, LinearSVC model, and optimized
decision threshold saved by `train.py`, and exposes a single
`predict_toxicity()` function that Streamlit (or anything else) can call.

This mirrors the notebook's own `predict_toxicity()` function
(Final Model - Custom Comment Testing section), but:
  - loads the artifacts from disk instead of relying on in-memory
    variables from a notebook session
  - additionally applies the IP-address replacement step so that
    preprocessing at inference time exactly matches preprocessing at
    training time (the notebook's inference cell skipped that one step
    by omission; train.py and this module both apply the full pipeline
    consistently)
  - returns a structured result instead of only printing

Label meaning (from the notebook, `toxic` column of the Jigsaw dataset):
  toxic == 1 -> "TOXIC"
  toxic == 0 -> "NON-TOXIC"

Note on confidence: LinearSVC does not implement predict_proba(), so no
probability score exists. What IS available is the raw SVM
decision_function() score -- a signed distance from the decision
boundary (more positive = more confidently toxic, more negative = more
confidently non-toxic). We surface that as a "decision score", never as
a fake percentage confidence.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from preprocessor import preprocess_text


def resolve_project_root() -> Path:
    """Return the real project root, whether the app lives one level below Data or next to it."""
    current = BASE_DIR
    for candidate in (current, current.parent):
        if (candidate / "Data").exists() or (candidate / "Notebook").exists():
            return candidate
    return current.parent if current.name != "toxic_comment_detector" else current


PROJECT_ROOT = resolve_project_root()
MODEL_DIR = PROJECT_ROOT / "model"

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
MODEL_PATH = MODEL_DIR / "toxic_svm_model.pkl"
THRESHOLD_PATH = MODEL_DIR / "decision_threshold.pkl"

LABELS = {0: "NON-TOXIC", 1: "TOXIC"}

_tfidf = None
_svm_model = None
_threshold = None


class ArtifactsNotFoundError(FileNotFoundError):
    """Raised when the trained model/vectorizer/threshold files are missing."""


def check_model_artifacts() -> bool:
    required = [VECTORIZER_PATH, MODEL_PATH, THRESHOLD_PATH]
    return all(path.exists() for path in required)


def load_artifacts(force_reload: bool = False):
    global _tfidf, _svm_model, _threshold

    if not force_reload and _tfidf is not None and _svm_model is not None:
        return _tfidf, _svm_model, _threshold

    if not check_model_artifacts():
        missing = "\n  - ".join(str(path) for path in [VECTORIZER_PATH, MODEL_PATH, THRESHOLD_PATH] if not path.exists())
        raise ArtifactsNotFoundError(
            "Trained model artifacts are missing:\n  - "
            f"{missing}\n\n"
            "Run 'python train.py' from the project root or use the automatic app initialization."
        )

    _tfidf = joblib.load(VECTORIZER_PATH)
    _svm_model = joblib.load(MODEL_PATH)
    _threshold = joblib.load(THRESHOLD_PATH)
    return _tfidf, _svm_model, _threshold


def predict_toxicity(comment: str) -> dict:
    if comment is None or not str(comment).strip():
        raise ValueError("Comment is empty. Please enter some text.")

    tfidf, svm_model, threshold = load_artifacts()
    cleaned = preprocess_text(comment)
    comment_tfidf = tfidf.transform([cleaned])
    decision_score = float(svm_model.decision_function(comment_tfidf)[0])

    prediction = int(decision_score >= threshold)
    label = LABELS[prediction]

    return {
        "comment": comment,
        "cleaned_text": cleaned,
        "label": label,
        "is_toxic": bool(prediction),
        "decision_score": decision_score,
        "threshold": threshold,
    }


if __name__ == "__main__":
    sample = input("Enter a comment: ")
    result = predict_toxicity(sample)
    print("=" * 50)
    print("Comment  :", result["comment"])
    print("Result   :", result["label"])
    print(f"Score    : {result['decision_score']:.4f}")
    print(f"Threshold: {result['threshold']:.2f}")
    print("=" * 50)
