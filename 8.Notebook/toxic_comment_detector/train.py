"""
train.py
--------
Reproduces the model-selection pipeline from `Notebook/main.ipynb` and
PERSISTS the trained artifacts to `model/` so the Streamlit app can load
them without retraining.

Pipeline (matches the notebook exactly):
    1. Load Data/train.csv (Jigsaw-style toxic comment dataset)
    2. Clean text with preprocessor.preprocess_text()
    3. Train/test split (80/20, stratified on the binary `toxic` label)
    4. TF-IDF vectorize:
         TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                          min_df=2, max_df=0.95, sublinear_tf=True)
    5. Train a baseline LinearSVC(random_state=42) -- this was the model
       selected in the notebook's final comparison (Section 4.4-4.8):
         it beat Logistic Regression / Naive Bayes / class-weight-balanced
         SVM on F1, and hyperparameter tuning (GridSearchCV over C) did not
         improve on the baseline, so the baseline SVM is kept.
    6. Sweep decision thresholds from -1.0 to 1.0 (step 0.05) over
       `decision_function()` scores on the test set and pick the threshold
       that maximizes F1 for the toxic class (Section 4.8). In the
       notebook this came out to ~-0.20.
    7. Save the fitted TfidfVectorizer, the fitted LinearSVC, and the
       chosen threshold to `model/` using joblib.

Run this once from the `toxic_comment_detector/` directory:

    python train.py

Requires `Data/train.csv` to exist one level up (i.e. `../Data/train.csv`),
matching the project's folder layout.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from preprocessor import preprocess_text


def resolve_project_root() -> Path:
    for candidate in (BASE_DIR, BASE_DIR.parent):
        if (candidate / "Data").exists() or (candidate / "Notebook").exists():
            return candidate
    return BASE_DIR.parent if BASE_DIR.name == "toxic_comment_detector" else BASE_DIR


PROJECT_ROOT = resolve_project_root()
DATA_PATH = PROJECT_ROOT / "Data" / "train.csv"
MODEL_DIR = PROJECT_ROOT / "model"

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
MODEL_PATH = MODEL_DIR / "toxic_svm_model.pkl"
THRESHOLD_PATH = MODEL_DIR / "decision_threshold.pkl"


def train_model() -> dict:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find training data at '{DATA_PATH}'. "
            "Make sure Data/train.csv exists at the project root."
        )

    print("Loading data from:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape)

    print("Cleaning text...")
    df["clean_text"] = df["comment_text"].fillna("").apply(preprocess_text)

    X = df["clean_text"]
    y = df["toxic"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print("X_train:", X_train.shape, "X_test:", X_test.shape)

    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print("Vocabulary size:", len(tfidf.vocabulary_))

    print("Training LinearSVC...")
    svm_model = LinearSVC(random_state=42)
    svm_model.fit(X_train_tfidf, y_train)

    y_pred = svm_model.predict(X_test_tfidf)
    print("Baseline LinearSVC:")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1 Score : {f1_score(y_test, y_pred):.4f}")

    print("Sweeping decision thresholds...")
    decision_scores = svm_model.decision_function(X_test_tfidf)
    thresholds = np.arange(-1.0, 1.01, 0.05)

    best_threshold = 0.0
    best_f1 = -1.0
    for threshold in thresholds:
        y_pred_t = (decision_scores >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred_t, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    y_pred_final = (decision_scores >= best_threshold).astype(int)
    print("Threshold-optimized LinearSVC:")
    print(f"  Threshold: {best_threshold:.2f}")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred_final):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_final):.4f}")
    print(f"  Recall   : {recall_score(y_test, y_pred_final):.4f}")
    print(f"  F1 Score : {f1_score(y_test, y_pred_final):.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(tfidf, VECTORIZER_PATH)
    joblib.dump(svm_model, MODEL_PATH)
    joblib.dump(best_threshold, THRESHOLD_PATH)

    print("\nSaved artifacts:")
    print(" -", VECTORIZER_PATH)
    print(" -", MODEL_PATH)
    print(" -", THRESHOLD_PATH)

    return {
        "vectorizer_path": str(VECTORIZER_PATH),
        "model_path": str(MODEL_PATH),
        "threshold_path": str(THRESHOLD_PATH),
        "best_threshold": best_threshold,
    }


def main():
    train_model()


if __name__ == "__main__":
    main()
