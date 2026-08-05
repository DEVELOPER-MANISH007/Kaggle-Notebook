"""
train_model.py
================
Training script extracted from Notebook/loan_approval.ipynb.

This script reproduces — WITHOUT ANY CHANGES — the exact data cleaning,
preprocessing (StandardScaler + OneHotEncoder) and final model
(tuned Gradient Boosting Classifier, selected in the notebook after
baseline comparison + RandomizedSearchCV hyperparameter tuning) from
the original notebook.

Run this script once to (re)generate the artifacts consumed by the
Streamlit app:
    models/model.pkl     -> trained GradientBoostingClassifier
    models/scaler.pkl    -> fitted StandardScaler (numerical features)
    models/encoder.pkl   -> fitted OneHotEncoder (categorical features)
    models/label_encoder.pkl -> fitted LabelEncoder for the target
    models/feature_columns.pkl -> ordered lists of numerical/categorical
                                   columns, needed to rebuild the exact
                                   ColumnTransformer column order at
                                   inference time.

NOTE: Training logic (algorithms, hyperparameter search space, random
states, scoring metric) is copied verbatim from the notebook. Nothing
about the ML approach has been altered — only reorganized into a
reusable script.
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "loan_approval_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_and_clean_data(path: Path) -> pd.DataFrame:
    """Step 3 / Step 4.1: load dataset, strip column names, drop loan_id.

    (No missing values / duplicates were found in the original notebook's
    EDA, so no imputation or duplicate handling is required here either.)
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()          # remove whitespace from headers
    df.drop("loan_id", axis=1, inplace=True)      # identifier column, no predictive value
    return df


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list, list]:
    """Step 5.2: Build the exact ColumnTransformer used in the notebook."""
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numerical_cols = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ]
    )
    return preprocessor, numerical_cols, categorical_cols


def train_and_tune(X_train_t, y_train):
    """Step 8.4: Gradient Boosting hyperparameter tuning via RandomizedSearchCV.

    This is the exact search space / settings used in the notebook for the
    model that was ultimately selected as the final model (Phase 9).
    """
    gt = GradientBoostingClassifier(random_state=42)
    params_dist = {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4, 5],
        "subsample": [0.8, 0.9, 1.0],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    gradient_search = RandomizedSearchCV(
        estimator=gt,
        param_distributions=params_dist,
        n_iter=30,
        cv=5,
        scoring="f1",
        random_state=42,
        n_jobs=-1,
    )
    gradient_search.fit(X_train_t, y_train)
    print("Best Parameters :", gradient_search.best_params_)
    print("Best CV F1 Score:", gradient_search.best_score_)
    return gradient_search.best_estimator_


def evaluate(model, X_test_t, y_test):
    """Phase 9: Final model evaluation (identical metrics to the notebook)."""
    y_pred = model.predict(X_test_t)
    print("-" * 50)
    print("GradientBoostingClassifier - Final Model")
    print("-" * 50)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report")
    print(classification_report(y_test, y_pred))


def main():
    # Step 4: Load + clean
    df = load_and_clean_data(DATA_PATH)

    # Step 5.1: Feature / target split
    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    # Step 5.2: Build preprocessing pipeline
    preprocessor, numerical_cols, categorical_cols = build_preprocessor(X)

    # Encode target (0 = Approved, 1 = Rejected -- matches notebook's LabelEncoder)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Step 5.4: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Fit preprocessing on train, transform both splits
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Phase 8: Hyperparameter tuning -> final selected model
    best_model = train_and_tune(X_train_t, y_train)

    # Phase 9: Evaluation
    evaluate(best_model, X_test_t, y_test)

    # --------------------------------------------------------------------- #
    # Persist artifacts for the Streamlit app
    # --------------------------------------------------------------------- #
    scaler = preprocessor.named_transformers_["num"]
    encoder = preprocessor.named_transformers_["cat"]

    with open(MODELS_DIR / "model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(MODELS_DIR / "encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    with open(MODELS_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    with open(MODELS_DIR / "feature_columns.pkl", "wb") as f:
        pickle.dump(
            {"numerical_cols": numerical_cols, "categorical_cols": categorical_cols}, f
        )

    print(f"\n✅ Artifacts saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
