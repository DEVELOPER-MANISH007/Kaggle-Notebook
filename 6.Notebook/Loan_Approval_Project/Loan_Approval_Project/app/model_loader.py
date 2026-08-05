"""
model_loader.py
================
Responsible for loading the trained model and preprocessing artifacts
(scaler, encoder, label encoder, feature column lists) from the
`models/` directory.

The model is NOT retrained here — it is loaded exactly as it was
saved by `train_model.py`, which reproduces the original notebook's
training logic without modification.

Streamlit's `st.cache_resource` is used so the (relatively expensive)
pickle loading only happens once per app session, not on every rerun.
"""

import pickle
from pathlib import Path

import streamlit as st

# app/ -> project root -> models/
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _load_pickle(filename: str):
    """Load a single pickle file from the models directory."""
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact '{filename}' not found in {MODELS_DIR}. "
            f"Run `python train_model.py` from the project root first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading model and preprocessing artifacts...")
def load_artifacts():
    """
    Load and return all artifacts required for inference.

    Returns
    -------
    dict with keys:
        model            -> trained GradientBoostingClassifier
        scaler           -> fitted StandardScaler (numerical features)
        encoder          -> fitted OneHotEncoder (categorical features)
        label_encoder    -> fitted LabelEncoder (target: Approved/Rejected)
        numerical_cols   -> list[str], numerical feature column order
        categorical_cols -> list[str], categorical feature column order
    """
    model = _load_pickle("model.pkl")
    scaler = _load_pickle("scaler.pkl")
    encoder = _load_pickle("encoder.pkl")
    label_encoder = _load_pickle("label_encoder.pkl")
    feature_columns = _load_pickle("feature_columns.pkl")

    return {
        "model": model,
        "scaler": scaler,
        "encoder": encoder,
        "label_encoder": label_encoder,
        "numerical_cols": feature_columns["numerical_cols"],
        "categorical_cols": feature_columns["categorical_cols"],
    }
