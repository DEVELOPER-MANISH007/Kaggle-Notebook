"""
preprocessing.py
=================
Replicates — exactly, with no changes to the logic — the preprocessing
pipeline built in the notebook (Phase 5 : Data Preprocessing):

    ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ]
    )

At inference time we don't have a live ColumnTransformer object (the
app loads the fitted `scaler` and `encoder` separately, per the
required models/ folder layout), so this module manually reproduces
the exact column order the ColumnTransformer produced:
    [scaled numerical columns] + [one-hot encoded categorical columns]

IMPORTANT (data quirk preserved intentionally):
The original dataset's categorical values contain a LEADING SPACE
(e.g. " Graduate", " Not Graduate", " No", " Yes") because only the
*column headers* were stripped in the notebook (`df.columns =
df.columns.str.strip()`) — the cell *values* were never stripped.
The OneHotEncoder was therefore fit on these exact strings. To keep
the ML logic byte-for-byte identical, this module maps clean UI
labels back to those exact trained category strings before encoding.
"""

import numpy as np
import pandas as pd

# Maps clean, human-friendly UI values -> exact category strings the
# OneHotEncoder was originally fit on (including the leading space
# quirk present in the source dataset).
_EDUCATION_MAP = {
    "Graduate": " Graduate",
    "Not Graduate": " Not Graduate",
}
_SELF_EMPLOYED_MAP = {
    "No": " No",
    "Yes": " Yes",
}


def build_raw_dataframe(user_input: dict) -> pd.DataFrame:
    """
    Convert the raw user input (from the Streamlit form) into a single-row
    DataFrame with the exact same column names/dtypes the model was
    trained on.

    Parameters
    ----------
    user_input : dict
        Keys: no_of_dependents, education, self_employed, income_annum,
        loan_amount, loan_term, cibil_score, residential_assets_value,
        commercial_assets_value, luxury_assets_value, bank_asset_value

    Returns
    -------
    pd.DataFrame with a single row.
    """
    row = {
        "no_of_dependents": user_input["no_of_dependents"],
        "education": _EDUCATION_MAP[user_input["education"]],
        "self_employed": _SELF_EMPLOYED_MAP[user_input["self_employed"]],
        "income_annum": user_input["income_annum"],
        "loan_amount": user_input["loan_amount"],
        "loan_term": user_input["loan_term"],
        "cibil_score": user_input["cibil_score"],
        "residential_assets_value": user_input["residential_assets_value"],
        "commercial_assets_value": user_input["commercial_assets_value"],
        "luxury_assets_value": user_input["luxury_assets_value"],
        "bank_asset_value": user_input["bank_asset_value"],
    }
    return pd.DataFrame([row])


def handle_missing_values(df: pd.DataFrame, numerical_cols: list, categorical_cols: list) -> pd.DataFrame:
    """
    Defensive missing-value handling for inference-time input.

    The original dataset had NO missing values (confirmed in the
    notebook's EDA), so no imputation strategy was trained. For
    robustness against unexpected blank/NaN input from the UI, we
    apply a simple, transparent fallback:
        - numerical columns  -> fill with 0
        - categorical columns -> fill with the first known category

    This does not alter the trained ML logic in any way — it only
    guards the app against malformed input before it reaches the
    scaler/encoder.
    """
    df = df.copy()
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "")
    return df


def transform_features(
    raw_df: pd.DataFrame,
    scaler,
    encoder,
    numerical_cols: list,
    categorical_cols: list,
) -> np.ndarray:
    """
    Apply the fitted StandardScaler + OneHotEncoder to raw input,
    concatenated in the same column order the notebook's
    ColumnTransformer produced: [num features] + [cat features].

    Returns
    -------
    np.ndarray ready to be passed directly into `model.predict()` /
    `model.predict_proba()`.
    """
    raw_df = handle_missing_values(raw_df, numerical_cols, categorical_cols)

    # Numerical: StandardScaler transform (order must match training)
    num_values = scaler.transform(raw_df[numerical_cols])

    # Categorical: OneHotEncoder transform (drop="first", handle_unknown="ignore")
    cat_values = encoder.transform(raw_df[categorical_cols])
    if hasattr(cat_values, "toarray"):
        cat_values = cat_values.toarray()

    features = np.hstack([num_values, cat_values])
    return features
