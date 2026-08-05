"""
utils.py
========
Shared helper functions: input validation, number/currency formatting,
and Streamlit session-state helpers. Keeping these in one place avoids
duplicating logic across the Home / Prediction / About pages.
"""

import streamlit as st


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def format_currency(value: float) -> str:
    """Format a number as Indian-Rupee-style currency, e.g. 5,000,000 -> ₹50,00,000."""
    try:
        value = int(value)
    except (ValueError, TypeError):
        return str(value)

    s = str(abs(value))
    if len(s) > 3:
        last3 = s[-3:]
        rest = s[:-3]
        rest_grouped = ""
        while len(rest) > 2:
            rest_grouped = "," + rest[-2:] + rest_grouped
            rest = rest[:-2]
        rest_grouped = rest + rest_grouped
        formatted = rest_grouped + "," + last3
    else:
        formatted = s

    sign = "-" if value < 0 else ""
    return f"{sign}₹{formatted}"


def format_percent(value: float) -> str:
    """Format a 0-1 float as a percentage string, e.g. 0.9871 -> '98.71%'."""
    return f"{value * 100:.2f}%"


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def validate_inputs(user_input: dict) -> list:
    """
    Validate user-entered loan application values.

    Returns
    -------
    list[str] : validation error messages. Empty list means input is valid.
    """
    errors = []

    if user_input["no_of_dependents"] < 0 or user_input["no_of_dependents"] > 10:
        errors.append("Number of dependents must be between 0 and 10.")

    if user_input["income_annum"] <= 0:
        errors.append("Annual income must be greater than 0.")

    if user_input["loan_amount"] <= 0:
        errors.append("Loan amount must be greater than 0.")

    if user_input["loan_term"] <= 0:
        errors.append("Loan term must be greater than 0 years.")

    if not (300 <= user_input["cibil_score"] <= 900):
        errors.append("CIBIL score must be between 300 and 900.")

    for field in [
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
    ]:
        if user_input[field] < 0:
            errors.append(f"{field.replace('_', ' ').title()} cannot be negative.")

    if user_input["loan_amount"] > 0 and user_input["income_annum"] > 0:
        if user_input["loan_amount"] > user_input["income_annum"] * 50:
            errors.append(
                "Loan amount looks unusually high relative to annual income. "
                "Please double-check the values."
            )

    return errors


# --------------------------------------------------------------------------- #
# Session-state helpers
# --------------------------------------------------------------------------- #
def init_session_state():
    """Initialize all keys the app relies on in st.session_state."""
    defaults = {
        "prediction_result": None,
        "last_input": None,
        "prediction_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_prediction_to_history(user_input: dict, result: dict):
    """Append a prediction to the session's running history list."""
    entry = {**user_input, **result}
    st.session_state.prediction_history.append(entry)
