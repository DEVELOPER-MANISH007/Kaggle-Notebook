"""
prediction.py
=============
Prediction logic only. This module takes already-preprocessed feature
arrays and runs them through the trained model — it does not contain
any preprocessing or UI code, keeping concerns cleanly separated.
"""

import numpy as np


def predict_loan_status(model, label_encoder, features: np.ndarray) -> dict:
    """
    Run inference on a single (or batch of) preprocessed feature row(s).

    Parameters
    ----------
    model : trained GradientBoostingClassifier
    label_encoder : fitted LabelEncoder used on the target during training
                    (0 -> " Approved", 1 -> " Rejected")
    features : np.ndarray, shape (n_samples, n_features)
        Output of `preprocessing.transform_features`.

    Returns
    -------
    dict with:
        label            -> "Approved" or "Rejected" (cleaned, no leading space)
        raw_label        -> exact label_encoder inverse_transform output
        class_index      -> 0 or 1 (as predicted by the model)
        probability      -> float, model's confidence in the predicted class
        approve_prob     -> float, probability of "Approved"
        reject_prob      -> float, probability of "Rejected"
    """
    pred_class = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    raw_label = label_encoder.inverse_transform([pred_class])[0]
    clean_label = raw_label.strip()  # remove the dataset's leading-space quirk for display

    # Identify which encoded index corresponds to Approved / Rejected so the
    # probabilities are labeled correctly regardless of encoding order.
    classes = label_encoder.classes_  # e.g. [' Approved', ' Rejected']
    approve_idx = list(classes).index(
        next(c for c in classes if c.strip().lower() == "approved")
    )
    reject_idx = list(classes).index(
        next(c for c in classes if c.strip().lower() == "rejected")
    )

    return {
        "label": clean_label,
        "raw_label": raw_label,
        "class_index": int(pred_class),
        "probability": float(probabilities[pred_class]),
        "approve_prob": float(probabilities[approve_idx]),
        "reject_prob": float(probabilities[reject_idx]),
    }
