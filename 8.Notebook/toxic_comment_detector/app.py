"""Streamlit app with automatic model initialization."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from predict import ArtifactsNotFoundError, check_model_artifacts, load_artifacts, predict_toxicity
from train import train_model

st.set_page_config(
    page_title="Toxic Comment Detector",
    page_icon="🛡️",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def initialize_model():
    """Load existing artifacts or train them automatically if missing."""
    if check_model_artifacts():
        model, vectorizer, threshold = load_artifacts()
        st.success("✅ Model loaded successfully.")
        return model, vectorizer, threshold

    st.warning("🔧 Model not found. Initializing the ML pipeline...")
    try:
        with st.spinner("📚 Training model..."):
            train_model()
        model, vectorizer, threshold = load_artifacts()
        st.success("✅ Model trained successfully.")
        return model, vectorizer, threshold
    except Exception as exc:  # pragma: no cover - surfaced to UI
        st.error("⚠️ Model initialization failed.")
        st.code(traceback.format_exc())
        raise RuntimeError(f"Model initialization failed: {exc}") from exc


st.title("🛡️ Toxic Comment Detector")
st.subheader("Detect whether a comment is toxic using NLP and Machine Learning.")
st.write("")

try:
    _, _, _ = initialize_model()
    model_ready = True
except Exception as exc:
    model_ready = False
    st.error(f"Unable to initialize the model: {exc}")

comment = st.text_area(
    "Enter your comment here...",
    height=140,
    placeholder="Type or paste a comment to check...",
    disabled=not model_ready,
)

predict_clicked = st.button("Predict", type="primary", disabled=not model_ready)

if predict_clicked:
    if not comment or not comment.strip():
        st.warning("Please enter a comment before clicking Predict.")
    else:
        try:
            with st.spinner("Analyzing comment..."):
                result = predict_toxicity(comment)

            st.write("---")
            st.markdown("### Prediction Result")

            col1, col2 = st.columns(2)

            if result["is_toxic"]:
                with col1:
                    st.metric("Prediction", "🚨 TOXIC")
                st.error("This comment was classified as **TOXIC** by the model.")
            else:
                with col1:
                    st.metric("Prediction", "✅ NON-TOXIC")
                st.success("This comment was classified as **NON-TOXIC** by the model.")

            with col2:
                st.metric(
                    "Model Decision Score",
                    f"{result['decision_score']:.3f}",
                    help=(
                        "Raw SVM decision score, not a probability. "
                        f"Comment is predicted TOXIC when this score is >= {result['threshold']:.2f}."
                    ),
                )

            with st.expander("Comment & processing details"):
                st.write("**Comment:**")
                st.write(result["comment"])
                st.write("**Cleaned text (after preprocessing):**")
                st.code(result["cleaned_text"] or "(empty after cleaning)")
                st.write(f"**Decision threshold used:** {result['threshold']:.2f}")

        except ValueError as exc:
            st.warning(str(exc))
        except ArtifactsNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:  # pragma: no cover
            st.error("Something went wrong while making the prediction.")
            with st.expander("Technical details"):
                st.code(str(exc))

st.write("")
st.write("---")

with st.expander("ℹ️ How it works"):
    st.markdown(
        """
1. **Text preprocessing** — comment text is cleaned using the same steps from the notebook.
2. **Feature extraction** — the cleaned text is transformed with the saved TF-IDF vectorizer.
3. **Model prediction** — the saved Linear SVM scores the text.
4. **Classification** — the decision score is compared to the tuned threshold to output **TOXIC** or **NON-TOXIC**.
        """
    )

st.caption("Built on the trained NLP pipeline from the project notebook and training script.")
