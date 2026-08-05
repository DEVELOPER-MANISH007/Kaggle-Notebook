"""
app.py
======
Main Streamlit entry point for the Loan Approval Prediction application.

Pages:
    🏠 Home        -> project overview + model information
    🔍 Prediction  -> input form + live prediction
    ℹ️  About       -> dataset info, model details, tech stack

Run with:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow running via `streamlit run app/app.py` from any working directory
sys.path.append(str(Path(__file__).resolve().parent))

from model_loader import load_artifacts
from prediction import predict_loan_status
from preprocessing import build_raw_dataframe, transform_features
from utils import (
    format_currency,
    format_percent,
    init_session_state,
    save_prediction_to_history,
    validate_inputs,
)

BASE_DIR = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Custom theme / CSS
# --------------------------------------------------------------------------- #
CUSTOM_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    :root {
        --primary-color: #145DA0;
        --secondary-color: #0C2D48;
        --accent-color: #2E8BC0;
        --success-color: #1B998B;
        --danger-color: #E5533C;

        /* Text color scale used INSIDE white cards only (never applied globally) */
        --text-heading: #111827;
        --text-body: #374151;
        --text-label: #4B5563;
        --text-muted: #6B7280;
    }

    /* ---- Typography (scoped to the main content area, not the whole app) ---- */
    .main, .main .block-container {
        font-family: "Inter", sans-serif;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    .main p, .main li, .main span, .main label {
        font-size: 16px;
        line-height: 1.6;
    }
    /* Baseline heading sizes for the main content area. Component-specific
       rules below (e.g. .app-hero h1) intentionally come AFTER this block
       so they win the cascade for their own scoped context. */
    .main h1 { font-size: 42px; color: var(--text-heading); }
    .main h2 { font-size: 28px; color: var(--text-heading); }
    .main h3, .main h4 { font-size: 22px; color: var(--text-heading); }

    /* ============================================================= */
    /* Hero header - blue gradient, white text (intentional, on-brand) */
    /* ============================================================= */
    .app-hero {
        background: linear-gradient(135deg, var(--secondary-color) 0%, var(--primary-color) 60%, var(--accent-color) 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(12, 45, 72, 0.25);
    }
    .app-hero h1 {
        color: #FFFFFF;
        font-size: 42px;
        line-height: 1.25;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
    }
    .app-hero p {
        color: #E4F1FE;
        font-size: 17px;
        line-height: 1.6;
        margin: 0;
    }

    /* ============================================================= */
    /* White cards - metric cards, section/info cards                */
    /* All text inside these MUST be dark (fixes invisible-text bug) */
    /* ============================================================= */
    .metric-card,
    .section-card {
        background: #FFFFFF;
        border: 1px solid #E3ECF3;
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 4px 14px rgba(12, 45, 72, 0.08);
    }
    .metric-card { padding: 1.2rem 1.4rem; height: 100%; }
    .section-card { margin-bottom: 1.2rem; }

    .metric-card h3,
    .section-card h1,
    .section-card h2,
    .section-card h3 {
        color: var(--text-heading);
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }
    .metric-card h3 { font-size: 15px; color: var(--text-label); font-weight: 600; }
    .section-card h3 { font-size: 22px; line-height: 1.4; }

    .metric-card .value {
        font-size: 28px;
        font-weight: 800;
        color: var(--text-heading);
    }

    .section-card p,
    .section-card li,
    .section-card span {
        color: var(--text-body);
        font-size: 16px;
        line-height: 1.6;
    }
    .section-card ul { margin: 0.4rem 0 0 0; padding-left: 1.2rem; }
    .section-card b, .section-card strong { color: var(--text-heading); }

    /* Belt-and-braces: catch any stray text node inside a white card */
    .metric-card, .metric-card * { color: var(--text-heading); }
    .section-card, .section-card * { color: var(--text-body); }
    .section-card h1, .section-card h2, .section-card h3,
    .section-card h1 *, .section-card h2 *, .section-card h3 * {
        color: var(--text-heading);
    }

    /* ============================================================= */
    /* Result cards (prediction outcome)                              */
    /* ============================================================= */
    .result-approved {
        background: linear-gradient(135deg, #DDF6EE 0%, #C4F1E0 100%);
        border: 1.5px solid var(--success-color);
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        text-align: center;
        box-shadow: 0 4px 14px rgba(27, 153, 139, 0.15);
    }
    .result-approved h2 { color: #10715F; font-size: 28px; margin: 0 0 0.3rem 0; }
    .result-approved p  { color: #145643; font-size: 16px; margin: 0; }

    .result-rejected {
        background: linear-gradient(135deg, #FCE4DF 0%, #FAD1C8 100%);
        border: 1.5px solid var(--danger-color);
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        text-align: center;
        box-shadow: 0 4px 14px rgba(229, 83, 60, 0.15);
    }
    .result-rejected h2 { color: #B03A26; font-size: 28px; margin: 0 0 0.3rem 0; }
    .result-rejected p  { color: #7A2A1B; font-size: 16px; margin: 0; }

    /* ============================================================= */
    /* Buttons                                                        */
    /* ============================================================= */
    .stButton>button {
        background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
        color: #FFFFFF;
        font-weight: 600;
        font-family: "Inter", sans-serif;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1.4rem;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(20, 93, 160, 0.35);
    }

    /* ============================================================= */
    /* Sidebar - dark theme, kept as-is                                */
    /* ============================================================= */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0C2D48 0%, #145DA0 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #EAF4FC;
        font-family: "Inter", sans-serif;
    }
    section[data-testid="stSidebar"] .stRadio label { font-size: 16px; }

    /* ============================================================= */
    /* Data table (entered-values summary) - ensure dark, readable text */
    /* ============================================================= */
    div[data-testid="stDataFrame"] * {
        color: var(--text-body);
    }

    .footer-note {
        text-align: center;
        color: var(--text-muted);
        font-size: 14px;
        margin-top: 2rem;
    }

    /* Responsive tweaks */
    @media (max-width: 768px) {
        .app-hero h1 { font-size: 30px; }
        .section-card h3 { font-size: 19px; }
        .app-hero { padding: 1.6rem 1.4rem; }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Session state + artifacts
# --------------------------------------------------------------------------- #
init_session_state()
artifacts = load_artifacts()

# --------------------------------------------------------------------------- #
# Sidebar navigation
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 🏦 Loan Approval AI")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔍 Prediction", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Model:** Gradient Boosting Classifier")
    st.markdown("**Test Accuracy:** 98.71%")
    st.markdown("**F1 Score:** 98.26%")
    st.markdown("---")
    st.caption("Built with Streamlit · scikit-learn")


# =============================================================================
# HOME PAGE
# =============================================================================
def render_home():
    st.markdown(
        """
        <div class="app-hero">
            <h1>🏦 Loan Approval Prediction System</h1>
            <p>An AI-powered tool that predicts whether a loan application is likely
            to be Approved or Rejected, based on applicant demographics, financial
            status, employment information, and credit history.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("Accuracy", "98.71%"),
        ("Precision", "98.73%"),
        ("Recall", "97.80%"),
        ("F1 Score", "98.26%"),
    ]
    for col, (label, value) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>{label}</h3>
                    <div class="value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1.3, 1])
    with col_a:
        st.markdown(
            """
            <div class="section-card">
                <h3>📌 Problem Statement</h3>
                <p>Financial institutions receive thousands of loan applications every
                day. Manually reviewing each application is time-consuming and may
                lead to inconsistent decisions.</p>
                <p>This project builds a Machine Learning model that predicts loan
                approval status using applicant demographic details, financial
                status, employment information, and credit history — helping banks
                automate screening, reduce risk, and improve decision-making
                efficiency.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div class="section-card">
                <h3>🤖 Model Information</h3>
                <p><b>Algorithm:</b> Gradient Boosting Classifier</p>
                <p><b>Selection process:</b> 10 baseline models compared, top 3
                tuned via RandomizedSearchCV (5-fold CV, F1-optimized).</p>
                <p><b>Preprocessing:</b> StandardScaler (numerical features) +
                OneHotEncoder (categorical features).</p>
                <p><b>ROC-AUC:</b> 0.9986</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="section-card">
            <h3>🚀 Get Started</h3>
            <p>Head to the <b>Prediction</b> page from the sidebar to enter applicant
            details and get an instant loan approval prediction with confidence
            scores.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# PREDICTION PAGE
# =============================================================================
def render_prediction():
    st.markdown(
        """
        <div class="app-hero">
            <h1>🔍 Loan Prediction</h1>
            <p>Enter the applicant's details below to predict loan approval status.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        st.markdown("#### 👤 Applicant Details")
        c1, c2, c3 = st.columns(3)
        with c1:
            no_of_dependents = st.number_input(
                "Number of Dependents", min_value=0, max_value=10, value=0, step=1
            )
        with c2:
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        with c3:
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])

        st.markdown("#### 💰 Financial Details")
        c4, c5, c6 = st.columns(3)
        with c4:
            income_annum = st.number_input(
                "Annual Income (₹)", min_value=0, value=5_000_000, step=100_000
            )
        with c5:
            loan_amount = st.number_input(
                "Loan Amount (₹)", min_value=0, value=15_000_000, step=100_000
            )
        with c6:
            loan_term = st.number_input(
                "Loan Term (years)", min_value=1, max_value=30, value=10, step=1
            )

        st.markdown("#### 📊 Credit History")
        cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, value=650)

        st.markdown("#### 🏠 Asset Details")
        c7, c8 = st.columns(2)
        with c7:
            residential_assets_value = st.number_input(
                "Residential Assets Value (₹)", min_value=0, value=5_000_000, step=100_000
            )
            luxury_assets_value = st.number_input(
                "Luxury Assets Value (₹)", min_value=0, value=10_000_000, step=100_000
            )
        with c8:
            commercial_assets_value = st.number_input(
                "Commercial Assets Value (₹)", min_value=0, value=3_000_000, step=100_000
            )
            bank_asset_value = st.number_input(
                "Bank Asset Value (₹)", min_value=0, value=3_000_000, step=100_000
            )

        submitted = st.form_submit_button("🔮 Predict Loan Status", use_container_width=True)

    if submitted:
        user_input = {
            "no_of_dependents": no_of_dependents,
            "education": education,
            "self_employed": self_employed,
            "income_annum": income_annum,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil_score,
            "residential_assets_value": residential_assets_value,
            "commercial_assets_value": commercial_assets_value,
            "luxury_assets_value": luxury_assets_value,
            "bank_asset_value": bank_asset_value,
        }

        # ---- Input validation ----
        errors = validate_inputs(user_input)
        if errors:
            for err in errors:
                st.error(f"⚠️ {err}")
            return

        # ---- Show entered values before prediction ----
        st.markdown("#### 📝 Entered Values")
        summary_df = pd.DataFrame(
            {
                "Field": [
                    "Number of Dependents", "Education", "Self Employed",
                    "Annual Income", "Loan Amount", "Loan Term (years)",
                    "CIBIL Score", "Residential Assets", "Commercial Assets",
                    "Luxury Assets", "Bank Assets",
                ],
                "Value": [
                    no_of_dependents, education, self_employed,
                    format_currency(income_annum), format_currency(loan_amount), loan_term,
                    cibil_score, format_currency(residential_assets_value),
                    format_currency(commercial_assets_value),
                    format_currency(luxury_assets_value), format_currency(bank_asset_value),
                ],
            }
        )
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

        # ---- Preprocess + predict ----
        with st.spinner("Analyzing application..."):
            raw_df = build_raw_dataframe(user_input)
            features = transform_features(
                raw_df,
                scaler=artifacts["scaler"],
                encoder=artifacts["encoder"],
                numerical_cols=artifacts["numerical_cols"],
                categorical_cols=artifacts["categorical_cols"],
            )
            result = predict_loan_status(
                model=artifacts["model"],
                label_encoder=artifacts["label_encoder"],
                features=features,
            )

        st.session_state.prediction_result = result
        st.session_state.last_input = user_input
        save_prediction_to_history(user_input, result)

        # ---- Result display ----
        st.markdown("#### 🎯 Prediction Result")
        if result["label"] == "Approved":
            st.markdown(
                f"""
                <div class="result-approved">
                    <h2>✅ Loan Approved</h2>
                    <p>Confidence: <b>{format_percent(result['probability'])}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-rejected">
                    <h2>❌ Loan Rejected</h2>
                    <p>Confidence: <b>{format_percent(result['probability'])}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- Probability chart ----
        fig = go.Figure(
            data=[
                go.Bar(
                    x=["Approved", "Rejected"],
                    y=[result["approve_prob"] * 100, result["reject_prob"] * 100],
                    marker_color=["#1B998B", "#E5533C"],
                    text=[
                        f"{result['approve_prob']*100:.2f}%",
                        f"{result['reject_prob']*100:.2f}%",
                    ],
                    textposition="outside",
                )
            ]
        )
        fig.update_layout(
            title="Prediction Probability Breakdown",
            yaxis_title="Probability (%)",
            yaxis_range=[0, 105],
            height=350,
            margin=dict(t=50, b=20, l=20, r=20),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# ABOUT PAGE
# =============================================================================
def render_about():
    st.markdown(
        """
        <div class="app-hero">
            <h1>ℹ️ About This Project</h1>
            <p>Dataset information, model details, and the technologies used to
            build this application.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            <div class="section-card">
                <h3>📂 Dataset Information</h3>
                <ul>
                    <li><b>Records:</b> 4,269 loan applications</li>
                    <li><b>Target:</b> loan_status (Approved / Rejected)</li>
                    <li><b>Class balance:</b> 62.22% Approved, 37.78% Rejected</li>
                    <li><b>Features:</b> dependents, education, employment,
                        income, loan amount, loan term, CIBIL score, and
                        4 asset-value features</li>
                    <li><b>Missing values:</b> None</li>
                    <li><b>Duplicates:</b> None</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-card">
                <h3>🛠️ Technologies Used</h3>
                <ul>
                    <li>Python</li>
                    <li>Pandas &amp; NumPy — data handling</li>
                    <li>scikit-learn — preprocessing &amp; modeling</li>
                    <li>Matplotlib &amp; Seaborn — EDA (notebook)</li>
                    <li>Streamlit — web application</li>
                    <li>Plotly — interactive charts</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div class="section-card">
                <h3>🤖 Model Details</h3>
                <ul>
                    <li><b>Final model:</b> Gradient Boosting Classifier</li>
                    <li><b>Selected from:</b> 10 baseline models
                        (Logistic Regression, KNN, SVM, Decision Tree,
                        Random Forest, Extra Trees, AdaBoost, Gradient
                        Boosting, GaussianNB, LDA)</li>
                    <li><b>Tuning:</b> RandomizedSearchCV, 30 iterations,
                        5-fold CV, optimized for F1 score</li>
                    <li><b>Preprocessing:</b> StandardScaler (numerical) +
                        OneHotEncoder with drop-first (categorical)</li>
                </ul>
                <h3>📊 Final Performance</h3>
                <ul>
                    <li><b>Accuracy:</b> 98.71%</li>
                    <li><b>Precision:</b> 98.73%</li>
                    <li><b>Recall:</b> 97.80%</li>
                    <li><b>F1 Score:</b> 98.26%</li>
                    <li><b>ROC-AUC:</b> 0.9986</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-card">
                <h3>💡 Key Business Insights</h3>
                <ul>
                    <li>Applicants with higher CIBIL scores have a significantly
                        greater chance of loan approval.</li>
                    <li>Annual income, loan amount, and asset values also
                        contribute meaningfully to approval decisions.</li>
                    <li>Education and self-employment status alone show little
                        influence on approval outcomes.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="footer-note">Loan Approval Prediction · Built for educational purposes</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
if page == "🏠 Home":
    render_home()
elif page == "🔍 Prediction":
    render_prediction()
elif page == "ℹ️ About":
    render_about()
