# 🏦 Loan Approval Prediction

An end-to-end Machine Learning project that predicts whether a loan application
will be **Approved** or **Rejected**, based on applicant demographics, financial
status, employment information, and credit history — packaged as a
production-ready **Streamlit** web application.

## 📌 Problem Statement

Financial institutions receive thousands of loan applications every day.
Manually reviewing each one is time-consuming and can lead to inconsistent
decisions. This project trains a Gradient Boosting Classifier to automate
that screening step.

## 📂 Project Structure

```
Loan_Approval_Project/
│
├── app/
│   ├── app.py              # Streamlit entry point (Home / Prediction / About)
│   ├── prediction.py        # Model inference logic
│   ├── preprocessing.py     # Feature preprocessing (scaler + encoder)
│   ├── model_loader.py      # Loads pickled model/scaler/encoder artifacts
│   ├── utils.py              # Validation, formatting, session-state helpers
│   └── assets/               # Static assets (images, etc.)
│
├── data/
│   └── loan_approval_dataset.csv
│
├── models/
│   ├── model.pkl              # Trained Gradient Boosting Classifier
│   ├── scaler.pkl             # Fitted StandardScaler (numerical features)
│   ├── encoder.pkl            # Fitted OneHotEncoder (categorical features)
│   ├── label_encoder.pkl      # Fitted LabelEncoder (target)
│   └── feature_columns.pkl    # Numerical/categorical column ordering
│
├── Notebook/
│   └── loan_approval.ipynb    # Original EDA + model development notebook
│
├── train_model.py             # Training script (extracted from the notebook)
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Re)train the model — optional

The trained artifacts are already included under `models/`. If you want to
retrain from scratch (e.g. after updating the dataset), run:

```bash
python train_model.py
```

This regenerates `models/model.pkl`, `scaler.pkl`, `encoder.pkl`,
`label_encoder.pkl`, and `feature_columns.pkl` using the exact preprocessing
and modeling steps from the original notebook.

### 3. Run the Streamlit app

```bash
streamlit run app/app.py
```

## 🤖 Model

- **Algorithm:** Gradient Boosting Classifier (selected after comparing 10
  baseline models and tuning the top 3 with `RandomizedSearchCV`)
- **Preprocessing:** `StandardScaler` on numerical features +
  `OneHotEncoder(drop="first")` on categorical features
- **Final performance (test set):**
  - Accuracy: 98.71%
  - Precision: 98.73%
  - Recall: 97.80%
  - F1 Score: 98.26%
  - ROC-AUC: 0.9986

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · Streamlit · Plotly · Matplotlib · Seaborn

## 📊 Dataset

4,269 loan applications with features covering applicant dependents,
education, employment status, annual income, loan amount/term, CIBIL score,
and four asset-value categories. No missing values or duplicates were found.

## ⚠️ Disclaimer

This project is for educational purposes. Predictions should not be used as
the sole basis for real-world lending decisions.
