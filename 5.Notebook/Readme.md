# 📉 Customer Churn Prediction using Machine Learning

## 📌 Project Overview

Customer churn prediction is one of the most important business problems in customer relationship management.

The objective of this project is to build a Machine Learning model that predicts whether a customer is likely to churn based on demographic information, subscription details, payment history, and customer behavior.

---

# 📂 Dataset

Source:
Kaggle – Customer Churn Dataset

Training Samples:
64,374

Testing Samples:
64,374

Features:
12

Target:
Churn

---

# 🎯 Problem Statement

Predict whether a customer will churn or not using historical customer information.

Target Variable

- 0 → Customer will stay
- 1 → Customer will churn

---

# 🛠 Libraries Used

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

---

# 📊 Exploratory Data Analysis

Performed:

- Dataset Overview
- Missing Value Analysis
- Duplicate Analysis
- Numerical Statistics
- Categorical Statistics
- Correlation Heatmap
- Churn Distribution
- Feature Relationship Analysis

---

# ⚙ Data Preprocessing

- Removed CustomerID
- One-Hot Encoding
- Train Validation Split
- Standard Scaling (for distance-based models)

---

# 🤖 Machine Learning Models

The following models were trained and compared:

- Logistic Regression
- KNN
- SVM
- Decision Tree
- Random Forest
- Extra Trees
- AdaBoost
- Gradient Boosting
- Gaussian Naive Bayes
- Linear Discriminant Analysis

---

# 🏆 Best Model

Random Forest Classifier

Hyperparameter tuning performed using RandomizedSearchCV.

---

# 📈 Final Model Performance

Accuracy:
99.86%

Precision:
99.90%

Recall:
99.80%

F1-Score:
99.85%

---

# 🚀 Test Prediction

The final tuned Random Forest model was used to predict customer churn on the unseen testing dataset.

Predictions were exported to:

Customer_Churn_Predictions.csv

---

# 📌 Project Workflow

Problem Statement

↓

Data Collection

↓

EDA

↓

Data Cleaning

↓

Feature Engineering

↓

One-Hot Encoding

↓

Train-Test Split

↓

Feature Scaling

↓

Model Comparison

↓

Hyperparameter Tuning

↓

Model Evaluation

↓

Test Prediction

---

# 🧑‍💻 Author

**Manish Kumar**

Machine Learning | Data Science | AI

GitHub:
https://github.com/Developer-Manish007