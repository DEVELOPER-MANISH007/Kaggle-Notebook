# 💳 Credit Card Fraud Detection using Machine Learning

An end-to-end Machine Learning project that detects fraudulent credit card transactions using supervised classification algorithms. The project includes Exploratory Data Analysis (EDA), handling imbalanced data, SMOTE experiments, hyperparameter tuning, feature importance analysis, and model comparison.

---

# 📌 Project Overview

Credit card fraud is one of the biggest challenges faced by financial institutions. Since fraudulent transactions are extremely rare, traditional machine learning models often struggle to detect them.

The objective of this project is to build a robust fraud detection model that maximizes fraud detection while minimizing false alarms.

---

# 📂 Dataset

**Source:** Kaggle - Credit Card Fraud Detection Dataset

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Dataset Information

- Total Transactions: **284,807**
- Features: **31**
- Fraud Transactions: **492**
- Genuine Transactions: **284,315**
- Fraud Percentage: **0.173%**

---

# 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)

---

# 📋 Project Workflow

```
Problem Understanding
        │
Data Understanding
        │
Exploratory Data Analysis
        │
Preprocessing
        │
Train-Test Split
        │
Feature Scaling
        │
Baseline Model Training
        │
SMOTE Analysis
        │
Model Comparison
        │
Hyperparameter Tuning
        │
Feature Importance
        │
Threshold Analysis
        │
Final Model Selection
```

---

# 📊 Exploratory Data Analysis

Performed:

- Class Distribution Analysis
- Pie Chart Analysis
- Transaction Amount Distribution
- Transaction Time Distribution
- Amount vs Class
- Time vs Class
- Feature Correlation Analysis
- Feature Importance

---

# ⚠ Class Imbalance

The dataset is highly imbalanced.

| Class | Transactions |
|--------|-------------:|
| Genuine | 284,315 |
| Fraud | 492 |

Accuracy alone is not an appropriate evaluation metric for this problem.

Therefore the following metrics were used:

- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

---

# 🤖 Machine Learning Models

The following models were trained:

- Logistic Regression
- Decision Tree
- Random Forest

Each model was evaluated:

- Before SMOTE
- After SMOTE

---

# 🔄 SMOTE Analysis

SMOTE (Synthetic Minority Oversampling Technique) was applied only to the training dataset to handle class imbalance.

The experiment showed that:

- Logistic Regression achieved higher Recall after SMOTE but suffered a significant drop in Precision due to increased False Positives.
- Decision Tree showed a slight improvement in Recall but a noticeable decrease in Precision.
- Random Forest achieved the best balance on the original dataset and did not benefit significantly from SMOTE.

---

# 📈 Hyperparameter Tuning

Hyperparameter tuning was performed using:

- GridSearchCV / RandomizedSearchCV

Tuned Parameters:

- Number of Trees
- Maximum Depth
- Minimum Samples Split
- Minimum Samples Leaf
- Maximum Features

---

# 📊 Feature Importance

Random Forest Feature Importance was used to identify the most influential PCA components contributing to fraud detection.

---

# 🏆 Final Model Selection

After comparing all experiments, the **Random Forest model trained on the original dataset (without SMOTE)** was selected as the final model.

### Why?

- Highest Precision
- High Recall
- Best F1-Score
- Lowest False Positives
- Lowest False Negatives among the tested models
- Best overall business performance

---

# 📁 Project Structure

```
Credit-Card-Fraud-Detection/
│
├── Credit_Card_Fraud_Detection.ipynb
├── creditcard.csv
├── README.md
├── requirements.txt
└── images/
```

---

# 🚀 Future Improvements

- XGBoost
- LightGBM
- CatBoost
- Isolation Forest
- Autoencoder-based Fraud Detection
- Deep Learning Models
- Real-Time Fraud Detection API using FastAPI
- Model Deployment

---

# 📚 Key Learnings

- Imbalanced Dataset Handling
- SMOTE
- Precision vs Recall
- F1-Score
- ROC-AUC
- Threshold Analysis
- Random Forest
- Hyperparameter Tuning
- Feature Importance
- Business-Oriented Model Evaluation

---

# ⭐ Results

- Performed End-to-End Fraud Detection Pipeline
- Compared Multiple Machine Learning Models
- Evaluated Models Before and After SMOTE
- Conducted Hyperparameter Tuning
- Selected the Best Model Based on Business Metrics instead of Accuracy

---

# 👨‍💻 Author

**Manish Kumar**

Data Science | Machine Learning | NLP