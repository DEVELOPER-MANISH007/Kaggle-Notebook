# 🎬 IMDB Movie Sentiment Analysis using NLP & Machine Learning

An end-to-end Natural Language Processing (NLP) project that classifies IMDB movie reviews into **Positive** or **Negative** sentiments using classical Machine Learning techniques.

---

## 📌 Project Overview

This project demonstrates a complete NLP pipeline starting from raw text preprocessing to model training, evaluation, and hyperparameter tuning.

The IMDB dataset contains **50,000 movie reviews** labeled as positive or negative sentiments.

---

## 📂 Dataset

- **Source:** Kaggle - IMDB Movie Reviews Dataset
- **Total Reviews:** 50,000
- **Classes:**
  - Positive
  - Negative

After removing duplicate reviews:

- **Total Samples:** 49,582

---

# 🛠 Tech Stack

- Python
- Pandas
- NumPy
- BeautifulSoup
- Regex
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn

---

# 📋 Project Workflow

```
Dataset
   │
EDA
   │
Data Cleaning
   │
Text Preprocessing
   │
TF-IDF Vectorization
   │
Train-Test Split
   │
Model Training
   │
Model Comparison
   │
Hyperparameter Tuning
   │
Best Model Selection
```

---

# 🧹 Text Preprocessing

The following preprocessing steps were applied:

- Remove Duplicate Reviews
- Remove HTML Tags
- Convert Text to Lowercase
- Expand English Contractions
- Remove URLs
- Remove Special Characters
- Tokenization
- Smart Stopword Removal
- Lemmatization
- TF-IDF Vectorization

---

# 🤖 Machine Learning Models

The following models were trained and compared:

- Logistic Regression
- Linear SVM
- Multinomial Naive Bayes
- Decision Tree
- Random Forest
- Extra Trees
- AdaBoost
- Gradient Boosting

---

# 📊 Model Performance

| Model | Accuracy |
|--------|---------:|
| Linear SVM | **90.86%** |
| Logistic Regression | **90.43%** |
| Multinomial Naive Bayes | **88.88%** |
| Extra Trees | **87.87%** |
| Random Forest | **86.53%** |
| Gradient Boosting | **81.36%** |
| AdaBoost | **75.26%** |
| Decision Tree | **72.66%** |

---

# 🔧 Hyperparameter Tuning

GridSearchCV was applied to the top-performing models.

### Logistic Regression

Best Parameters

```python
{'C': 1}
```

Accuracy after tuning

```
91.01%
```

---

### Linear SVM

Best Parameters

```python
{'C': 1}
```

Accuracy after tuning

```
90.86%
```

---

### Multinomial Naive Bayes

Best Parameters

```python
{'alpha': 0.5}
```

Accuracy after tuning

```
88.80%
```

---

# 🏆 Best Model

**Logistic Regression**

Final Accuracy:

```
91.01%
```

---

# 📁 Project Structure

```
IMDB-Sentiment-Analysis/
│
├── IMDB_Sentiment_Analysis.ipynb
├── README.md
├── requirements.txt
├── imdb_dataset.csv
└── images/
```

---

# 🚀 Future Improvements

- Word2Vec
- GloVe Embeddings
- FastText
- LSTM
- GRU
- BERT
- RoBERTa
- Hugging Face Transformers
- Model Deployment using FastAPI / Streamlit

---

# 📚 Key Learnings

- Text Cleaning
- Regular Expressions
- BeautifulSoup
- Tokenization
- Stopword Removal
- Lemmatization
- TF-IDF Vectorization
- Sparse Matrix Representation
- Classical NLP Pipeline
- Model Comparison
- Hyperparameter Tuning
- Sentiment Classification

---

# ⭐ Results

- End-to-End NLP Pipeline
- 8 Machine Learning Models Compared
- Hyperparameter Tuning
- Final Accuracy of **91.01%**
- Production-ready Text Preprocessing Pipeline

---

## 👨‍💻 Author

**Manish Kumar**

Data Science | Machine Learning | NLP