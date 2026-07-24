📝 Problem Statement

Develop a machine learning model that predicts the sentiment (Positive, Neutral, or Negative) of AI-related social media posts using the post text and relevant metadata.



## Root Cause Analysis

The model achieved only ~35% accuracy despite following a standard NLP pipeline
(Text Cleaning → TF-IDF → Logistic Regression).

After investigating the dataset, the following issues were identified:

- The dataset contains only **10 unique text templates** repeated across 2200 samples.
- The same text appears with **different sentiment labels** (Negative, Neutral, and Positive).
- This creates **label inconsistency**, meaning identical inputs correspond to different outputs.
- Such ambiguity prevents any supervised machine learning model from learning a reliable mapping between text and sentiment.
- As a result, the model performs close to random guessing (~33% for three balanced classes).

### Conclusion

The poor performance is caused primarily by **dataset quality issues**, not by the choice of preprocessing or machine learning algorithm.