import numpy as np
import pandas as pd

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# === Train Data Processing ===
train_df = pd.read_csv("data/processed/train.csv")
print("Available columns in train_df:", train_df.columns.tolist())
train_df = train_df.dropna(subset=["content"])  # Drop rows with missing content
X_train_text = train_df["content"].values
y_train = train_df["sentiment"].values

# Vectorize train text
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train_text)

# Convert to DataFrame
train_bow_df = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())
train_bow_df["sentiment"] = y_train

# Save train BoW
train_bow_df.to_csv("data/interim/train_bow.csv", index=False)

# === Test Data Processing ===
test_df = pd.read_csv("data/processed/test.csv")
test_df = test_df.dropna(subset=["content"])  # Drop rows with missing content
X_test_text = test_df["content"]
y_test = test_df["sentiment"]

# Use the same vectorizer for test
X_test = vectorizer.transform(X_test_text)

# Convert to DataFrame
test_bow_df = pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names_out())
test_bow_df["sentiment"] = y_test

# Save test BoW
test_bow_df.to_csv("data/interim/test_bow.csv", index=False)

print("Feature engineering completed successfully.")
