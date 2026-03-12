"""
train_model.py
--------------
Trains a TF-IDF + Logistic Regression cyberbullying classifier
on 'Suspicious Communication on Social Platforms.csv' and saves
the trained model and vectorizer as .pkl files.

Run:
    python train_model.py
"""

import re
import string
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─── Config ───────────────────────────────────────────────────────────────────
CSV_PATH       = "Suspicious Communication on Social Platforms.csv"
MODEL_PATH     = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
RANDOM_STATE   = 42
TEST_SIZE      = 0.2

# ─── Text Cleaning ────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase, remove URLs, punctuation, and extra whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)               # remove mentions/hashtags
    text = re.sub(r"[^a-z\s]", " ", text)               # keep only letters
    text = re.sub(r"\s+", " ", text).strip()            # collapse whitespace
    return text

# ─── Load & Prepare Data ──────────────────────────────────────────────────────
print("📂  Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"    Shape: {df.shape}")
print(f"    Columns: {list(df.columns)}")

# Drop nulls just in case
df.dropna(subset=["comments", "tagging"], inplace=True)

print("\n🔢  Label distribution:")
print(df["tagging"].value_counts())

# Clean text
print("\n🧹  Cleaning text...")
df["clean_comment"] = df["comments"].apply(clean_text)

X = df["clean_comment"]
y = df["tagging"]

# ─── Train / Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)
print(f"\n📊  Split → Train: {len(X_train)}  |  Test: {len(X_test)}")

# ─── TF-IDF Vectorization ─────────────────────────────────────────────────────
print("\n🔤  Fitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=50_000,
    ngram_range=(1, 2),       # unigrams + bigrams
    sublinear_tf=True,        # apply log normalization
    min_df=2,                 # ignore very rare terms
    strip_accents="unicode",
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)
print(f"    Vocabulary size: {len(vectorizer.vocabulary_)}")

# ─── Model Training ───────────────────────────────────────────────────────────
print("\n🤖  Training Logistic Regression...")
model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="lbfgs",
    class_weight="balanced",  # handles class imbalance (61% vs 39%)
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model.fit(X_train_vec, y_train)

# ─── Evaluation ───────────────────────────────────────────────────────────────
print("\n📈  Evaluating on test set...")
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅  Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Bullying", "Bullying"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ─── Save Artifacts ───────────────────────────────────────────────────────────
print(f"\n💾  Saving model  → {MODEL_PATH}")
joblib.dump(model, MODEL_PATH)

print(f"💾  Saving vectorizer → {VECTORIZER_PATH}")
joblib.dump(vectorizer, VECTORIZER_PATH)

print("\n🎉  Done! Model and vectorizer saved successfully.")
print("    You can now start the API with:  uvicorn main:app --reload --port 8000")
