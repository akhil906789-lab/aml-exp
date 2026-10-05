"""
train_model.py
Trains a spam classifier (TF-IDF + Logistic Regression) and saves it.

Run:  python train_model.py
"""
import json
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join("dataset", "spam.csv")
MODEL_DIR = "model"

# 1. Load the dataset (columns: label, message)
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} messages")
print(df["label"].value_counts().to_string(), "\n")

# 2. Clean / prepare
df = df.dropna(subset=["label", "message"])
df = df.drop_duplicates(subset="message")          # remove repeated messages
print(f"After removing duplicates: {len(df)} messages\n")

# 3 + 4. Separate text and labels, convert labels to numbers (spam=1, ham=0)
X = df["message"].astype(str)
y = df["label"].map({"ham": 0, "spam": 1})

# 5. Train / test split (80% / 20%), keeping the spam ratio the same in both
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. TF-IDF features (lowercase + remove English stop words is built in)
vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)   # learn vocabulary on TRAIN only
X_test_vec = vectorizer.transform(X_test)

# 7. Train the classifier
model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model.fit(X_train_vec, y_train)

# 8 + 9. Evaluate and print metrics
pred = model.predict(X_test_vec)
metrics = {
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "accuracy": round(accuracy_score(y_test, pred), 4),
    "precision": round(precision_score(y_test, pred), 4),
    "recall": round(recall_score(y_test, pred), 4),
    "f1_score": round(f1_score(y_test, pred), 4),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
}
print("Test-set results (spam = positive class)")
for k in ("accuracy", "precision", "recall", "f1_score"):
    print(f"  {k:10s}: {metrics[k]}")
print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_test, pred))
print("\n", classification_report(y_test, pred, target_names=["NOT SPAM", "SPAM"]))

# 10 + 11. Save model and vectorizer
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved: model/model.pkl, model/vectorizer.pkl, model/metrics.json")
