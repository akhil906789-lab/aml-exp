"""
app.py - Flask web application for Spam Detection.
Run:  python app.py   then open http://127.0.0.1:5000
"""
import os

import joblib
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "vectorizer.pkl")
MAX_LENGTH = 2000  # characters

# Load the SAME model and vectorizer that train_model.py saved
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    load_error = None
except FileNotFoundError:
    model = vectorizer = None
    load_error = "Model files not found. Run 'python train_model.py' first."


@app.route("/")
def home():
    """Show the main page."""
    return render_template("index.html", load_error=load_error)


@app.route("/predict", methods=["POST"])
def predict():
    """Receive a message, vectorize it, predict, return JSON."""
    if model is None:
        return jsonify(error=load_error), 500

    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify(error="Please enter a message first."), 400
    if len(message) > MAX_LENGTH:
        return jsonify(error=f"Message is too long (max {MAX_LENGTH} characters)."), 400

    try:
        features = vectorizer.transform([message])          # text -> TF-IDF numbers
        label = int(model.predict(features)[0])             # 1 = spam, 0 = not spam
        spam_probability = float(model.predict_proba(features)[0][1])
    except Exception:
        return jsonify(error="Something went wrong while predicting."), 500

    return jsonify(
        label="SPAM" if label == 1 else "NOT SPAM",
        is_spam=bool(label),
        spam_probability=round(spam_probability * 100, 1),
    )


if __name__ == "__main__":
    app.run(debug=True)
