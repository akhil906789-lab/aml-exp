# Spam Email Detection Web Application Using Flask

## Overview
A small web application that classifies a text message as **SPAM** or **NOT SPAM**.
A machine-learning model is trained separately (`train_model.py`), saved with joblib,
and then loaded by a Flask app (`app.py`) that serves a web page.

## Features
- Type or paste a message and get SPAM / NOT SPAM
- Shows the model's spam probability
- Handles empty and very long input
- Clear button and two one-click examples
- Works on desktop and mobile screens

## Technologies Used
Python, scikit-learn, pandas, joblib, Flask, HTML, CSS, JavaScript

## Project Architecture
```
Dataset -> Preprocessing -> TF-IDF -> Train classifier -> model.pkl + vectorizer.pkl
                                                              |
User message -> Web page -> Flask /predict -> TF-IDF -> Classifier -> SPAM / NOT SPAM -> Web page
```

## Dataset
SMS Spam Collection (UCI), 5,572 messages, labels `ham` / `spam`.
Details and licence note: see `dataset/README.md`.

## Machine Learning Approach
1. Remove duplicate messages (5,572 -> 5,169)
2. Labels: ham = 0, spam = 1
3. 80/20 train-test split (stratified, `random_state=42`)
4. TF-IDF vectorizer (lowercase, English stop words removed), fitted on training data only
5. Logistic Regression (`class_weight="balanced"` because spam is the minority class)

## Installation
```
python -m venv venv
venv\Scripts\activate          (Windows)
source venv/bin/activate        (Mac/Linux)
pip install -r requirements.txt
```

## How to Run
```
python train_model.py
python app.py
```
Then open **http://127.0.0.1:5000** in your browser.
(`model/model.pkl` and `model/vectorizer.pkl` are included, so `python app.py` also works directly.
Re-run `train_model.py` if you get a scikit-learn version warning.)

## Example Inputs
- SPAM: `Congratulations! You have won a free prize. Click here to claim.`
- NOT SPAM: `Hi, our meeting is scheduled for 10 AM tomorrow.`

## Model Evaluation
Results on the held-out test set (1034 messages), from running `train_model.py`:

| Metric | Value |
|--------|-------|
| Accuracy | 0.9739 |
| Precision (spam) | 0.9062 |
| Recall (spam) | 0.8855 |
| F1-score (spam) | 0.8958 |

Confusion matrix: TN=891, FP=12, FN=15, TP=116.
Numbers can change slightly if you change the split, library version or settings.

## Screenshots
Real screenshots of the running app are in `screenshots/`.
Add your own (VS Code, terminal, GitHub page) in the same folder.

## Future Improvements
- Try Naive Bayes / SVM and compare
- Use a larger email dataset
- Add a confidence threshold and message history
- Retrain automatically with user feedback

## Author
- Name: A. Varsha
- Roll number: YOUR ROLL NUMBER
- Department / University: Department of Computer Science & Engineering, FET
