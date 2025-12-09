import joblib
from config import TFIDF_VECTORIZER_PATH, TFIDF_MODEL_PATH
from .preprocessing import basic_clean

class TfidfClassifier:
    def __init__(self):
        self.vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
        self.model = joblib.load(TFIDF_MODEL_PATH)
    def predict_proba(self, text: str) -> float:
        cleaned = basic_clean(text)
        X = self.vectorizer.transform([cleaned])
        proba = self.model.predict_proba(X)[0][1]
        return float(proba)
    def predict_label(self, text: str, threshold: float = 0.5) -> str:
        proba = self.predict_proba(text)
        return "phishing" if proba >= threshold else "legitimate"
