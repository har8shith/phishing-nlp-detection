import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "emails.csv"
MODELS_DIR = BASE_DIR / "models"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
TFIDF_MODEL_PATH = MODELS_DIR / "tfidf_logreg.pkl"
BERT_MODEL_DIR = MODELS_DIR / "bert_model"
DB_URL = os.getenv("PHISHING_DB_URL", f"sqlite:///{BASE_DIR / 'phishing_emails.db'}")
API_HOST = "0.0.0.0"
API_PORT = int(os.getenv("API_PORT", 8000))
