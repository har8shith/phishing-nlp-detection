from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from config import BERT_MODEL_DIR
from .preprocessing import basic_clean

class BertClassifier:
    def __init__(self, device: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    @torch.no_grad()
    def predict_proba(self, text: str) -> float:
        cleaned = basic_clean(text)
        inputs = self.tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        return float(probs[1])
    def predict_label(self, text: str, threshold: float = 0.5) -> str:
        proba = self.predict_proba(text)
        return "phishing" if proba >= threshold else "legitimate"
