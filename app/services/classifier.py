from typing import Literal
from ..nlp.tfidf_model import TfidfClassifier
from ..nlp.bert_model import BertClassifier

_tfidf_clf = None
_bert_clf = None

def get_tfidf_clf():
    global _tfidf_clf
    if _tfidf_clf is None:
        _tfidf_clf = TfidfClassifier()
    return _tfidf_clf

def get_bert_clf():
    global _bert_clf
    if _bert_clf is None:
        _bert_clf = BertClassifier()
    return _bert_clf

def classify_email(email_text: str, model_name: Literal["tfidf", "bert"] = "bert"):
    if model_name == "tfidf":
        clf = get_tfidf_clf()
        model_used = "tfidf"
    else:
        clf = get_bert_clf()
        model_used = "bert"
    proba = clf.predict_proba(email_text)
    label = "phishing" if proba >= 0.5 else "legitimate"
    return label, float(proba), model_used
