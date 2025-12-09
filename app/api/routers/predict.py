from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ...schemas import PredictRequest, PredictResponse, AnalyzeResponse, EmailRecordSchema
from ...database import get_db
from ...services.classifier import classify_email
from ...services.extractor import extract_information
from ...services.history import log_prediction, get_recent_records

router = APIRouter(prefix="/api", tags=["phishing"])

@router.post("/predict", response_model=PredictResponse)
def predict_email(request: PredictRequest, db: Session = Depends(get_db)):
    label, proba, model_used = classify_email(request.email_text, request.model)
    log_prediction(db, request.email_text, label, proba, model_used, extracted_info=None)
    return PredictResponse(label=label, probability=proba, model_used=model_used)

@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_email(request: PredictRequest, db: Session = Depends(get_db)):
    label, proba, model_used = classify_email(request.email_text, request.model)
    extracted_info = {}
    if label == "phishing":
        extracted_info = extract_information(request.email_text)
    log_prediction(db, request.email_text, label, proba, model_used, extracted_info)
    return AnalyzeResponse(label=label, probability=proba, model_used=model_used, extracted_info=extracted_info)

@router.get("/history", response_model=list[EmailRecordSchema])
def history(limit: int = 20, db: Session = Depends(get_db)):
    records = get_recent_records(db, limit)
    return records
