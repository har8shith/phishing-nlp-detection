from sqlalchemy.orm import Session
from .. import models
from .extractor import serialize_extracted_info

def log_prediction(db: Session, email_text: str, label: str, probability: float, model_used: str, extracted_info: dict | None = None):
    record = models.EmailRecord(
        email_text=email_text,
        predicted_label=label,
        probability=probability,
        model_used=model_used,
        extracted_info=serialize_extracted_info(extracted_info) if extracted_info else None,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_recent_records(db: Session, limit: int = 20):
    return db.query(models.EmailRecord).order_by(models.EmailRecord.created_at.desc()).limit(limit).all()
