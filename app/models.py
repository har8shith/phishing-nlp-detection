from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from sqlalchemy.sql import func
from .database import Base

class EmailRecord(Base):
    __tablename__ = "email_records"
    id = Column(Integer, primary_key=True, index=True)
    email_text = Column(Text, nullable=False)
    predicted_label = Column(String(20), nullable=False)
    probability = Column(Float, nullable=False)
    model_used = Column(String(20), nullable=False)
    extracted_info = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
