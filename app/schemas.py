from typing import Optional, Dict, Any
from pydantic import BaseModel

class PredictRequest(BaseModel):
    email_text: str
    model: str = "bert"

class PredictResponse(BaseModel):
    label: str
    probability: float
    model_used: str

class AnalyzeResponse(BaseModel):
    label: str
    probability: float
    model_used: str
    extracted_info: Dict[str, Any]

class EmailRecordSchema(BaseModel):
    id: int
    email_text: str
    predicted_label: str
    probability: float
    model_used: str
    extracted_info: Optional[str]
    class Config:
        orm_mode = True
