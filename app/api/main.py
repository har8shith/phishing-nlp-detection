from fastapi import FastAPI
from ..database import Base, engine
from .. import models
from .routers import predict as predict_router

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Phishing Email Detection API", version="1.0.0")

app.include_router(predict_router.router)

@app.get("/")
def root():
    return {"message": "Phishing detection API is running"}
