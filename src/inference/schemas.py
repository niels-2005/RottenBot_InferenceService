from pydantic import BaseModel


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_class_name: str
    confidence: float
