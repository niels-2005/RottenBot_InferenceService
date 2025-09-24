from fastapi import APIRouter, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from .service import InferenceService
import numpy as np


router = APIRouter()
inference_service = InferenceService()


def confidence_below_threshold(confidence: float, threshold: float = 0.9) -> bool:
    return confidence < threshold


@router.post("/predict")
async def predict(file: UploadFile, save_prediction: bool, request: Request):
    # this is optional, type checking should be handled in the frontend
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPEG, JPG, and PNG are supported.",
        )

    prediction, confidence = await inference_service.predict(
        file,
        request.app.state.model,
        save_prediction,
    )

    if prediction is None or confidence is None:
        raise HTTPException(status_code=500, detail="Error during prediction.")

    if confidence_below_threshold(confidence, threshold=0.9):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "info": "Low confidence in prediction. Use the provided hints to improve the image quality.",
                "confidence": confidence,
            },
        )
    else:
        # TODO: Log class names to mlflow and get them here.
        predicted_class = np.argmax(prediction)
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "predicted_class": predicted_class,  # class names needed here: class_names[predicted_class]
                "confidence": confidence,
            },
        )
