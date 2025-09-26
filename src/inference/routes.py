from fastapi import (
    APIRouter,
    UploadFile,
    HTTPException,
    Request,
    Depends,
    BackgroundTasks,
)
from fastapi.responses import JSONResponse
from .service import InferenceService
from .schemas import PredictionResponse
import numpy as np
from src.db.main import get_session
from sqlmodel.ext.asyncio.session import AsyncSession
import uuid
from datetime import datetime

inference_router = APIRouter()
inference_service = InferenceService()


def generate_image_path(filename: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    extension = filename.split(".")[-1] if "." in filename else "jpg"
    return f"{timestamp}_{unique_id}.{extension}"


@inference_router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile,
    save_prediction: bool,
    request: Request,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    # this is optional, type checking should be handled in the frontend
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPEG, JPG, and PNG are supported.",
        )

    contents = await file.read()
    await file.seek(0)

    # get the predicted_class, predicted_class_name and confidence from the model
    prediction_info = await inference_service.predict(
        contents,
        request.app.state.model,
        request.app.state.index_to_class,
    )

    # raise an error if the prediction failed
    if prediction_info is None:
        raise HTTPException(
            status_code=500, detail="Ooops! Something went wrong during prediction."
        )

    if save_prediction:
        image_path = generate_image_path(file.filename)
        # FastAPI Background tasks is a MVP solution, migrate to Celery for a more scalable solution.
        # for a absolute scalable solution, build a seperate microservice
        # save the prediction in the background, only if the user accepts
        background_tasks.add_task(
            inference_service.save_prediction_to_db,
            prediction_info,
            image_path,
            session,
        )

        # save the image to s3 in the background
        background_tasks.add_task(
            inference_service.save_image_to_s3,
            image_path,
            contents,
        )

    return PredictionResponse(**prediction_info)
