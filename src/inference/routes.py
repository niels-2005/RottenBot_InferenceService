import logging
import time
import uuid

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
    UploadFile,
)
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.main import get_session

from .dependencies import AccessTokenBearer
from .schemas import PredictionResponse
from .service import InferenceService
from .setup_observability import get_meter, get_tracer
from .utils import (
    generate_image_path,
    record_inference_api_duration,
    increase_inference_api_counter,
)

tracer = get_tracer(__name__)
meter = get_meter(__name__)

logger = logging.getLogger(__name__)

inference_router = APIRouter()
inference_service = InferenceService()


@inference_router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile,
    save_prediction: bool,
    user_uid: uuid.UUID,
    request: Request,
    background_tasks: BackgroundTasks,
    token_details: dict = Depends(AccessTokenBearer()),
    session: AsyncSession = Depends(get_session),
    endpoint_config: dict[str, str] = {
        "endpoint": "/predict",
        "method": "POST",
        "service_name": "inference_service",
    },
):
    """Handles prediction requests for uploaded images.

    Processes an uploaded image file to perform model inference, returning the prediction results.
    Optionally saves the prediction data and image to the database and S3 storage if requested.
    Includes observability features like tracing, metrics, and logging for monitoring.

    Args:
        file (UploadFile): The uploaded image file for prediction.
        save_prediction (bool): Whether to save the prediction results and image to storage.
        user_uid (uuid.UUID): The unique identifier of the user making the request.
        request (Request): The FastAPI request object, containing app state like the model.
        background_tasks (BackgroundTasks): FastAPI background tasks for async operations.
        session (AsyncSession): The database session for saving predictions.

    Returns:
        PredictionResponse: The prediction results including class, confidence, etc.

    Raises:
        HTTPException: If the file type is invalid or if prediction fails internally.
    """
    try:
        with tracer.start_as_current_span("predict_endpoint") as prediction_entry_span:
            start_time = time.time()
            increase_inference_api_counter(endpoint_config)

            logger.info(
                f"User {user_uid} requested a prediction with save_prediction={save_prediction}."
            )
            logger.info(
                f"User {user_uid} uploaded file {file.filename} with content_type: {file.content_type}."
            )

            # set attributes for the span
            prediction_entry_span.set_attribute("user_uid", str(user_uid))
            prediction_entry_span.set_attribute("save_prediction", save_prediction)

            # this is optional, type checking should be handled in the frontend
            if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
                logger.warning(
                    f"Invalid file type: {file.content_type}. This should be handled in the frontend.",
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type. Only JPEG, JPG, and PNG are supported. Got {file.content_type}",
                )

            # span for reading the image contents
            with tracer.start_as_current_span("read_image_contents") as read_image_span:
                logger.info(
                    f"Reading image contents for file {file.filename} with content_type: {file.content_type}."
                )
                read_image_span.set_attribute("filename", file.filename)
                read_image_span.set_attribute("content_type", file.content_type)
                contents = await file.read()
                await file.seek(0)

            # span for model prediction
            with tracer.start_as_current_span(
                "model_prediction"
            ) as model_prediction_span:
                logger.info(f"Started model prediction for user {user_uid}.")
                # get the predicted_class, predicted_class_name and confidence from the model
                prediction_info = await inference_service.predict(
                    contents,
                    request.app.state.model,
                    request.app.state.index_to_class,
                    user_uid,
                )
                logger.info(
                    f"Completed model prediction for user {user_uid}. With prediction_info: {prediction_info}"
                )

            # raise an error if the prediction failed
            if prediction_info is None:
                # logging in inference_service.predict() already done
                raise HTTPException(
                    status_code=500,
                    detail="Ooops! Something went wrong during prediction.",
                )

            # if the user wants to save the prediction, save it to the db and s3. For later use.
            if save_prediction:
                image_path = generate_image_path(file.filename)

                if image_path is None:
                    logger.error(
                        f"Error generating image path for user {user_uid} with filename {file.filename}. Cannot save prediction to DB and S3.",
                    )
                else:
                    with tracer.start_as_current_span(
                        "save_prediction_db"
                    ) as save_prediction_db_span:
                        logger.info(
                            f"Saving prediction to DB for user {user_uid} with image_path: {image_path} and prediction_info: {prediction_info}."
                        )
                        # FastAPI Background tasks is a MVP solution, migrate to Celery for a more scalable solution.
                        # for a absolute scalable solution, build a seperate microservice
                        # save the prediction in the background, only if the user accepts
                        background_tasks.add_task(
                            inference_service.save_prediction_to_db,
                            prediction_info,
                            image_path,
                            user_uid,
                            session,
                        )

                    with tracer.start_as_current_span(
                        "save_image_s3"
                    ) as save_image_s3_span:
                        logger.info(
                            f"Saving image to S3 for user {user_uid} with image_path: {image_path}."
                        )
                        # save the image to s3 in the background
                        background_tasks.add_task(
                            inference_service.save_image_to_s3,
                            image_path,
                            contents,
                            user_uid,
                        )
            logger.info(
                f"Returning prediction response for user {user_uid}. Completed successfully."
            )
            return PredictionResponse(**prediction_info)
    finally:
        duration_ms = (time.time() - start_time) * 1000
        record_inference_api_duration(endpoint_config, duration_ms)
