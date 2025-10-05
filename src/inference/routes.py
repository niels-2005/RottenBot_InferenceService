from fastapi import (
    APIRouter,
    UploadFile,
    HTTPException,
    Request,
    Depends,
    BackgroundTasks,
)
from .utils import generate_image_path
from .service import InferenceService
from .schemas import PredictionResponse
from src.db.main import get_session
from sqlmodel.ext.asyncio.session import AsyncSession
import uuid
from .setup_observability import setup_observability
import logging
import uuid
import time

tracer, meter = setup_observability("inference_service")
logger = logging.getLogger(__name__)

inference_router = APIRouter()
inference_service = InferenceService()

prediction_api_counter = meter.create_counter(
    name="prediction_api_requests_total",
    description="Total number of prediction API requests",
    unit="1",
)

prediction_api_duration = meter.create_histogram(
    name="prediction_api_duration_seconds",
    description="Prediction API request duration",
    unit="s",
)


@inference_router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile,
    save_prediction: bool,
    user_uid: uuid.UUID,
    request: Request,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    start_time = time.time()
    prediction_api_counter.add(
        1,
        {"endpoint": "/predict", "method": "POST", "service_name": "inference_service"},
    )
    with tracer.start_as_current_span("predict_endpoint") as prediction_entry_span:
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
        with tracer.start_as_current_span("model_prediction") as model_prediction_span:
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
                status_code=500, detail="Ooops! Something went wrong during prediction."
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
        duration = time.time() - start_time
        prediction_api_duration.record(
            duration,
            {
                "endpoint": "/predict",
                "method": "POST",
                "service_name": "inference_service",
            },
        )
        logger.info(
            f"Returning prediction response for user {user_uid}. Completed successfully."
        )
        return PredictionResponse(**prediction_info)
