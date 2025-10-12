import io
import logging
import uuid
from typing import Any

import numpy as np
import tensorflow as tf
from sqlmodel.ext.asyncio.session import AsyncSession

from .models import Prediction
from .setup_observability import get_tracer
from .utils import connect_to_s3, get_prediction, preprocess_image

tracer = get_tracer(__name__)
logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(self):
        self.s3 = connect_to_s3()

    async def predict(
        self,
        contents: bytes,
        model: tf.keras.Model,
        index_to_class: dict[str, str],
        user_uid: uuid.UUID,
    ) -> dict[str, Any] | None:
        """Performs model inference on the provided image contents.

        Preprocesses the image bytes, runs prediction using the given model,
        and maps the predicted class index to its name using the index_to_class dictionary.

        Args:
            contents (bytes): The raw image data as bytes.
            model (tf.keras.Model): The trained TensorFlow/Keras model for inference.
            index_to_class (dict[str, str]): A mapping from class index to class name.
            user_uid (uuid.UUID): The unique identifier of the user requesting the prediction.

        Returns:
            dict[str, Any] or None: A dictionary containing predicted_class, predicted_class_name,
            and confidence if successful, otherwise None.
        """
        try:
            with tracer.start_as_current_span(
                "preprocess_image"
            ) as preprocess_image_span:
                preprocess_image_span.set_attribute("image_size", "(224, 224)")
                IMAGE_SIZE = (224, 224)
                logger.info(
                    f"Preprocessing image for model inference with Image shape: {IMAGE_SIZE} for user {user_uid}."
                )
                image = await preprocess_image(contents, IMAGE_SIZE)

                if image is None:
                    # error already logged in preprocess_image
                    return None

            with tracer.start_as_current_span("get_prediction") as get_prediction_span:
                logger.info(f"Running model prediction for user {user_uid}.")
                prediction = get_prediction(model, image)

                if prediction is None:
                    # error already logged in get_prediction
                    return None

            # get the highest confidence class
            predicted_class = np.argmax(prediction)

            return {
                "predicted_class": predicted_class,
                "predicted_class_name": index_to_class[str(predicted_class)],
                "confidence": float(np.max(prediction)),
            }
        except Exception as e:
            logger.error(
                f"Error during prediction for user {user_uid}: {e}", exc_info=True
            )
            return None

    async def save_prediction_to_db(
        self,
        prediction_info: dict[str, Any],
        image_path: str,
        user_uid: uuid.UUID,
        session: AsyncSession,
    ):
        """Saves the prediction information to the database.

        Creates a new Prediction record with the provided details and commits it to the database.

        Args:
            prediction_info (dict[str, Any]): A dictionary containing prediction details like
                predicted_class, predicted_class_name, and confidence.
            image_path (str): The path where the image is stored.
            user_uid (uuid.UUID): The unique identifier of the user.
            session (AsyncSession): The database session for committing the changes.
        """
        try:
            logger.info(
                f"Saving prediction to DB for user {user_uid} with image_path: {image_path} and prediction_info: {prediction_info}."
            )
            new_prediction = Prediction(
                image_path=image_path,
                **prediction_info,
                user_uid=user_uid,
            )
            session.add(new_prediction)
            # push new prediction to the db
            await session.commit()
        except Exception as e:
            logger.error(
                f"Error saving prediction to DB for user {user_uid} with image_path: {image_path} and prediction_info: {prediction_info}. Error: {e}",
                exc_info=True,
            )

    async def save_image_to_s3(
        self,
        image_path: str,
        contents: bytes,
        user_uid: uuid.UUID,
    ):
        """Uploads the image contents to S3 storage.

        Saves the provided image bytes to the specified path in the S3 bucket.

        Args:
            image_path (str): The path in S3 where the image should be stored.
            contents (bytes): The raw image data as bytes.
            user_uid (uuid.UUID): The unique identifier of the user.
        """
        try:
            if self.s3 is not None:
                # upload the image to the bucket
                self.s3.upload_fileobj(io.BytesIO(contents), "images", image_path)
            else:
                logger.error(
                    f"S3 client is None, cannot upload image for user {user_uid} with image_path: {image_path}.",
                )
        except Exception as e:
            logger.error(
                f"Error saving image to S3 for user {user_uid}. Error: {e}",
                exc_info=True,
            )
