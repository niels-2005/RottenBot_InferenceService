from fastapi import UploadFile
from .utils import preprocess_image, get_prediction, connect_to_s3
import numpy as np
import tensorflow as tf
from .models import Prediction
from sqlmodel.ext.asyncio.session import AsyncSession
import uuid
from src.config import Config
from typing import Any
import io


class InferenceService:
    def __init__(self):
        self.s3 = connect_to_s3()

    async def predict(
        self,
        contents: bytes,
        model: tf.keras.Model,
        index_to_class: dict[str, str],
    ) -> dict[str, Any] | None:
        try:
            image = await preprocess_image(contents, (224, 224))

            prediction = get_prediction(model, image)
            predicted_class = np.argmax(prediction)

            return {
                "predicted_class": predicted_class,
                "predicted_class_name": index_to_class[str(predicted_class)],
                "confidence": float(np.max(prediction)),
            }
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

    async def save_prediction_to_db(
        self,
        prediction_info: dict[str, Any],
        image_path: str,
        user_uid: uuid.UUID,
        session: AsyncSession,
    ):
        try:
            new_prediction = Prediction(
                image_path=image_path,
                **prediction_info,
                user_uid=user_uid,
            )
            session.add(new_prediction)
            # push new prediction to the db
            await session.commit()
        except Exception as e:
            print(f"Error saving prediction: {e}")

    async def save_image_to_s3(
        self,
        image_path: str,
        contents: bytes,
    ):
        try:
            # upload the image to the bucket
            self.s3.upload_fileobj(io.BytesIO(contents), "images", image_path)
        except Exception as e:
            print(f"Error saving image to S3: {e}")
