from fastapi import UploadFile, BackgroundTasks
from .utils import preprocess_image, get_prediction
import numpy as np
import tensorflow as tf


class InferenceService:
    async def predict(
        file: UploadFile,
        model: tf.keras.Model,
        save_prediction: bool,
        background_tasks: BackgroundTasks,
    ):
        try:
            image = await preprocess_image(file, (224, 224))

            prediction = get_prediction(model, image)
            confidence = float(np.max(prediction))

            if save_prediction:
                pass

            return prediction, confidence
        except Exception as e:
            return None, None

    async def save_prediction():
        pass
