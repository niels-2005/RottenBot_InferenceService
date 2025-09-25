from fastapi import UploadFile
from typing import Tuple
import tensorflow as tf
import numpy as np
import io
import boto3
from src.config import Config


async def preprocess_image(contents: bytes, image_size: Tuple[int, int]) -> tf.Tensor:
    """Preprocesses an uploaded image file for model inference.

    Loads the image from the uploaded file, resizes it to the specified dimensions,
    converts it to a numpy array, and adds a batch dimension for TensorFlow model input.

    Args:
        file (UploadFile): The uploaded image file from the FastAPI request.
        image_size (Tuple[int, int]): The target size (height, width) to resize the image to.

    Returns:
        tf.Tensor: The preprocessed image as a 4D tensor with shape (1, height, width, 3).
    """
    image = tf.keras.preprocessing.image.load_img(
        io.BytesIO(contents), target_size=image_size
    )
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, 0)
    return image


def get_prediction(model: tf.keras.Model, image: tf.Tensor) -> np.ndarray:
    """Runs inference on the preprocessed image using the given model.

    Performs prediction on the input image tensor and returns the model's output.

    Args:
        model (tf.keras.Model): The trained TensorFlow/Keras model to use for inference.
        image (tf.Tensor): The preprocessed image tensor with shape (1, height, width, channels).

    Returns:
        numpy.ndarray: The model's prediction output.
    """
    return model.predict(image)


def connect_to_s3():
    return boto3.client(
        "s3",
        aws_access_key_id=Config.MINIO_ROOT_USER,
        aws_secret_access_key=Config.MINIO_ROOT_PASSWORD,
        endpoint_url=Config.LOCAL_S3_PROXY_SERVICE_URL,
    )
