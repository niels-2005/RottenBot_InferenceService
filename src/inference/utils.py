import io
import logging
import uuid
from datetime import datetime
from typing import Tuple

import boto3
import numpy as np
import tensorflow as tf

from src.config import Config

logger = logging.getLogger(__name__)


async def preprocess_image(contents: bytes, image_size: Tuple[int, int]) -> tf.Tensor:
    """Preprocesses an uploaded image file for model inference.

    Loads the image from the provided bytes, resizes it to the specified dimensions,
    converts it to a numpy array, and adds a batch dimension for TensorFlow model input.

    Args:
        contents (bytes): The image file contents as bytes.
        image_size (Tuple[int, int]): The target size (height, width) to resize the image to.

    Returns:
        tf.Tensor or None: The preprocessed image as a 4D tensor with shape (1, height, width, 3), or None if an error occurs.
    """
    try:
        image = tf.keras.preprocessing.image.load_img(
            io.BytesIO(contents), target_size=image_size
        )
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.expand_dims(image, 0)
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}", exc_info=True)
        return None


def get_prediction(model: tf.keras.Model, image: tf.Tensor) -> np.ndarray:
    """Runs inference on the preprocessed image using the given model.

    Performs prediction on the input image tensor and returns the model's output.

    Args:
        model (tf.keras.Model): The trained TensorFlow/Keras model to use for inference.
        image (tf.Tensor): The preprocessed image tensor with shape (1, height, width, channels).

    Returns:
        numpy.ndarray or None: The model's prediction output, or None if an error occurs.
    """
    try:
        return model.predict(image)
    except Exception as e:
        logger.error(f"Error getting prediction: {e}", exc_info=True)
        return None


def connect_to_s3():
    """Connects to an S3-compatible service using boto3.

    Creates a boto3 client for interacting with an S3-compatible storage service,
    such as MinIO, using the provided configuration credentials and endpoint URL.

    Returns:
        boto3.client or None: The S3 client instance if successful, otherwise None.
    """
    try:
        return boto3.client(
            "s3",
            aws_access_key_id=Config.MINIO_ROOT_USER,
            aws_secret_access_key=Config.MINIO_ROOT_PASSWORD,
            endpoint_url=Config.LOCAL_S3_PROXY_SERVICE_URL,
        )
    except Exception as e:
        logger.error(f"Error connecting to S3: {e}", exc_info=True)
        return None


def generate_image_path(filename: str) -> str:
    """Generates a unique image path based on the provided filename.

    Creates a unique file path by combining a timestamp, a short UUID, and the file extension
    from the original filename, ensuring uniqueness for storage purposes.

    Args:
        filename (str): The original filename, used to extract the file extension.

    Returns:
        str or None: The generated unique image path if successful, otherwise None.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        extension = filename.split(".")[-1] if "." in filename else "jpg"
        return f"{timestamp}_{unique_id}.{extension}"
    except Exception as e:
        logger.error(f"Error generating image path: {e}", exc_info=True)
        return None
