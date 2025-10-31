import io
import logging
import uuid
from datetime import datetime
from typing import Tuple

import jwt
import boto3
import numpy as np
import tensorflow as tf

from src.config import Config
from .setup_observability import get_meter

meter = get_meter(__name__)

logger = logging.getLogger(__name__)


# a meter to track inference API requests
inference_api_counter = meter.create_counter(
    name="inference_api_requests_total",
    description="Total number of inference API requests",
    unit="1",
)

# a histogram to track inference API request durations
inference_api_duration = meter.create_histogram(
    name="inference_api_duration_milliseconds",
    description="Inference API request duration",
    unit="ms",
)


def increase_inference_api_counter(endpoint_config: dict[str, str]) -> None:
    """Increases the inference API counter for a specific endpoint.

    Args:
        endpoint_config (dict[str, str]): Configuration for the endpoint.
    """
    try:
        inference_api_counter.add(1, attributes=endpoint_config)
    except Exception as e:
        logger.error(
            f"Error increasing inference API counter for endpoint {endpoint_config['endpoint']}: {e}",
            exc_info=True,
        )


def record_inference_api_duration(
    duration_ms: float, endpoint_config: dict[str, str]
) -> None:
    """Records the duration of an inference API request for a specific endpoint.

    Args:
        duration_ms (float): The duration of the API request in milliseconds.
        endpoint_config (dict[str, str]): Configuration for the endpoint.
    """
    try:
        inference_api_duration.record(duration_ms, attributes=endpoint_config)
    except Exception as e:
        logger.error(
            f"Error recording auth API duration for endpoint {endpoint_config['endpoint']}: {e}",
            exc_info=True,
        )


async def preprocess_image(contents: bytes, image_size: Tuple[int, int]) -> tf.Tensor:
    """Preprocesses an uploaded image file for model inference.

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
        image = tf.expand_dims(image, axis=0)
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}", exc_info=True)
        return None


def get_prediction(model: tf.keras.Model, image: tf.Tensor) -> np.ndarray:
    """Runs inference on the preprocessed image using the given model.

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


def decode_token(token: str) -> dict:
    """Decodes a JWT token and returns its payload.

    Args:
        token: The JWT token to decode.

    Returns:
        A dictionary containing the decoded token payload.
    """
    try:
        return jwt.decode(
            jwt=token, algorithms=[Config.JWT_ALGORITHM], key=Config.JWT_SECRET
        )
    except Exception as e:
        logger.error(f"Error decoding token: {e}", exc_info=True)
