import json
from typing import Dict

import mlflow
import tensorflow as tf

import logging

from src.inference.setup_observability import setup_observability

setup_observability("inference-service")

logger = logging.getLogger(__name__)


def load_model_from_mlflow(model_uri: str, dst_path: str) -> tf.keras.Model:
    """Loads a TensorFlow model from MLflow.

    Downloads and loads the specified TensorFlow model from the given MLflow model URI,
    saving it to the destination path.

    Args:
        model_uri (str): The URI of the MLflow model to load.
        dst_path (str): The local destination path where the model will be saved.

    Returns:
        tf.keras.Model: The loaded TensorFlow model.

    Raises:
        Exception: If loading the model fails.
    """
    try:
        # load the model from mlflow
        model = mlflow.tensorflow.load_model(model_uri, dst_path=dst_path)
        return model
    except Exception as e:
        logger.error(f"Error loading model from MLflow: {e}", exc_info=True)
        # raise error because this is critical
        raise e


def load_classes_from_mlflow(run_id: str, dst_path: str) -> Dict[str, str]:
    """Loads class names from MLflow.

    Downloads the index_to_class.json artifact from the specified MLflow run
    and loads it into a dictionary mapping class indices to class names.

    Args:
        run_id (str): The ID of the MLflow run containing the artifact.
        dst_path (str): The local destination path where the artifact will be saved.

    Returns:
        Dict[str, str]: A dictionary mapping class indices to class names.

    Raises:
        Exception: If loading the class names fails.
    """
    try:
        # load the index_to_class.json from mlflow
        class_names = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="index_to_class.json",
            dst_path=dst_path,
        )

        # load the json file
        with open(class_names, "r") as f:
            class_names = json.load(f)
        return class_names
    except Exception as e:
        logger.error(f"Error loading class names from MLflow: {e}", exc_info=True)
        # raise error because this is critical
        raise e
