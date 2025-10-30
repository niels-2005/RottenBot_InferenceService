import json
from typing import Dict
import mlflow
import tensorflow as tf

import logging
from src.config import Config

logger = logging.getLogger(__name__)


def load_model_from_mlflow(
    model_uri: str, dst_path: str, use_local: bool = True
) -> tf.keras.Model:
    """Loads a TensorFlow model from MLflow.

    Downloads and loads the specified TensorFlow model from the given MLflow model URI,
    saving it to the destination path.

    Args:
        model_uri (str): The URI of the MLflow model to load.
        dst_path (str): The local destination path where the model will be saved.
        use_local (bool): Whether to load the model from a local path instead of MLflow. Defaults to True.

    Returns:
        tf.keras.Model: The loaded TensorFlow model.

    Raises:
        Exception: If loading the model fails.
    """
    try:
        if use_local:
            print("DEBUG: Loading model from local path")
            logger.info(f"Loading model from local path:")
            model = mlflow.tensorflow.load_model(model_uri)
            return model
        else:
            print("DEBUG: Loading model from MLflow")
            # load the model from mlflow
            logger.info(f"Loading model from MLflow URI: {model_uri}")
            model = mlflow.tensorflow.load_model(model_uri, dst_path=dst_path)
            return model
    except Exception as e:
        print("DEBUG: Exception occurred while loading model")
        logger.error(f"Error loading model from MLflow: {e}", exc_info=True)
        # raise error because this is critical
        raise e


def load_classes_from_mlflow(
    run_id: str, dst_path: str, use_local: bool = True
) -> Dict[str, str]:
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
        if use_local:
            print("DEBUG: Loading class names from local path")
            logger.info(f"Loading class names from local path: {run_id}")
            with open(run_id, "r") as f:
                class_names = json.load(f)
            return class_names
        else:
            print("DEBUG: Loading class names from MLflow")
            logger.info(f"Loading class names from MLflow run ID: {run_id}")
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
        print("DEBUG: Exception occurred while loading class names")
        logger.error(f"Error loading class names from MLflow: {e}", exc_info=True)
        # raise error because this is critical
        raise e
