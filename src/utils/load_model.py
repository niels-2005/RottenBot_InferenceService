import mlflow
import tensorflow as tf
from typing import Dict
import json


def load_model_from_mlflow(model_uri: str, dst_path: str) -> tf.keras.Model:
    """Load a TensorFlow model from MLflow.

    Returns:
        tf.keras.Model: The loaded TensorFlow model.
    """
    # load the model from mlflow
    model = mlflow.tensorflow.load_model(model_uri, dst_path=dst_path)
    return model


def load_classes_from_mlflow(run_id: str, dst_path: str) -> Dict[str, str]:
    """Load class names from MLflow.

    Returns:
        Dict[str, str]: A dictionary mapping class indices to class names.
    """
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
