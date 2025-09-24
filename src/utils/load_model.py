from src.config import Config
import mlflow
import os
import tensorflow as tf


def load_model_from_mlflow() -> tf.keras.Model:
    """Load a TensorFlow model from MLflow.

    Returns:
        tf.keras.Model: The loaded TensorFlow model.
    """
    # connect to mlflow tracking server
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)

    # create a local directory to store the model
    dst_path = "./model"
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    # load the model from mlflow
    model = mlflow.tensorflow.load_model(Config.MODEL_URI, dst_path=dst_path)
    return model
