from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str
    MLFLOW_TRACKING_URI: str
    MODEL_URI: str
    RUN_ID: str
    MINIO_ROOT_USER: str
    MINIO_ROOT_PASSWORD: str
    LOCAL_S3_PROXY_SERVICE_URL: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# make it usable throughout the app
Config = Settings()
