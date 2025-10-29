from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str
    MLFLOW_TRACKING_URI: str
    MODEL_URI: str
    RUN_ID: str
    MINIO_ROOT_USER: str
    MINIO_ROOT_PASSWORD: str
    LOCAL_S3_PROXY_SERVICE_URL: str
    ALLOY_ENDPOINT: str
    LOCAL_MODEL_PATH: str
    LOCAL_INDEX_TO_JSON_PATH: str
    REDIS_PASSWORD: str
    REDIS_HOST: str
    REDIS_PORT: int
    JWT_SECRET: str
    JWT_ALGORITHM: str
    USE_LOCAL: bool
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# make it usable throughout the app
Config = Settings()
