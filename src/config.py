from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str
    MLFLOW_TRACKING_URI: str
    MODEL_URI: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# make it usable throughout the app
Config = Settings()
