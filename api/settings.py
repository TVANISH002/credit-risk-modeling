from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODEL_PATH: str = "artifacts/model_data_v1.joblib"
    MODEL_VERSION: str = "v1"

    class Config:
        env_file = ".env"


settings = Settings()
