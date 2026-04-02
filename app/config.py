from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    API_KEY: str = "changeme"
    CALLBACK_URL: str = "https://platform-integrator.transtrack.co/cloud-result"
    RECORDS_DIR: str = "./records"
    VIDEO_DOWNLOAD_TIMEOUT: int = 60
    HTTPX_TIMEOUT: int = 30
    KEEP_TMP_VIDEOS: bool = False

    TRANSTRACK_BASE_URL: str = ""
    TRANSTRACK_USERNAME: str = ""
    TRANSTRACK_PASSWORD: str = ""

    MODEL_PATH: str = "models/classifier/best_val_f1.pth"
    MODEL_NAME: str = "MultiScaleTCN"


settings = Settings()
