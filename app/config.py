from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Callback
    CALLBACK_URL: str = "https://platform-integrator.transtrack.co/cloud-result"
    PUBLIC_BASE_URL: str = "https://cloud.transtrack.co"

    # Storage
    RECORDS_DIR: str = "./records"
    RECORD_TTL_SECONDS: int = 30 * 60   # JSON records: 30 min
    VIDEO_TTL_SECONDS: int = 45 * 60    # processed videos: 45 min

    # HTTP
    VIDEO_DOWNLOAD_TIMEOUT: int = 60
    HTTPX_TIMEOUT: int = 30

    # Model
    MODEL_PATH: str = "models/classifier/best_val_f1.pth"
    MODEL_NAME: str = "MultiScaleTCN"
    MODEL_DOWNLOAD_URL: str = ""

    # Queue
    REDIS_URL: str = "redis://localhost:6379/0"
    MAX_QUEUE_SIZE: int = 5000


settings = Settings()
