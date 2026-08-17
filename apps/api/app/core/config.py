from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    app_name: str = "JobPilot API"
    environment: str = "development"

    database_url: str = "postgresql+asyncpg://jobpilot:jobpilot@localhost:5432/jobpilot"
    redis_url: str = "redis://localhost:6379/0"

    # External providers (dummy keys in development; refill for testing)
    openai_api_key: str = "sk-dummy"
    sarvam_api_key: str = "sarvam-dummy"
    dodo_payments_api_key: str = "dodo-dummy"

    google_oauth_client_id: str = ""
    google_oauth_client_secret: str = ""

    s3_endpoint_url: str = ""
    s3_bucket: str = "jobpilot"
    s3_access_key: str = ""
    s3_secret_key: str = ""

    cors_origins: list[str] = ["http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
