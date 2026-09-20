"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings; no model is loaded while constructing them."""

    model_config = SettingsConfigDict(env_prefix="ERIS_", env_file=".env", extra="ignore")
    environment: str = "development"
    model_bundle_path: Path = Path("artifacts/models/eris_xgboost_v1.joblib")
    model_metadata_path: Path = Path("artifacts/models/eris_xgboost_v1.metadata.json")
    model_checksum_path: Path = Path("artifacts/models/eris_xgboost_v1.sha256")
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_log_level: str = "INFO"
    max_batch_size: int = 100
    cors_origins: str = ""
    service_token: str | None = None
