"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings; no model is loaded while constructing them."""

    model_config = SettingsConfigDict(env_prefix="ERIS_", env_file=".env", extra="ignore")
    environment: str = "development"
    model_bundle_path: Path = Path("artifacts/models/model_bundle.joblib")
    log_level: str = "INFO"
