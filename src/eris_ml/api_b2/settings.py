"""Environment-only B2 configuration; no .env or implicit service secret."""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from eris_ml.api_b2.schemas import MODEL_SHA256

ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ERIS_", env_file=None, extra="forbid")

    bundle_path: Path = ROOT / "artifacts/models/v3_final_15/b2"
    adaboost_bundle_path: Path = ROOT / "artifacts/models/adaboost_v3/s1"
    explain_queue_size: int = Field(default=2, ge=0, le=16)
    explain_timeout_seconds: float = Field(default=60, gt=0, le=300)
    expected_model_sha256: str = MODEL_SHA256
    service_api_key: SecretStr = SecretStr("")
    max_batch_size: int = Field(default=100, ge=1, le=100)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
