"""Centralized configuration via Pydantic Settings -- all values from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # -- Database ----------------------------------------------------------------
    database_url: str = "postgresql+psycopg://trader:trader_dev@localhost:5432/trader_agent"
    async_database_url: str = "postgresql+asyncpg://trader:trader_dev@localhost:5432/trader_agent"
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # -- HTTP / API --------------------------------------------------------------
    http_timeout_seconds: float = 30.0

    # -- API Keys (never commit real values -- use .env or secrets manager) -------
    finnhub_api_key: str = ""
    newsapi_api_key: str = ""
    twitter_bearer_token: str = ""
    stocktwits_access_token: str = ""

    # -- NLP / Model -------------------------------------------------------------
    finbert_model_name: str = "ProsusAI/finbert"
    finbert_device: str = "cpu"

    # -- Alerting ----------------------------------------------------------------
    webhook_url: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    alert_recipients: str = ""  # comma-separated emails

    # -- Scheduling --------------------------------------------------------------
    ingest_fundamentals_cron: str = "0 6 * * 1-5"  # weekdays 06:00 UTC
    ingest_insider_trades_cron: str = "0 7 * * 1-5"
    ingest_news_interval_minutes: int = 30
    ingest_social_interval_minutes: int = 15

    # -- Prediction --------------------------------------------------------------
    prediction_horizon_days: int = 90
    default_model_type: str = "lightgbm"

    @property
    def alert_recipient_list(self) -> list[str]:
        return [r.strip() for r in self.alert_recipients.split(",") if r.strip()]


settings = Settings()
