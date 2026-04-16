"""
Configuration management for the Data Science Platform.
Uses pydantic settings with environment variable support.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    # Application
    app_name: str = Field(
        default="Data Science Platform", description="Application name"
    )
    app_env: str = Field(
        default="development",
        description="Environment: development, staging, production",
    )
    debug: bool = Field(default=True, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # Database
    database_url: str = Field(
        default="sqlite:///./data.db", description="Database connection URL"
    )

    # API
    api_host: str = Field(default="0.0.0.0", description="FastAPI host")
    api_port: int = Field(default=8000, description="FastAPI port")
    api_workers: int = Field(default=4, description="Number of API workers")

    # Streamlit
    streamlit_host: str = Field(default="0.0.0.0", description="Streamlit host")
    streamlit_port: int = Field(default=8501, description="Streamlit port")
    streamlit_theme: str = Field(default="light", description="Streamlit theme")

    # Caching
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    redis_url: Optional[str] = Field(default=None, description="Redis URL for caching")

    # Security
    secret_key: str = Field(
        default="change-this-in-production", description="Secret key for security"
    )
    allowed_hosts: str = Field(
        default="localhost,127.0.0.1",
        description="Comma-separated list of allowed hosts",
    )

    # Analytics (optional)
    google_analytics_id: Optional[str] = Field(
        default=None, description="Google Analytics ID"
    )
    sentry_dsn: Optional[str] = Field(
        default=None, description="Sentry DSN for error tracking"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.app_env == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env == "production"

    @property
    def allowed_hosts_list(self) -> list[str]:
        """Get allowed hosts as a list."""
        return [host.strip() for host in self.allowed_hosts.split(",") if host.strip()]

    def get_database_path(self) -> Path:
        """Get the database file path for SQLite."""
        if self.database_url.startswith("sqlite:///"):
            db_path = self.database_url.replace("sqlite:///", "")
            return Path(db_path).resolve()
        raise ValueError("Not a SQLite database URL")


# Global settings instance
settings = Settings()


# Create .env file if it doesn't exist
def ensure_env_file() -> None:
    """Create .env file from .env.example if it doesn't exist."""
    env_path = Path(".env")
    env_example_path = Path(".env.example")

    if not env_path.exists() and env_example_path.exists():
        env_example_path.copy_to(env_path)
        print(f"Created {env_path} from {env_example_path}")


# Log configuration on startup
if __name__ == "__main__":
    print(f"Configuration loaded for {settings.app_name}")
    print(f"Environment: {settings.app_env}")
    print(f"Debug: {settings.debug}")
    print(f"Database: {settings.database_url}")
