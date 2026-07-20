"""Application configuration loaded from environment variables."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# Look for .env in current dir, then parent dirs
_env_file = Path(".env")
if not _env_file.exists():
    _env_file = Path("../.env")
    if not _env_file.exists():
        _env_file = Path("../../.env")


class Settings(BaseSettings):
    # LLM Provider
    llm_provider: str = "venus"
    env_venus_openapi_secret_id: str = ""
    venus_llm_proxy_url: str = "http://v2.open.venus.oa.com/llmproxy"
    venus_llm_model: str = "gpt-4o-2024-11-20"

    # OpenAI fallback
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"

    # Paper source API keys
    semantic_scholar_api_key: str = ""
    openalex_email: str = ""
    ieee_api_key: str = ""
    core_api_key: str = ""

    # Agent params
    max_rounds: int = 5
    queries_per_round: int = 5
    papers_per_source_per_query: int = 15
    high_priority_target: int = 15

    # Database
    database_url: str = "sqlite:///./deep_research.db"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Authentication (P0-4): if set, POST/PUT/DELETE on /api/tasks require X-API-Key header
    api_key: str = ""

    model_config = SettingsConfigDict(
        env_file=str(_env_file),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
