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

    # Concurrency limits (P0-1: prevent SQLite write contention)
    max_concurrent_agents: int = 2

    # RAG / PDF download
    enable_scihub: bool = False  # P0-4: Sci-Hub disabled by default for legal compliance

    # Scoring weights (P1-7: authority bumped from 0.25 to 0.30)
    score_weight_relevance: float = 0.25
    score_weight_authority: float = 0.30
    score_weight_recency: float = 0.15
    score_weight_novelty: float = 0.15
    score_weight_idea_potential: float = 0.15

    # Rate limiting (P1-9: per-source request budget)
    rate_limit_s2_per_min: int = 100
    rate_limit_openalex_per_min: int = 100
    rate_limit_default_per_min: int = 60

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
