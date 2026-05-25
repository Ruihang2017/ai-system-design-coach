"""Application settings loaded from environment / .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Providers
    llm_provider: str = "openai"          # "openai" | "fake"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    embed_provider: str = "fastembed"     # "fastembed" | "openai" | "fake"
    embed_model: str = "BAAI/bge-small-en-v1.5"

    # Vector DB
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "ai_coach_docs"

    # Pipeline
    top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 150
    score_threshold: float = 0.30
    rewrite_enabled: bool = True

    # Logging
    log_dir: str = "reports/eval_runs"

    # LangSmith (optional; tracing only fires when enabled + key present)
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "ai-system-design-coach"


settings = Settings()
