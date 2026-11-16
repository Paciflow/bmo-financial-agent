from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5-5"
    anthropic_interpretation_model: str = "claude-sonnet-5-5"

    # AWS / S3
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket_name: str = ""

    # Embeddings
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "

    # Qdrant
    qdrant_local_path: str = "./data/qdrant"
    qdrant_sections_collection: str = "bmo_sections"
    qdrant_chunks_collection: str = "bmo_chunks"

    # BM25
    bm25_index_path: str = "./data/bm25"

    # Document
    bmo_pdf_url: str = "https://www.bmo.com/ir/archive/en/bmo_ar2025_MDA.pdf"
    document_id: str = "bmo_ar2025_mda"

    # Retrieval
    dense_top_k: int = 30
    bm25_top_k: int = 30
    rrf_k: int = 60
    rerank_top_k: int = 20
    evidence_top_k: int = 10

    # App
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def s3_configured(self) -> bool:
        # A bucket name is the only thing we can check locally. Explicit keys are
        # one way to authenticate (local/.env setup) but not the only way -- an
        # ECS task role provides credentials implicitly with no keys in sight.
        # Whether those implicit credentials actually work is validated for real
        # by storage.s3_store.check_bucket_access(), not by this property.
        return bool(self.s3_bucket_name)

    @property
    def anthropic_configured(self) -> bool:
        return bool(self.anthropic_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
