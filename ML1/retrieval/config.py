from __future__ import annotations

import os
from dataclasses import dataclass, field

SCHEMA_VERSION = 1
MAX_CHARS = 1450
OVERLAP = 160
RRF_K = 30
CANDIDATE_DEPTH = 50
LOCAL_VECTOR_MODEL = "corpus-char-tfidf-3-4-v1"


@dataclass(frozen=True)
class Settings:
    base_url: str = ""
    api_key: str = field(default="", repr=False)
    embedding_model: str = "BAAI/bge-m3"
    embedding_path: str = "/embeddings"
    reranker_model: str = ""
    reranker_path: str = "/rerank"
    reranker_top_n: int = 20
    api_timeout: float = 3.0
    build_timeout: float = 12.0
    retries: int = 1

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            base_url=os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL", ""),
            api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
            embedding_model=os.getenv("EMB_MODEL") or "BAAI/bge-m3",
            embedding_path=os.getenv("RETRIEVAL_EMBEDDING_PATH", "/embeddings"),
            reranker_model=os.getenv("RERANKER_MODEL", ""),
            reranker_path=os.getenv("RETRIEVAL_RERANKER_PATH", "/rerank"),
            reranker_top_n=max(0, min(30, int(os.getenv("RERANKER_TOP_N", "20")))),
            api_timeout=max(0.1, float(os.getenv("RETRIEVAL_API_TIMEOUT", "3"))),
            build_timeout=max(0.1, float(os.getenv("RETRIEVAL_BUILD_TIMEOUT", "12"))),
            retries=max(0, min(2, int(os.getenv("RETRIEVAL_API_RETRIES", "1")))),
        )

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key)
