"""FastAPI application exposing the RAG pipeline."""

import logging
from functools import lru_cache

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.logging_config import configure_logging
from app.rag.models import AnswerResult

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="AI System Design Coach", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class QueryRequest(BaseModel):
    query: str


@lru_cache(maxsize=1)
def _orchestrator():
    from app.rag.orchestrator import build_orchestrator

    return build_orchestrator(settings)


def get_orchestrator():
    return _orchestrator()


@app.post("/query", response_model=AnswerResult)
def query(req: QueryRequest, orchestrator=Depends(get_orchestrator)) -> AnswerResult:
    return orchestrator.answer(req.query)


@app.get("/health")
def health() -> dict:
    qdrant_ok = False
    try:
        from qdrant_client import QdrantClient

        QdrantClient(url=settings.qdrant_url).get_collections()
        qdrant_ok = True
    except Exception as exc:  # noqa: BLE001 - health check must not raise
        logger.warning("Qdrant health check failed: %s", exc)
    return {"status": "ok", "qdrant": qdrant_ok}
