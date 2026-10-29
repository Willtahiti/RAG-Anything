"""Build a configured RAGAnything from environment variables."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from raganything import RAGAnything, RAGAnythingConfig
from raganything.app.models import build_models


def lightrag_kwargs_from_env(env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    env = dict(os.environ if env is None else env)
    kwargs: Dict[str, Any] = {}
    if env.get("RAG_WORKSPACE"):
        kwargs["workspace"] = env["RAG_WORKSPACE"]
    # Quality over speed: few concurrent embedding calls, modest batches
    kwargs["embedding_func_max_async"] = int(env.get("RAG_EMBED_CONCURRENCY", "2"))
    kwargs["embedding_batch_num"] = int(env.get("RAG_EMBED_BATCH", "8"))
    kwargs["llm_model_max_async"] = int(env.get("RAG_LLM_CONCURRENCY", "2"))
    kwargs["max_parallel_insert"] = int(env.get("RAG_PARALLEL_INSERT", "2"))
    if env.get("RAG_STORAGE", "file").lower() == "postgres":
        kwargs.update(
            kv_storage="PGKVStorage",
            vector_storage="PGVectorStorage",
            doc_status_storage="PGDocStatusStorage",
            graph_storage=env.get("RAG_GRAPH_STORAGE", "NetworkXStorage"),
        )
    return kwargs


def build_rag_from_env(
    env: Optional[Dict[str, str]] = None,
) -> tuple[RAGAnything, dict]:
    """Return ``(rag, health_checks)``."""
    models = build_models(env)
    health = models.pop("health")
    rag = RAGAnything(
        config=RAGAnythingConfig(),
        lightrag_kwargs=lightrag_kwargs_from_env(env),
        **models,
    )
    return rag, health
