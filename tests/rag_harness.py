"""A real RAGAnything on fake models, for storage-level tests.

LightRAG, its storages (JSON/NanoVectorDB/NetworkX files, or PostgreSQL +
pgvector when ``RAG_TEST_POSTGRES`` is set) and RAGAnything's pipeline are
real; only the parser, the LLM and the embedding model are fakes, so no
network, MinerU or tokenizer download is needed.
"""

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from lightrag.utils import EmbeddingFunc, Tokenizer
from raganything import RAGAnything, RAGAnythingConfig
from raganything.parser import Parser, register_parser

EMBED_DIM = 16


class CharTokenizer:
    def encode(self, content):
        return [ord(c) for c in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


def fake_vector(text: str) -> np.ndarray:
    seed = int.from_bytes(
        __import__("hashlib").sha256(text.encode()).digest()[:4], "big"
    )
    vec = np.random.RandomState(seed).rand(EMBED_DIM).astype(np.float32) + 0.1
    return vec / np.linalg.norm(vec)


async def fake_embed(texts, **kwargs):
    return np.array([fake_vector(t) for t in texts], dtype=np.float32)


_ENTITIES = ("Alice", "Bob", "ACME")


async def fake_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    if "high_level_keywords" in prompt:
        return json.dumps(
            {"high_level_keywords": ["work"], "low_level_keywords": ["ACME"]}
        )
    lines = []
    present = [e for e in _ENTITIES if e in prompt]
    if len(present) >= 2 and "Entity_types" in prompt:
        d = "<|#|>"
        for name in present:
            lines.append(
                f"entity{d}{name}{d}organization{d}{name} is mentioned in the text."
            )
        lines.append(
            f"relation{d}{present[0]}{d}{present[1]}{d}works with{d}{present[0]} works with {present[1]}."
        )
        lines.append("<|COMPLETE|>")
        return "\n".join(lines)
    return "<|COMPLETE|>" if "Entity_types" in prompt else "ok"


class FakeParser(Parser):
    """Plain-text parser: one text block per paragraph, one page per form feed."""

    def check_installation(self) -> bool:
        return True

    def parse_document(
        self, file_path, method="auto", output_dir=None, lang=None, **kw
    ):
        text = Path(file_path).read_text(encoding="utf-8")
        blocks: List[Dict[str, Any]] = []
        for page, chunk in enumerate(text.split("\f")):
            for para in re.split(r"\n\s*\n", chunk):
                if para.strip():
                    blocks.append(
                        {"type": "text", "text": para.strip(), "page_idx": page}
                    )
        return blocks

    parse_pdf = parse_document


def postgres_env(database: str = "") -> Dict[str, str]:
    return {
        "POSTGRES_HOST": os.environ.get("POSTGRES_HOST", "localhost"),
        "POSTGRES_PORT": os.environ.get("POSTGRES_PORT", "5432"),
        "POSTGRES_USER": os.environ.get("POSTGRES_USER", "rag"),
        "POSTGRES_PASSWORD": os.environ.get("POSTGRES_PASSWORD", "rag"),
        "POSTGRES_DATABASE": database or os.environ.get("POSTGRES_DATABASE", "rag"),
    }


def make_rag(
    tmp_path: Path,
    backend: str = "file",
    workspace: str = "",
    monkeypatch=None,
    database: str = "",
    embedding_model: str = "fake-embed",
    embed_func=None,
    **config_overrides: Any,
) -> RAGAnything:
    """Build a RAGAnything. Each instance needs its own ``workspace``: LightRAG
    keeps the in-memory state of a workspace per process."""
    register_parser("fake", FakeParser)  # other tests clear the registry
    workspace = workspace or f"ws{uuid.uuid4().hex[:10]}"
    kwargs: Dict[str, Any] = {
        "tokenizer": Tokenizer("chars", CharTokenizer()),
        "chunk_token_size": 400,
        "chunk_overlap_token_size": 40,
        "workspace": workspace,
    }
    if backend == "postgres":
        env = {**postgres_env(database), "POSTGRES_WORKSPACE": workspace}
        for key, value in env.items():
            if monkeypatch is not None:
                monkeypatch.setenv(key, value)
            else:
                os.environ[key] = value
        kwargs.update(
            kv_storage="PGKVStorage",
            vector_storage="PGVectorStorage",
            graph_storage="NetworkXStorage",
            doc_status_storage="PGDocStatusStorage",
        )
    config = RAGAnythingConfig(
        working_dir=str(tmp_path / "rag_storage"),
        parser_output_dir=str(tmp_path / "output"),
        parser="fake",
        enable_drawing_processing=False,
        **config_overrides,
    )
    return RAGAnything(
        config=config,
        llm_model_func=fake_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM,
            max_token_size=8192,
            func=embed_func or fake_embed,
            model_name=embedding_model,
        ),
        lightrag_kwargs=kwargs,
    )
