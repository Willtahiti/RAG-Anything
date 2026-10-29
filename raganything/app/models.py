"""Model functions for OpenAI-compatible servers (Ollama, vLLM, LM Studio...).

Configured by environment variables:

``RAG_LLM_BASE_URL`` (default ``http://localhost:11434/v1``), ``RAG_LLM_MODEL``,
``RAG_VISION_MODEL`` (defaults to the LLM), ``RAG_EMBED_BASE_URL`` (defaults to
the LLM URL), ``RAG_EMBED_MODEL``, ``RAG_EMBED_DIM`` and ``RAG_API_KEY``.

Every callable is wrapped so that server disconnections are waited out (see
``raganything.resilient_embedding``) and embeddings are validated.
"""

from __future__ import annotations

import base64
import os
from typing import Any, Dict, List, Optional, Sequence

import httpx
import numpy as np

from lightrag.utils import EmbeddingFunc
from raganything.image_signature import image_mime_type
from raganything.resilient_embedding import make_resilient_embedding, make_resilient_llm


class _HttpStatusError(Exception):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}: {body[:300]}")
        self.status_code = status_code


class OpenAICompatClient:
    def __init__(
        self, base_url: str, api_key: str = "", timeout: float = 600.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.timeout = timeout

    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}{path}", json=payload, headers=self.headers
            )
        if response.status_code >= 400:
            raise _HttpStatusError(response.status_code, response.text)
        return response.json()

    async def chat(
        self, model: str, messages: List[Dict[str, Any]], **options: Any
    ) -> str:
        data = await self._post(
            "/chat/completions", {"model": model, "messages": messages, **options}
        )
        return data["choices"][0]["message"].get("content") or ""

    async def embed(self, model: str, texts: Sequence[str]) -> np.ndarray:
        data = await self._post("/embeddings", {"model": model, "input": list(texts)})
        rows = sorted(data["data"], key=lambda item: item.get("index", 0))
        return np.asarray([row["embedding"] for row in rows], dtype=np.float32)

    async def healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(
                    f"{self.base_url}/models", headers=self.headers
                )
            return response.status_code < 500
        except Exception:
            return False


def _history(history_messages: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return [dict(m) for m in history_messages or []]


def build_models(env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Return ``llm_model_func``, ``vision_model_func`` and ``embedding_func``."""
    env = dict(os.environ if env is None else env)
    base = env.get("RAG_LLM_BASE_URL", "http://localhost:11434/v1")
    key = env.get("RAG_API_KEY", "")
    llm_model = env.get("RAG_LLM_MODEL", "")
    vision_model = env.get("RAG_VISION_MODEL", llm_model)
    embed_model = env.get("RAG_EMBED_MODEL", "")
    if not (llm_model and embed_model and env.get("RAG_EMBED_DIM")):
        raise RuntimeError(
            "Set RAG_LLM_MODEL, RAG_EMBED_MODEL and RAG_EMBED_DIM "
            "(and RAG_LLM_BASE_URL if the server is not Ollama on localhost)"
        )
    dim = int(env["RAG_EMBED_DIM"])
    llm = OpenAICompatClient(base, key)
    emb = OpenAICompatClient(env.get("RAG_EMBED_BASE_URL", base), key, timeout=300.0)

    async def llm_model_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> str:
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(_history(history_messages))
        messages.append({"role": "user", "content": prompt})
        return await llm.chat(llm_model, messages, temperature=0)

    async def vision_model_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, Any]]] = None,
        image_data: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> str:
        if messages:
            return await llm.chat(vision_model, messages, temperature=0)
        if not image_data:
            return await llm_model_func(prompt, system_prompt, history_messages)
        mime = image_mime_type(base64.b64decode(image_data[:32])) or "image/png"
        chat: List[Dict[str, Any]] = []
        if system_prompt:
            chat.append({"role": "system", "content": system_prompt})
        chat.extend(_history(history_messages))
        chat.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_data}"},
                    },
                ],
            }
        )
        return await llm.chat(vision_model, chat, temperature=0)

    async def embed(texts: List[str], **kwargs: Any) -> np.ndarray:
        return await emb.embed(embed_model, texts)

    max_outage = float(env.get("RAG_MAX_OUTAGE_S", "3600"))

    async def health() -> bool:
        return await emb.healthy()

    resilient_embed = make_resilient_embedding(
        embed, expected_dim=dim, max_outage_s=max_outage, health_check=health
    )
    return {
        "llm_model_func": make_resilient_llm(llm_model_func, max_outage_s=max_outage),
        "vision_model_func": make_resilient_llm(
            vision_model_func, max_outage_s=max_outage
        ),
        "embedding_func": EmbeddingFunc(
            embedding_dim=dim,
            max_token_size=int(env.get("RAG_EMBED_MAX_TOKENS", "8192")),
            func=resilient_embed,
            model_name=embed_model,
        ),
        "health": {"llm": llm.healthy, "embedding": emb.healthy},
    }
