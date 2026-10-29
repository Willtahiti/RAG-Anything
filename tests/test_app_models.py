"""OpenAI-compatible model functions: payloads, resilience, configuration."""

import base64
import json

import httpx
import numpy as np
import pytest

from raganything.app import models
from raganything.app.factory import lightrag_kwargs_from_env

ENV = {
    "RAG_LLM_BASE_URL": "http://llm.test/v1",
    "RAG_LLM_MODEL": "qwen-vl",
    "RAG_EMBED_MODEL": "bge-m3",
    "RAG_EMBED_DIM": "3",
    "RAG_MAX_OUTAGE_S": "5",
}
PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 20).decode()


@pytest.fixture
def server(monkeypatch):
    calls = []
    state = {"down": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        calls.append((request.url.path, body))
        if state["down"] > 0 and request.url.path.endswith("embeddings"):
            state["down"] -= 1
            raise httpx.ConnectError("refused")
        if request.url.path.endswith("embeddings"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": i, "embedding": [1.0, 2.0, 3.0]}
                        for i, _ in reversed(list(enumerate(body["input"])))
                    ]
                },
            )
        if request.url.path.endswith("chat/completions"):
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "réponse"}}]}
            )
        return httpx.Response(200, json={"data": []})

    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient
    monkeypatch.setattr(
        models.httpx, "AsyncClient", lambda **kw: real(transport=transport, **kw)
    )
    monkeypatch.setattr(
        "raganything.resilient_embedding.ResilientCaller._delay", lambda self, a: 0.0
    )
    return calls, state


@pytest.mark.asyncio
async def test_chat_and_vision_payloads(server):
    calls, _ = server
    m = models.build_models(ENV)
    out = await m["llm_model_func"](
        "hello",
        system_prompt="sys",
        history_messages=[{"role": "user", "content": "h"}],
    )
    assert out == "réponse"
    msgs = calls[-1][1]["messages"]
    assert [x["role"] for x in msgs] == ["system", "user", "user"] and calls[-1][1][
        "model"
    ] == "qwen-vl"

    await m["vision_model_func"]("describe", image_data=PNG)
    content = calls[-1][1]["messages"][-1]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    await m["vision_model_func"]("no image")
    assert isinstance(calls[-1][1]["messages"][-1]["content"], str)


@pytest.mark.asyncio
async def test_embeddings_in_order_and_survive_outage(server):
    calls, state = server
    m = models.build_models(ENV)
    state["down"] = 3
    vectors = await m["embedding_func"].func(["a", "b"])
    assert vectors.shape == (2, 3) and vectors.dtype == np.float32
    assert sum(1 for p, _ in calls if p.endswith("embeddings")) == 4
    assert (
        m["embedding_func"].embedding_dim == 3
        and m["embedding_func"].model_name == "bge-m3"
    )


def test_missing_configuration_is_explained():
    with pytest.raises(RuntimeError, match="RAG_EMBED_DIM"):
        models.build_models({"RAG_LLM_MODEL": "x"})


def test_lightrag_kwargs_favor_quality_and_postgres():
    kw = lightrag_kwargs_from_env({})
    assert kw["embedding_func_max_async"] == 2 and "kv_storage" not in kw
    kw = lightrag_kwargs_from_env(
        {"RAG_STORAGE": "postgres", "RAG_WORKSPACE": "chantier1"}
    )
    assert (
        kw["kv_storage"] == "PGKVStorage" and kw["vector_storage"] == "PGVectorStorage"
    )
    assert kw["workspace"] == "chantier1" and kw["graph_storage"] == "NetworkXStorage"
