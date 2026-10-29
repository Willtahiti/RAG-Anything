"""Resilient embedding wrapper and resume of unfinished documents."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from lightrag.base import DocStatus
from raganything.resilient_embedding import (
    EmbeddingUnavailableError,
    EmbeddingValidationError,
    is_transient_error,
    make_resilient_embedding,
    make_resilient_llm,
    validate_embeddings,
)
from raganything.resume import (
    list_unfinished_documents,
    resolve_source_file,
    resume_unfinished,
)

FAST = dict(base_delay=0.0, max_delay=0.0, health_interval=0.0)


class FlakyServer:
    """Embedding server that is down for the first ``down_calls`` calls."""

    def __init__(self, down_calls=0, dim=4, exc=None):
        self.down_calls = down_calls
        self.dim = dim
        self.calls = 0
        self.exc = exc or ConnectionError("connection refused")

    async def __call__(self, texts):
        self.calls += 1
        if self.calls <= self.down_calls:
            raise self.exc
        return [[float(i + 1)] * self.dim for i, _ in enumerate(texts)]


@pytest.mark.asyncio
async def test_survives_outage_and_returns_validated_vectors():
    server = FlakyServer(down_calls=3)
    outage, recovered = [], []
    embed = make_resilient_embedding(
        server,
        expected_dim=4,
        on_outage=lambda exc: outage.append(exc),
        on_recover=lambda s: recovered.append(s),
        **FAST,
    )
    out = await embed(["a", "b"])
    assert isinstance(out, np.ndarray) and out.shape == (2, 4)
    assert server.calls == 4
    assert len(outage) == 1 and len(recovered) == 1
    assert embed.stats["retries"] == 3 and embed.stats["outages"] == 1


@pytest.mark.asyncio
async def test_gives_up_after_max_outage_with_clear_error():
    server = FlakyServer(down_calls=10**9)
    embed = make_resilient_embedding(server, max_outage_s=0.05, **FAST)
    with pytest.raises(EmbeddingUnavailableError, match="resume"):
        await embed(["a"])


@pytest.mark.asyncio
async def test_permanent_errors_are_not_retried():
    class BadRequest(Exception):
        status_code = 400

    server = FlakyServer(down_calls=10, exc=BadRequest("input too long"))
    embed = make_resilient_embedding(server, **FAST)
    with pytest.raises(BadRequest):
        await embed(["a"])
    assert server.calls == 1


@pytest.mark.asyncio
async def test_5xx_and_429_are_retried():
    class ServerError(Exception):
        def __init__(self, code):
            self.status_code = code

    for code in (500, 503, 429):
        server = FlakyServer(down_calls=1, exc=ServerError(code))
        embed = make_resilient_embedding(server, **FAST)
        assert (await embed(["a"])).shape == (1, 4)
        assert server.calls == 2


@pytest.mark.asyncio
async def test_wrong_dimension_is_rejected_after_retries():
    server = FlakyServer(dim=3)
    embed = make_resilient_embedding(
        server, expected_dim=4, max_validation_retries=2, **FAST
    )
    with pytest.raises(EmbeddingValidationError, match="dimension 3"):
        await embed(["a"])
    assert server.calls == 3


@pytest.mark.asyncio
async def test_truncated_answer_then_good_answer():
    calls = []

    async def server(texts):
        calls.append(1)
        if len(calls) == 1:
            return [[1.0, 2.0]]  # one vector for two texts
        return [[1.0, 2.0]] * len(texts)

    embed = make_resilient_embedding(server, expected_dim=2, **FAST)
    assert (await embed(["a", "b"])).shape == (2, 2)
    assert embed.stats["validation_failures"] == 1


@pytest.mark.asyncio
async def test_request_timeout_counts_as_transient():
    calls = []

    async def slow_then_fast(texts):
        calls.append(1)
        if len(calls) == 1:
            await asyncio.sleep(5)
        return [[1.0]] * len(texts)

    embed = make_resilient_embedding(slow_then_fast, request_timeout=0.05, **FAST)
    assert (await embed(["a"])).shape == (1, 1)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_health_check_gates_retries():
    server = FlakyServer(down_calls=1)
    checks = []

    async def health():
        checks.append(1)
        return len(checks) >= 3

    embed = make_resilient_embedding(server, health_check=health, **FAST)
    await embed(["a"])
    assert len(checks) == 3 and server.calls == 2


@pytest.mark.asyncio
async def test_concurrency_limit():
    running = peak = 0

    async def server(texts):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return [[1.0]] * len(texts)

    embed = make_resilient_embedding(server, max_concurrency=2, **FAST)
    await asyncio.gather(*(embed(["x"]) for _ in range(8)))
    assert peak == 2


@pytest.mark.asyncio
async def test_llm_wrapper_passes_args_through():
    seen = []

    async def llm(prompt, system_prompt=None, **kw):
        seen.append((prompt, system_prompt, kw))
        if len(seen) == 1:
            raise TimeoutError()
        return "ok"

    wrapped = make_resilient_llm(llm, **FAST)
    assert await wrapped("p", system_prompt="s", image_data="img") == "ok"
    assert seen[-1] == ("p", "s", {"image_data": "img"})


def test_validate_embeddings_rejects_bad_vectors():
    good = validate_embeddings([[1, 2], [3, 4]], 2, 2)
    assert good.dtype == np.float32
    for bad in ([[float("nan"), 1.0]], [[0.0, 0.0]], [1.0, 2.0], "abc"):
        with pytest.raises(EmbeddingValidationError):
            validate_embeddings(bad, 1, 2)


def test_error_classification():
    assert is_transient_error(ConnectionResetError())
    assert is_transient_error(asyncio.TimeoutError())
    assert not is_transient_error(ValueError("x"))
    assert not is_transient_error(KeyError("x"))

    class ServerDisconnectedError(Exception):
        pass

    assert is_transient_error(ServerDisconnectedError())


# --- resume ---------------------------------------------------------------


class FakeDocStatus:
    def __init__(self, records):
        self.records = records

    async def get_docs_by_statuses(self, statuses):
        return {k: v for k, v in self.records.items() if v.status in statuses}


def _status(path, status, updated="2026-01-01"):
    return SimpleNamespace(
        file_path=path, status=status, error_msg=None, updated_at=updated
    )


class FakeRag:
    def __init__(self, records, fail=()):
        self.lightrag = SimpleNamespace(doc_status=FakeDocStatus(records))
        self.processed = []
        self.fail = set(fail)

    async def _ensure_lightrag_initialized(self):
        return {"success": True}

    async def process_document_complete(self, path, **kw):
        if path.endswith(tuple(self.fail)):
            raise RuntimeError("still down")
        self.processed.append((path, kw))


@pytest.mark.asyncio
async def test_resume_finds_and_reruns_unfinished(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.pdf").write_bytes(b"x")
    (tmp_path / "c.pdf").write_bytes(b"x")
    rag = FakeRag(
        {
            "d1": _status("a.pdf", DocStatus.FAILED, "2026-01-02"),
            "d2": _status("b.pdf", DocStatus.PREPROCESSED, "2026-01-01"),
            "d3": _status("gone.pdf", DocStatus.PENDING),
            "d4": _status("done.pdf", DocStatus.PROCESSED),
            "d5": _status("c.pdf", DocStatus.PROCESSING, "2026-01-03"),
        },
        fail={"c.pdf"},
    )
    unfinished = await list_unfinished_documents(rag)
    assert [d["doc_id"] for d in unfinished] == ["d2", "d3", "d1", "d5"]
    assert "d4" not in {d["doc_id"] for d in unfinished}

    progress = []
    outcome = await resume_unfinished(
        rag, [tmp_path], on_progress=progress.append, lang="fr"
    )
    assert {e["doc_id"] for e in outcome["resumed"]} == {"d1", "d2"}
    assert [e["doc_id"] for e in outcome["failed"]] == ["d5"]
    assert outcome["failed"][0]["error_msg"] == "still down"
    assert [e["doc_id"] for e in outcome["missing"]] == ["d3"]
    assert all(kw == {"lang": "fr"} for _, kw in rag.processed)
    assert len(progress) == 3


def test_resolve_source_file(tmp_path):
    f = tmp_path / "x" / "doc.pdf"
    f.parent.mkdir()
    f.write_bytes(b"x")
    assert resolve_source_file(str(f), []) == f
    assert resolve_source_file("doc.pdf", [tmp_path]) == f
    assert resolve_source_file("nope.pdf", [tmp_path]) is None
    assert resolve_source_file("", [tmp_path]) is None
