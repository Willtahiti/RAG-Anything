"""Web application: ingestion jobs, cited search, files, auth."""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from raganything.app.server import create_app
from raganything.app.service import KnowledgeService, make_excerpt, safe_filename
from rag_harness import make_rag

DOC = "Alice works with Bob at ACME.\n\nA second paragraph about ACME and cables.\fPage two mentions Alice and ACME again."


def _wait(client, predicate, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = client.get("/api/jobs").json()
        if jobs and predicate(jobs):
            return jobs
        time.sleep(0.2)
    raise AssertionError(f"timeout, jobs={client.get('/api/jobs').json()}")


@pytest.fixture
def make_client(tmp_path):
    clients = []

    def factory(**app_kwargs):
        rag = make_rag(tmp_path / "rag")
        service = KnowledgeService(rag)
        app = create_app(service, **app_kwargs)
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client, service

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


def test_ingest_search_and_sources(make_client):
    client, service = make_client()
    r = client.post(
        "/api/ingest", files={"files": ("rapport.txt", DOC.encode(), "text/plain")}
    )
    assert r.status_code == 200 and len(r.json()["jobs"]) == 1
    jobs = _wait(client, lambda js: all(j["status"] in ("done", "failed") for j in js))
    assert jobs[0]["status"] == "done", jobs

    docs = client.get("/api/documents").json()
    assert [d["file"] for d in docs] == ["rapport.txt"] and docs[0]["has_original"]

    result = client.post("/api/search", json={"question": "Who works at ACME?"}).json()
    assert result["answer"]
    src = result["sources"][0]
    assert src["file"] == "rapport.txt" and src["kind"] == "text"
    assert src["page"] == 1 and src["page_end"] == 2
    assert "ACME" in src["excerpt"]
    assert src["link"].startswith(f"/api/documents/{docs[0]['doc_id']}/view?chunk=")

    # the link opens the parsed text with the cited chunk highlighted
    page = client.get(src["link"].split("#")[0])
    assert (
        page.status_code == 200 and 'class="hit"' in page.text and "ACME" in page.text
    )
    # the original is served inline
    orig = client.get(f"/api/files/{docs[0]['doc_id']}")
    assert orig.status_code == 200 and orig.content == DOC.encode()


def test_pdf_sources_link_to_the_cited_page(make_client):
    client, service = make_client()
    source = {
        "doc_id": "doc-1",
        "file": "manuel.PDF",
        "page": 7,
        "has_original": True,
        "chunk_id": "chunk-1",
    }
    from raganything.app.server import _source_link

    assert _source_link(service, source) == "/api/files/doc-1#page=7"
    source["has_original"] = False
    assert _source_link(service, source).startswith("/api/documents/doc-1/view")


def test_same_file_gets_the_same_document_and_resumes(make_client):
    client, service = make_client()
    for _ in range(2):
        client.post(
            "/api/ingest", files={"files": ("a.txt", DOC.encode(), "text/plain")}
        )
    jobs = _wait(
        client,
        lambda js: len(js) == 2 and all(j["status"] in ("done", "failed") for j in js),
    )
    assert {j["doc_id"] for j in jobs} == {jobs[0]["doc_id"]}
    assert len(client.get("/api/documents").json()) == 1


def test_failed_job_can_be_resumed(make_client, monkeypatch):
    client, service = make_client()
    calls = {"n": 0}
    real = service.rag.process_document_complete

    async def flaky(path, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("embedding server went away")
        return await real(path, **kw)

    monkeypatch.setattr(service.rag, "process_document_complete", flaky)
    client.post("/api/ingest", files={"files": ("a.txt", DOC.encode(), "text/plain")})
    jobs = _wait(client, lambda js: js[0]["status"] == "failed")
    assert "went away" in jobs[0]["error"]
    assert client.post(f"/api/jobs/{jobs[0]['id']}/resume").status_code == 200
    jobs = _wait(client, lambda js: js[0]["status"] == "done")
    assert calls["n"] == 2


def test_interrupted_jobs_are_requeued_on_start(tmp_path):
    rag = make_rag(tmp_path / "rag")
    first = KnowledgeService(rag, auto_resume=True)
    src = tmp_path / "a.txt"
    src.write_text(DOC)
    job = first.add_file(src)
    first.jobs.update(job, status="running")  # the process died mid-ingestion

    second = KnowledgeService(rag, auto_resume=True)
    assert second.jobs.jobs[job.id].status == "running"
    import asyncio

    async def go():
        await second.start()
        for _ in range(100):
            if second.jobs.jobs[job.id].status in ("done", "failed"):
                break
            await asyncio.sleep(0.2)
        status = second.jobs.jobs[job.id].status
        await second.stop()
        return status

    assert asyncio.run(go()) == "done"


def test_upload_limit_and_validation(make_client):
    client, _ = make_client(max_upload_mb=0)
    r = client.post("/api/ingest", files={"files": ("a.txt", b"x", "text/plain")})
    assert r.status_code == 413
    client2, _ = make_client()
    assert client2.post("/api/search", json={"question": "  "}).status_code == 400
    assert (
        client2.post("/api/search", json={"question": "x", "mode": "bogus"}).status_code
        == 400
    )


def test_assets_cannot_escape_output_dir(make_client, tmp_path):
    client, service = make_client()
    secret = tmp_path / "secret.txt"
    secret.write_text("nope")
    assert client.get("/api/assets", params={"path": str(secret)}).status_code == 404
    assert (
        client.get("/api/assets", params={"path": "../../etc/passwd"}).status_code
        == 404
    )
    inside = Path(service.rag.config.parser_output_dir)
    inside.mkdir(parents=True, exist_ok=True)
    (inside / "fig.png").write_bytes(b"png")
    assert (
        client.get("/api/assets", params={"path": str(inside / "fig.png")}).content
        == b"png"
    )
    # a symlink out of the output dir is not followed
    (inside / "link.txt").symlink_to(secret)
    assert (
        client.get("/api/assets", params={"path": str(inside / "link.txt")}).status_code
        == 404
    )


def test_server_path_ingestion_is_off_by_default(make_client, tmp_path):
    client, _ = make_client()
    assert (
        client.post("/api/ingest/path", json={"path": str(tmp_path)}).status_code == 403
    )


def test_server_path_ingestion_is_restricted(make_client, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "x.txt").write_text(DOC)
    (allowed / "skip.bin").write_text("x")
    (tmp_path / "other.txt").write_text(DOC)

    client, _ = make_client(ingest_roots=[allowed])
    assert (
        client.post(
            "/api/ingest/path", json={"path": str(tmp_path / "other.txt")}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/ingest/path", json={"path": str(allowed / ".." / "other.txt")}
        ).status_code
        == 403
    )
    r = client.post("/api/ingest/path", json={"path": str(allowed)})
    assert r.status_code == 200 and len(r.json()["jobs"]) == 1


def test_token_protects_everything(make_client):
    client, _ = make_client(token="s3cret-é")  # non-ASCII token: compared as bytes
    assert client.get("/api/documents").status_code == 401
    assert client.get("/").status_code == 401
    assert (
        client.get(
            "/api/documents", headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )
    assert (
        client.get(
            "/login", params={"token_value": "bad"}, follow_redirects=False
        ).status_code
        == 401
    )
    r = client.get(
        "/login",
        params={"token_value": "s3cret-é", "next": "/ingest"},
        follow_redirects=False,
    )
    assert r.status_code == 307 and r.headers["location"] == "/ingest"
    assert client.get("/api/documents").status_code == 200  # cookie kept by the client
    assert client.get("/").status_code == 200


def test_pages_and_static_files(make_client):
    client, _ = make_client()
    assert "Rechercher" in client.get("/").text
    assert "Glissez" in client.get("/ingest").text
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/static/..%2f..%2fservice.py").status_code == 404
    assert client.get("/api/health").json()["ok"] is True


def test_delete_document(make_client):
    client, service = make_client()
    client.post("/api/ingest", files={"files": ("a.txt", DOC.encode(), "text/plain")})
    _wait(client, lambda js: js[0]["status"] == "done")
    doc = client.get("/api/documents").json()[0]
    assert client.delete(f"/api/documents/{doc['doc_id']}").status_code == 200
    assert client.get("/api/documents").json() == []
    assert client.get(f"/api/files/{doc['doc_id']}").status_code == 404


def test_helpers():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("a\x00b?.pdf").endswith(".pdf")
    long = "x " * 500 + "cable section " + "y " * 500
    excerpt = make_excerpt(long, "quelle cable section")
    assert "cable section" in excerpt and excerpt.startswith("…") and len(excerpt) < 700
    assert make_excerpt("short text", "q") == "short text"
