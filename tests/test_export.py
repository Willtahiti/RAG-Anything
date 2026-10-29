"""Snapshot and portable-bundle export/import, on file storage and PostgreSQL."""

import json
import os
import shutil
import subprocess
import uuid
import zipfile
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio

from lightrag.base import DocStatus
from raganything.export import (
    create_snapshot,
    export_bundle,
    import_bundle,
    kv_ids,
    read_manifest,
    restore_snapshot,
    vector_ids,
)
from rag_harness import fake_vector, make_rag, postgres_env

DOC_A = "Alice works with Bob at ACME.\n\nA second paragraph about ACME.\fPage two mentions Alice and ACME again."
DOC_B = "Bob visited ACME headquarters. Alice was there too."


def _psql(sql: str, database: str = "postgres") -> subprocess.CompletedProcess:
    env = postgres_env()
    return subprocess.run(
        [
            "psql",
            "-q",
            "-h",
            env["POSTGRES_HOST"],
            "-p",
            env["POSTGRES_PORT"],
            "-U",
            env["POSTGRES_USER"],
            "-d",
            database,
            "-c",
            sql,
        ],
        env={**os.environ, "PGPASSWORD": env["POSTGRES_PASSWORD"]},
        capture_output=True,
        text=True,
    )


def _pg_available() -> bool:
    try:
        import asyncpg  # noqa: F401

        return shutil.which("psql") is not None and _psql("SELECT 1").returncode == 0
    except Exception:
        return False


PG_OK = _pg_available()
needs_pg = pytest.mark.skipif(not PG_OK, reason="no PostgreSQL+pgvector server")


@pytest.fixture(scope="session")
def pg_database():
    """One throwaway database for the whole session (LightRAG shares a single
    PostgreSQL connection pool per process); tests isolate by workspace."""
    if not PG_OK:
        yield ""
        return
    name = f"rag_test_{uuid.uuid4().hex[:8]}"
    assert _psql(f'CREATE DATABASE "{name}"').returncode == 0
    assert _psql("CREATE EXTENSION IF NOT EXISTS vector", name).returncode == 0
    yield name
    _psql(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest_asyncio.fixture
async def rags():
    """Instances created by a test; their storages are finalized on teardown so
    the PostgreSQL pool does not outlive the test's event loop."""
    created = []
    yield created
    for rag in created:
        await rag.finalize_storages()


BACKENDS = ["file", pytest.param("postgres", marks=needs_pg)]


async def _populate(rag, tmp_path):
    for name, text in (("a.txt", DOC_A), ("b.txt", DOC_B)):
        doc = tmp_path / name
        doc.write_text(text, encoding="utf-8")
        await rag.process_document_complete(str(doc))


async def _state(rag):
    lr = rag.lightrag
    docs = await lr.doc_status.get_docs_by_statuses(list(DocStatus))
    chunks = {}
    for doc_id, status in docs.items():
        for chunk_id, rec in zip(
            status.chunks_list, await lr.text_chunks.get_by_ids(status.chunks_list)
        ):
            chunks[chunk_id] = rec["content"]
    nodes = {
        n["id"]: n.get("description", "")
        for n in await lr.chunk_entity_relation_graph.get_all_nodes()
    }
    edges = {
        frozenset((e["source"], e["target"]))
        for e in await lr.chunk_entity_relation_graph.get_all_edges()
    }
    vectors = {}
    for name, vdb in (
        ("chunks", lr.chunks_vdb),
        ("entities", lr.entities_vdb),
        ("relationships", lr.relationships_vdb),
    ):
        ids = await vector_ids(vdb)
        vecs = await vdb.get_vectors_by_ids(ids)
        vectors[name] = {k: np.asarray(v) for k, v in vecs.items()}
    return {
        "docs": {k: (v.file_path, v.status, v.chunks_count) for k, v in docs.items()},
        "chunks": chunks,
        "nodes": nodes,
        "edges": edges,
        "vectors": vectors,
        "parse_cache_keys": sorted(await kv_ids(rag.parse_cache)),
    }


def _assert_same(a, b, vector_tol=1e-3):
    assert a["docs"] == b["docs"]
    assert a["chunks"] == b["chunks"]
    assert a["nodes"] == b["nodes"]
    assert a["edges"] == b["edges"]
    assert a["parse_cache_keys"] == b["parse_cache_keys"]
    for name in a["vectors"]:
        assert set(a["vectors"][name]) == set(b["vectors"][name]), name
        for key, vec in a["vectors"][name].items():
            assert np.allclose(vec, b["vectors"][name][key], atol=vector_tol), (
                name,
                key,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("source", BACKENDS)
@pytest.mark.parametrize("target", BACKENDS)
async def test_bundle_round_trip_between_backends(
    tmp_path, monkeypatch, pg_database, rags, source, target
):
    src = make_rag(
        tmp_path / "src", source, monkeypatch=monkeypatch, database=pg_database
    )
    rags.append(src)
    await _populate(src, tmp_path / "src")
    before = await _state(src)
    assert len(before["docs"]) == 2 and before["nodes"] and before["edges"]
    assert before["parse_cache_keys"]

    bundle = tmp_path / "kb.zip"
    manifest = await export_bundle(src, bundle)
    assert manifest["counts"]["docs"] == 2
    assert manifest["embedding"] == {"dim": 16, "model": "fake-embed"}
    assert read_manifest(bundle)["format"] == "raganything-bundle"

    dst = make_rag(
        tmp_path / "dst", target, monkeypatch=monkeypatch, database=pg_database
    )
    rags.append(dst)
    result = await import_bundle(dst, bundle)
    assert result["reused_vectors"] is True
    assert result["imported"]["docs"] == 2
    after = await _state(dst)
    _assert_same(before, after)

    # the imported knowledge base answers queries like the source one
    q = "ACME"
    from lightrag.base import QueryParam

    hits_src = await src.lightrag.chunks_vdb.query(q, top_k=5)
    hits_dst = await dst.lightrag.chunks_vdb.query(q, top_k=5)
    assert [h["id"] for h in hits_src] == [h["id"] for h in hits_dst]
    ctx = await dst.lightrag.aquery(
        q, param=QueryParam(mode="naive", only_need_context=True)
    )
    assert "ACME" in str(ctx)


# --- bundle options ---------------------------------------------------------


@pytest.mark.asyncio
async def test_vectors_are_reused_unless_reembed(tmp_path, rags):
    src = make_rag(tmp_path / "src")
    rags.append(src)
    await _populate(src, tmp_path / "src")
    bundle = tmp_path / "kb.zip"
    await export_bundle(src, bundle)

    calls = []

    async def counting_embed(texts, **kw):
        calls.append(len(texts))
        return np.array([fake_vector("other" + t) for t in texts], dtype=np.float32)

    reuse = make_rag(tmp_path / "reuse", embed_func=counting_embed)
    rags.append(reuse)
    result = await import_bundle(reuse, bundle)
    assert result["reused_vectors"] is True and calls == []
    ids = await vector_ids(reuse.lightrag.chunks_vdb)
    src_ids = await vector_ids(src.lightrag.chunks_vdb)
    assert sorted(ids) == sorted(src_ids)

    fresh = make_rag(tmp_path / "fresh", embed_func=counting_embed)
    rags.append(fresh)
    result = await import_bundle(fresh, bundle, reembed=True)
    assert result["reused_vectors"] is False and sum(calls) > 0
    vec_src = (await src.lightrag.chunks_vdb.get_vectors_by_ids(src_ids[:1]))[
        src_ids[0]
    ]
    vec_new = (await fresh.lightrag.chunks_vdb.get_vectors_by_ids(src_ids[:1]))[
        src_ids[0]
    ]
    assert not np.allclose(vec_src, vec_new, atol=1e-2)


@pytest.mark.asyncio
async def test_incompatible_embedding_model_is_refused(tmp_path, rags):
    src = make_rag(tmp_path / "src")
    rags.append(src)
    await _populate(src, tmp_path / "src")
    bundle = tmp_path / "kb.zip"
    await export_bundle(src, bundle)

    other = make_rag(tmp_path / "other", embedding_model="another-model")
    rags.append(other)
    with pytest.raises(ValueError, match="reembed=True"):
        await import_bundle(other, bundle)
    assert await vector_ids(other.lightrag.chunks_vdb) == []


@pytest.mark.asyncio
async def test_replace_mode_requires_confirmation_and_empties_target(tmp_path, rags):
    a = make_rag(tmp_path / "a")
    rags.append(a)
    await _populate(a, tmp_path / "a")
    bundle = tmp_path / "kb.zip"
    await export_bundle(a, bundle)

    b = make_rag(tmp_path / "b")
    rags.append(b)
    extra = tmp_path / "b" / "extra.txt"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text("Completely unrelated document.", encoding="utf-8")
    await b.process_document_complete(str(extra))
    assert len(await b.lightrag.doc_status.get_docs_by_statuses(list(DocStatus))) == 1

    with pytest.raises(ValueError, match="confirm_replace"):
        await import_bundle(b, bundle, mode="replace")
    await import_bundle(b, bundle, mode="replace", confirm_replace=True)
    docs = await b.lightrag.doc_status.get_docs_by_statuses(list(DocStatus))
    assert sorted(d.file_path for d in docs.values()) == ["a.txt", "b.txt"]

    # merge keeps what is there
    await b.process_document_complete(str(extra))
    await import_bundle(b, bundle)
    docs = await b.lightrag.doc_status.get_docs_by_statuses(list(DocStatus))
    assert len(docs) == 3


@pytest.mark.asyncio
async def test_assets_are_exported_and_paths_rewritten(tmp_path, rags):
    src = make_rag(tmp_path / "src")
    rags.append(src)
    await _populate(src, tmp_path / "src")
    out = Path(src.config.parser_output_dir)
    (out / "doc").mkdir(parents=True)
    (out / "doc" / "fig.png").write_bytes(b"png-bytes")
    await src.parse_cache.upsert(
        {"k": {"img_path": str((out / "doc" / "fig.png").resolve())}}
    )
    bundle = tmp_path / "kb.zip"
    manifest = await export_bundle(src, bundle, include_assets=True)
    assert manifest["counts"]["assets"] == 1

    dst = make_rag(tmp_path / "dst")
    rags.append(dst)
    await import_bundle(dst, bundle)
    new_path = Path((await dst.parse_cache.get_by_id("k"))["img_path"])
    assert (
        new_path == (Path(dst.config.parser_output_dir) / "doc" / "fig.png").resolve()
    )
    assert new_path.read_bytes() == b"png-bytes"


@pytest.mark.asyncio
async def test_malicious_asset_path_is_rejected(tmp_path, rags):
    src = make_rag(tmp_path / "src")
    rags.append(src)
    await _populate(src, tmp_path / "src")
    bundle = tmp_path / "kb.zip"
    await export_bundle(src, bundle)
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(bundle) as zin, zipfile.ZipFile(evil, "w") as zout:
        for name in zin.namelist():
            data = zin.read(name)
            if name == "manifest.json":
                m = json.loads(data)
                m["include_assets"] = True
                data = json.dumps(m).encode()
            zout.writestr(name, data)
        zout.writestr("assets/output/../../escape.txt", b"x")
    dst = make_rag(tmp_path / "dst")
    rags.append(dst)
    with pytest.raises(ValueError, match="Unsafe asset path"):
        await import_bundle(dst, evil)
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.asyncio
async def test_not_a_bundle_is_rejected(tmp_path, rags):
    junk = tmp_path / "x.zip"
    with zipfile.ZipFile(junk, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"format": "other"}))
    dst = make_rag(tmp_path / "dst")
    rags.append(dst)
    with pytest.raises(ValueError, match="Not a RAGAnything bundle"):
        await import_bundle(dst, junk)


# --- native snapshots -------------------------------------------------------


@pytest.mark.asyncio
async def test_file_snapshot_round_trip(tmp_path, rags):
    src = make_rag(tmp_path / "src")
    rags.append(src)
    await _populate(src, tmp_path / "src")
    (Path(src.config.parser_output_dir) / "fig.png").parent.mkdir(
        parents=True, exist_ok=True
    )
    (Path(src.config.parser_output_dir) / "fig.png").write_bytes(b"png")

    archive = tmp_path / "snap.tar.gz"
    manifest = await create_snapshot(src, archive)
    assert (
        manifest["postgres_dump"] is False
        and read_manifest(archive)["format"] == "raganything-snapshot"
    )

    work, out = tmp_path / "restored" / "work", tmp_path / "restored" / "out"
    restore_snapshot(archive, work, out)
    ws = src.lightrag.workspace
    for name in (
        "kv_store_text_chunks.json",
        "kv_store_full_docs.json",
        "vdb_chunks.json",
    ):
        assert (work / ws / name).read_bytes() == (
            Path(src.config.working_dir) / ws / name
        ).read_bytes()
    assert (out / "fig.png").read_bytes() == b"png"

    with pytest.raises(FileExistsError):
        restore_snapshot(archive, work)
    restore_snapshot(archive, work, overwrite=True)


def test_snapshot_rejects_path_traversal(tmp_path):
    import tarfile

    evil = tmp_path / "evil.tar.gz"
    manifest = json.dumps({"format": "raganything-snapshot", "version": 1}).encode()
    with tarfile.open(evil, "w:gz") as tar:
        import io

        info = tarfile.TarInfo("manifest.json")
        info.size = len(manifest)
        tar.addfile(info, io.BytesIO(manifest))
        bad = tarfile.TarInfo("../escape.txt")
        bad.size = 1
        tar.addfile(bad, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe path"):
        restore_snapshot(evil, tmp_path / "w")
    assert not (tmp_path / "escape.txt").exists()


@needs_pg
@pytest.mark.skipif(shutil.which("pg_dump") is None, reason="pg_dump missing")
@pytest.mark.asyncio
async def test_postgres_snapshot_dumps_and_restores(
    tmp_path, monkeypatch, pg_database, rags
):
    src = make_rag(
        tmp_path / "src", "postgres", monkeypatch=monkeypatch, database=pg_database
    )
    rags.append(src)
    await _populate(src, tmp_path / "src")
    archive = tmp_path / "snap.tar.gz"
    manifest = await create_snapshot(src, archive)
    assert manifest["postgres_dump"] is True
    with __import__("tarfile").open(archive) as tar:
        assert "postgres.dump" in tar.getnames()

    target = f"{pg_database}_restore"
    assert _psql(f'CREATE DATABASE "{target}"').returncode == 0
    try:
        assert _psql("CREATE EXTENSION IF NOT EXISTS vector", target).returncode == 0
        monkeypatch.setenv("POSTGRES_DATABASE", target)
        restore_snapshot(archive, tmp_path / "restored")
        counts = lambda db: subprocess.run(  # noqa: E731
            [
                "psql",
                "-tA",
                "-h",
                "localhost",
                "-U",
                "rag",
                "-d",
                db,
                "-c",
                "SELECT count(*) FROM lightrag_doc_chunks",
            ],
            env={**os.environ, "PGPASSWORD": "rag"},
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert counts(target) == counts(pg_database) != "0"
    finally:
        _psql(f'DROP DATABASE IF EXISTS "{target}" WITH (FORCE)')
