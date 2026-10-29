"""
Export / import of a RAGAnything knowledge base.

Two levels, for both the default file storages and a PostgreSQL server:

* **Snapshot** (``create_snapshot`` / ``restore_snapshot``): an exact copy of
  the storage, fast, to restore on the same kind of backend. File storages are
  archived as a directory tree; a PostgreSQL backend is dumped with
  ``pg_dump`` (whole database, custom format) next to the working directory,
  which still holds the parts PostgreSQL does not (graph file, parse cache).

* **Bundle** (``export_bundle`` / ``import_bundle``): a backend-independent
  zip (JSON lines + float32 ``.npy`` vectors) that moves a knowledge base
  between file storages and PostgreSQL, or between servers. It is written and
  read through LightRAG's storage interfaces, so what is exported (documents,
  chunks, graph, vectors, parse cache) can be imported into any backend.
  Vectors are written back without recomputing them when the embedding model
  is unchanged; ``reembed=True`` recomputes them with the target model.
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np

from lightrag.base import DocStatus
from lightrag.constants import GRAPH_FIELD_SEP

logger = logging.getLogger(__name__)

BUNDLE_FORMAT = "raganything-bundle"
SNAPSHOT_FORMAT = "raganything-snapshot"
FORMAT_VERSION = 1
KV_BATCH = 200
VECTOR_PART_ROWS = 2000

ProgressCallback = Callable[[str, int], None]

# Fields some vector backends keep next to the vector (NanoVectorDB drops them).
_VDB_PAYLOAD_KEYS = {"tokens", "chunk_order_index", "full_doc_id", "file_path"}


# --- helpers ----------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _dumps(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n").encode(
        "utf-8"
    )


def _is_postgres(storage_name: Optional[str]) -> bool:
    return bool(storage_name) and str(storage_name).startswith("PG")


def _lightrag_backends(rag: Any) -> Dict[str, str]:
    lr = rag.lightrag
    return {
        "kv": str(getattr(lr, "kv_storage", "")),
        "vector": str(getattr(lr, "vector_storage", "")),
        "graph": str(getattr(lr, "graph_storage", "")),
        "doc_status": str(getattr(lr, "doc_status_storage", "")),
    }


def _kv_storages(rag: Any, include_llm_cache: bool) -> Dict[str, Any]:
    lr = rag.lightrag
    storages = {
        "full_docs": lr.full_docs,
        "text_chunks": lr.text_chunks,
        "full_entities": lr.full_entities,
        "full_relations": lr.full_relations,
        "entity_chunks": lr.entity_chunks,
        "relation_chunks": lr.relation_chunks,
    }
    if include_llm_cache:
        storages["llm_response_cache"] = lr.llm_response_cache
    if getattr(rag, "parse_cache", None) is not None:
        storages["parse_cache"] = rag.parse_cache
    if getattr(rag, "multimodal_status_cache", None) is not None:
        storages["multimodal_status"] = rag.multimodal_status_cache
    return storages


def _vector_storages(rag: Any) -> Dict[str, Any]:
    lr = rag.lightrag
    return {
        "chunks": lr.chunks_vdb,
        "entities": lr.entities_vdb,
        "relationships": lr.relationships_vdb,
    }


async def _flush(rag: Any) -> None:
    lr = rag.lightrag
    storages = [
        *_kv_storages(rag, True).values(),
        *_vector_storages(rag).values(),
        lr.chunk_entity_relation_graph,
        lr.doc_status,
    ]
    for storage in storages:
        callback = getattr(storage, "index_done_callback", None)
        if callback is not None:
            await callback()


def _embedding_info(rag: Any) -> Dict[str, Any]:
    func = rag.lightrag.embedding_func
    return {
        "dim": getattr(func, "embedding_dim", None),
        "model": getattr(func, "model_name", None),
    }


# --- reading storages through whatever interface the backend offers -----------


async def kv_ids(storage: Any) -> List[str]:
    """All keys of a KV storage (JSON files or PostgreSQL)."""
    data = getattr(storage, "_data", None)
    if data is not None:
        return [str(k) for k in list(data.keys())]
    db = getattr(storage, "db", None)
    if db is not None:
        from lightrag.kg.postgres_impl import namespace_to_table_name

        table = namespace_to_table_name(storage.namespace)
        if table:
            rows = await db.query(
                f"SELECT id FROM {table} WHERE workspace=$1",
                [storage.workspace],
                multirows=True,
            )
            return [str(r["id"]) for r in rows or []]
    raise NotImplementedError(
        f"Cannot list the keys of {type(storage).__name__} "
        f"({getattr(storage, 'namespace', '?')}); use a snapshot instead"
    )


async def vector_ids(storage: Any) -> List[str]:
    """All ids of a vector storage (NanoVectorDB file or pgvector)."""
    path = getattr(storage, "_client_file_name", None)
    if path:
        await storage.index_done_callback()
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [str(d["__id__"]) for d in json.load(f).get("data", [])]
    db = getattr(storage, "db", None)
    table = getattr(storage, "table_name", None)
    if db is not None and table:
        rows = await db.query(
            f"SELECT id FROM {table} WHERE workspace=$1",
            [storage.workspace],
            multirows=True,
        )
        return [str(r["id"]) for r in rows or []]
    raise NotImplementedError(
        f"Cannot list the ids of {type(storage).__name__}; use a snapshot instead"
    )


def _batches(items: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _clean_kv_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in record.items() if k not in ("workspace",)}


# --- native snapshot -----------------------------------------------------------


def _pg_env() -> Dict[str, str]:
    env = dict(os.environ)
    env["PGPASSWORD"] = os.environ.get("POSTGRES_PASSWORD", "")
    return env


def _pg_args() -> List[str]:
    return [
        "-h",
        os.environ.get("POSTGRES_HOST", "localhost"),
        "-p",
        os.environ.get("POSTGRES_PORT", "5432"),
        "-U",
        os.environ.get("POSTGRES_USER", "postgres"),
        "-d",
        os.environ.get("POSTGRES_DATABASE", "postgres"),
    ]


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"{name} is required to snapshot a PostgreSQL backend "
            "(install the PostgreSQL client tools)"
        )
    return path


def _safe_members(tar: tarfile.TarFile, dest: Path) -> List[tarfile.TarInfo]:
    members = []
    dest = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(dest)):
            raise ValueError(f"Unsafe path in snapshot: {member.name}")
        if member.issym() or member.islnk() or member.isdev():
            raise ValueError(f"Unsupported member in snapshot: {member.name}")
        members.append(member)
    return members


async def create_snapshot(
    rag: Any,
    out_path: Union[str, Path],
    *,
    include_assets: bool = True,
) -> Dict[str, Any]:
    """Archive the knowledge base exactly as stored (``.tar.gz``)."""
    init = await rag._ensure_lightrag_initialized()
    if not init or not init.get("success"):
        raise RuntimeError(f"LightRAG initialization failed: {init}")
    await _flush(rag)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    backends = _lightrag_backends(rag)
    uses_pg = any(_is_postgres(name) for name in backends.values())
    working_dir = Path(rag.config.working_dir)
    output_dir = Path(rag.config.parser_output_dir)

    manifest = {
        "format": SNAPSHOT_FORMAT,
        "version": FORMAT_VERSION,
        "created_at": _now(),
        "backends": backends,
        "embedding": _embedding_info(rag),
        "postgres_dump": uses_pg,
        "workspace": getattr(rag.lightrag, "workspace", ""),
        "include_assets": include_assets,
    }
    with tempfile.TemporaryDirectory() as tmp:
        dump_path = Path(tmp) / "postgres.dump"
        if uses_pg:
            subprocess.run(
                [
                    _require_tool("pg_dump"),
                    "-Fc",
                    "--no-owner",
                    "-f",
                    str(dump_path),
                    *_pg_args(),
                ],
                check=True,
                env=_pg_env(),
                capture_output=True,
            )
        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        with tarfile.open(out_path, "w:gz") as tar:
            info = tarfile.TarInfo("manifest.json")
            info.size = len(manifest_bytes)
            info.mtime = int(datetime.now().timestamp())
            tar.addfile(info, io.BytesIO(manifest_bytes))
            if working_dir.exists():
                tar.add(working_dir, arcname="working_dir")
            if include_assets and output_dir.exists():
                tar.add(output_dir, arcname="output")
            if uses_pg:
                tar.add(dump_path, arcname="postgres.dump")
    logger.info("Snapshot written to %s", out_path)
    return manifest


def restore_snapshot(
    archive: Union[str, Path],
    working_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    *,
    restore_postgres: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Restore a snapshot. Call it *before* creating the RAGAnything instance."""
    archive = Path(archive)
    working_dir = Path(working_dir)
    if working_dir.exists() and any(working_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{working_dir} is not empty; pass overwrite=True to replace it"
        )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(archive, "r:*") as tar:
            members = _safe_members(tar, tmp_path)
            tar.extractall(tmp_path, members=members)  # noqa: S202 - validated above
        manifest = json.loads((tmp_path / "manifest.json").read_text("utf-8"))
        if manifest.get("format") != SNAPSHOT_FORMAT:
            raise ValueError(f"{archive} is not a RAGAnything snapshot")

        if working_dir.exists() and overwrite:
            shutil.rmtree(working_dir)
        if (tmp_path / "working_dir").exists():
            shutil.copytree(tmp_path / "working_dir", working_dir)
        else:
            working_dir.mkdir(parents=True, exist_ok=True)
        if output_dir is not None and (tmp_path / "output").exists():
            output_dir = Path(output_dir)
            shutil.copytree(tmp_path / "output", output_dir, dirs_exist_ok=True)

        dump = tmp_path / "postgres.dump"
        if dump.exists() and restore_postgres:
            subprocess.run(
                [
                    _require_tool("pg_restore"),
                    "--clean",
                    "--if-exists",
                    "--no-owner",
                    *_pg_args(),
                    str(dump),
                ],
                check=True,
                env=_pg_env(),
                capture_output=True,
            )
    return manifest


# --- portable bundle: export ---------------------------------------------------


def _zip_writer(zf: zipfile.ZipFile, name: str):
    return zf.open(name, "w", force_zip64=True)


async def export_bundle(
    rag: Any,
    out_path: Union[str, Path],
    *,
    include_llm_cache: bool = False,
    include_assets: bool = False,
    on_progress: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Write the knowledge base as a backend-independent zip bundle."""
    init = await rag._ensure_lightrag_initialized()
    if not init or not init.get("success"):
        raise RuntimeError(f"LightRAG initialization failed: {init}")
    await _flush(rag)
    lr = rag.lightrag

    def progress(stage: str, count: int) -> None:
        if on_progress is not None:
            on_progress(stage, count)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        # documents and their processing state
        records = await lr.doc_status.get_docs_by_statuses(list(DocStatus))
        with _zip_writer(zf, "docs.jsonl") as out:
            for doc_id, status in records.items():
                row = {
                    "content_summary": status.content_summary,
                    "content_length": status.content_length,
                    "file_path": status.file_path,
                    "status": getattr(status.status, "value", status.status),
                    "created_at": status.created_at,
                    "updated_at": status.updated_at,
                    "track_id": status.track_id,
                    "chunks_count": status.chunks_count,
                    "chunks_list": list(status.chunks_list or []),
                    "error_msg": status.error_msg,
                    "metadata": status.metadata or {},
                }
                if status.multimodal_processed is not None:
                    row["multimodal_processed"] = status.multimodal_processed
                out.write(_dumps({"id": doc_id, "data": row}))
        counts["docs"] = len(records)
        progress("docs", len(records))

        # key-value namespaces
        for name, storage in _kv_storages(rag, include_llm_cache).items():
            ids = await kv_ids(storage)
            written = 0
            with _zip_writer(zf, f"kv/{name}.jsonl") as out:
                for batch in _batches(ids, KV_BATCH):
                    fetched = await storage.get_by_ids(batch)
                    for key, record in zip(batch, fetched):
                        if record is None:
                            continue
                        out.write(_dumps({"id": key, "data": _clean_kv_record(record)}))
                        written += 1
            counts[f"kv/{name}"] = written
            progress(f"kv/{name}", written)

        # knowledge graph
        graph = lr.chunk_entity_relation_graph
        nodes = await graph.get_all_nodes()
        with _zip_writer(zf, "graph/nodes.jsonl") as out:
            for node in nodes:
                node = dict(node)
                node_id = node.pop("id", None)
                out.write(_dumps({"id": node_id, "data": node}))
        edges = await graph.get_all_edges()
        with _zip_writer(zf, "graph/edges.jsonl") as out:
            for edge in edges:
                edge = dict(edge)
                source, target = edge.pop("source", None), edge.pop("target", None)
                out.write(_dumps({"source": source, "target": target, "data": edge}))
        counts["graph/nodes"], counts["graph/edges"] = len(nodes), len(edges)
        progress("graph", len(nodes) + len(edges))

        # vectors, in parts so memory stays bounded
        for name, vdb in _vector_storages(rag).items():
            ids = await vector_ids(vdb)
            meta_keys = set(getattr(vdb, "meta_fields", set())) | {"content"}
            total = 0
            for part, batch in enumerate(_batches(ids, VECTOR_PART_ROWS), start=1):
                metas = await vdb.get_by_ids(batch)
                vectors = await vdb.get_vectors_by_ids(batch)
                rows, matrix = [], []
                for key, meta in zip(batch, metas):
                    vector = vectors.get(key)
                    if meta is None or vector is None:
                        continue
                    payload = {
                        k: v
                        for k, v in meta.items()
                        if k in meta_keys or k in _VDB_PAYLOAD_KEYS
                    }
                    # pgvector names the relation's endpoints source_id/target_id
                    if name == "relationships" and "target_id" in meta:
                        payload["src_id"] = meta["source_id"]
                        payload["tgt_id"] = meta["target_id"]
                        payload.pop("source_id", None)
                    # ...and keeps the source chunks as an array column
                    if "source_id" not in payload and meta.get("chunk_ids") is not None:
                        chunk_ids = meta["chunk_ids"]
                        if isinstance(chunk_ids, str):
                            chunk_ids = json.loads(chunk_ids)
                        payload["source_id"] = GRAPH_FIELD_SEP.join(chunk_ids)
                    rows.append({"id": key, "data": payload})
                    matrix.append(vector)
                if not rows:
                    continue
                base = f"vectors/{name}/part-{part:05d}"
                with _zip_writer(zf, base + ".jsonl") as out:
                    for row in rows:
                        out.write(_dumps(row))
                buf = io.BytesIO()
                np.save(buf, np.asarray(matrix, dtype=np.float32), allow_pickle=False)
                zf.writestr(base + ".npy", buf.getvalue())
                total += len(rows)
            counts[f"vectors/{name}"] = total
            progress(f"vectors/{name}", total)

        if include_assets:
            out_dir = Path(rag.config.parser_output_dir)
            if out_dir.exists():
                n = 0
                for file in out_dir.rglob("*"):
                    if file.is_file():
                        zf.write(
                            file,
                            "assets/output/" + file.relative_to(out_dir).as_posix(),
                        )
                        n += 1
                counts["assets"] = n
                progress("assets", n)

        manifest = {
            "format": BUNDLE_FORMAT,
            "version": FORMAT_VERSION,
            "created_at": _now(),
            "backends": _lightrag_backends(rag),
            "embedding": _embedding_info(rag),
            "workspace": getattr(lr, "workspace", ""),
            "counts": counts,
            "include_llm_cache": include_llm_cache,
            "include_assets": include_assets,
            "source_dirs": {
                "output_dir": str(Path(rag.config.parser_output_dir).resolve()),
                "working_dir": str(Path(rag.config.working_dir).resolve()),
            },
        }
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
    logger.info("Bundle written to %s (%s)", out_path, counts)
    return manifest


def read_manifest(path: Union[str, Path]) -> Dict[str, Any]:
    """Manifest of a bundle (.zip) or snapshot (.tar.gz), without importing."""
    path = Path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            return json.loads(zf.read("manifest.json"))
    with tarfile.open(path, "r:*") as tar:
        member = tar.extractfile("manifest.json")
        if member is None:
            raise ValueError(f"{path} has no manifest")
        return json.loads(member.read())


# --- portable bundle: import ---------------------------------------------------


class _ReplayEmbedding:
    """Embedding function that answers from stored vectors, else the real one."""

    def __init__(self, original: Any) -> None:
        self._original = original
        self.table: Dict[str, np.ndarray] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    async def __call__(self, texts: List[str], *args: Any, **kwargs: Any) -> np.ndarray:
        result: List[Optional[np.ndarray]] = [self.table.get(t) for t in texts]
        missing = [i for i, v in enumerate(result) if v is None]
        if missing:
            fresh = await self._original([texts[i] for i in missing], *args, **kwargs)
            for i, vec in zip(missing, np.asarray(fresh)):
                result[i] = vec
        return np.asarray(result, dtype=np.float32)


def _check_compatibility(manifest: Dict[str, Any], rag: Any, reembed: bool) -> bool:
    """Whether stored vectors can be reused; raises on an incompatible target."""
    if manifest.get("format") != BUNDLE_FORMAT:
        raise ValueError("Not a RAGAnything bundle")
    if manifest.get("version", 0) > FORMAT_VERSION:
        raise ValueError(
            f"Bundle version {manifest.get('version')} is newer than supported "
            f"({FORMAT_VERSION})"
        )
    source, target = manifest.get("embedding", {}), _embedding_info(rag)
    same_dim = source.get("dim") == target.get("dim")
    same_model = (
        not source.get("model")
        or not target.get("model")
        or source.get("model") == target.get("model")
    )
    if reembed:
        return False
    if not same_dim:
        raise ValueError(
            f"Embedding dimension differs (bundle {source.get('dim')}, target "
            f"{target.get('dim')}); pass reembed=True to recompute the vectors"
        )
    if not same_model:
        raise ValueError(
            f"Embedding model differs (bundle {source.get('model')!r}, target "
            f"{target.get('model')!r}); pass reembed=True to recompute the vectors"
        )
    return True


def _read_jsonl(zf: zipfile.ZipFile, name: str, rewrite: Callable[[str], str]):
    if name not in zf.namelist():
        return
    with zf.open(name) as f:
        for line in io.TextIOWrapper(f, encoding="utf-8"):
            line = line.strip()
            if line:
                yield json.loads(rewrite(line))


async def import_bundle(
    rag: Any,
    path: Union[str, Path],
    *,
    mode: str = "merge",
    confirm_replace: bool = False,
    reembed: bool = False,
    assets_dir: Optional[Union[str, Path]] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Load a bundle into the current backend.

    ``mode="merge"`` upserts into what is already there; ``mode="replace"``
    first empties every storage and needs ``confirm_replace=True``.
    """
    if mode not in ("merge", "replace"):
        raise ValueError("mode must be 'merge' or 'replace'")
    if mode == "replace" and not confirm_replace:
        raise ValueError(
            "mode='replace' deletes the current data: pass confirm_replace=True"
        )
    init = await rag._ensure_lightrag_initialized()
    if not init or not init.get("success"):
        raise RuntimeError(f"LightRAG initialization failed: {init}")
    lr = rag.lightrag

    def progress(stage: str, count: int) -> None:
        if on_progress is not None:
            on_progress(stage, count)

    path = Path(path)
    imported: Dict[str, int] = {}
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        reuse_vectors = _check_compatibility(manifest, rag, reembed)

        # rewrite absolute asset paths from the source machine to this one
        source_out = manifest.get("source_dirs", {}).get("output_dir")
        target_out = str(Path(assets_dir or rag.config.parser_output_dir).resolve())

        def rewrite(line: str) -> str:
            if source_out and source_out != target_out:
                return line.replace(
                    json.dumps(source_out)[1:-1], json.dumps(target_out)[1:-1]
                )
            return line

        if mode == "replace":
            for storage in [
                *_kv_storages(rag, True).values(),
                *_vector_storages(rag).values(),
                lr.chunk_entity_relation_graph,
                lr.doc_status,
            ]:
                await storage.drop()

        if manifest.get("include_assets"):
            dest = Path(target_out)
            for name in zf.namelist():
                if name.startswith("assets/output/") and not name.endswith("/"):
                    rel = Path(name[len("assets/output/") :])
                    target = (dest / rel).resolve()
                    if not str(target).startswith(str(dest.resolve())):
                        raise ValueError(f"Unsafe asset path in bundle: {name}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(name) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)

        # key-value namespaces
        chunk_records: Dict[str, Dict[str, Any]] = {}
        for name, storage in _kv_storages(rag, True).items():
            member = f"kv/{name}.jsonl"
            count = 0
            batch: Dict[str, Any] = {}
            for row in _read_jsonl(zf, member, rewrite):
                batch[row["id"]] = row["data"]
                if name == "text_chunks":
                    chunk_records[row["id"]] = row["data"]
                if len(batch) >= KV_BATCH:
                    await storage.upsert(batch)
                    count += len(batch)
                    batch = {}
            if batch:
                await storage.upsert(batch)
                count += len(batch)
            if member in zf.namelist():
                imported[f"kv/{name}"] = count
                progress(f"kv/{name}", count)

        # graph
        graph = lr.chunk_entity_relation_graph
        n_nodes = n_edges = 0
        for row in _read_jsonl(zf, "graph/nodes.jsonl", rewrite):
            await graph.upsert_node(row["id"], row["data"])
            n_nodes += 1
        for row in _read_jsonl(zf, "graph/edges.jsonl", rewrite):
            await graph.upsert_edge(row["source"], row["target"], row["data"])
            n_edges += 1
        imported["graph/nodes"], imported["graph/edges"] = n_nodes, n_edges
        progress("graph", n_nodes + n_edges)

        # vectors
        for name, vdb in _vector_storages(rag).items():
            original = vdb.embedding_func
            replay = _ReplayEmbedding(original)
            vdb.embedding_func = replay
            total = 0
            try:
                parts = sorted(
                    n[: -len(".jsonl")]
                    for n in zf.namelist()
                    if n.startswith(f"vectors/{name}/") and n.endswith(".jsonl")
                )
                for base in parts:
                    rows = list(_read_jsonl(zf, base + ".jsonl", rewrite))
                    if reuse_vectors:
                        matrix = np.load(io.BytesIO(zf.read(base + ".npy")))
                        replay.table = {
                            row["data"]["content"]: matrix[i]
                            for i, row in enumerate(rows)
                            if "content" in row["data"]
                        }
                    else:
                        replay.table = {}
                    if name == "chunks":
                        # NanoVectorDB does not store these; pgvector requires them
                        for row in rows:
                            record = chunk_records.get(row["id"], {})
                            for field in (
                                "tokens",
                                "chunk_order_index",
                                "full_doc_id",
                                "file_path",
                            ):
                                if row["data"].get(field) is None and field in record:
                                    row["data"][field] = record[field]
                    await vdb.upsert({row["id"]: row["data"] for row in rows})
                    total += len(rows)
            finally:
                vdb.embedding_func = original
            imported[f"vectors/{name}"] = total
            progress(f"vectors/{name}", total)

        # documents last: an interrupted import leaves no half-visible document
        docs = {
            row["id"]: row["data"] for row in _read_jsonl(zf, "docs.jsonl", rewrite)
        }
        if docs:
            await lr.doc_status.upsert(docs)
        imported["docs"] = len(docs)
        progress("docs", len(docs))

    await _flush(rag)
    logger.info("Bundle imported from %s (%s)", path, imported)
    return {"manifest": manifest, "imported": imported, "reused_vectors": reuse_vectors}


# --- command line (inspection only; full commands are in raganything.cli) ------


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect a RAGAnything bundle (.zip) or snapshot (.tar.gz). "
        "Export/import need a configured instance: use `python -m raganything.cli`."
    )
    parser.add_argument("archive")
    args = parser.parse_args(argv)
    print(json.dumps(read_manifest(args.archive), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
