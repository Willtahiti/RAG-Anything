"""Ingestion jobs and cited search on top of a RAGAnything instance.

Used by the web server and the command line, so both behave the same.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from lightrag import QueryParam
from lightrag.base import DocStatus

logger = logging.getLogger(__name__)

CITATION_INSTRUCTIONS = (
    "Answer in the language of the question. Support every statement with the "
    "sources given in the context, citing them inline with their reference "
    "numbers in square brackets, e.g. [1] or [2][3]. Do not add a reference "
    "list at the end. If the context does not contain the answer, say so."
)

_IMAGE_PATH_RE = re.compile(r"^\s*(?:Image|Table)\s*Path:\s*(.+?)\s*$", re.MULTILINE)
_SAFE_NAME = re.compile(r"[^\w.\-() ]+", re.UNICODE)


def safe_filename(name: str) -> str:
    base = _SAFE_NAME.sub("_", Path(name.replace("\\", "/")).name).strip(" .")
    return (base or "upload")[:150]


def file_doc_id(path: Path) -> str:
    """Stable id from the file's bytes: the same file resumes the same document."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return "doc-" + digest.hexdigest()[:32]


@dataclass
class Job:
    id: str
    file: str
    doc_id: str
    status: str = "queued"  # queued | running | done | failed | cancelled | interrupted
    stage: str = ""
    done_pages: int = 0
    total_pages: int = 0
    error: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class JobStore:
    """Jobs persisted as JSON so an interrupted ingestion can be resumed."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.jobs: Dict[str, Job] = {}
        if path.exists():
            try:
                for raw in json.loads(path.read_text("utf-8")):
                    self.jobs[raw["id"]] = Job(**raw)
            except (OSError, ValueError, TypeError):
                logger.warning("Could not read job store %s", path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps([asdict(j) for j in self.jobs.values()], ensure_ascii=False),
            "utf-8",
        )
        os.replace(tmp, self.path)

    def update(self, job: Job, **changes: Any) -> None:
        for key, value in changes.items():
            setattr(job, key, value)
        job.updated_at = time.time()
        self.save()


class KnowledgeService:
    def __init__(
        self,
        rag: Any,
        *,
        data_dir: Optional[Path] = None,
        auto_resume: bool = True,
        process_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.rag = rag
        self.data_dir = Path(data_dir or rag.config.working_dir) / "app"
        self.sources_dir = self.data_dir / "sources"
        self.sources_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.data_dir / "sources_index.json"
        self.index: Dict[str, Dict[str, str]] = {}
        if self.index_path.exists():
            try:
                self.index = json.loads(self.index_path.read_text("utf-8"))
            except (OSError, ValueError):
                self.index = {}
        self.jobs = JobStore(self.data_dir / "jobs.json")
        self.auto_resume = auto_resume
        self.process_kwargs = process_kwargs or {}
        self._queue: "asyncio.Queue[str]" = asyncio.Queue()
        self._worker: Optional[asyncio.Task] = None
        self._cancel: set = set()
        self.output_root = Path(rag.config.parser_output_dir).resolve()

    # --- ingestion ------------------------------------------------------

    async def start(self) -> None:
        """Start the worker; requeue jobs a previous run left unfinished."""
        for job in self.jobs.jobs.values():
            if job.status in ("running", "queued"):
                self.jobs.update(job, status="interrupted")
        if self.auto_resume:
            for job in list(self.jobs.jobs.values()):
                if job.status == "interrupted" and self._stored_file(job):
                    self.jobs.update(job, status="queued", error="")
                    self._queue.put_nowait(job.id)
        self._worker = asyncio.create_task(self._run_worker())

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):
                pass

    def _stored_file(self, job: Job) -> Optional[Path]:
        entry = self.index.get(job.doc_id)
        if entry and Path(entry["path"]).is_file():
            return Path(entry["path"])
        return None

    def add_file(self, source: Path, original_name: Optional[str] = None) -> Job:
        """Keep a copy of ``source`` as the document's original and queue it."""
        doc_id = file_doc_id(source)
        name = safe_filename(original_name or source.name)
        target = self.sources_dir / doc_id / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        self.index[doc_id] = {"path": str(target), "name": name}
        self._save_index()
        job = Job(id=uuid.uuid4().hex[:12], file=name, doc_id=doc_id)
        self.jobs.jobs[job.id] = job
        self.jobs.save()
        self._queue.put_nowait(job.id)
        return job

    def _save_index(self) -> None:
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.index, ensure_ascii=False), "utf-8")
        os.replace(tmp, self.index_path)

    def retry(self, job_id: str) -> Job:
        job = self.jobs.jobs[job_id]
        if job.status in ("running", "queued"):
            return job
        self.jobs.update(job, status="queued", error="", stage="")
        self._cancel.discard(job_id)
        self._queue.put_nowait(job_id)
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.jobs.jobs[job_id]
        if job.status == "queued":
            self.jobs.update(job, status="cancelled")
        elif job.status == "running":
            self._cancel.add(job_id)
        return job

    async def _run_worker(self) -> None:
        while True:
            job_id = await self._queue.get()
            job = self.jobs.jobs.get(job_id)
            if job is None or job.status != "queued":
                continue
            await self._run_job(job)

    async def _run_job(self, job: Job) -> None:
        path = self._stored_file(job)
        if path is None:
            self.jobs.update(job, status="failed", error="original file is missing")
            return
        loop = asyncio.get_running_loop()

        def on_window(done: int, total: int, _range: Any) -> None:
            loop.call_soon_threadsafe(
                self.jobs.update, job, done_pages=done, total_pages=total
            )

        self.jobs.update(job, status="running", stage="parsing + indexing", error="")
        try:
            if await self._already_indexed(job.doc_id):
                # LightRAG would record a failed "duplicate" entry for it
                self.jobs.update(job, status="done", stage="already indexed")
                return
            extra = (
                {"window_progress": on_window} if path.suffix.lower() == ".pdf" else {}
            )
            task = asyncio.create_task(
                self.rag.process_document_complete(
                    str(path),
                    doc_id=job.doc_id,
                    **extra,
                    **self.process_kwargs,
                )
            )
            try:
                while not task.done():
                    await asyncio.wait({task}, timeout=1.0)
                    if job.id in self._cancel:
                        task.cancel()
            except asyncio.CancelledError:
                # the worker itself is being stopped: stop the ingestion too
                task.cancel()
                self.jobs.update(job, status="interrupted", stage="")
                raise
            try:
                await task
            except asyncio.CancelledError:
                self._cancel.discard(job.id)
                self.jobs.update(job, status="cancelled", stage="")
                return
            self.jobs.update(job, status="done", stage="")
        except Exception as exc:
            logger.exception("Ingestion of %s failed", job.file)
            self.jobs.update(job, status="failed", error=str(exc)[:500], stage="")

    async def _already_indexed(self, doc_id: str) -> bool:
        await self.rag._ensure_lightrag_initialized()
        record = await self.rag.lightrag.doc_status.get_by_id(doc_id)
        if not record or record.get("status") != DocStatus.PROCESSED:
            return False
        return await self.rag._get_multimodal_processed_flag(doc_id, record)

    def list_jobs(self) -> List[Dict[str, Any]]:
        return [
            asdict(j)
            for j in sorted(
                self.jobs.jobs.values(), key=lambda j: j.created_at, reverse=True
            )
        ]

    # --- documents --------------------------------------------------------

    async def list_documents(self) -> List[Dict[str, Any]]:
        await self.rag._ensure_lightrag_initialized()
        records = await self.rag.lightrag.doc_status.get_docs_by_statuses(
            list(DocStatus)
        )
        docs = []
        for doc_id, status in records.items():
            docs.append(
                {
                    "doc_id": doc_id,
                    "file": status.file_path,
                    "status": getattr(status.status, "value", str(status.status)),
                    "chunks": status.chunks_count,
                    "updated_at": status.updated_at,
                    "error": status.error_msg,
                    "has_original": doc_id in self.index,
                }
            )
        return sorted(docs, key=lambda d: d["updated_at"] or "", reverse=True)

    async def delete_document(self, doc_id: str) -> None:
        await self.rag._ensure_lightrag_initialized()
        await self.rag.lightrag.adelete_by_doc_id(doc_id)
        entry = self.index.pop(doc_id, None)
        if entry:
            shutil.rmtree(Path(entry["path"]).parent, ignore_errors=True)
            self._save_index()

    def original_file(self, doc_id: str) -> Optional[Path]:
        entry = self.index.get(doc_id)
        if entry:
            path = Path(entry["path"]).resolve()
            if path.is_file() and str(path).startswith(str(self.sources_dir.resolve())):
                return path
        return None

    async def document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        status = await self.rag.lightrag.doc_status.get_by_id(doc_id)
        if not status:
            return []
        ids = list(status.get("chunks_list") or [])
        records = await self.rag.lightrag.text_chunks.get_by_ids(ids)
        return [
            {"chunk_id": cid, **(rec or {})} for cid, rec in zip(ids, records) if rec
        ]

    def asset_path(self, raw: str) -> Optional[Path]:
        """An extracted image/tile, only if it lies under the output directory."""
        try:
            path = Path(raw).resolve()
        except (OSError, ValueError):
            return None
        if path.is_file() and str(path).startswith(str(self.output_root) + os.sep):
            return path
        return None

    # --- search -----------------------------------------------------------

    async def search(
        self,
        question: str,
        *,
        mode: str = "mix",
        top_k: int = 40,
        chunk_top_k: int = 20,
    ) -> Dict[str, Any]:
        await self.rag._ensure_lightrag_initialized()
        param = QueryParam(
            mode=mode,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            include_references=True,
            user_prompt=CITATION_INSTRUCTIONS,
        )
        result = await self.rag.lightrag.aquery_llm(question, param)
        answer = (result.get("llm_response") or {}).get("content") or ""
        chunks = (result.get("data") or {}).get("chunks") or []
        sources = await self.build_sources(question, chunks)
        return {"answer": answer, "sources": sources, "mode": mode}

    async def build_sources(
        self, question: str, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """One source per cited reference, with page, kind, excerpt and links."""
        by_ref: Dict[str, Dict[str, Any]] = {}
        for chunk in chunks:
            ref = str(chunk.get("reference_id", ""))
            if not ref or ref in by_ref:
                continue
            record = (
                await self.rag.lightrag.text_chunks.get_by_id(chunk["chunk_id"])
                if chunk.get("chunk_id")
                else None
            ) or {}
            content = chunk.get("content") or record.get("content") or ""
            doc_id = record.get("full_doc_id", "")
            file_name = chunk.get("file_path") or record.get("file_path") or ""
            page = record.get("page_idx")
            page_end = record.get("page_idx_end")
            asset = None
            match = _IMAGE_PATH_RE.search(content)
            if match:
                asset = self.asset_path(match.group(1))
            kind = self._kind(file_name, content, asset)
            by_ref[ref] = {
                "n": int(ref) if ref.isdigit() else len(by_ref) + 1,
                "doc_id": doc_id,
                "chunk_id": chunk.get("chunk_id"),
                "file": file_name,
                "page": page + 1 if isinstance(page, int) else None,
                "page_end": page_end + 1 if isinstance(page_end, int) else None,
                "kind": kind,
                "excerpt": make_excerpt(content, question),
                "asset": str(asset) if asset else None,
                "has_original": doc_id in self.index,
            }
        return sorted(by_ref.values(), key=lambda s: s["n"])

    @staticmethod
    def _kind(file_name: str, content: str, asset: Optional[Path]) -> str:
        if "Table Analysis" in content or "Table Path:" in content:
            return "table"
        if asset is not None or "Image Content Analysis" in content:
            if (
                "technical_drawing" in content
                or "drawings" in str(asset or "")
                or "Plan technique" in content
            ):
                return "plan"
            return "image"
        if file_name.lower().endswith(".msg"):
            return "email"
        return "text"


def make_excerpt(content: str, question: str, width: int = 600) -> str:
    """A passage of ``content`` around the first question term it contains."""
    text = re.sub(r"\s+", " ", content).strip()
    if len(text) <= width:
        return text
    terms = [t for t in re.findall(r"\w{4,}", question.lower())]
    lower = text.lower()
    positions = [lower.find(t) for t in terms if lower.find(t) >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - width // 4)
    end = min(len(text), start + width)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix + text[start:end] + suffix
