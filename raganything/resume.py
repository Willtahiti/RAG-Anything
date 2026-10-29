"""
Resume interrupted ingestions.

An ingestion that died halfway (server disconnection, crash, reboot) leaves
its documents PENDING / PROCESSING / PREPROCESSED / FAILED in LightRAG's
``doc_status``. Re-running ``process_document_complete`` on the same file is
safe and cheap: the parse result comes from the parse cache, LightRAG only
reprocesses what is not PROCESSED, and the multimodal stage is skipped once
flagged complete. These helpers find the unfinished documents and re-run them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Union

from lightrag.base import DocStatus

logger = logging.getLogger(__name__)

UNFINISHED_STATUSES = [
    DocStatus.PENDING,
    DocStatus.PROCESSING,
    DocStatus.PREPROCESSED,
    DocStatus.FAILED,
]


async def list_unfinished_documents(rag: Any) -> List[Dict[str, Any]]:
    """Documents whose ingestion is not complete, oldest first."""
    init = await rag._ensure_lightrag_initialized()
    if not init or not init.get("success"):
        raise RuntimeError(f"LightRAG initialization failed: {init}")
    records = await rag.lightrag.doc_status.get_docs_by_statuses(UNFINISHED_STATUSES)
    docs = []
    for doc_id, status in records.items():
        docs.append(
            {
                "doc_id": doc_id,
                "file_path": getattr(status, "file_path", ""),
                "status": getattr(getattr(status, "status", ""), "value", str(status)),
                "error_msg": getattr(status, "error_msg", None),
                "updated_at": getattr(status, "updated_at", ""),
            }
        )
    return sorted(docs, key=lambda d: d["updated_at"] or "")


def resolve_source_file(
    file_path: str, search_dirs: Iterable[Union[str, Path]]
) -> Optional[Path]:
    """Find the original file of a doc_status record (full path or basename)."""
    if not file_path:
        return None
    direct = Path(file_path)
    if direct.is_file():
        return direct
    name = direct.name
    for directory in search_dirs:
        directory = Path(directory)
        candidate = directory / name
        if candidate.is_file():
            return candidate
        if directory.is_dir():
            found = next(directory.rglob(name), None)
            if found is not None and found.is_file():
                return found
    return None


async def resume_unfinished(
    rag: Any,
    search_dirs: Iterable[Union[str, Path]] = (),
    *,
    on_progress: Optional[
        Callable[[Dict[str, Any]], Union[None, Awaitable[None]]]
    ] = None,
    **process_kwargs: Any,
) -> Dict[str, List[Dict[str, Any]]]:
    """Re-run ingestion for every unfinished document whose file can be found.

    Returns ``{"resumed": [...], "failed": [...], "missing": [...]}``.
    """
    import asyncio

    search_dirs = list(search_dirs)
    outcome: Dict[str, List[Dict[str, Any]]] = {
        "resumed": [],
        "failed": [],
        "missing": [],
    }
    for doc in await list_unfinished_documents(rag):
        source = resolve_source_file(doc["file_path"], search_dirs)
        if source is None:
            logger.warning("Cannot resume %s: file not found", doc["file_path"])
            outcome["missing"].append(doc)
            continue
        entry = {**doc, "source": str(source)}
        try:
            await rag.process_document_complete(str(source), **process_kwargs)
            outcome["resumed"].append(entry)
        except Exception as exc:
            logger.error("Resuming %s failed: %s", source, exc)
            outcome["failed"].append({**entry, "error_msg": str(exc)})
        if on_progress is not None:
            result = on_progress(entry)
            if asyncio.iscoroutine(result):
                await result
    return outcome
