"""FastAPI application: ingestion page, search page and their JSON API."""

from __future__ import annotations

import asyncio
import html
import json
import os
import secrets
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    RedirectResponse,
    StreamingResponse,
)
from pydantic import BaseModel

from raganything.app.service import KnowledgeService, safe_filename

STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_CHUNK = 1024 * 1024


class SearchRequest(BaseModel):
    question: str
    mode: str = "mix"
    top_k: int = 40


class PathRequest(BaseModel):
    path: str


def create_app(
    service: KnowledgeService,
    *,
    token: Optional[str] = None,
    max_upload_mb: int = 2048,
    ingest_roots: Optional[List[Path]] = None,
    health_checks: Optional[dict] = None,
    manage_service: bool = True,
) -> FastAPI:
    """Build the app around ``service``.

    ``token`` (``RAG_APP_TOKEN``) protects every route when the app is reachable
    from other machines. ``ingest_roots`` lists the server directories the
    "ingest a server path" endpoint may read; none means that endpoint is off.
    """

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if manage_service:
            await service.start()
        yield
        if manage_service:
            await service.stop()

    app = FastAPI(title="RAG-Anything", lifespan=lifespan)
    roots = [Path(r).resolve() for r in (ingest_roots or [])]

    def require_auth(request: Request) -> None:
        if not token:
            return
        supplied = request.headers.get("authorization", "").removeprefix(
            "Bearer "
        ).strip() or request.cookies.get("rag_token", "")
        if not secrets.compare_digest(supplied.encode(), token.encode()):
            raise HTTPException(status_code=401, detail="authentication required")

    auth = [Depends(require_auth)]

    @app.get("/login")
    async def login(token_value: str = "", next: str = "/"):
        if token and secrets.compare_digest(token_value.encode(), token.encode()):
            response = RedirectResponse(next if next.startswith("/") else "/")
            response.set_cookie(
                "rag_token", token, httponly=True, samesite="strict", max_age=86400 * 30
            )
            return response
        raise HTTPException(status_code=401, detail="bad token")

    # --- pages ---------------------------------------------------------------

    @app.get("/", dependencies=auth, response_class=HTMLResponse)
    async def search_page():
        return (STATIC_DIR / "search.html").read_text("utf-8")

    @app.get("/ingest", dependencies=auth, response_class=HTMLResponse)
    async def ingest_page():
        return (STATIC_DIR / "ingest.html").read_text("utf-8")

    @app.get("/static/{name}", dependencies=auth)
    async def static_file(name: str):
        path = STATIC_DIR / Path(name).name
        if not path.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(path)

    # --- ingestion API -----------------------------------------------------------

    @app.post("/api/ingest", dependencies=auth)
    async def ingest(files: List[UploadFile] = File(...)):
        limit = max_upload_mb * 1024 * 1024
        created = []
        for upload in files:
            suffix = Path(safe_filename(upload.filename or "upload")).suffix
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, dir=service.data_dir
            ) as tmp:
                size = 0
                while True:
                    block = await upload.read(UPLOAD_CHUNK)
                    if not block:
                        break
                    size += len(block)
                    if size > limit:
                        tmp.close()
                        os.unlink(tmp.name)
                        raise HTTPException(
                            413, f"{upload.filename} exceeds {max_upload_mb} MB"
                        )
                    tmp.write(block)
            try:
                job = await asyncio.to_thread(
                    service.add_file, Path(tmp.name), upload.filename
                )
                created.append(job.id)
            finally:
                os.unlink(tmp.name)
        return {"jobs": created}

    @app.post("/api/ingest/path", dependencies=auth)
    async def ingest_path(body: PathRequest):
        if not roots:
            raise HTTPException(
                403, "server-side ingestion is disabled (RAG_APP_INGEST_ROOTS)"
            )
        target = Path(body.path).resolve()
        if not any(
            str(target).startswith(str(r) + os.sep) or target == r for r in roots
        ):
            raise HTTPException(403, "path is outside the allowed ingestion roots")
        if not target.exists():
            raise HTTPException(404, "no such file or folder")
        files = (
            [target]
            if target.is_file()
            else sorted(p for p in target.rglob("*") if p.is_file())
        )
        supported = {
            e.strip().lower() for e in service.rag.config.supported_file_extensions
        }
        files = [f for f in files if f.suffix.lower() in supported]
        created = []
        for f in files:
            job = await asyncio.to_thread(service.add_file, f)
            created.append(job.id)
        return {"jobs": created}

    @app.get("/api/jobs", dependencies=auth)
    async def jobs():
        return service.list_jobs()

    @app.get("/api/jobs/stream", dependencies=auth)
    async def jobs_stream(request: Request):
        async def events():
            last = None
            while not await request.is_disconnected():
                payload = json.dumps(service.list_jobs())
                if payload != last:
                    yield f"data: {payload}\n\n"
                    last = payload
                await asyncio.sleep(1.0)

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/api/jobs/{job_id}/{action}", dependencies=auth)
    async def job_action(job_id: str, action: str):
        if job_id not in service.jobs.jobs:
            raise HTTPException(404, "unknown job")
        if action == "resume":
            return service.retry(job_id).__dict__
        if action == "cancel":
            return service.cancel(job_id).__dict__
        raise HTTPException(400, "action must be resume or cancel")

    @app.get("/api/documents", dependencies=auth)
    async def documents():
        return await service.list_documents()

    @app.delete("/api/documents/{doc_id}", dependencies=auth)
    async def delete_document(doc_id: str):
        await service.delete_document(doc_id)
        return {"deleted": doc_id}

    @app.get("/api/files/{doc_id}", dependencies=auth)
    async def original(doc_id: str):
        path = service.original_file(doc_id)
        if path is None:
            raise HTTPException(404, "original file not available")
        return FileResponse(path, content_disposition_type="inline")

    @app.get(
        "/api/documents/{doc_id}/view", dependencies=auth, response_class=HTMLResponse
    )
    async def view_document(doc_id: str, chunk: str = ""):
        chunks = await service.document_chunks(doc_id)
        if not chunks:
            raise HTTPException(404, "unknown document")
        name = service.index.get(doc_id, {}).get("name", doc_id)
        parts = []
        for c in chunks:
            page = c.get("page_idx")
            label = f"p.{page + 1}" if isinstance(page, int) else ""
            cls = "hit" if c["chunk_id"] == chunk else ""
            parts.append(
                f'<section id="{html.escape(c["chunk_id"])}" class="{cls}">'
                f"<small>{html.escape(label)}</small>"
                f"<pre>{html.escape(c.get('content', ''))}</pre></section>"
            )
        body = "".join(parts)
        return HTMLResponse(
            "<!doctype html><meta charset=utf-8>"
            f"<title>{html.escape(name)}</title>"
            "<style>body{font:15px system-ui;max-width:860px;margin:2rem auto;padding:0 1rem}"
            "pre{white-space:pre-wrap;margin:.2rem 0}section{padding:.4rem .6rem;border-left:3px solid #ddd;margin:.5rem 0}"
            ".hit{border-color:#d97706;background:#fff7ed}small{color:#777}</style>"
            f"<h1>{html.escape(name)}</h1>{body}"
            "<script>const h=document.querySelector('.hit');if(h)h.scrollIntoView({block:'center'})</script>"
        )

    @app.get("/api/assets", dependencies=auth)
    async def asset(path: str):
        resolved = service.asset_path(path)
        if resolved is None:
            raise HTTPException(404, "asset not available")
        return FileResponse(resolved)

    # --- search API --------------------------------------------------------------

    @app.post("/api/search", dependencies=auth)
    async def search(body: SearchRequest):
        if not body.question.strip():
            raise HTTPException(400, "empty question")
        if body.mode not in ("local", "global", "hybrid", "naive", "mix"):
            raise HTTPException(400, "invalid mode")
        result = await service.search(
            body.question.strip(), mode=body.mode, top_k=max(1, min(body.top_k, 200))
        )
        for source in result["sources"]:
            source["link"] = _source_link(service, source)
            source["asset_url"] = (
                "/api/assets?path=" + _quote(source["asset"])
                if source.get("asset")
                else None
            )
            source.pop("asset", None)
        return result

    @app.get("/api/health", dependencies=auth)
    async def health():
        status = {}
        for name, check in (health_checks or {}).items():
            try:
                status[name] = bool(await check())
            except Exception:
                status[name] = False
        return {"ok": all(status.values()) if status else True, "services": status}

    return app


def _quote(value: str) -> str:
    from urllib.parse import quote

    return quote(value, safe="")


def _source_link(service: KnowledgeService, source: dict) -> Optional[str]:
    """PDFs open at the cited page; everything else opens the parsed text."""
    doc_id = source.get("doc_id")
    if not doc_id:
        return None
    name = source.get("file", "").lower()
    if name.endswith(".pdf") and source.get("has_original"):
        page = source.get("page") or 1
        return f"/api/files/{doc_id}#page={page}"
    chunk = source.get("chunk_id") or ""
    return f"/api/documents/{doc_id}/view?chunk={_quote(chunk)}#{chunk}"
