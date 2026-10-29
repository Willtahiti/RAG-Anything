"""``python -m raganything.app``: serve the ingestion and search pages."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from raganything.app.factory import build_rag_from_env
from raganything.app.server import create_app
from raganything.app.service import KnowledgeService


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("RAG_APP_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("RAG_APP_PORT", "8080"))
    )
    args = parser.parse_args()

    token = os.environ.get("RAG_APP_TOKEN") or None
    if args.host not in ("127.0.0.1", "localhost", "::1") and not token:
        parser.error("set RAG_APP_TOKEN before listening on a non-local address")

    rag, health = build_rag_from_env()
    service = KnowledgeService(
        rag,
        auto_resume=os.environ.get("RAG_APP_AUTO_RESUME", "true").lower() != "false",
    )
    roots = [
        Path(p)
        for p in os.environ.get("RAG_APP_INGEST_ROOTS", "").split(os.pathsep)
        if p
    ]
    app = create_app(
        service,
        token=token,
        max_upload_mb=int(os.environ.get("RAG_APP_MAX_UPLOAD_MB", "2048")),
        ingest_roots=roots,
        health_checks=health,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
