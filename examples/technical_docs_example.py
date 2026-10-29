"""
Technical documentation with local models: .msg mails, huge PDFs, plans.

Runs the ingestion and search stack against an OpenAI-compatible local server
(Ollama, vLLM, LM Studio), optionally with PostgreSQL + pgvector storage.

    # Ollama example
    ollama pull qwen2.5vl:7b && ollama pull bge-m3
    export RAG_LLM_BASE_URL=http://localhost:11434/v1
    export RAG_LLM_MODEL=qwen2.5vl:7b RAG_EMBED_MODEL=bge-m3 RAG_EMBED_DIM=1024

    # optional: PostgreSQL (docker compose -f docker-compose.pg.yml up -d)
    export RAG_STORAGE=postgres POSTGRES_PASSWORD=... POSTGRES_USER=rag POSTGRES_DATABASE=rag

    python examples/technical_docs_example.py mail.msg manuel.pdf "Quelle section de câble ?"

Useful settings (environment variables):
    PDF_WINDOW_PAGES=200      pages parsed per MinerU run, with resumable checkpoints
    PDF_WINDOW_TIMEOUT=3600   seconds allowed per window
    ENABLE_DRAWING_PROCESSING=true   plans are rendered in tiles and described by the VLM
    MAX_IMAGE_SIDE=1792       downscale images before they reach a local vision model
    MIN_IMAGE_PIXELS=10000    skip logos and bullets
    RAG_MAX_OUTAGE_S=3600     how long to wait for a model server that went away

For the web interface use ``python -m raganything.app`` (ingestion page + search page).
"""

import asyncio
import sys
from pathlib import Path

from raganything.app.factory import build_rag_from_env
from raganything.app.service import KnowledgeService
from raganything.cli import format_sources


async def main(paths: list[str], question: str) -> None:
    rag, _ = build_rag_from_env()
    service = KnowledgeService(rag)
    await service.start()
    try:
        ids = [service.add_file(Path(p)).id for p in paths]
        while any(service.jobs.jobs[i].status in ("queued", "running") for i in ids):
            for i in ids:
                job = service.jobs.jobs[i]
                print(f"{job.file}: {job.status} {job.done_pages}/{job.total_pages}")
            await asyncio.sleep(5)
        result = await service.search(question)
        print(result["answer"], "\n")
        print(format_sources(result["sources"]))
    finally:
        await service.stop()
        await rag.finalize_storages()


if __name__ == "__main__":
    *files, ask = sys.argv[1:]
    asyncio.run(main(files, ask))
