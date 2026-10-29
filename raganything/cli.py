"""Command line: ingest, search, resume, export and import.

    python -m raganything.cli ingest report.pdf mail.msg folder/
    python -m raganything.cli query "Quelle section de câble pour le TGBT ?"
    python -m raganything.cli docs
    python -m raganything.cli export kb.zip [--assets]
    python -m raganything.cli import kb.zip [--replace --yes] [--reembed]
    python -m raganything.cli snapshot backup.tar.gz
    python -m raganything.cli restore backup.tar.gz ./rag_storage
    python -m raganything.cli inspect kb.zip

The models and storage come from the same environment variables as the web
application (see ``raganything.app.models`` and ``raganything.app.factory``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

KIND_EMOJI = {"text": "📄", "image": "🖼️", "table": "📊", "plan": "📐", "email": "✉️"}


def _services():
    from raganything.app.factory import build_rag_from_env
    from raganything.app.service import KnowledgeService

    rag, _ = build_rag_from_env()
    return rag, KnowledgeService(rag)


def format_sources(sources: List[dict]) -> str:
    lines = []
    for s in sources:
        page = f" p.{s['page']}" if s.get("page") else ""
        lines.append(
            f"[{s['n']}] {KIND_EMOJI.get(s['kind'], '📄')} {s['file']}{page} — « {s['excerpt'][:160]} »"
        )
    return "\n".join(lines)


async def _wait_jobs(service, job_ids: List[str]) -> int:
    failed = 0
    last = {}
    while True:
        pending = False
        for job_id in job_ids:
            job = service.jobs.jobs[job_id]
            line = f"{job.file}: {job.status}"
            if job.total_pages:
                line += f" ({job.done_pages}/{job.total_pages} pages)"
            if last.get(job_id) != line:
                print(line, flush=True)
                last[job_id] = line
            if job.status in ("queued", "running"):
                pending = True
            elif job.status in ("failed", "interrupted"):
                failed += 1
                print(f"  error: {job.error}", file=sys.stderr)
        if not pending:
            return failed
        await asyncio.sleep(1)


async def _run(args: argparse.Namespace) -> int:
    if args.command == "inspect":
        from raganything.export import read_manifest

        print(json.dumps(read_manifest(args.archive), indent=2, ensure_ascii=False))
        return 0
    if args.command == "restore":
        from raganything.export import restore_snapshot

        restore_snapshot(
            args.archive, args.working_dir, args.output_dir, overwrite=args.overwrite
        )
        print("restored")
        return 0

    rag, service = _services()
    try:
        if args.command == "ingest":
            await service.start()
            from raganything.config import RAGAnythingConfig  # noqa: F401

            supported = {e.lower() for e in rag.config.supported_file_extensions}
            files = []
            for raw in args.paths:
                path = Path(raw)
                found = (
                    [path]
                    if path.is_file()
                    else sorted(p for p in path.rglob("*") if p.is_file())
                )
                files += [f for f in found if f.suffix.lower() in supported]
            if not files:
                print("no supported file found", file=sys.stderr)
                return 1
            ids = [service.add_file(f).id for f in files]
            return 1 if await _wait_jobs(service, ids) else 0
        if args.command == "resume":
            await service.start()
            ids = [
                j.id
                for j in service.jobs.jobs.values()
                if j.status in ("queued", "running")
            ]
            ids += [
                service.retry(j.id).id
                for j in list(service.jobs.jobs.values())
                if j.status in ("failed", "interrupted", "cancelled")
            ]
            if not ids:
                print("nothing to resume")
                return 0
            return 1 if await _wait_jobs(service, ids) else 0
        if args.command == "query":
            result = await service.search(args.question, mode=args.mode)
            print(result["answer"], "\n")
            print(format_sources(result["sources"]))
            return 0
        if args.command == "docs":
            for d in await service.list_documents():
                print(
                    f"{d['status']:<10} {d['chunks'] or 0:>6}  {d['file']}  ({d['doc_id']})"
                )
            return 0
        if args.command == "export":
            from raganything.export import export_bundle

            manifest = await export_bundle(
                rag,
                args.archive,
                include_assets=args.assets,
                include_llm_cache=args.llm_cache,
                on_progress=lambda stage, n: print(f"{stage}: {n}", flush=True),
            )
            print(json.dumps(manifest["counts"]))
            return 0
        if args.command == "import":
            from raganything.export import import_bundle

            result = await import_bundle(
                rag,
                args.archive,
                mode="replace" if args.replace else "merge",
                confirm_replace=args.yes,
                reembed=args.reembed,
                on_progress=lambda stage, n: print(f"{stage}: {n}", flush=True),
            )
            print(json.dumps(result["imported"]))
            return 0
        if args.command == "snapshot":
            from raganything.export import create_snapshot

            await create_snapshot(rag, args.archive, include_assets=not args.no_assets)
            print(f"snapshot written to {args.archive}")
            return 0
    finally:
        await service.stop()
        await rag.finalize_storages()
    return 2


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raganything.cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ingest", help="ingest files or folders")
    p.add_argument("paths", nargs="+")
    sub.add_parser("resume", help="resume interrupted or failed ingestions")
    p = sub.add_parser(
        "query", help="ask a question; sources are listed after the answer"
    )
    p.add_argument("question")
    p.add_argument(
        "--mode", default="mix", choices=["local", "global", "hybrid", "naive", "mix"]
    )
    sub.add_parser("docs", help="list indexed documents")
    p = sub.add_parser("export", help="write a portable bundle (.zip)")
    p.add_argument("archive")
    p.add_argument("--assets", action="store_true", help="include extracted images")
    p.add_argument(
        "--llm-cache", action="store_true", help="include the LLM response cache"
    )
    p = sub.add_parser("import", help="load a portable bundle")
    p.add_argument("archive")
    p.add_argument("--replace", action="store_true", help="empty the target first")
    p.add_argument("--yes", action="store_true", help="confirm --replace")
    p.add_argument(
        "--reembed",
        action="store_true",
        help="recompute vectors with the current model",
    )
    p = sub.add_parser(
        "snapshot", help="exact backup (.tar.gz; pg_dump for PostgreSQL)"
    )
    p.add_argument("archive")
    p.add_argument("--no-assets", action="store_true")
    p = sub.add_parser("restore", help="restore a snapshot before starting the app")
    p.add_argument("archive")
    p.add_argument("working_dir")
    p.add_argument("--output-dir")
    p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("inspect", help="print a bundle's or snapshot's manifest")
    p.add_argument("archive")
    return asyncio.run(_run(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
