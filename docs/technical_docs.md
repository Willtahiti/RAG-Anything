# Technical documentation: mails, huge PDFs, plans

This fork adds what is needed to index Outlook mails and large technical PDFs
(text, images, plans, multi-column tables) with local models.

| Need | How |
|---|---|
| Outlook `.msg` | `pip install -e ".[email]"`. Headers and body become text; attachments (PDF, Office, text, images, nested `.msg`) are parsed with the active parser, each on its own page range. `.msg` is in the default `SUPPORTED_FILE_EXTENSIONS`. |
| PDFs up to 12,000 pages | Above `PDF_WINDOW_THRESHOLD` pages (default 400) the PDF is parsed in windows of `PDF_WINDOW_PAGES` (default 200). Each window is checkpointed; rerunning resumes. A page that keeps failing is isolated and skipped (reported in the log). Tables split across a page break are rejoined. Windowing needs a parser that sets `SUPPORTS_PDF_WINDOWING` (MinerU, Docling). |
| Plans (electrical, plumbing…) | `ENABLE_DRAWING_PROCESSING` (default on) detects drawing pages, renders an overview plus overlapping tiles (`DRAWING_DPI`, `DRAWING_TILE_PX`, `DRAWING_MAX_TILES`) and describes each with a drawing-specific prompt (`drawing_vision_prompt`). Vector text (title block, tags) is indexed verbatim. |
| Disconnections | `raganything.resilient_embedding.make_resilient_embedding` / `make_resilient_llm` wait for a server to come back, validate every vector (count, dimension, NaN, zeros) and never truncate. `raganything.resume.resume_unfinished` reruns documents left unfinished. The web app requeues interrupted jobs at startup. |
| Storage | Files by default, or PostgreSQL + pgvector: see [postgres_setup.md](postgres_setup.md). |
| Export / import | `raganything.export` (snapshot or portable bundle between file and PostgreSQL storage), also via `python -m raganything.cli export|import|snapshot|restore|inspect`. |
| Interfaces | `python -m raganything.app` (ingestion page + search page, answers with clickable sources) and `python -m raganything.cli`. Install with `pip install -e ".[app]"`. |

## Web application

```bash
export RAG_LLM_BASE_URL=http://localhost:11434/v1
export RAG_LLM_MODEL=qwen2.5vl:7b RAG_EMBED_MODEL=bge-m3 RAG_EMBED_DIM=1024
python -m raganything.app            # http://127.0.0.1:8080
```

* `/ingest`: drag and drop files, follow each job (pages done), resume or cancel,
  list and delete indexed documents.
* `/` (search): the answer cites its sources as `[1]`, `[2]`… shown as badges
  (📄 text, 🖼️ image, 📊 table, 📐 plan, ✉️ mail). Hover or focus a badge for the
  passage (query terms highlighted) and a thumbnail for images and plans;
  click it to open the original PDF at the cited page, or the parsed text of
  the mail/document with the passage highlighted.
* Security: it listens on `127.0.0.1` only. To expose it set `RAG_APP_HOST`
  and `RAG_APP_TOKEN` (required for non-local addresses; open `/login?token_value=…`
  once in the browser). Server-side folder ingestion is off unless
  `RAG_APP_INGEST_ROOTS` lists allowed directories (separated by `:`).

Settings: `RAG_LLM_MODEL`, `RAG_VISION_MODEL` (defaults to the LLM),
`RAG_EMBED_MODEL`, `RAG_EMBED_DIM`, `RAG_STORAGE=postgres`, `RAG_WORKSPACE`,
`RAG_EMBED_CONCURRENCY` (default 2) and `RAG_EMBED_BATCH` (default 8): defaults
favor embedding quality and reliability over speed.
