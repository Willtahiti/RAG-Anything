"""
Window-by-window parsing of very large PDFs.

A 12,000-page PDF parsed in one MinerU/Docling run has no checkpoint, no
progress, and loses everything on a crash or timeout. This module slices the
PDF into windows of ``window_pages`` pages (with pypdfium2), parses each window
as an independent PDF, checkpoints the result on disk, and merges the windows
back with absolute ``page_idx`` values. A rerun resumes from the checkpoints.

A window that keeps failing is bisected down to the offending page(s), which
are skipped and reported, so one corrupt page does not abort the document.
Tables that continue across a page break are stitched back together.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from lightrag.utils import get_env_value

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_PAGES = 200
DEFAULT_WINDOW_THRESHOLD = 400
DEFAULT_WINDOW_RETRIES = 1

# Environment problems, not bad pages: bisecting would only repeat them.
_FATAL_ERRORS = (FileNotFoundError, ImportError, PermissionError, MemoryError)
DEFAULT_MAX_FAILED_PAGES = 20

# parse_window(window_pdf, out_dir) -> content list with window-relative page_idx
WindowParser = Callable[[Path, Path], List[Dict[str, Any]]]
# progress(done_pages, total_pages, (start, end)) after each finished window
ProgressCallback = Callable[[int, int, Tuple[int, int]], None]


def _pdfium():
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - mineru depends on it
        raise ImportError(
            "Windowed PDF parsing needs pypdfium2: pip install pypdfium2"
        ) from exc
    return pdfium


def count_pdf_pages(pdf_path: Union[str, Path]) -> int:
    pdf = _pdfium().PdfDocument(str(pdf_path))
    try:
        return len(pdf)
    finally:
        pdf.close()


def plan_windows(total_pages: int, window_pages: int) -> List[Tuple[int, int]]:
    """Return ``[(start, end), ...]`` half-open page ranges covering the PDF."""
    if window_pages < 1:
        raise ValueError("window_pages must be >= 1")
    return [
        (start, min(start + window_pages, total_pages))
        for start in range(0, total_pages, window_pages)
    ]


def window_settings(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Pop the windowing options from ``kwargs`` (env vars as defaults)."""
    pages = kwargs.pop("window_pages", None)
    threshold = kwargs.pop("window_threshold", None)
    resume = kwargs.pop("window_resume", None)
    window_timeout = kwargs.pop("window_timeout", None)
    retries = kwargs.pop("window_retries", None)
    progress = kwargs.pop("window_progress", None)
    return {
        "pages": int(
            pages
            if pages is not None
            else get_env_value("PDF_WINDOW_PAGES", DEFAULT_WINDOW_PAGES, int)
        ),
        "threshold": int(
            threshold
            if threshold is not None
            else get_env_value("PDF_WINDOW_THRESHOLD", DEFAULT_WINDOW_THRESHOLD, int)
        ),
        "resume": bool(
            resume if resume is not None else get_env_value("PDF_RESUME", True, bool)
        ),
        "timeout": window_timeout
        if window_timeout is not None
        else get_env_value("PDF_WINDOW_TIMEOUT", None, int),
        "retries": int(
            retries
            if retries is not None
            else get_env_value("PDF_WINDOW_RETRIES", DEFAULT_WINDOW_RETRIES, int)
        ),
        "progress": progress,
    }


def _slice_pdf(src, start: int, end: int, target: Path) -> None:
    pdfium = _pdfium()
    dst = pdfium.PdfDocument.new()
    try:
        dst.import_pages(src, list(range(start, end)))
        target.parent.mkdir(parents=True, exist_ok=True)
        dst.save(str(target))
    finally:
        dst.close()


def _file_fingerprint(pdf_path: Path, extra: Dict[str, Any]) -> str:
    stat = pdf_path.stat()
    digest = hashlib.sha256()
    digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    digest.update(json.dumps(extra, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, path)


def _window_name(start: int, end: int) -> str:
    return f"w{start:06d}_{end:06d}"


def _offset_pages(blocks: List[Dict[str, Any]], offset: int) -> List[Dict[str, Any]]:
    shifted = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        item = dict(block)
        item["page_idx"] = int(item.get("page_idx", 0) or 0) + offset
        shifted.append(item)
    return shifted


# --- table stitching ------------------------------------------------------

_ROW_RE = re.compile(r"<tr\b.*?</tr>", re.IGNORECASE | re.DOTALL)
_CELL_RE = re.compile(r"<t[dh]\b", re.IGNORECASE)


def _table_body(block: Dict[str, Any]) -> Optional[str]:
    body = block.get("table_body")
    return body if isinstance(body, str) and body.strip() else None


def _html_rows(body: str) -> List[str]:
    return _ROW_RE.findall(body)


def _column_count(body: str) -> Optional[int]:
    if "<table" in body.lower():
        rows = _html_rows(body)
        if not rows:
            return None
        return len(_CELL_RE.findall(rows[0]))
    lines = [ln for ln in body.strip().splitlines() if ln.strip().startswith("|")]
    if not lines:
        return None
    return max(lines[0].strip().strip("|").count("|") + 1, 1)


def _merge_bodies(first: str, second: str) -> Optional[str]:
    if "<table" in first.lower() and "<table" in second.lower():
        rows_a, rows_b = _html_rows(first), _html_rows(second)
        if not rows_a or not rows_b:
            return None
        # A repeated header row at the top of the continuation is dropped.
        if _norm_row(rows_b[0]) == _norm_row(rows_a[0]):
            rows_b = rows_b[1:]
        head = first[: first.lower().find("<tr")]
        tail = "</table>"
        return head + "".join(rows_a + rows_b) + tail
    if "<table" not in first.lower() and "<table" not in second.lower():
        lines_a = first.strip().splitlines()
        lines_b = second.strip().splitlines()
        if len(lines_b) >= 2 and _is_md_separator(lines_b[1]):
            if lines_b[0].strip() == lines_a[0].strip():
                lines_b = lines_b[2:]
        return "\n".join(lines_a + lines_b)
    return None


def _norm_row(row: str) -> str:
    return re.sub(r"\s+", "", re.sub(r"<[^>]+>", "|", row)).lower()


def _is_md_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\s*\|?[\s:\-|]+\|?\s*", line)) and "-" in line


def stitch_cross_page_tables(
    content_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge a table with its continuation on the next page.

    Two adjacent table blocks on consecutive pages with the same number of
    columns, where the continuation has no caption of its own, are one table.
    """
    merged: List[Dict[str, Any]] = []
    for block in content_list:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and isinstance(block, dict)
            and block.get("type") == "table"
            and prev.get("type") == "table"
            and int(block.get("page_idx", 0))
            == int(prev.get("merged_end_page", prev.get("page_idx", 0))) + 1
            and not _caption_text(block.get("table_caption"))
        ):
            body_a, body_b = _table_body(prev), _table_body(block)
            if body_a and body_b:
                cols_a, cols_b = _column_count(body_a), _column_count(body_b)
                if cols_a and cols_a == cols_b:
                    combined = _merge_bodies(body_a, body_b)
                    if combined:
                        prev["table_body"] = combined
                        prev["merged_end_page"] = int(block.get("page_idx", 0))
                        foot = list(block.get("table_footnote") or [])
                        if foot:
                            prev["table_footnote"] = (
                                list(prev.get("table_footnote") or []) + foot
                            )
                        continue
        merged.append(dict(block) if isinstance(block, dict) else block)
    for block in merged:
        if isinstance(block, dict):
            block.pop("merged_end_page", None)
    return merged


def _caption_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value).strip()
    return ""


# --- main entry -----------------------------------------------------------


def parse_pdf_in_windows(
    parse_window: WindowParser,
    pdf_path: Union[str, Path],
    work_dir: Union[str, Path],
    *,
    window_pages: int = DEFAULT_WINDOW_PAGES,
    resume: bool = True,
    retries: int = DEFAULT_WINDOW_RETRIES,
    progress: Optional[ProgressCallback] = None,
    fingerprint_extra: Optional[Dict[str, Any]] = None,
    total_pages: Optional[int] = None,
    max_failed_pages: int = DEFAULT_MAX_FAILED_PAGES,
) -> List[Dict[str, Any]]:
    """Parse ``pdf_path`` window by window and return the merged content list."""
    pdf_path = Path(pdf_path)
    work_dir = Path(work_dir)
    total = total_pages if total_pages is not None else count_pdf_pages(pdf_path)
    windows = plan_windows(total, window_pages)
    ckpt_dir = work_dir / "windows"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = _file_fingerprint(
        pdf_path, {"window_pages": window_pages, **(fingerprint_extra or {})}
    )
    manifest_path = ckpt_dir / "manifest.json"
    if manifest_path.exists():
        try:
            same = (
                json.loads(manifest_path.read_text("utf-8")).get("fingerprint")
                == fingerprint
            )
        except (OSError, ValueError):
            same = False
        if not (resume and same):
            logger.info("Discarding stale window checkpoints in %s", ckpt_dir)
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        manifest_path,
        {
            "fingerprint": fingerprint,
            "total_pages": total,
            "window_pages": window_pages,
        },
    )

    pdfium = _pdfium()
    src = pdfium.PdfDocument(str(pdf_path))
    failed_pages: List[int] = []
    results: List[List[Dict[str, Any]]] = []
    done_pages = 0

    def run_range(start: int, end: int, attempts_left: int) -> List[Dict[str, Any]]:
        name = _window_name(start, end)
        ckpt = ckpt_dir / f"{name}.json"
        if resume and ckpt.exists():
            try:
                return json.loads(ckpt.read_text("utf-8"))
            except (OSError, ValueError):
                logger.warning("Corrupt checkpoint %s, reparsing", ckpt)
        window_pdf = ckpt_dir / f"{name}.pdf"
        out_dir = ckpt_dir / name
        try:
            _slice_pdf(src, start, end, window_pdf)
            blocks = _offset_pages(parse_window(window_pdf, out_dir), start)
            _write_json_atomic(ckpt, blocks)
            return blocks
        except _FATAL_ERRORS:
            raise
        except Exception as exc:
            if attempts_left > 0:
                logger.warning("Pages %d-%d failed (%s); retrying", start + 1, end, exc)
                return run_range(start, end, attempts_left - 1)
            if end - start > 1:
                mid = (start + end) // 2
                logger.warning(
                    "Pages %d-%d failed (%s); splitting to isolate the bad page",
                    start + 1,
                    end,
                    exc,
                )
                return run_range(start, mid, retries) + run_range(mid, end, retries)
            logger.error("Page %d skipped after repeated failures: %s", start + 1, exc)
            failed_pages.append(start)
            if len(failed_pages) > max_failed_pages:
                raise RuntimeError(
                    f"More than {max_failed_pages} pages failed to parse; "
                    f"aborting (last error on page {start + 1}: {exc}). "
                    "Checkpoints are kept: fix the cause and rerun to resume."
                ) from exc
            placeholder = [
                {
                    "type": "text",
                    "text": f"[Page {start + 1} non analysable : {exc}]",
                    "page_idx": start,
                    "parse_error": True,
                }
            ]
            _write_json_atomic(ckpt, placeholder)
            return placeholder
        finally:
            if window_pdf.exists():
                window_pdf.unlink()

    try:
        for start, end in windows:
            results.append(run_range(start, end, retries))
            done_pages = end
            logger.info("PDF window pages %d-%d/%d done", start + 1, end, total)
            if progress is not None:
                try:
                    progress(done_pages, total, (start, end))
                except Exception:  # a UI callback must not break parsing
                    logger.debug("progress callback failed", exc_info=True)
    finally:
        src.close()

    merged = stitch_cross_page_tables([b for chunk in results for b in chunk])
    if failed_pages:
        logger.warning(
            "%d page(s) could not be parsed: %s",
            len(failed_pages),
            ", ".join(str(p + 1) for p in failed_pages[:50]),
        )
    return merged
