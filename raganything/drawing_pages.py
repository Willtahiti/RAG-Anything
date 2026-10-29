"""
Technical drawings (electrical, plumbing, HVAC, architectural plans) in PDFs.

Document parsers treat a plan as one picture or lose it entirely, and a
vision model cannot read symbols and legends from a whole A1 sheet shrunk to a
thumbnail. This module finds the drawing pages of a PDF (little text, many
vector paths, or a large scanned sheet), renders them at high resolution, and
emits an overview plus overlapping tiles as ``image`` blocks flagged
``image_kind="technical_drawing"``. The image processor then describes each
tile with a drawing-specific prompt, and the vector text (title block, tags,
notes) is added as a text block so it is searchable verbatim.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union

from lightrag.utils import get_env_value

logger = logging.getLogger(__name__)

DRAWING_IMAGE_KIND = "technical_drawing"
# A3 is 1191 x 842 pt; anything this big is a sheet, not a letter page.
_SHEET_AREA_PT2 = 0.9 * 1191 * 842


@dataclass
class DrawingConfig:
    enabled: bool = True
    dpi: int = 200
    tile_px: int = 1536
    overlap_px: int = 128
    max_tiles: int = 12
    overview_px: int = 1792
    max_render_px: int = 9000
    min_path_objects: int = 300
    max_text_chars: int = 3000
    include_vector_text: bool = True

    @classmethod
    def from_env(cls, **overrides: Any) -> "DrawingConfig":
        cfg = cls(
            enabled=get_env_value("ENABLE_DRAWING_PROCESSING", True, bool),
            dpi=get_env_value("DRAWING_DPI", 200, int),
            tile_px=get_env_value("DRAWING_TILE_PX", 1536, int),
            overlap_px=get_env_value("DRAWING_OVERLAP_PX", 128, int),
            max_tiles=get_env_value("DRAWING_MAX_TILES", 12, int),
            overview_px=get_env_value("DRAWING_OVERVIEW_PX", 1792, int),
            min_path_objects=get_env_value("DRAWING_MIN_PATH_OBJECTS", 300, int),
            max_text_chars=get_env_value("DRAWING_MAX_TEXT_CHARS", 3000, int),
        )
        for key, value in overrides.items():
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg


@dataclass
class PageStats:
    index: int
    width_pt: float
    height_pt: float
    chars: int
    path_objects: int
    image_objects: int
    image_coverage: float  # share of the page area covered by images (0..1)

    @property
    def area(self) -> float:
        return self.width_pt * self.height_pt


def _pdfium():
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Drawing detection needs pypdfium2: pip install pypdfium2"
        ) from exc
    return pdfium


def page_stats(page: Any, index: int, cfg: DrawingConfig) -> PageStats:
    """Cheap per-page measurements; stops scanning objects once decisive."""
    import pypdfium2.raw as raw

    width, height = page.get_size()
    textpage = page.get_textpage()
    try:
        chars = textpage.count_chars()
    finally:
        textpage.close()

    paths = images = 0
    covered = 0.0
    if chars <= cfg.max_text_chars:
        # Text-heavy pages are never drawings: skip the (costly) object scan.
        for obj in page.get_objects(
            filter=[raw.FPDF_PAGEOBJ_PATH, raw.FPDF_PAGEOBJ_IMAGE], max_depth=2
        ):
            if obj.type == raw.FPDF_PAGEOBJ_PATH:
                paths += 1
            else:
                images += 1
                try:
                    left, bottom, right, top = obj.get_pos()
                    covered += max(0.0, right - left) * max(0.0, top - bottom)
                except Exception:
                    pass
            if paths >= cfg.min_path_objects and images >= 1:
                break
    area = max(width * height, 1.0)
    return PageStats(
        index=index,
        width_pt=float(width),
        height_pt=float(height),
        chars=int(chars),
        path_objects=paths,
        image_objects=images,
        image_coverage=min(covered / area, 1.0),
    )


def is_drawing_page(stats: PageStats, cfg: DrawingConfig) -> bool:
    """True for plan sheets: vector drawings or large scanned sheets."""
    if stats.chars > cfg.max_text_chars:
        return False
    large_sheet = stats.area >= _SHEET_AREA_PT2
    if stats.path_objects >= cfg.min_path_objects:
        return True
    if large_sheet and stats.path_objects >= cfg.min_path_objects // 3:
        return True
    # Scanned plan: a big image on a large sheet (scanned A4 text is excluded).
    return large_sheet and stats.image_coverage >= 0.5 and stats.chars < 200


def detect_drawing_pages(
    pdf_path: Union[str, Path],
    cfg: Optional[DrawingConfig] = None,
    pages: Optional[Iterable[int]] = None,
) -> List[PageStats]:
    """Return the stats of the pages classified as technical drawings."""
    cfg = cfg or DrawingConfig.from_env()
    pdfium = _pdfium()
    pdf = pdfium.PdfDocument(str(pdf_path))
    found: List[PageStats] = []
    try:
        indexes = range(len(pdf)) if pages is None else list(pages)
        for index in indexes:
            page = pdf[index]
            try:
                stats = page_stats(page, index, cfg)
            except Exception as exc:
                logger.debug("Page %d stats failed: %s", index + 1, exc)
                continue
            finally:
                page.close()
            if is_drawing_page(stats, cfg):
                found.append(stats)
    finally:
        pdf.close()
    return found


def tile_grid(width: int, height: int, cfg: DrawingConfig) -> List[tuple]:
    """Overlapping tile boxes ``(left, top, right, bottom, row, col)``.

    The tile size grows when the sheet would need more than ``max_tiles``.
    """
    tile = max(cfg.tile_px, 256)
    overlap = min(max(cfg.overlap_px, 0), tile // 2)
    while True:
        step = tile - overlap
        cols = max(1, math.ceil((width - overlap) / step))
        rows = max(1, math.ceil((height - overlap) / step))
        if rows * cols <= cfg.max_tiles or tile > 8 * cfg.tile_px:
            break
        tile = int(tile * 1.25)
    boxes = []
    step = tile - overlap
    for row in range(rows):
        for col in range(cols):
            left = min(col * step, max(width - tile, 0))
            top = min(row * step, max(height - tile, 0))
            boxes.append(
                (left, top, min(left + tile, width), min(top + tile, height), row, col)
            )
    return boxes


def _render_page(page: Any, cfg: DrawingConfig):
    width, height = page.get_size()
    scale = cfg.dpi / 72.0
    longest = max(width, height) * scale
    if longest > cfg.max_render_px:
        scale *= cfg.max_render_px / longest
    return page.render(scale=scale).to_pil().convert("RGB")


def _vector_text(page: Any) -> str:
    textpage = page.get_textpage()
    try:
        return (textpage.get_text_range() or "").strip()
    finally:
        textpage.close()


def drawing_blocks_for_page(
    page: Any,
    stats: PageStats,
    out_dir: Path,
    cfg: DrawingConfig,
    *,
    include_text: bool,
) -> List[Dict[str, Any]]:
    """Render one drawing page into overview + tile ``image`` blocks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    page_no = stats.index + 1
    image = _render_page(page, cfg)
    width, height = image.size
    blocks: List[Dict[str, Any]] = []

    overview = image.copy()
    overview.thumbnail((cfg.overview_px, cfg.overview_px))
    overview_path = out_dir / f"p{page_no:06d}_overview.png"
    overview.save(overview_path)
    blocks.append(
        {
            "type": "image",
            "img_path": str(overview_path.resolve()),
            "image_caption": [f"Plan technique, page {page_no} — vue d'ensemble"],
            "image_footnote": [],
            "page_idx": stats.index,
            "image_kind": DRAWING_IMAGE_KIND,
            "drawing_tile": {"role": "overview", "page_size_px": [width, height]},
        }
    )

    boxes = tile_grid(width, height, cfg)
    if len(boxes) > 1:
        rows = max(b[4] for b in boxes) + 1
        cols = max(b[5] for b in boxes) + 1
        for left, top, right, bottom, row, col in boxes:
            tile_path = out_dir / f"p{page_no:06d}_r{row + 1}c{col + 1}.png"
            image.crop((left, top, right, bottom)).save(tile_path)
            blocks.append(
                {
                    "type": "image",
                    "img_path": str(tile_path.resolve()),
                    "image_caption": [
                        f"Plan technique, page {page_no} — détail ligne {row + 1}/{rows}, "
                        f"colonne {col + 1}/{cols}"
                    ],
                    "image_footnote": [],
                    "page_idx": stats.index,
                    "image_kind": DRAWING_IMAGE_KIND,
                    "drawing_tile": {
                        "role": "tile",
                        "row": row + 1,
                        "col": col + 1,
                        "rows": rows,
                        "cols": cols,
                        "bbox_px": [left, top, right, bottom],
                        "page_size_px": [width, height],
                    },
                }
            )

    if include_text:
        text = _vector_text(page)
        if text:
            blocks.append(
                {
                    "type": "text",
                    "text": f"Plan technique, page {page_no} — texte du plan "
                    f"(cartouche, repères, notes) :\n{text}",
                    "page_idx": stats.index,
                }
            )
    return blocks


def _insert_after_page(
    content_list: List[Dict[str, Any]], page_idx: int, new_blocks: List[Dict[str, Any]]
) -> None:
    position = None
    for i, block in enumerate(content_list):
        page = block.get("page_idx") if isinstance(block, dict) else None
        if not isinstance(page, int):
            continue
        if page <= page_idx:
            position = i + 1
        elif position is None:
            position = i
            break
    if position is None:
        position = len(content_list)
    content_list[position:position] = new_blocks


def add_drawing_blocks(
    content_list: List[Dict[str, Any]],
    pdf_path: Union[str, Path],
    output_dir: Union[str, Path],
    cfg: Optional[DrawingConfig] = None,
) -> List[Dict[str, Any]]:
    """Add overview/tile image blocks for the drawing pages of ``pdf_path``.

    Images the parser extracted from those pages are replaced by the tiles
    (they would otherwise be described twice, at lower resolution). Text the
    parser already extracted from a drawing page is kept, and the vector text
    is only added for pages the parser returned no text for.
    """
    cfg = cfg or DrawingConfig.from_env()
    if not cfg.enabled:
        return content_list
    try:
        drawings = detect_drawing_pages(pdf_path, cfg)
    except Exception as exc:
        logger.warning("Drawing detection failed for %s: %s", pdf_path, exc)
        return content_list
    if not drawings:
        return content_list

    drawing_pages: Set[int] = {d.index for d in drawings}
    logger.info(
        "Detected %d technical drawing page(s) in %s",
        len(drawings),
        Path(pdf_path).name,
    )
    result = [
        b
        for b in content_list
        if not (
            isinstance(b, dict)
            and b.get("type") == "image"
            and b.get("image_kind") != DRAWING_IMAGE_KIND
            and b.get("page_idx") in drawing_pages
        )
    ]
    text_pages = {
        b.get("page_idx")
        for b in result
        if isinstance(b, dict)
        and b.get("type") == "text"
        and str(b.get("text", "")).strip()
    }

    pdfium = _pdfium()
    pdf = pdfium.PdfDocument(str(pdf_path))
    out_dir = Path(output_dir) / "drawings"
    try:
        for stats in drawings:
            page = pdf[stats.index]
            try:
                blocks = drawing_blocks_for_page(
                    page,
                    stats,
                    out_dir,
                    cfg,
                    include_text=cfg.include_vector_text
                    and stats.index not in text_pages,
                )
            except Exception as exc:
                logger.warning(
                    "Could not render drawing page %d: %s", stats.index + 1, exc
                )
                continue
            finally:
                page.close()
            _insert_after_page(result, stats.index, blocks)
    finally:
        pdf.close()
    return result
