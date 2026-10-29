"""Technical drawing pages: detection, tiling, prompts, image downscaling."""

import base64
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from raganything.drawing_pages import (
    DRAWING_IMAGE_KIND,
    DrawingConfig,
    add_drawing_blocks,
    detect_drawing_pages,
    tile_grid,
)
from raganything.modalprocessors import ImageModalProcessor
from raganything.prompt import PROMPTS
from raganything.utils import encode_image_to_base64

A1 = (1684, 2384)


def _make_pdf(path: Path) -> Path:
    """Page 0: text. Page 1: vector plan on A1. Page 2: a ruled table on A4."""
    c = canvas.Canvas(str(path))
    c.setPageSize(A4)
    for i in range(60):
        c.drawString(40, 800 - i * 12, f"Ligne de texte numéro {i} " * 4)
    c.showPage()

    c.setPageSize(A1)
    for i in range(400):
        c.line(50 + i * 3, 100, 60 + i * 3, 2200)
        c.circle(100 + i * 3, 1200, 5)
    c.drawString(100, 60, "CARTOUCHE PLAN-ELEC-01 echelle 1/50")
    c.showPage()

    c.setPageSize(A4)
    for i in range(400):  # grid lines, but a lot of text
        c.line(30, 30 + i, 560, 30 + i)
    for i in range(70):
        c.drawString(40, 820 - i * 11, "cellule " * 30)
    c.showPage()
    c.save()
    return path


@pytest.fixture
def pdf(tmp_path):
    return _make_pdf(tmp_path / "doc.pdf")


def test_detects_only_the_plan_page(pdf):
    found = detect_drawing_pages(pdf, DrawingConfig(min_path_objects=300))
    assert [s.index for s in found] == [1]
    assert found[0].path_objects >= 300


def test_blocks_for_plan_page(pdf, tmp_path):
    cfg = DrawingConfig(dpi=60, tile_px=400, overlap_px=40, max_tiles=6, overview_px=500)
    content = [
        {"type": "text", "text": "intro", "page_idx": 0},
        {"type": "image", "img_path": "/x/lowres.png", "page_idx": 1},  # parser's copy
        {"type": "image", "img_path": "/x/photo.png", "page_idx": 0},
        {"type": "text", "text": "page 3", "page_idx": 2},
    ]
    out = add_drawing_blocks(content, pdf, tmp_path / "o", cfg)

    drawings = [b for b in out if b.get("image_kind") == DRAWING_IMAGE_KIND]
    assert drawings[0]["drawing_tile"]["role"] == "overview"
    tiles = [b for b in drawings if b["drawing_tile"]["role"] == "tile"]
    assert 2 <= len(tiles) <= 6
    assert all(b["page_idx"] == 1 for b in drawings)
    for b in drawings:
        assert Path(b["img_path"]).is_absolute() and Path(b["img_path"]).exists()
    assert "page 2" in drawings[0]["image_caption"][0]

    # parser image on the drawing page replaced, other pages untouched
    assert not any(b.get("img_path") == "/x/lowres.png" for b in out)
    assert any(b.get("img_path") == "/x/photo.png" for b in out)
    # vector text became a text block (parser gave no text for that page)
    text = [b for b in out if b["type"] == "text" and b["page_idx"] == 1]
    assert text and "PLAN-ELEC-01" in text[0]["text"]
    # order: page 1 blocks sit between page 0 and page 2 blocks
    pages = [b["page_idx"] for b in out]
    assert pages == sorted(pages)


def test_vector_text_not_duplicated_when_parser_has_text(pdf, tmp_path):
    cfg = DrawingConfig(dpi=40, tile_px=500, max_tiles=2)
    content = [{"type": "text", "text": "OCR text of the plan", "page_idx": 1}]
    out = add_drawing_blocks(content, pdf, tmp_path / "o", cfg)
    texts = [b for b in out if b["type"] == "text"]
    assert len(texts) == 1


def test_disabled_and_failure_return_input(pdf, tmp_path):
    content = [{"type": "text", "text": "x", "page_idx": 0}]
    assert (
        add_drawing_blocks(content, pdf, tmp_path / "o", DrawingConfig(enabled=False))
        is content
    )
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    assert add_drawing_blocks(content, bad, tmp_path / "o") == content


def test_tile_grid_respects_max_tiles_and_covers_sheet():
    cfg = DrawingConfig(tile_px=1000, overlap_px=100, max_tiles=4)
    boxes = tile_grid(6000, 4000, cfg)
    assert len(boxes) <= 4
    assert max(b[2] for b in boxes) == 6000 and max(b[3] for b in boxes) == 4000
    assert min(b[0] for b in boxes) == 0 and min(b[1] for b in boxes) == 0
    assert len(tile_grid(500, 400, cfg)) == 1


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("DRAWING_DPI", "123")
    monkeypatch.setenv("ENABLE_DRAWING_PROCESSING", "false")
    cfg = DrawingConfig.from_env(max_tiles=3)
    assert cfg.dpi == 123 and cfg.enabled is False and cfg.max_tiles == 3


# --- vision prompt --------------------------------------------------------


def _png(path: Path, size=(64, 64)) -> Path:
    Image.new("RGB", size, "white").save(path)
    return path


class _FakeProcessor(ImageModalProcessor):
    def __init__(self):  # skip LightRAG wiring
        self.calls = []
        self.max_image_side = None

    async def modal_caption_func(self, prompt, image_data=None, system_prompt=None):
        self.calls.append((prompt, system_prompt))
        return json.dumps(
            {
                "detailed_description": "desc",
                "entity_info": {
                    "entity_name": "n",
                    "entity_type": "technical_drawing",
                    "summary": "s",
                },
            }
        )

    def _get_context_for_item(self, item_info):
        return "contexte voisin"


@pytest.mark.asyncio
async def test_drawing_tiles_use_the_drawing_prompt(tmp_path):
    proc = _FakeProcessor()
    img = _png(tmp_path / "t.png")
    await proc.generate_description_only(
        {
            "img_path": str(img),
            "image_kind": DRAWING_IMAGE_KIND,
            "image_caption": ["Plan technique, page 2"],
        },
        "image",
        item_info={"page_idx": 1},
    )
    prompt, system = proc.calls[0]
    assert "technical drawing" in prompt and "contexte voisin" in prompt
    assert "Plan technique, page 2" in prompt
    assert system == PROMPTS["DRAWING_ANALYSIS_SYSTEM"]

    await proc.generate_description_only(
        {"img_path": str(img)}, "image", item_info={"page_idx": 1}
    )
    assert proc.calls[1][1] == PROMPTS["IMAGE_ANALYSIS_SYSTEM"]
    assert "technical drawing" not in proc.calls[1][0]


# --- downscaling ----------------------------------------------------------


def test_encode_downscales_only_when_requested(tmp_path):
    big = _png(tmp_path / "big.png", (3000, 1500))
    raw = base64.b64decode(encode_image_to_base64(str(big)))
    assert Image.open(BytesIO(raw)).size == (3000, 1500)

    small = base64.b64decode(encode_image_to_base64(str(big), max_side=1000))
    assert Image.open(BytesIO(small)).size == (1000, 500)

    untouched = base64.b64decode(encode_image_to_base64(str(big), max_side=5000))
    assert untouched == big.read_bytes()


def test_image_processor_passes_max_side(tmp_path):
    proc = _FakeProcessor()
    proc.max_image_side = 800
    big = _png(tmp_path / "big.png", (2000, 2000))
    data = base64.b64decode(proc._encode_image_to_base64(str(big)))
    assert Image.open(BytesIO(data)).size == (800, 800)


# --- tiny image filter ----------------------------------------------------


def test_tiny_images_are_dropped_but_not_drawings(tmp_path):
    from raganything.processor import ProcessorMixin

    class P(ProcessorMixin):
        pass

    class L:
        def info(self, *a, **k):
            pass

    proc = P()
    proc.logger = L()
    proc.config = SimpleNamespace(min_image_pixels=1000)
    tiny = _png(tmp_path / "tiny.png", (10, 10))
    ok = _png(tmp_path / "ok.png", (100, 100))
    items = [
        {"type": "image", "img_path": str(tiny)},
        {"type": "image", "img_path": str(ok)},
        {"type": "image", "img_path": str(tiny), "image_kind": DRAWING_IMAGE_KIND},
        {"type": "table", "table_body": "x"},
    ]
    kept = proc._drop_tiny_images(items)
    assert [i.get("img_path") for i in kept] == [str(ok), str(tiny), None]
    proc.config = SimpleNamespace(min_image_pixels=0)
    assert proc._drop_tiny_images(items) == items
