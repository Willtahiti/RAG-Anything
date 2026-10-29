"""Window-by-window parsing of large PDFs: offsets, resume, bisect, stitching."""

import json
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from raganything.parser import MineruParser, Parser
from raganything.pdf_windowing import (
    parse_pdf_in_windows,
    plan_windows,
    stitch_cross_page_tables,
    window_settings,
)


def _make_pdf(path: Path, pages: int) -> Path:
    c = canvas.Canvas(str(path))
    for i in range(pages):
        c.drawString(72, 720, f"page {i}")
        c.showPage()
    c.save()
    return path


def _text_per_page(window_pdf: Path, out_dir: Path):
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(window_pdf))
    blocks = []
    for i in range(len(pdf)):
        text = pdf[i].get_textpage().get_text_range().strip()
        blocks.append({"type": "text", "text": text, "page_idx": i})
    pdf.close()
    return blocks


def test_plan_windows():
    assert plan_windows(0, 10) == []
    assert plan_windows(25, 10) == [(0, 10), (10, 20), (20, 25)]
    with pytest.raises(ValueError):
        plan_windows(5, 0)


def test_windows_merge_with_absolute_page_idx(tmp_path):
    pdf = _make_pdf(tmp_path / "doc.pdf", 25)
    seen = []

    def parse_window(window_pdf, out_dir):
        seen.append(window_pdf.name)
        return _text_per_page(window_pdf, out_dir)

    blocks = parse_pdf_in_windows(parse_window, pdf, tmp_path / "work", window_pages=10)

    assert [b["page_idx"] for b in blocks] == list(range(25))
    assert [b["text"] for b in blocks] == [f"page {i}" for i in range(25)]
    assert len(seen) == 3
    # window PDFs are removed once checkpointed
    assert not list((tmp_path / "work" / "windows").glob("*.pdf"))


def test_resume_skips_finished_windows(tmp_path):
    pdf = _make_pdf(tmp_path / "doc.pdf", 25)
    calls = []

    def flaky(window_pdf, out_dir):
        calls.append(window_pdf.name)
        if len(calls) == 3:
            raise FileNotFoundError("mineru vanished")  # fatal: no bisecting
        return _text_per_page(window_pdf, out_dir)

    with pytest.raises(FileNotFoundError):
        parse_pdf_in_windows(flaky, pdf, tmp_path / "work", window_pages=10)
    assert len(calls) == 3

    calls.clear()

    def ok(window_pdf, out_dir):
        calls.append(window_pdf.name)
        return _text_per_page(window_pdf, out_dir)

    blocks = parse_pdf_in_windows(ok, pdf, tmp_path / "work", window_pages=10)
    assert len(calls) == 1  # only the last window is redone
    assert [b["page_idx"] for b in blocks] == list(range(25))


def test_changed_file_or_options_invalidate_checkpoints(tmp_path):
    pdf = _make_pdf(tmp_path / "doc.pdf", 12)
    calls = []

    def parse(window_pdf, out_dir):
        calls.append(1)
        return _text_per_page(window_pdf, out_dir)

    parse_pdf_in_windows(parse, pdf, tmp_path / "w", window_pages=6)
    parse_pdf_in_windows(parse, pdf, tmp_path / "w", window_pages=6)
    assert len(calls) == 2
    parse_pdf_in_windows(
        parse, pdf, tmp_path / "w", window_pages=6, fingerprint_extra={"lang": "fr"}
    )
    assert len(calls) == 4
    parse_pdf_in_windows(parse, pdf, tmp_path / "w", window_pages=6, resume=False)
    assert len(calls) == 6


def test_bad_page_is_isolated_and_skipped(tmp_path):
    pdf = _make_pdf(tmp_path / "doc.pdf", 8)

    def parse(window_pdf, out_dir):
        blocks = _text_per_page(window_pdf, out_dir)
        if any(b["text"] == "page 5" for b in blocks):
            raise RuntimeError("corrupt page")
        return blocks

    blocks = parse_pdf_in_windows(parse, pdf, tmp_path / "w", window_pages=8, retries=0)
    texts = {b["page_idx"]: b for b in blocks}
    assert sorted(texts) == list(range(8))
    assert texts[5].get("parse_error") is True
    assert "Page 6" in texts[5]["text"]
    assert texts[4]["text"] == "page 4" and texts[6]["text"] == "page 6"


def test_too_many_failures_abort(tmp_path):
    pdf = _make_pdf(tmp_path / "doc.pdf", 6)

    def always_fail(window_pdf, out_dir):
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="pages failed"):
        parse_pdf_in_windows(
            always_fail,
            pdf,
            tmp_path / "w",
            window_pages=6,
            retries=0,
            max_failed_pages=2,
        )


def test_progress_callback(tmp_path):
    pdf = _make_pdf(tmp_path / "doc.pdf", 10)
    events = []
    parse_pdf_in_windows(
        _text_per_page,
        pdf,
        tmp_path / "w",
        window_pages=4,
        progress=lambda done, total, rng: events.append((done, total, rng)),
    )
    assert events == [(4, 10, (0, 4)), (8, 10, (4, 8)), (10, 10, (8, 10))]


# --- table stitching -----------------------------------------------------

_T1 = "<table><tr><td>Ref</td><td>Qté</td></tr><tr><td>A</td><td>1</td></tr></table>"
_T2_REPEAT = (
    "<table><tr><td>Ref</td><td>Qté</td></tr><tr><td>B</td><td>2</td></tr></table>"
)


def test_stitch_html_tables_with_repeated_header():
    blocks = [
        {"type": "text", "text": "intro", "page_idx": 0},
        {"type": "table", "table_body": _T1, "page_idx": 0},
        {"type": "table", "table_body": _T2_REPEAT, "page_idx": 1},
        {"type": "text", "text": "after", "page_idx": 1},
    ]
    out = stitch_cross_page_tables(blocks)
    assert [b["type"] for b in out] == ["text", "table", "text"]
    body = out[1]["table_body"]
    assert body.count("<tr>") == 3 and body.count("<table") == 1
    assert "<td>B</td>" in body
    assert "merged_end_page" not in out[1]
    assert blocks[1]["table_body"] == _T1  # input untouched


def test_stitch_markdown_tables_and_chain_of_three():
    md = lambda rows: "| a | b |\n| --- | --- |\n" + "\n".join(rows)  # noqa: E731
    blocks = [
        {"type": "table", "table_body": md(["| 1 | 2 |"]), "page_idx": 3},
        {"type": "table", "table_body": md(["| 3 | 4 |"]), "page_idx": 4},
        {"type": "table", "table_body": md(["| 5 | 6 |"]), "page_idx": 5},
    ]
    out = stitch_cross_page_tables(blocks)
    assert len(out) == 1
    # one header + one separator + three data rows
    assert len(out[0]["table_body"].splitlines()) == 5
    assert "| 5 | 6 |" in out[0]["table_body"]
    assert out[0]["page_idx"] == 3


def test_no_stitch_when_columns_differ_caption_or_gap():
    wide = "<table><tr><td>a</td><td>b</td><td>c</td></tr></table>"
    cases = [
        [
            {"type": "table", "table_body": _T1, "page_idx": 0},
            {"type": "table", "table_body": wide, "page_idx": 1},
        ],
        [
            {"type": "table", "table_body": _T1, "page_idx": 0},
            {
                "type": "table",
                "table_body": _T2_REPEAT,
                "table_caption": ["Tableau 2"],
                "page_idx": 1,
            },
        ],
        [
            {"type": "table", "table_body": _T1, "page_idx": 0},
            {"type": "table", "table_body": _T2_REPEAT, "page_idx": 2},
        ],
        [
            {"type": "table", "table_body": _T1, "page_idx": 0},
            {"type": "text", "text": "x", "page_idx": 1},
            {"type": "table", "table_body": _T2_REPEAT, "page_idx": 1},
        ],
    ]
    for blocks in cases:
        assert len(stitch_cross_page_tables(blocks)) == len(blocks)


# --- integration with the Parser base -------------------------------------


class _RecordingParser(MineruParser):
    """MinerU parser whose backend is replaced by a per-page text reader."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def parse_pdf(self, pdf_path, output_dir=None, method="auto", lang=None, **kwargs):
        windowed = self._maybe_parse_windowed(
            Path(pdf_path), output_dir, method, lang, kwargs
        )
        if windowed is not None:
            return windowed
        self.calls.append((Path(pdf_path).name, dict(kwargs)))
        return _text_per_page(Path(pdf_path), Path(output_dir or "."))


def test_small_pdf_is_parsed_whole(tmp_path):
    pdf = _make_pdf(tmp_path / "small.pdf", 5)
    parser = _RecordingParser()
    blocks = parser.parse_pdf(
        pdf, str(tmp_path / "o"), window_pages=2, window_threshold=10
    )
    assert len(blocks) == 5 and len(parser.calls) == 1
    assert "window_pages" not in parser.calls[0][1]


def test_large_pdf_goes_through_windows(tmp_path):
    pdf = _make_pdf(tmp_path / "big.pdf", 9)
    parser = _RecordingParser()
    blocks = parser.parse_pdf(
        pdf, str(tmp_path / "o"), window_pages=4, window_threshold=5, lang="fr"
    )
    assert [b["page_idx"] for b in blocks] == list(range(9))
    assert len(parser.calls) == 3
    for _, kwargs in parser.calls:
        assert "_in_window" not in kwargs and "window_pages" not in kwargs


def test_window_settings_pops_options_and_reads_env(monkeypatch):
    monkeypatch.setenv("PDF_WINDOW_PAGES", "50")
    kwargs = {"window_threshold": 7, "lang": "en", "window_progress": None}
    settings = window_settings(kwargs)
    assert kwargs == {"lang": "en"}
    assert settings["pages"] == 50 and settings["threshold"] == 7
    assert settings["resume"] is True and settings["timeout"] is None


def test_paddle_does_not_advertise_windowing():
    from raganything.parser import PaddleOCRParser

    assert Parser.SUPPORTS_PDF_WINDOWING is False
    assert PaddleOCRParser.SUPPORTS_PDF_WINDOWING is False
    assert MineruParser.SUPPORTS_PDF_WINDOWING is True


def test_checkpoint_json_is_valid(tmp_path):
    pdf = _make_pdf(tmp_path / "doc.pdf", 4)
    parse_pdf_in_windows(_text_per_page, pdf, tmp_path / "w", window_pages=2)
    files = sorted((tmp_path / "w" / "windows").glob("w*.json"))
    assert [f.name for f in files] == ["w000000_000002.json", "w000002_000004.json"]
    assert json.loads(files[1].read_text())[0]["page_idx"] == 2
