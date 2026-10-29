"""Outlook .msg ingestion: blocks, attachments, routing."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from raganything.batch_parser import BatchParser
from raganything.config import RAGAnythingConfig
from raganything.email_parser import html_to_text, parse_msg, safe_attachment_name
from raganything.parser import MineruParser, Parser
from raganything.processor import ProcessorMixin


class FakeAttachment:
    def __init__(self, name, data, nested=False):
        self.longFilename = name
        self.shortFilename = name
        self.data = data
        self.type = SimpleNamespace(name="MSG" if nested else "DATA")


class FakeMessage:
    def __init__(self, subject="Sujet", body="Bonjour", attachments=(), html=None):
        self.sender = "alice@example.com"
        self.to = "bob@example.com"
        self.cc = ""
        self.date = "2026-01-02"
        self.subject = subject
        self.body = body
        self.htmlBody = html
        self.attachments = list(attachments)
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def fake_extract_msg(monkeypatch):
    messages = {}

    def open_msg(path):
        return messages[Path(path).name]

    monkeypatch.setitem(sys.modules, "extract_msg", SimpleNamespace(openMsg=open_msg))
    return messages


def _msg_file(tmp_path, name="mail.msg"):
    path = tmp_path / name
    path.write_bytes(b"x")
    return path


def test_headers_body_and_pdf_attachment(tmp_path, fake_extract_msg):
    fake_extract_msg["mail.msg"] = FakeMessage(
        attachments=[FakeAttachment("plan.pdf", b"%PDF-1.4 data")]
    )
    seen = []

    def handler(path, out_dir):
        seen.append(path)
        return [
            {"type": "text", "text": "p0", "page_idx": 0},
            {"type": "text", "text": "p1", "page_idx": 1},
        ]

    blocks = parse_msg(_msg_file(tmp_path), tmp_path / "out", handler)

    assert blocks[0]["text"].startswith("De : alice@example.com")
    assert "Pièces jointes : plan.pdf" in blocks[0]["text"]
    assert blocks[1] == {"type": "text", "text": "Bonjour", "page_idx": 0}
    assert "plan.pdf" in blocks[2]["text"] and blocks[2]["page_idx"] == 1
    assert [b["page_idx"] for b in blocks[3:]] == [1, 2]
    assert seen[0].read_bytes() == b"%PDF-1.4 data"
    assert seen[0].parent == (tmp_path / "out" / "attachments")


def test_second_attachment_pages_follow_first(tmp_path, fake_extract_msg):
    fake_extract_msg["mail.msg"] = FakeMessage(
        attachments=[
            FakeAttachment("a.pdf", b"a" * 10),
            FakeAttachment("b.pdf", b"b" * 10),
        ]
    )
    handler = lambda p, o: [  # noqa: E731
        {"type": "text", "text": p.name, "page_idx": 0},
        {"type": "text", "text": p.name, "page_idx": 2},
    ]
    blocks = parse_msg(_msg_file(tmp_path), tmp_path / "out", handler)
    pages = {}
    for b in blocks:
        if b["text"].endswith(".pdf"):
            pages.setdefault(b["text"], b["page_idx"])
    assert pages["00_a.pdf"] == 1
    assert pages["01_b.pdf"] == 4  # a.pdf used pages 1..3


def test_html_body_fallback(tmp_path, fake_extract_msg):
    fake_extract_msg["mail.msg"] = FakeMessage(
        body="",
        html=b"<html><body><p>Hello&nbsp;<b>you</b></p><p>Bye</p></body></html>",
    )
    blocks = parse_msg(_msg_file(tmp_path), tmp_path / "out")
    assert blocks[1]["text"] == "Hello you\n\nBye" or blocks[1]["text"].startswith(
        "Hello you"
    )


def test_image_attachment_becomes_image_block_and_small_ones_skipped(
    tmp_path, fake_extract_msg
):
    fake_extract_msg["mail.msg"] = FakeMessage(
        attachments=[
            FakeAttachment("schema.png", b"\x89PNG" + b"0" * 20000),
            FakeAttachment("logo.png", b"\x89PNG" + b"1" * 50),
        ]
    )
    blocks = parse_msg(_msg_file(tmp_path), tmp_path / "out")
    images = [b for b in blocks if b["type"] == "image"]
    assert len(images) == 1
    assert Path(images[0]["img_path"]).is_absolute()
    assert Path(images[0]["img_path"]).exists()


def test_duplicate_attachments_parsed_once(tmp_path, fake_extract_msg):
    fake_extract_msg["mail.msg"] = FakeMessage(
        attachments=[FakeAttachment("a.pdf", b"same"), FakeAttachment("b.pdf", b"same")]
    )
    calls = []
    parse_msg(
        _msg_file(tmp_path),
        tmp_path / "out",
        lambda p, o: calls.append(p) or [{"type": "text", "text": "x", "page_idx": 0}],
    )
    assert len(calls) == 1


def test_unsupported_attachment_is_noted(tmp_path, fake_extract_msg):
    fake_extract_msg["mail.msg"] = FakeMessage(
        attachments=[FakeAttachment("data.bin", b"\x00\x01")]
    )
    blocks = parse_msg(_msg_file(tmp_path), tmp_path / "out", lambda p, o: None)
    assert any("non analysée" in b["text"] for b in blocks)


def test_attachment_failure_does_not_lose_mail(tmp_path, fake_extract_msg):
    fake_extract_msg["mail.msg"] = FakeMessage(
        attachments=[FakeAttachment("a.pdf", b"boom")]
    )

    def handler(path, out_dir):
        raise RuntimeError("parser exploded")

    blocks = parse_msg(_msg_file(tmp_path), tmp_path / "out", handler)
    assert blocks[1]["text"] == "Bonjour"
    assert any("parser exploded" in b["text"] for b in blocks)


def test_nested_msg_respects_depth(tmp_path, fake_extract_msg):
    inner = FakeMessage(subject="Inner", body="inner body")
    middle = FakeMessage(
        subject="Middle",
        body="middle body",
        attachments=[FakeAttachment("i.msg", inner, True)],
    )
    fake_extract_msg["mail.msg"] = FakeMessage(
        attachments=[FakeAttachment("m.msg", middle, True)]
    )
    blocks = parse_msg(_msg_file(tmp_path), tmp_path / "out")
    texts = [b["text"] for b in blocks]
    assert "middle body" in texts and "inner body" in texts

    blocks = parse_msg(_msg_file(tmp_path), tmp_path / "out", max_depth=1)
    texts = [b["text"] for b in blocks]
    assert "middle body" in texts and "inner body" not in texts


def test_message_closed(tmp_path, fake_extract_msg):
    message = FakeMessage()
    fake_extract_msg["mail.msg"] = message
    parse_msg(_msg_file(tmp_path), tmp_path / "out")
    assert message.closed


@pytest.mark.parametrize(
    "name",
    ["../../etc/passwd", "..\\..\\evil.exe", "/abs/path.pdf", "", None, "a\x00b.pdf"],
)
def test_attachment_names_cannot_escape(name):
    safe = safe_attachment_name(name, 3)
    assert "/" not in safe and "\\" not in safe and ".." not in safe.split(".")[0]
    assert safe.startswith("03_")


def test_malicious_name_is_written_inside_attachment_dir(tmp_path, fake_extract_msg):
    fake_extract_msg["mail.msg"] = FakeMessage(
        attachments=[FakeAttachment("../../escape.pdf", b"data")]
    )
    parse_msg(
        _msg_file(tmp_path),
        tmp_path / "out",
        lambda p, o: [{"type": "text", "text": "x", "page_idx": 0}],
    )
    assert not (tmp_path / "escape.pdf").exists()
    assert list((tmp_path / "out" / "attachments").iterdir())


def test_missing_extract_msg_gives_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "extract_msg", None)
    with pytest.raises(ImportError, match="raganything\\[email\\]"):
        parse_msg(_msg_file(tmp_path), tmp_path / "out")


def test_rejects_wrong_extension(tmp_path):
    bad = tmp_path / "x.eml"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError):
        parse_msg(bad, tmp_path)


def test_html_to_text_strips_scripts():
    assert html_to_text("<style>a{}</style><p>Hi</p><script>x()</script>") == "Hi"


# --- routing -------------------------------------------------------------


def test_parser_attachments_dispatch(tmp_path, fake_extract_msg, monkeypatch):
    fake_extract_msg["mail.msg"] = FakeMessage(
        attachments=[
            FakeAttachment("doc.pdf", b"pdf"),
            FakeAttachment("sheet.xlsx", b"xlsx"),
            FakeAttachment("note.txt", b"txt"),
        ]
    )
    parser = MineruParser()
    calls = []
    monkeypatch.setattr(
        parser, "parse_pdf", lambda **kw: calls.append("pdf") or [_blk("pdf")]
    )
    monkeypatch.setattr(
        parser, "parse_office_doc", lambda **kw: calls.append("office") or [_blk("o")]
    )
    monkeypatch.setattr(
        parser,
        "parse_text_file",
        lambda *a, **kw: calls.append("text") or [_blk("t")],
    )
    blocks = parser.parse_document(_msg_file(tmp_path), output_dir=str(tmp_path / "o"))
    assert calls == ["pdf", "office", "text"]
    assert blocks[0]["text"].startswith("De :")


def _blk(text):
    return {"type": "text", "text": text, "page_idx": 0}


def test_email_formats_in_supported_extensions():
    assert ".msg" in Parser.EMAIL_FORMATS
    assert (
        ".msg"
        in BatchParser(
            parser_type="mineru", skip_installation_check=True
        ).get_supported_extensions()
    )
    assert ".msg" in RAGAnythingConfig().supported_file_extensions


class _Logger:
    def debug(self, *a, **k):
        pass

    info = warning = error = debug


class _Cache:
    async def get_by_id(self, key):
        return None

    async def upsert(self, entries):
        pass

    async def index_done_callback(self):
        pass


class _P(ProcessorMixin):
    pass


@pytest.mark.asyncio
async def test_processor_routes_msg_to_parse_email(tmp_path):
    seen = {}

    class EmailParser:
        def parse_email(self, email_path, output_dir=None, method="auto", **kw):
            seen.update(path=Path(email_path), method=method, kw=kw)
            return [{"type": "text", "text": "mail", "page_idx": 0}]

    proc = _P()
    proc.config = SimpleNamespace(
        parser="t",
        parser_output_dir=str(tmp_path / "o"),
        parse_method="auto",
        display_content_stats=False,
        use_full_path=False,
    )
    proc.logger = _Logger()
    proc.parse_cache = _Cache()
    proc.doc_parser = EmailParser()
    doc = _msg_file(tmp_path)
    content, _ = await proc.parse_document(str(doc), lang="fr")
    assert content[0]["text"] == "mail"
    assert seen["path"] == doc and seen["kw"] == {"lang": "fr"}
