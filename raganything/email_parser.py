"""
Outlook ``.msg`` ingestion.

Turns a ``.msg`` message into a MinerU-style content list: one text block for
the headers, one for the body, and the content of every attachment (parsed
through a caller-supplied handler, so the active document parser is reused).
Nested ``.msg`` attachments are followed up to ``max_depth`` levels.

``extract-msg`` is imported lazily (``pip install raganything[email]``).
The message objects are only used through a handful of attributes, so tests
can stand in a duck-typed fake.
"""

from __future__ import annotations

import hashlib
import html as html_lib
import logging
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

EMAIL_FILE_EXTENSIONS = frozenset({".msg"})

DEFAULT_MAX_DEPTH = 3
DEFAULT_MAX_ATTACHMENT_BYTES = 200 * 1024 * 1024
# Signature logos and tracking pixels are not worth a vision-model call.
DEFAULT_MIN_IMAGE_BYTES = 8 * 1024

_IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
)
_UNSAFE_NAME_CHARS = re.compile(r"[^\w.\-() ]+", re.UNICODE)

# ``handler(path, output_dir)`` returns the attachment's content blocks, or
# ``None`` when the format is not supported.
AttachmentHandler = Callable[[Path, Path], Optional[List[Dict[str, Any]]]]


class _HTMLToText(HTMLParser):
    _BLOCK_TAGS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "table"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self._skip += 1
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")
        elif tag in ("td", "th"):
            self._parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self._skip = max(0, self._skip - 1)
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def html_to_text(html: str) -> str:
    """Strip tags from an HTML mail body, keeping paragraph/row breaks."""
    parser = _HTMLToText()
    try:
        parser.feed(html)
        parser.close()
        text = parser.text()
    except Exception:  # malformed HTML: fall back to a crude strip
        text = html_lib.unescape(re.sub(r"<[^>]+>", " ", html))
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _decode_bytes(raw: bytes) -> str:
    for encoding in ("utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def safe_attachment_name(name: Optional[str], index: int) -> str:
    """Return a file name that cannot escape the attachment directory."""
    base = (name or "").replace("\\", "/").split("/")[-1]
    base = _UNSAFE_NAME_CHARS.sub("_", base).strip(" .")
    if not base or base in (".", ".."):
        base = "attachment"
    stem, dot, suffix = base.rpartition(".")
    if dot and stem:
        base = f"{stem[:100]}.{suffix[:10]}"
    else:
        base = base[:110]
    return f"{index:02d}_{base}"


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return _decode_bytes(value)
    return str(value).strip()


def _attachment_name(att: Any) -> Optional[str]:
    for attr in ("longFilename", "shortFilename", "name", "displayName"):
        value = getattr(att, attr, None)
        if value:
            return _text(value)
    return None


def _is_nested_message(att: Any) -> bool:
    type_name = getattr(getattr(att, "type", None), "name", None)
    if type_name == "MSG":
        return True
    return hasattr(getattr(att, "data", None), "attachments")


def _message_body(message: Any) -> str:
    body = _text(getattr(message, "body", None))
    if body:
        return re.sub(r"\r\n?", "\n", body).strip()
    html = getattr(message, "htmlBody", None)
    if html:
        return html_to_text(_text(html))
    return ""


def _headers_block(message: Any, attachment_names: List[str]) -> str:
    lines = []
    for label, attr in (
        ("De", "sender"),
        ("À", "to"),
        ("Cc", "cc"),
        ("Date", "date"),
        ("Objet", "subject"),
    ):
        value = _text(getattr(message, attr, None))
        if value:
            lines.append(f"{label} : {value}")
    if attachment_names:
        lines.append("Pièces jointes : " + ", ".join(attachment_names))
    return "\n".join(lines)


def _shift_pages(blocks: List[Dict[str, Any]], offset: int) -> int:
    """Offset ``page_idx`` of ``blocks`` in place; return the pages used."""
    highest = -1
    for block in blocks:
        page = int(block.get("page_idx", 0) or 0) + offset
        block["page_idx"] = page
        highest = max(highest, page)
    return max(0, highest - offset + 1) if blocks else 0


def _convert_message(
    message: Any,
    output_dir: Path,
    attachment_handler: Optional[AttachmentHandler],
    *,
    depth: int,
    max_depth: int,
    max_attachment_bytes: int,
    min_image_bytes: int,
    page_offset: int,
) -> List[Dict[str, Any]]:
    attachments = list(getattr(message, "attachments", None) or [])
    subject = _text(getattr(message, "subject", None)) or "(sans objet)"
    names = [
        _attachment_name(a) or f"attachment_{i}" for i, a in enumerate(attachments)
    ]

    blocks: List[Dict[str, Any]] = []
    header = _headers_block(message, names)
    if header:
        blocks.append(
            {"type": "text", "text": header, "text_level": 1, "page_idx": page_offset}
        )
    body = _message_body(message)
    if body:
        blocks.append({"type": "text", "text": body, "page_idx": page_offset})

    next_page = page_offset + 1
    seen_hashes: set = set()
    attach_dir = output_dir / "attachments"

    for index, att in enumerate(attachments):
        name = names[index]
        label = f"[Pièce jointe : {name} — mail : {subject}]"
        try:
            if _is_nested_message(att):
                if depth >= max_depth:
                    logger.warning("Nested .msg depth limit reached for %s", name)
                    continue
                nested = getattr(att, "data", None)
                nested_blocks = _convert_message(
                    nested,
                    output_dir / f"nested_{index:02d}",
                    attachment_handler,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_attachment_bytes=max_attachment_bytes,
                    min_image_bytes=min_image_bytes,
                    page_offset=next_page,
                )
                if nested_blocks:
                    blocks.append(
                        {"type": "text", "text": label, "page_idx": next_page}
                    )
                    blocks.extend(nested_blocks)
                    next_page = (
                        max(int(b.get("page_idx", 0)) for b in nested_blocks) + 1
                    )
                continue

            data = getattr(att, "data", None)
            if not isinstance(data, (bytes, bytearray)) or not data:
                continue
            if len(data) > max_attachment_bytes:
                blocks.append(
                    {
                        "type": "text",
                        "text": f"{label} ignorée : taille {len(data)} octets "
                        "au-delà de la limite.",
                        "page_idx": next_page,
                    }
                )
                next_page += 1
                continue

            digest = hashlib.sha256(data).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)

            suffix = Path(name).suffix.lower()
            if suffix in _IMAGE_SUFFIXES and len(data) < min_image_bytes:
                continue

            attach_dir.mkdir(parents=True, exist_ok=True)
            target = attach_dir / safe_attachment_name(name, index)
            target.write_bytes(bytes(data))

            if suffix in _IMAGE_SUFFIXES:
                blocks.append(
                    {
                        "type": "image",
                        "img_path": str(target.resolve()),
                        "image_caption": [f"Pièce jointe {name} du mail « {subject} »"],
                        "image_footnote": [],
                        "page_idx": next_page,
                    }
                )
                next_page += 1
                continue

            sub_blocks = (
                attachment_handler(target, attach_dir / f"{index:02d}_parsed")
                if attachment_handler
                else None
            )
            if sub_blocks is None:
                blocks.append(
                    {
                        "type": "text",
                        "text": f"{label} non analysée (format non supporté).",
                        "page_idx": next_page,
                    }
                )
                next_page += 1
                continue
            if not sub_blocks:
                continue
            blocks.append({"type": "text", "text": label, "page_idx": next_page})
            used = _shift_pages(sub_blocks, next_page)
            blocks.extend(sub_blocks)
            next_page += max(used, 1)
        except Exception as exc:  # one bad attachment must not lose the mail
            logger.warning("Attachment %s of mail %r failed: %s", name, subject, exc)
            blocks.append(
                {
                    "type": "text",
                    "text": f"{label} non analysée (erreur : {exc}).",
                    "page_idx": next_page,
                }
            )
            next_page += 1

    return blocks


def parse_msg(
    msg_path: Union[str, Path],
    output_dir: Union[str, Path],
    attachment_handler: Optional[AttachmentHandler] = None,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_attachment_bytes: int = DEFAULT_MAX_ATTACHMENT_BYTES,
    min_image_bytes: int = DEFAULT_MIN_IMAGE_BYTES,
) -> List[Dict[str, Any]]:
    """Parse an Outlook ``.msg`` file into content-list blocks."""
    msg_path = Path(msg_path)
    if not msg_path.exists():
        raise FileNotFoundError(f"Email file does not exist: {msg_path}")
    if msg_path.suffix.lower() not in EMAIL_FILE_EXTENSIONS:
        raise ValueError(f"Unsupported email format: {msg_path.suffix}")
    try:
        import extract_msg
    except ImportError as exc:
        raise ImportError(
            "Reading .msg files requires extract-msg: "
            "pip install 'raganything[email]'"
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    message = extract_msg.openMsg(str(msg_path))
    try:
        return _convert_message(
            message,
            output_dir,
            attachment_handler,
            depth=0,
            max_depth=max_depth,
            max_attachment_bytes=max_attachment_bytes,
            min_image_bytes=min_image_bytes,
            page_offset=0,
        )
    finally:
        close = getattr(message, "close", None)
        if close:
            close()
