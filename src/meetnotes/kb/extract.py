"""Text extraction: turn a source file into heading-aware blocks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md", ".txt", ".html", ".htm"}


class ExtractionError(Exception):
    """The file could not be read or has no extractable text."""


@dataclass(frozen=True, slots=True)
class Block:
    """A contiguous run of text under one heading breadcrumb."""

    text: str
    heading: str  # breadcrumb like "Pricing > Enterprise"; "" at top level
    page: int | None  # 1-based PDF page, None elsewhere
    start_char: int  # offset into the document's concatenated text


@dataclass(frozen=True, slots=True)
class ExtractedDoc:
    title: str
    blocks: list[Block]


def extract_file(path: Path) -> ExtractedDoc:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".docx":
        return _extract_docx(path)
    if suffix == ".md":
        return _extract_markdown(path.read_text(encoding="utf-8", errors="replace"), path.stem)
    if suffix in (".html", ".htm"):
        return _extract_text_blocks(
            _strip_html(path.read_text(encoding="utf-8", errors="replace")), path.stem
        )
    if suffix == ".txt":
        return _extract_text_blocks(path.read_text(encoding="utf-8", errors="replace"), path.stem)
    raise ExtractionError(f"Unsupported file type: {path.suffix}")


def extract_markdown_text(text: str, title: str) -> ExtractedDoc:
    """Public entry for already-in-memory markdown (past meeting notes)."""
    return _extract_markdown(text, title)


# -- per-format extractors ----------------------------------------------------


def _extract_pdf(path: Path) -> ExtractedDoc:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise ExtractionError(f"Could not read PDF: {exc}") from exc
    blocks: list[Block] = []
    offset = 0
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        blocks.append(Block(text=text, heading="", page=page_no, start_char=offset))
        offset += len(text) + 1
    if not blocks:
        raise ExtractionError("No extractable text in PDF")
    title = path.stem
    if reader.metadata and reader.metadata.title:
        title = str(reader.metadata.title)
    return ExtractedDoc(title=title, blocks=blocks)


def _extract_docx(path: Path) -> ExtractedDoc:
    import docx

    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise ExtractionError(f"Could not read .docx: {exc}") from exc
    blocks: list[Block] = []
    breadcrumb: list[str] = []
    offset = 0
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name if para.style is not None else "") or ""
        match = re.match(r"Heading (\d)", style)
        if match:
            level = int(match.group(1))
            breadcrumb = [*breadcrumb[: level - 1], text]
        else:
            blocks.append(
                Block(text=text, heading=" > ".join(breadcrumb), page=None, start_char=offset)
            )
        offset += len(text) + 1
    if not blocks:
        raise ExtractionError("No extractable text in .docx")
    return ExtractedDoc(title=path.stem, blocks=blocks)


def _extract_markdown(text: str, fallback_title: str) -> ExtractedDoc:
    blocks: list[Block] = []
    breadcrumb: list[str] = []
    title = fallback_title
    paragraph: list[str] = []
    offset = 0
    para_start = 0

    def flush() -> None:
        nonlocal paragraph
        joined = "\n".join(paragraph).strip()
        if joined:
            blocks.append(
                Block(
                    text=joined,
                    heading=" > ".join(breadcrumb),
                    page=None,
                    start_char=para_start,
                )
            )
        paragraph = []

    for line in text.split("\n"):
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            flush()
            level = len(heading.group(1))
            name = heading.group(2).strip()
            if level == 1 and title == fallback_title:
                title = name
            breadcrumb = [*breadcrumb[: level - 1], name]
        elif line.strip() == "":
            flush()
        else:
            if not paragraph:
                para_start = offset
            paragraph.append(line)
        offset += len(line) + 1
    flush()
    if not blocks:
        raise ExtractionError("No extractable text")
    return ExtractedDoc(title=title, blocks=blocks)


def _extract_text_blocks(text: str, title: str) -> ExtractedDoc:
    blocks: list[Block] = []
    offset = 0
    for raw in re.split(r"\n\s*\n", text):
        para = raw.strip()
        if para:
            blocks.append(Block(text=para, heading="", page=None, start_char=offset))
        offset += len(raw) + 2
    if not blocks:
        raise ExtractionError("No extractable text")
    return ExtractedDoc(title=title, blocks=blocks)


class _HTMLTextParser(HTMLParser):
    _SKIP = frozenset({"script", "style", "head", "noscript"})
    _BREAK = frozenset(
        {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section"}
    )

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BREAK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def _strip_html(html: str) -> str:
    parser = _HTMLTextParser()
    parser.feed(html)
    return "".join(parser.parts)
