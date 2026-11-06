"""Read a PDF for the deterministic checks.

Claude receives the original file. The text pulled here is used to verify Claude's quotes and to
scan for hostile content, so it has to include what a human reader would never see.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber

from loan_triage.domain.enums import FlagCode
from loan_triage.domain.models import IntegrityFinding

_MIN_VISIBLE_FONT_SIZE = 2.0
_WHITE_COLORS = ((1,), (1, 1, 1), (0, 0, 0, 0))
_PDF_MAGIC = b"%PDF-"


class InvalidDocument(ValueError):
    """The upload is not a readable PDF."""


@dataclass(frozen=True)
class ParsedPdf:
    sha256: str
    pages: list[str]
    findings: list[IntegrityFinding] = field(default_factory=list)

    @property
    def has_text_layer(self) -> bool:
        return any(page.strip() for page in self.pages)

    @property
    def content_sha256(self) -> str:
        """Fingerprint of the words, not the bytes: re-saving a PDF changes only the latter."""
        if not self.has_text_layer:
            return self.sha256
        text = re.sub(r"\s+", " ", " ".join(self.pages)).strip().casefold()
        return hashlib.sha256(text.encode()).hexdigest()


def validate_pdf_bytes(data: bytes, *, max_bytes: int) -> None:
    if len(data) > max_bytes:
        raise InvalidDocument(f"file exceeds the {max_bytes // (1024 * 1024)} MB upload limit")
    if not data.startswith(_PDF_MAGIC):
        raise InvalidDocument("file is not a PDF")


def _is_white(color: Any) -> bool:
    if color is None:
        return False
    values = tuple(color) if isinstance(color, list | tuple) else (color,)
    return any(
        len(values) == len(white)
        and all(abs(a - b) < 0.02 for a, b in zip(values, white, strict=True))
        for white in _WHITE_COLORS
    )


def _on_dark_fill(char: dict[str, Any], rects: list[dict[str, Any]]) -> bool:
    """White text on a coloured band is ordinary design, not concealment."""
    for rect in rects:
        if not rect.get("fill") or _is_white(rect.get("non_stroking_color")):
            continue
        inside = (
            rect["x0"] <= char["x0"]
            and char["x1"] <= rect["x1"]
            and rect["top"] <= char["top"]
            and char["bottom"] <= rect["bottom"]
        )
        if inside:
            return True
    return False


def _hidden_text(page: Any) -> str:
    rects = page.rects
    hidden = [
        char["text"]
        for char in page.chars
        if char["size"] < _MIN_VISIBLE_FONT_SIZE
        or (_is_white(char.get("non_stroking_color")) and not _on_dark_fill(char, rects))
    ]
    return "".join(hidden).strip()


def parse_pdf(data: bytes) -> ParsedPdf:
    pages: list[str] = []
    findings: list[IntegrityFinding] = []
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for number, page in enumerate(pdf.pages, start=1):
                pages.append(page.extract_text() or "")
                hidden = _hidden_text(page)
                if hidden:
                    findings.append(
                        IntegrityFinding(
                            code=FlagCode.HIDDEN_TEXT_DETECTED,
                            detail="Text is present in the file but invisible on the page.",
                            page=number,
                            excerpt=hidden[:240],
                        )
                    )
    except Exception as exc:  # pdfminer raises many unrelated types on malformed input
        raise InvalidDocument(f"could not read PDF: {exc}") from exc
    if not pages:
        raise InvalidDocument("PDF has no pages")
    return ParsedPdf(sha256=hashlib.sha256(data).hexdigest(), pages=pages, findings=findings)


def load_pdf(path: Path) -> ParsedPdf:
    return parse_pdf(path.read_bytes())
