"""Render synthetic documents to PDF."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pdfplumber
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from loan_triage.domain.enums import DocumentType
from loan_triage.synth.layout import LABELS, TITLES

if TYPE_CHECKING:
    from loan_triage.synth.generator import SyntheticDocument

_WIDTH, _HEIGHT = LETTER
_MARGIN = 54
_INK = HexColor("#1F2933")
_MUTED = HexColor("#616E7C")
_RULE = HexColor("#D5DAE0")
_ACCENTS = {
    DocumentType.LOAN_APPLICATION: HexColor("#1E3A5F"),
    DocumentType.PAYSTUB: HexColor("#22543D"),
    DocumentType.W2: HexColor("#4A3B6B"),
    DocumentType.BANK_STATEMENT: HexColor("#0F4C5C"),
    DocumentType.CREDIT_REPORT: HexColor("#7B341E"),
}
_ROW_HEIGHT = 24


def render_pdf(document: SyntheticDocument) -> bytes:
    buffer = io.BytesIO()
    canvas = Canvas(buffer, pagesize=LETTER)
    canvas.setTitle(TITLES[document.document_type])

    canvas.setFillColor(_ACCENTS[document.document_type])
    canvas.rect(0, _HEIGHT - 84, _WIDTH, 84, fill=1, stroke=0)
    canvas.setFillColor(white)
    canvas.setFont("Helvetica-Bold", 17)
    canvas.drawString(_MARGIN, _HEIGHT - 46, TITLES[document.document_type])
    canvas.setFont("Helvetica", 9.5)
    canvas.drawString(_MARGIN, _HEIGHT - 64, document.issuer)

    y = _HEIGHT - 124
    labels = LABELS[document.document_type]
    for name, label in labels.items():
        value = document.fields.get(name)
        if value is None:
            continue
        canvas.setFillColor(_MUTED)
        canvas.setFont("Helvetica", 10)
        canvas.drawString(_MARGIN, y, label)
        canvas.setFillColor(_INK)
        canvas.setFont("Helvetica-Bold", 10.5)
        canvas.drawRightString(_WIDTH - _MARGIN, y, value)
        canvas.setStrokeColor(_RULE)
        canvas.setLineWidth(0.5)
        canvas.line(_MARGIN, y - 8, _WIDTH - _MARGIN, y - 8)
        y -= _ROW_HEIGHT

    if document.visible_note:
        canvas.setFillColor(_MUTED)
        canvas.setFont("Helvetica", 8)
        text = canvas.beginText(_MARGIN, y - 24)
        for line in _wrap(document.visible_note, 96):
            text.textLine(line)
        canvas.drawText(text)

    if document.hidden_note:
        canvas.setFillColor(white)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(_MARGIN, 96, document.hidden_note)

    canvas.setFillColor(_MUTED)
    canvas.setFont("Helvetica-Oblique", 8)
    canvas.drawString(
        _MARGIN, 40, "Synthetic test document. All people, companies, and figures are fictional."
    )
    canvas.showPage()
    canvas.save()
    return buffer.getvalue()


def _wrap(text: str, width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    return [*lines, current] if current else lines


def to_scanned_pdf(pdf_bytes: bytes, resolution: int = 130) -> bytes:
    """Flatten a PDF to page images, the way a scanner would: no text layer survives."""
    buffer = io.BytesIO()
    canvas = Canvas(buffer, pagesize=LETTER)
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            image = page.to_image(resolution=resolution).original
            canvas.drawImage(ImageReader(image), 0, 0, width=_WIDTH, height=_HEIGHT)
            canvas.showPage()
    canvas.save()
    return buffer.getvalue()
