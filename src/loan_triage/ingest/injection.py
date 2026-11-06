"""Spot text in a document that is addressed to the model rather than to a human reader.

This is a tripwire, not the defence. The defence is architectural: Claude only transcribes, its
values must be found verbatim in the file, and code makes every routing decision. A successful
injection can at worst misstate a field, which cross-document checks are built to catch. The
tripwire exists so the attempt itself reaches an underwriter, because it is strong evidence of
fraud whether or not it worked.
"""

from __future__ import annotations

import re

from loan_triage.domain.enums import FlagCode
from loan_triage.domain.models import IntegrityFinding

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in (
        (
            "instruction override",
            r"\b(ignore|disregard|forget|override)\b.{0,40}\b"
            r"(instructions?|rules?|prompts?|guidelines?|polic(y|ies))\b",
        ),
        (
            "role reassignment",
            r"\byou\s+are\s+now\b|\bact\s+as\s+(an?\s+)?(ai|assistant|underwriter)\b",
        ),
        (
            "addressing the model",
            r"\b(ai|llm|language model|assistant|chatbot|claude|gpt)\b.{0,40}"
            r"\b(must|should|shall|please|reading|processing)\b",
        ),
        ("prompt syntax", r"(^|\s)(system|assistant)\s*:\s|<\s*/?\s*(system|instructions?)\s*>"),
        ("reference to the prompt", r"\bsystem\s+prompt\b"),
        ("score manipulation", r"\b(set|report|mark|output|return)\b.{0,40}\bconfidence\b"),
        (
            "outcome manipulation",
            r"\b(auto[-\s]?approve|pre[-\s]?approved|must\s+be\s+approved|approve\s+this)\b",
        ),
        ("suppressing review", r"\b(do\s+not|don'?t|never)\s+(flag|report|mention|escalate)\b"),
    )
)


def scan_pages(pages: list[str]) -> list[IntegrityFinding]:
    """At most one finding per page, naming every pattern it tripped."""
    findings: list[IntegrityFinding] = []
    for number, page in enumerate(pages, start=1):
        text = re.sub(r"\s+", " ", page)
        matches = [(label, match) for label, p in _PATTERNS if (match := p.search(text))]
        if not matches:
            continue
        start = max(min(match.start() for _, match in matches) - 40, 0)
        end = max(match.end() for _, match in matches) + 40
        findings.append(
            IntegrityFinding(
                code=FlagCode.PROMPT_INJECTION_SUSPECTED,
                detail=f"Text aimed at the reader: {', '.join(label for label, _ in matches)}.",
                page=number,
                excerpt=text[start:end],
            )
        )
    return findings
