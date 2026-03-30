"""Shared data models for the legal_word_drift pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ParagraphRecord:
    """A single numbered paragraph extracted from a judgment document.

    Attributes
    ----------
    celex : CELEX identifier of the parent judgment.
    para_id : Unique paragraph ID (``{celex}_P{num:04d}`` for grounds,
              ``{celex}_OP{num:04d}`` for operative part).
    para_num : The paragraph number as it appears in the document.
    section : ``"grounds"``, ``"operative"``, or ``"other"``.
    text : Normalized paragraph text (whitespace-collapsed).
    ref_cases : Case citation strings found in this paragraph.
    ref_oj : OJ citation strings found in this paragraph.
    """

    celex: str
    para_id: str
    para_num: str
    section: str
    text: str
    ref_cases: List[str] = field(default_factory=list)
    ref_oj: List[str] = field(default_factory=list)
