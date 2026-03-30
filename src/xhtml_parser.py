"""Parse XHTML judgment files into paragraph records.

Handles two structural variants that CELLAR uses across different eras:

1. **Anchor-based** (pre-2014): Sections marked by ``<a name="MO">`` /
   ``<a name="DI">`` anchors. Paragraphs are plain ``<p>`` elements whose
   text starts with a number.

2. **Class-based** (2014+): Paragraphs use ``<p class="[prefix]count">``
   inside ``<table><tr><td>`` grids, with text in sibling
   ``<p class="[prefix]normal">`` elements.  The prefix is either ``coj-``
   or empty (``""``), detected automatically.

Unlike FMX4, XHTML has no structured reference elements (REF.DOC.ECR,
REF.DOC.OJ), so all citation extraction relies on text regex patterns.
"""

from __future__ import annotations

import re
from typing import Literal

from lxml import etree

from .models import ParagraphRecord

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# "On those grounds" marker separating grounds from operative part.
# Works across all eras and court compositions.
_OPERATIVE_RE = re.compile(r"On\s+those\s+grounds", re.IGNORECASE)

# Pattern: paragraph text starts with a number, followed by period/space/nbsp
_PARA_NUM_RE = re.compile(r"^(\d+)\s*[.\s\xa0]")

# Whitespace normalization: collapse runs of whitespace (incl. newlines) to single space
_WS_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    """Collapse all whitespace runs (including newlines) to a single space."""
    return _WS_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

Variant = Literal["anchor", "class"]


def _detect_variant(tree: etree._Element) -> tuple[Variant, str]:
    """Detect which XHTML structural variant a document uses.

    Returns
    -------
    (variant, prefix) where variant is "anchor" or "class", and prefix is
    the CSS class prefix ("coj-" or "") for class-based documents.
    """
    # Walk all elements once and check for distinguishing features.
    for elem in tree.iter():
        if not isinstance(elem.tag, str):
            continue
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

        # Anchor-based: <a name="MO"> or <a name="DI">
        if tag == "a":
            name = elem.get("name", "")
            if name in ("MO", "DI"):
                return ("anchor", "")

        # Class-based with coj- prefix
        if tag == "p":
            cls = elem.get("class", "")
            if "coj-count" in cls:
                return ("class", "coj-")
            # Class-based without prefix (plain "count")
            if cls == "count":
                return ("class", "")

    # Fallback: anchor-based (graceful degradation for unrecognized docs)
    return ("anchor", "")


# ---------------------------------------------------------------------------
# Anchor-based parser (pre-2014)
# ---------------------------------------------------------------------------

# Section anchor names -> section labels
_SECTION_ANCHORS = {
    "MO": "grounds",
    "DI": "operative",
}


def _parse_anchor_based(tree: etree._Element, celex: str) -> list[ParagraphRecord]:
    """Parse anchor-based XHTML (pre-2014 format)."""
    # Build a map: anchor_name -> element position (for section detection).
    section_starts: dict[int, str] = {}
    all_elements = list(tree.iter())

    for i, elem in enumerate(all_elements):
        if not isinstance(elem.tag, str):
            continue
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag == "a":
            name = elem.get("name", "")
            if name in _SECTION_ANCHORS:
                section_starts[i] = _SECTION_ANCHORS[name]

    # Assign section to each <p> based on its position relative to anchors
    sorted_anchors = sorted(section_starts.items())

    def _get_section(elem_index: int) -> str:
        section = "other"
        for anchor_idx, sec in sorted_anchors:
            if elem_index >= anchor_idx:
                section = sec
        return section

    paragraphs: list[ParagraphRecord] = []

    for i, elem in enumerate(all_elements):
        if not isinstance(elem.tag, str):
            continue
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag != "p":
            continue

        text = _normalize_text("".join(elem.itertext()))
        if not text:
            continue

        m = _PARA_NUM_RE.match(text)
        if not m:
            continue

        para_num = m.group(1)
        # Strip the leading number prefix for consistency with class-based
        # parser (where the number lives in a separate cell).
        text = text[m.end():].strip()
        if not text:
            continue

        section = _get_section(i)

        if section == "operative":
            para_id = f"{celex}_OP{int(para_num):04d}"
        else:
            para_id = f"{celex}_P{int(para_num):04d}"

        paragraphs.append(ParagraphRecord(
            celex=celex,
            para_id=para_id,
            para_num=para_num,
            section=section,
            text=text,
            ref_cases=[],
            ref_oj=[],
        ))

    return paragraphs


# ---------------------------------------------------------------------------
# Class-based parser (2014+)
# ---------------------------------------------------------------------------

def _tag_local(elem: etree._Element) -> str:
    """Return the local tag name, stripping any namespace."""
    tag = elem.tag
    if not isinstance(tag, str):
        return ""
    return tag.split("}")[-1] if "}" in tag else tag


def _get_class(elem: etree._Element) -> str:
    """Return the class attribute (empty string if absent)."""
    return elem.get("class", "")


def _parse_class_based(
    tree: etree._Element,
    celex: str,
    prefix: str,
) -> list[ParagraphRecord]:
    """Parse class-based XHTML (2014+ format).

    Each numbered paragraph lives in a ``<table>`` with two ``<td>`` cells:
    - First ``<td>`` contains ``<p class="[prefix]count">``: the number
    - Second ``<td>`` contains one or more ``<p class="[prefix]normal">``: the text

    Section detection uses the "On those grounds" marker to split grounds
    from operative part.
    """
    count_cls = f"{prefix}count"
    normal_cls = f"{prefix}normal"

    paragraphs: list[ParagraphRecord] = []
    in_operative = False

    # Find all <p> elements with the count class
    for elem in tree.iter():
        if _tag_local(elem) != "p":
            continue

        cls = _get_class(elem)

        # Check for "On those grounds" marker in normal paragraphs
        if count_cls not in cls and normal_cls in cls:
            raw = _normalize_text("".join(elem.itertext()))
            if _OPERATIVE_RE.search(raw):
                in_operative = True
            continue

        # Not a count element -> skip
        if count_cls not in cls:
            continue

        # Extract the paragraph number from the count element
        raw_num = "".join(elem.itertext()).strip().rstrip(".")
        if not raw_num.isdigit():
            continue
        para_num = raw_num

        # Navigate: <p class="count"> is inside a <td>.
        # The sibling <td> contains the text paragraphs.
        td_count = elem.getparent()
        if td_count is None or _tag_local(td_count) != "td":
            continue

        tr = td_count.getparent()
        if tr is None or _tag_local(tr) != "tr":
            continue

        # Find the sibling <td> (the one that is NOT the count cell)
        td_text = None
        for child in tr:
            if _tag_local(child) == "td" and child is not td_count:
                td_text = child
                break

        if td_text is None:
            continue

        # Collect text from all <p class="[prefix]normal"> children in the text cell
        text_parts: list[str] = []
        for p_elem in td_text.iter():
            if _tag_local(p_elem) != "p":
                continue
            p_cls = _get_class(p_elem)
            if normal_cls not in p_cls:
                continue
            t = _normalize_text("".join(p_elem.itertext()))
            if t:
                text_parts.append(t)
                if _OPERATIVE_RE.search(t):
                    in_operative = True

        text = " ".join(text_parts)
        if not text:
            continue

        section = "operative" if in_operative else "grounds"

        if section == "operative":
            para_id = f"{celex}_OP{int(para_num):04d}"
        else:
            para_id = f"{celex}_P{int(para_num):04d}"

        paragraphs.append(ParagraphRecord(
            celex=celex,
            para_id=para_id,
            para_num=para_num,
            section=section,
            text=text,
            ref_cases=[],
            ref_oj=[],
        ))

    return paragraphs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_summary_only(xhtml_bytes: bytes) -> bool:
    """Check whether an XHTML document is a summary-only notice.

    CELLAR sometimes serves a ``listNotice`` document containing only
    "Subject of the case" + "Operative part", with no grounds section.
    This pre-screening function detects that case *before* full parsing.

    Returns ``True`` for summary-only documents, ``False`` for full text.
    """
    try:
        tree = etree.fromstring(xhtml_bytes, etree.HTMLParser())
    except etree.XMLSyntaxError:
        return False  # can't parse → let parse_xhtml handle the error

    has_mo_anchor = False
    has_count_class = False
    has_list_notice = False

    for elem in tree.iter():
        if not isinstance(elem.tag, str):
            continue
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

        if tag == "a" and elem.get("name") == "MO":
            has_mo_anchor = True
        elif tag == "p":
            cls = elem.get("class", "")
            if "coj-count" in cls or cls == "count":
                has_count_class = True
        elif tag == "div" and "listNotice" in elem.get("class", ""):
            has_list_notice = True

    # Full text: has grounds anchor or numbered-paragraph classes
    if has_mo_anchor or has_count_class:
        return False
    # Summary-only: listNotice wrapper without grounds markers
    if has_list_notice:
        return True
    # Unknown structure → assume full text (let parser try)
    return False


def parse_xhtml(xhtml_bytes: bytes, celex: str) -> list[ParagraphRecord]:
    """Parse an XHTML judgment document into paragraph records.

    Auto-detects the structural variant (anchor-based or class-based) and
    dispatches to the appropriate parser.

    Parameters
    ----------
    xhtml_bytes : raw XHTML/HTML content
    celex : CELEX number of the parent judgment

    Returns
    -------
    List of ParagraphRecord dicts, one per numbered paragraph.
    """
    try:
        tree = etree.fromstring(xhtml_bytes, etree.HTMLParser())
    except etree.XMLSyntaxError as e:
        print(f"  WARNING: {celex} XHTML parse error: {e}")
        return []

    variant, prefix = _detect_variant(tree)

    if variant == "class":
        return _parse_class_based(tree, celex, prefix)
    else:
        return _parse_anchor_based(tree, celex)
