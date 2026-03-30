"""Legal text preprocessing pipeline for word2vec training.

Transforms raw paragraph text into clean tokenized sentences suitable
for gensim's LineSentence format.  Pipeline steps:

1. Citation normalization  (case refs, ECR, OJ, article refs -> typed tokens)
2. Latin phrase protection (multi-word Latin terms -> single underscore-joined tokens)
3. Tokenization            (lowercase, split, filter stopwords, min length)
"""
from __future__ import annotations

import re
from typing import List

# ---------------------------------------------------------------------------
# 1. Citation normalization
# ---------------------------------------------------------------------------

# Case references: "Case C-123/45", "Joined Cases C-456/78 and C-789/01", etc.
_CASE_REF_RE = re.compile(
    r"(?:Joined\s+)?Cases?\s+[A-Z][\-\u2010\u2011]?\d+/\d+"
    r"(?:\s*(?:and|,|&)\s*[A-Z][\-\u2010\u2011]?\d+/\d+)*",
    re.IGNORECASE,
)

# ECR / report references: "[1998] ECR I-1234", "EU:C:2020:123"
_ECR_REF_RE = re.compile(
    r"\[\d{4}\]\s*ECR\s*[I\-]+\d+"
    r"|EU:[A-Z]:\d{4}:\d+",
    re.IGNORECASE,
)

# OJ references: "OJ 1998 L 123, p. 45", "OJ L 123/45"
_OJ_REF_RE = re.compile(
    r"OJ\s*\d{0,4}\s*[LCS]\s*\d+(?:[/,]\s*(?:p\.?\s*)?\d+)?",
    re.IGNORECASE,
)

# Article references with treaty: "Article 234 EC", "Article 101(1) TFEU",
# "Articles 34 and 36 TFEU"
_ARTICLE_RE = re.compile(
    r"Articles?\s+(\d+)(?:\((\d+)\))?"
    r"(?:\s+(?:and|to)\s+\d+(?:\(\d+\))?)?"
    r"\s+(EC|EEC|TFEU|TEU|ECSC|Euratom|CFR|ECHR)",
    re.IGNORECASE,
)

# Paragraph references: "paragraph 45", "para. 45", "paragraphs 10 to 15"
_PARA_REF_RE = re.compile(
    r"paragraphs?\s*\.?\s*\d+(?:\s*(?:to|and|,)\s*\d+)*",
    re.IGNORECASE,
)

# Standalone numbers (not part of other patterns)
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?(?:\s*%)?(?:\s*(?:EUR|euros?))?(?=\s|$|\))", re.IGNORECASE)


def normalize_citations(text: str) -> str:
    """Replace legal citations with typed placeholder tokens."""
    # Order matters: article refs before case refs (article patterns are more specific)
    # Article refs -> article_NNN_treaty
    def _article_repl(m: re.Match) -> str:
        num = m.group(1)
        treaty = m.group(3).lower()
        return f"article_{num}_{treaty}"

    text = _ARTICLE_RE.sub(_article_repl, text)
    text = _CASE_REF_RE.sub("__CASEREF__", text)
    text = _ECR_REF_RE.sub("__ECRREF__", text)
    text = _OJ_REF_RE.sub("__OJREF__", text)
    text = _PARA_REF_RE.sub("__PARAREF__", text)
    text = _NUM_RE.sub("__NUM__", text)
    return text


# ---------------------------------------------------------------------------
# 2. Latin phrase protection
# ---------------------------------------------------------------------------

# Common Latin phrases in EU legal text.
# These are joined with underscores so they survive tokenization as single tokens.
LATIN_PHRASES = [
    "ab initio",
    "acquis communautaire",
    "acte clair",
    "acte eclaire",
    "ad hoc",
    "amicus curiae",
    "bona fide",
    "comitas gentium",
    "contra legem",
    "de facto",
    "de jure",
    "de minimis",
    "de novo",
    "erga omnes",
    "ex ante",
    "ex officio",
    "ex post",
    "ex tunc",
    "ex nunc",
    "forum non conveniens",
    "habeas corpus",
    "in absentia",
    "in camera",
    "in casu",
    "in extenso",
    "in fine",
    "in limine",
    "in personam",
    "in rem",
    "in situ",
    "in toto",
    "inter alia",
    "inter partes",
    "ius cogens",
    "jus cogens",
    "locus standi",
    "modus operandi",
    "mutatis mutandis",
    "nemo iudex in causa sua",
    "non bis in idem",
    "non liquet",
    "nullum crimen sine lege",
    "obiter dictum",
    "obiter dicta",
    "pacta sunt servanda",
    "per se",
    "prima facie",
    "pro rata",
    "pro tempore",
    "ratio decidendi",
    "ratione materiae",
    "ratione temporis",
    "ratione personae",
    "res judicata",
    "res iudicata",
    "sine qua non",
    "stare decisis",
    "sui generis",
    "ultra vires",
    "vis major",
]

# Build regex: sort longest-first so "acquis communautaire" matches before "acquis"
_LATIN_PHRASES_SORTED = sorted(LATIN_PHRASES, key=len, reverse=True)
_LATIN_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in _LATIN_PHRASES_SORTED) + r")\b",
    re.IGNORECASE,
)


def protect_latin_phrases(text: str) -> str:
    """Replace multi-word Latin phrases with underscore-joined tokens."""
    def _repl(m: re.Match) -> str:
        return m.group(0).lower().replace(" ", "_")
    return _LATIN_RE.sub(_repl, text)


# ---------------------------------------------------------------------------
# 3. Tokenization
# ---------------------------------------------------------------------------

# English stopwords + legal-specific stopwords
STOPWORDS = {
    # Standard English
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "must", "it", "its", "this", "that", "these", "those", "he", "she",
    "they", "we", "you", "i", "me", "him", "her", "us", "them", "my",
    "his", "our", "your", "their", "which", "who", "whom", "what",
    "where", "when", "how", "not", "no", "nor", "if", "then", "than",
    "so", "very", "also", "just", "about", "up", "out", "into", "over",
    "after", "before", "between", "under", "above", "below", "through",
    "during", "each", "all", "both", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "too", "any", "there",
    "here", "once", "again", "further", "because", "until", "while",
    "however", "therefore", "thus", "hence", "moreover", "furthermore",
    "although", "though", "whether", "since", "yet", "still", "already",
    # Legal-specific stopwords (formulaic, procedural)
    "hereby", "whereas", "thereof", "therein", "thereto", "thereby",
    "hereinafter", "herein", "hereto", "whereof", "wherein", "aforesaid",
    "abovementioned", "aforementioned", "hereunder", "thereunder",
    "notwithstanding", "pursuant", "inasmuch",
}

# Tokenization: split on non-alphanumeric (preserving underscores for joined phrases)
_TOKEN_RE = re.compile(r"[a-z_][a-z0-9_]*")


def tokenize(text: str) -> List[str]:
    """Tokenize preprocessed legal text into a list of clean tokens.

    Assumes text has already been passed through normalize_citations()
    and protect_latin_phrases().
    """
    text = text.lower()
    tokens = _TOKEN_RE.findall(text)
    return [
        t for t in tokens
        if len(t) >= 2
        and t not in STOPWORDS
        and not t.startswith("_")  # skip stray underscores
    ]


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess_paragraph(text: str) -> List[str]:
    """Run the full preprocessing pipeline on a single paragraph.

    Returns a list of tokens ready for word2vec training.
    """
    text = normalize_citations(text)
    text = protect_latin_phrases(text)
    return tokenize(text)


def preprocess_paragraphs(paragraphs: List[str]) -> List[List[str]]:
    """Run the full preprocessing pipeline on a list of paragraphs.

    Returns a list of token lists (one per paragraph).
    Empty results (paragraphs that produce no tokens) are filtered out.
    """
    results = []
    for p in paragraphs:
        tokens = preprocess_paragraph(p)
        if tokens:
            results.append(tokens)
    return results
