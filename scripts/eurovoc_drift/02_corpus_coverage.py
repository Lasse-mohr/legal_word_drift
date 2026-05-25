"""Compute coverage of EuroVoc labels in the CJEU judgment corpus.

For each EuroVoc label (pref + alt), tokenise it with the same pipeline used
to build the corpus vocab (`preprocess_paragraph`) and count contiguous
n-gram matches in the per-year paragraph JSONL files. Aggregate to per-concept
counts, stratify by domain and decade, and emit a top-100 markdown report.

Outputs (data/processed/eurovoc_coverage/):
    label_celex_counts.parquet  long: concept_id, label, label_type, celex, year, count
    label_year_counts.parquet   long: concept_id, label, label_type, year, count
                                  (derived from celex table)
    concept_coverage.parquet    per-concept: counts, years_present, decade vec, metadata
    coverage_by_domain.csv      21-row summary
    coverage_by_decade.csv      decade x domain matrix
    coverage_top100.md          top-100 concepts ranked by years_present then total_count
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.preprocessing.legal_tokenizer import preprocess_paragraph
from src.utils.config import PARAGRAPHS_DIR, PROCESSED_DIR

EUROVOC_DIR = Path(PROCESSED_DIR) / "eurovoc"
OUT_DIR = Path(PROCESSED_DIR) / "eurovoc_coverage"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(1990, 2026))
MAX_LABEL_LEN = 10 # cap n-gram length; labels longer than this are dropped (counted)


def decade_of(year: int) -> str:
    return f"{(year // 10) * 10}s"


def load_labels() -> tuple[list[tuple[str, str, str, tuple[str, ...]]], dict, int, int]:
    """Read labels_en.csv, tokenise each label, return list of
    (concept_id, label, label_type, token_tuple) plus stats."""
    df = pd.read_csv(EUROVOC_DIR / "labels_en.csv")
    out = []
    n_empty = 0
    n_too_long = 0
    for row in df.itertuples(index=False):
        toks = tuple(preprocess_paragraph(str(row.label)))
        if not toks:
            n_empty += 1
            continue
        if len(toks) > MAX_LABEL_LEN:
            n_too_long += 1
            continue
        out.append((str(row.concept_id), str(row.label), str(row.label_type), toks))
    concepts_meta = pd.read_csv(EUROVOC_DIR / "concepts_enriched.csv", dtype={"concept_id": str})
    meta_idx = concepts_meta.set_index("concept_id").to_dict("index")
    return out, meta_idx, n_empty, n_too_long


def count_year(year: int, labels_by_len: dict[int, set[tuple]]) -> dict[str, Counter]:
    """Stream one year's JSONL, return {celex: Counter[token_tuple] -> count}."""
    path = Path(PARAGRAPHS_DIR) / f"{year}.jsonl"
    per_celex: dict[str, Counter] = {}
    if not path.exists():
        return per_celex

    lengths = sorted(labels_by_len.keys())
    label_sets = {n: labels_by_len[n] for n in lengths}

    with path.open() as f:
        for line in f:
            doc = json.loads(line)
            celex = doc["celex"]
            counter = per_celex.setdefault(celex, Counter())
            for para in doc.get("paragraphs", []):
                toks = preprocess_paragraph(para)
                L = len(toks)
                for n in lengths:
                    if n > L:
                        break
                    s = label_sets[n]
                    for i in range(L - n + 1):
                        ng = tuple(toks[i:i + n])
                        if ng in s:
                            counter[ng] += 1
    return per_celex


def main() -> None:
    print("Loading EuroVoc labels...")
    labels, meta_idx, n_empty, n_too_long = load_labels()
    print(f"  loaded:        {len(labels):,} labels")
    print(f"  empty after tokenise: {n_empty:,}")
    print(f"  longer than {MAX_LABEL_LEN} tokens: {n_too_long:,}")

    # token_tuple -> list of (concept_id, label, label_type)
    tuple_to_labels: dict[tuple, list[tuple[str, str, str]]] = defaultdict(list)
    labels_by_len: dict[int, set[tuple]] = defaultdict(set)
    for cid, label, ltype, toks in labels:
        tuple_to_labels[toks].append((cid, label, ltype))
        labels_by_len[len(toks)].add(toks)
    print(f"  unique token tuples: {len(tuple_to_labels):,}")
    print(f"  length histogram:    {sorted({n: len(s) for n, s in labels_by_len.items()}.items())}")

    # Per-year scan; rows are per-celex so per-year is a downstream groupby
    rows = []  # (concept_id, label, label_type, celex, year, count)
    for year in YEARS:
        print(f"  scanning {year}...", end=" ", flush=True)
        per_celex = count_year(year, labels_by_len)
        n_hits = 0
        distinct_labels: set = set()
        for celex, cnt in per_celex.items():
            for ng, c in cnt.items():
                for cid, label, ltype in tuple_to_labels[ng]:
                    rows.append((cid, label, ltype, celex, year, c))
                    n_hits += c
                    distinct_labels.add(ng)
        print(f"{n_hits:,} hits across {len(distinct_labels):,} distinct labels in {len(per_celex):,} celexes")

    label_celex = pd.DataFrame(
        rows, columns=["concept_id", "label", "label_type", "celex", "year", "count"]
    )
    label_celex.to_parquet(OUT_DIR / "label_celex_counts.parquet", index=False)
    print(f"wrote {OUT_DIR / 'label_celex_counts.parquet'}  ({len(label_celex):,} rows)")

    # Derive per-year table by groupby over celex
    label_year = (
        label_celex.groupby(
            ["concept_id", "label", "label_type", "year"], as_index=False
        )["count"]
        .sum()
    )
    label_year.to_parquet(OUT_DIR / "label_year_counts.parquet", index=False)
    print(f"wrote {OUT_DIR / 'label_year_counts.parquet'}  ({len(label_year):,} rows)")

    # Per-concept aggregation: sum counts across labels (pref + alt) per (concept, year)
    cy = label_year.groupby(["concept_id", "year"], as_index=False)["count"].sum()
    # Per-concept totals
    totals = cy.groupby("concept_id")["count"].sum().rename("total_count")
    years_present = cy[cy["count"] > 0].groupby("concept_id")["year"].nunique().rename("years_present")
    # Per-decade
    cy["decade"] = cy["year"].apply(decade_of)
    decade_pivot = cy.pivot_table(index="concept_id", columns="decade", values="count", aggfunc="sum", fill_value=0)
    for d in ["1990s", "2000s", "2010s", "2020s"]:
        if d not in decade_pivot.columns:
            decade_pivot[d] = 0
    decade_pivot = decade_pivot[["1990s", "2000s", "2010s", "2020s"]].astype(int)

    # Build concept_coverage frame; include all concepts (even those with zero hits) so domain stats are honest
    all_concepts = pd.DataFrame(
        [{"concept_id": cid, **{k: v for k, v in m.items()}} for cid, m in meta_idx.items()]
    )
    cov = all_concepts.merge(totals, on="concept_id", how="left").merge(years_present, on="concept_id", how="left")
    cov = cov.merge(decade_pivot, on="concept_id", how="left")
    cov[["total_count", "years_present", "1990s", "2000s", "2010s", "2020s"]] = (
        cov[["total_count", "years_present", "1990s", "2000s", "2010s", "2020s"]].fillna(0).astype(int)
    )
    cov.to_parquet(OUT_DIR / "concept_coverage.parquet", index=False)
    print(f"wrote {OUT_DIR / 'concept_coverage.parquet'}  ({len(cov):,} concepts)")

    # Coverage by domain
    by_dom = (
        cov.assign(
            has_any=(cov["total_count"] > 0).astype(int),
            ge10=(cov["total_count"] >= 10).astype(int),
            yrs_ge10=(cov["years_present"] >= 10).astype(int),
        )
        .groupby("domain_name", dropna=False)
        .agg(
            n_concepts=("concept_id", "count"),
            n_with_any_hit=("has_any", "sum"),
            n_ge10_hits=("ge10", "sum"),
            n_present_ge10_years=("yrs_ge10", "sum"),
            median_count=("total_count", "median"),
            total_count=("total_count", "sum"),
        )
        .reset_index()
        .sort_values("n_present_ge10_years", ascending=False)
    )
    by_dom.to_csv(OUT_DIR / "coverage_by_domain.csv", index=False)
    print(f"wrote {OUT_DIR / 'coverage_by_domain.csv'}")

    # Decade x domain
    dec_dom = cov.groupby("domain_name", dropna=False)[["1990s", "2000s", "2010s", "2020s"]].sum().reset_index()
    dec_dom.to_csv(OUT_DIR / "coverage_by_decade.csv", index=False)
    print(f"wrote {OUT_DIR / 'coverage_by_decade.csv'}")

    # Top-100 markdown
    top = cov.sort_values(["years_present", "total_count"], ascending=[False, False]).head(100).reset_index(drop=True)
    lines = []
    lines.append("# EuroVoc × CJEU corpus — top 100 concepts by coverage\n")
    lines.append(f"Generated from `02_corpus_coverage.py`. Years scanned: {YEARS[0]}–{YEARS[-1]}.\n")
    lines.append(
        f"- Concepts total: {len(cov):,} | with any corpus hit: {(cov['total_count'] > 0).sum():,} "
        f"| present in ≥10 years: {(cov['years_present'] >= 10).sum():,}\n"
    )
    lines.append(f"- Labels parsed: {len(labels):,} | empty after tokenisation: {n_empty:,} | dropped (>{MAX_LABEL_LEN} tokens): {n_too_long:,}\n")
    lines.append("\n## Coverage by domain (sorted by # concepts present in ≥10 years)\n")
    lines.append("| Domain | Concepts | With hit | ≥10 hits | Present ≥10 yrs | Total count |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in by_dom.itertuples(index=False):
        lines.append(
            f"| {r.domain_name or '(unassigned)'} | {r.n_concepts} | {r.n_with_any_hit} | "
            f"{r.n_ge10_hits} | {r.n_present_ge10_years} | {int(r.total_count):,} |"
        )

    lines.append("\n## Top 100 concepts (rank by years_present, then total_count)\n")
    lines.append("| # | Term | Domain | Microthesaurus | Total | Years | 1990s | 2000s | 2010s | 2020s |")
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|---:|")
    for i, (_, r) in enumerate(top.iterrows(), start=1):
        lines.append(
            f"| {i} | {r['pref_label_en']} | {r['domain_name'] or ''} | {r['microthesaurus_name'] or ''} | "
            f"{int(r['total_count']):,} | {int(r['years_present'])} | "
            f"{int(r['1990s'])} | {int(r['2000s'])} | {int(r['2010s'])} | {int(r['2020s'])} |"
        )
    (OUT_DIR / "coverage_top100.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_DIR / 'coverage_top100.md'}")

    # Sanity prints
    print("\nSanity checks:")
    for probe in ["regulation", "directive", "court", "tribunal"]:
        toks = tuple(preprocess_paragraph(probe))
        if toks in tuple_to_labels:
            cids = {c for c, _, _ in tuple_to_labels[toks]}
            sub = cov[cov["concept_id"].isin(cids)]
            if len(sub):
                print(f"  '{probe}': total={int(sub['total_count'].sum()):,} years_present(max)={int(sub['years_present'].max())}")
        else:
            print(f"  '{probe}': not a EuroVoc label")


if __name__ == "__main__":
    main()
