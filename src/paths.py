"""Centralised filesystem paths for the project.

`Paths` is the canonical pathlib-based directory map. New scripts should
import `PATHS` from here. The legacy string-based constants in
`src.utils.config` continue to work for older scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    project_root: Path = _PROJECT_ROOT

    # ── data root ─────────────────────────────────────────────────────────
    data: Path = _PROJECT_ROOT / "data"

    # raw
    raw: Path = _PROJECT_ROOT / "data" / "raw"
    raw_metadata: Path = _PROJECT_ROOT / "data" / "raw" / "metadata"
    raw_xhtml: Path = _PROJECT_ROOT / "data" / "raw" / "xhtml"
    raw_texts: Path = _PROJECT_ROOT / "data" / "raw" / "texts"

    # processed
    processed: Path = _PROJECT_ROOT / "data" / "processed"
    paragraphs: Path = _PROJECT_ROOT / "data" / "processed" / "paragraphs"
    sentences: Path = _PROJECT_ROOT / "data" / "processed" / "sentences"
    vocab: Path = _PROJECT_ROOT / "data" / "processed" / "vocab"
    eurovoc: Path = _PROJECT_ROOT / "data" / "processed" / "eurovoc"
    eurovoc_coverage: Path = _PROJECT_ROOT / "data" / "processed" / "eurovoc_coverage"
    eurovoc_coverage_label_celex_counts: Path = (
        _PROJECT_ROOT / "data" / "processed" / "eurovoc_coverage" / "label_celex_counts.parquet"
    )
    eurovoc_coverage_label_year_counts: Path = (
        _PROJECT_ROOT / "data" / "processed" / "eurovoc_coverage" / "label_year_counts.parquet"
    )

    # models
    models: Path = _PROJECT_ROOT / "data" / "models"
    w2v: Path = _PROJECT_ROOT / "data" / "models" / "word2vec"
    aligned: Path = _PROJECT_ROOT / "data" / "models" / "aligned"

    bert: Path = _PROJECT_ROOT / "data" / "models" / "bert"
    bert_usage_index: Path = _PROJECT_ROOT / "data" / "models" / "bert" / "usage_index"
    bert_sampled: Path = _PROJECT_ROOT / "data" / "models" / "bert" / "sampled_usages"
    bert_embeddings: Path = _PROJECT_ROOT / "data" / "models" / "bert" / "embeddings"
    bert_centroids: Path = _PROJECT_ROOT / "data" / "models" / "bert" / "centroids.npz"
    bert_cross_period_apd: Path = _PROJECT_ROOT / "data" / "models" / "bert" / "cross_period_apd.npz"

    # eurovoc-driven drift pipeline
    eurovoc_drift: Path = _PROJECT_ROOT / "data" / "models" / "eurovoc_drift"
    # Legacy alias — points to the year-unit file. Keeps 04/11/12 working.
    eurovoc_drift_selected_labels: Path = (
        _PROJECT_ROOT / "data" / "processed" / "eurovoc_drift" / "selected_labels_year.parquet"
    )
    # Legacy alias — points to the year-unit usage index dir.
    eurovoc_drift_usage_index: Path = (
        _PROJECT_ROOT / "data" / "models" / "eurovoc_drift" / "usage_index" / "year"
    )
    eurovoc_drift_sampled: Path = (
        _PROJECT_ROOT / "data" / "models" / "eurovoc_drift" / "sampled_usages"
    )
    eurovoc_drift_embeddings: Path = (
        _PROJECT_ROOT / "data" / "models" / "eurovoc_drift" / "embeddings"
    )
    eurovoc_drift_apd_npz: Path = (
        _PROJECT_ROOT / "data" / "models" / "eurovoc_drift" / "cross_period_apd.npz"
    )
    eurovoc_drift_mw_null_pool: Path = (
        _PROJECT_ROOT / "data" / "models" / "eurovoc_drift" / "mw_null_pool"
    )

    # results
    results: Path = _PROJECT_ROOT / "data" / "results"
    metrics: Path = _PROJECT_ROOT / "data" / "results" / "metrics"
    figures: Path = _PROJECT_ROOT / "data" / "results" / "figures"
    eurovoc_drift_figures: Path = _PROJECT_ROOT / "data" / "results" / "figures" / "eurovoc_drift"
    eurovoc_drift_ranking: Path = (
        _PROJECT_ROOT / "data" / "results" / "metrics" / "eurovoc_drift_ranking.parquet"
    )
    eurovoc_drift_linear_probe_results: Path = (
        _PROJECT_ROOT / "data" / "results" / "metrics" / "eurovoc_drift_linear_probe"
    )
    eurovoc_drift_linear_probe_figures: Path = (
        _PROJECT_ROOT / "data" / "results" / "figures" / "eurovoc_drift" / "linear_probe"
    )
    eurovoc_drift_centroid_auc_results: Path = (
        _PROJECT_ROOT / "data" / "results" / "metrics" / "eurovoc_drift_centroid_auc"
    )
    eurovoc_drift_centroid_auc_figures: Path = (
        _PROJECT_ROOT / "data" / "results" / "figures" / "eurovoc_drift" / "centroid_auc"
    )
    eurovoc_drift_mw_null_results: Path = (
        _PROJECT_ROOT / "data" / "results" / "metrics" / "eurovoc_drift_mw_null"
    )
    eurovoc_drift_mw_null_figures: Path = (
        _PROJECT_ROOT / "data" / "results" / "figures" / "eurovoc_drift" / "mw_null"
    )
    eurovoc_drift_usage_counts: Path = (
        _PROJECT_ROOT / "data" / "processed" / "eurovoc_drift" / "usage_counts_by_year.parquet"
    )
    eurovoc_drift_figures_drift_vs_time: Path = (
        _PROJECT_ROOT / "data" / "results" / "figures" / "eurovoc_drift" / "drift_vs_time"
    )

    # logs
    logs: Path = _PROJECT_ROOT / "logs"

    def paragraphs_year(self, year: int) -> Path:
        return self.paragraphs / f"{year}.jsonl"

    def eurovoc_drift_usage_index_year(self, year: int) -> Path:
        return self.eurovoc_drift_usage_index / f"{year}.jsonl"

    def eurovoc_drift_selected_labels_for(self, unit: str) -> Path:
        if unit not in ("year", "judgment"):
            raise ValueError(f"unit must be 'year' or 'judgment', got {unit!r}")
        return (
            _PROJECT_ROOT / "data" / "processed" / "eurovoc_drift"
            / f"selected_labels_{unit}.parquet"
        )

    def eurovoc_drift_usage_index_for(self, unit: str) -> Path:
        if unit not in ("year", "judgment"):
            raise ValueError(f"unit must be 'year' or 'judgment', got {unit!r}")
        return self.eurovoc_drift / "usage_index" / unit

    def eurovoc_drift_usage_index_year_for(self, unit: str, year: int) -> Path:
        return self.eurovoc_drift_usage_index_for(unit) / f"{year}.jsonl"

    def eurovoc_drift_sampled_year(self, year: int) -> Path:
        return self.eurovoc_drift_sampled / f"{year}.jsonl"

    def eurovoc_drift_embeddings_year(self, year: int) -> Path:
        return self.eurovoc_drift_embeddings / f"{year}.npz"


PATHS = Paths()
