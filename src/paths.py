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
    # Legacy aliases — now resolve to the year/ subdir of unit-routed layouts.
    eurovoc_drift_sampled: Path = (
        _PROJECT_ROOT / "data" / "models" / "eurovoc_drift" / "sampled_usages" / "year"
    )
    eurovoc_drift_embeddings: Path = (
        _PROJECT_ROOT / "data" / "models" / "eurovoc_drift" / "embeddings" / "year"
    )
    eurovoc_drift_embeddings_index: Path = (
        _PROJECT_ROOT / "data" / "models" / "eurovoc_drift" / "embeddings_index" / "year"
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

    # ── unit-routed helpers for year/judgment granularity ─────────────────

    @staticmethod
    def _check_unit(unit: str) -> None:
        if unit not in ("year", "judgment"):
            raise ValueError(f"unit must be 'year' or 'judgment', got {unit!r}")

    def _eurovoc_drift_root(self, model: str | None) -> Path:
        """eurovoc_drift root, optionally namespaced under ``models/<slug>/``.

        ``model=None`` returns the legacy root so the default (eurlex) encoder's
        outputs and the judgment-unit work stay byte-for-byte where they are.
        """
        return self.eurovoc_drift if not model else self.eurovoc_drift / "models" / model

    def eurovoc_drift_sampled_for(self, unit: str, model: str | None = None) -> Path:
        self._check_unit(unit)
        return self._eurovoc_drift_root(model) / "sampled_usages" / unit

    def eurovoc_drift_embeddings_for(self, unit: str, model: str | None = None) -> Path:
        self._check_unit(unit)
        return self._eurovoc_drift_root(model) / "embeddings" / unit

    def eurovoc_drift_embeddings_index_for(self, unit: str, model: str | None = None) -> Path:
        self._check_unit(unit)
        return self._eurovoc_drift_root(model) / "embeddings_index" / unit

    def eurovoc_drift_sampled_year_for(self, unit: str, year: int, model: str | None = None) -> Path:
        return self.eurovoc_drift_sampled_for(unit, model) / f"{year}.jsonl"

    def eurovoc_drift_embeddings_year_for(self, unit: str, year: int, model: str | None = None) -> Path:
        return self.eurovoc_drift_embeddings_for(unit, model) / f"{year}.npz"

    def eurovoc_drift_embeddings_index_year_for(self, unit: str, year: int, model: str | None = None) -> Path:
        return self.eurovoc_drift_embeddings_index_for(unit, model) / f"{year}.jsonl"

    # ── model-aware locations for the heatmap (04/05) and drift-vs-time (12) ──
    # outputs that older code built inline. ``model=None`` reproduces the legacy
    # year-mode paths exactly.

    def eurovoc_drift_apd_npz_for(self, unit: str, model: str | None = None) -> Path:
        self._check_unit(unit)
        root = self._eurovoc_drift_root(model)
        return root / "cross_period_apd.npz" if unit == "year" else root / unit / "cross_period_apd.npz"

    def eurovoc_drift_ranking_for(self, unit: str, model: str | None = None) -> Path:
        self._check_unit(unit)
        base = self.eurovoc_drift_ranking
        if model:
            base = base.with_name(f"{base.stem}_{model}{base.suffix}")
        if unit == "year":
            return base
        return base.with_name(f"{base.stem}_{unit}{base.suffix}")

    def eurovoc_drift_figures_for(self, model: str | None = None) -> Path:
        """Per-domain heatmap figure root (script 05)."""
        return self.eurovoc_drift_figures if not model else self.eurovoc_drift_figures / "models" / model

    def eurovoc_drift_figures_drift_vs_time_for(self, unit: str, model: str | None = None) -> Path:
        self._check_unit(unit)
        base = self.eurovoc_drift_figures_drift_vs_time
        if model:
            base = base / "models" / model
        return base if unit == "year" else base / unit


PATHS = Paths()
