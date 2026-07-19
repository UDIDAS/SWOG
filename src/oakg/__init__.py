"""OAKG: Observability-Aware Knowledge Graph Reasoning for Heterogeneous Medical Imaging.

Reproducible retrieval-evaluation package for the AAAI 2027 study. The public
API mirrors the experiment sections in the guidelines; see ``notebooks/`` for a
thin driver that reproduces every table and figure.
"""
from __future__ import annotations

from .config import ORGANS, TOP_K, Config
from .data import (
    Corpus,
    ExperimentData,
    generate_demo_data,
    load_data,
    load_experiment_data,
    validate_data,
)
from .masking import (
    MaskingRealization,
    apply_mask,
    build_default_realizations,
    make_asymmetric_mask,
    make_dataset_style_mask,
    make_random_mask,
    make_uniform_mask,
)
from .metrics import (
    aggregate_metrics,
    average_precision,
    ndcg_at_k,
    precision_at_k,
    query_metric_row,
    recall_at_k,
)
from .baselines import BASELINE_FUNCTIONS
from .oakg import oakg_scores, observation_overlap, oakg_similarity_components, rank_policy
from .benchmark import (
    OAKG_POLICIES,
    candidate_pool,
    evaluate_realization,
    evaluate_vector_methods,
    run_full_benchmark,
    summarize,
)
from .stats import (
    bootstrap_metric_ci,
    choose_policy,
    comparison_family,
    paired_bootstrap_difference,
    upstream_degradation,
)
from .selective import selective_curve
from .consistency import ranking_consistency, rank_ids, top_k_overlap
from .semantics import structured_query_evaluation
from .diagnostics import query_diagnostics
from .graph import case_graph, wl_embeddings, wl_feature_dict
from .neural import (
    apply_pairwise_observability_filter,
    cross_backbone_gain,
    embedding_scores,
    hybrid_scores,
    load_npz_embeddings,
    select_hybrid_alpha,
)
from .tables import format_metric_ci, publication_table
from .phenotypes import DatasetSpec, extract_phenotypes, voxel_volume_cm3

__version__ = "0.1.0"

__all__ = [
    "Config", "ORGANS", "TOP_K",
    "ExperimentData", "Corpus", "generate_demo_data", "load_data",
    "load_experiment_data", "validate_data",
    "MaskingRealization", "apply_mask", "build_default_realizations",
    "make_uniform_mask", "make_random_mask", "make_dataset_style_mask", "make_asymmetric_mask",
    "precision_at_k", "recall_at_k", "average_precision", "ndcg_at_k",
    "query_metric_row", "aggregate_metrics",
    "BASELINE_FUNCTIONS",
    "observation_overlap", "oakg_similarity_components", "rank_policy", "oakg_scores",
    "OAKG_POLICIES", "candidate_pool", "evaluate_vector_methods", "evaluate_realization",
    "run_full_benchmark", "summarize",
    "paired_bootstrap_difference", "comparison_family", "bootstrap_metric_ci",
    "upstream_degradation", "choose_policy",
    "selective_curve", "ranking_consistency", "rank_ids", "top_k_overlap",
    "structured_query_evaluation", "query_diagnostics",
    "case_graph", "wl_feature_dict", "wl_embeddings",
    "load_npz_embeddings", "embedding_scores", "hybrid_scores", "select_hybrid_alpha",
    "apply_pairwise_observability_filter", "cross_backbone_gain",
    "publication_table", "format_metric_ci",
    "DatasetSpec", "extract_phenotypes", "voxel_volume_cm3",
]
