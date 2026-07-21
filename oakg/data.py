"""Data model, reproducible demo generator, loader, and fitted corpus state.

The notebook kept the fitted feature matrices and schema in module-level globals.
Here they are collected into an explicit :class:`Corpus` object so every
downstream function receives its state explicitly (no hidden globals).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import Config

# Feature specifications used by the demo generator: (name, type, support organs).
FEATURE_SPECS: list[tuple[str, str, set[str]]] = [
    ("tumor_burden", "numeric", set()),
    ("lesion_multiplicity", "numeric", set()),
    ("liver_tumor_present", "binary", {"liver"}),
    ("pancreas_tumor_present", "binary", {"pancreas"}),
    ("kidney_tumor_present", "binary", {"left_kidney", "right_kidney"}),
    ("liver_containment", "binary", {"liver"}),
    ("pancreas_containment", "binary", {"pancreas"}),
]


@dataclass(frozen=True)
class ExperimentData:
    """The five raw tables that fully define a retrieval benchmark."""

    cases: pd.DataFrame
    phenotypes_ref: pd.DataFrame
    phenotypes_pred: pd.DataFrame
    queries: pd.DataFrame
    relevance: pd.DataFrame


def organs_to_text(organs: Iterable[str]) -> str:
    return "|".join(sorted(set(organs)))


def parse_organs(value: Any) -> frozenset[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return frozenset()
    if isinstance(value, (set, frozenset, list, tuple)):
        return frozenset(map(str, value))
    text = str(value).strip()
    return frozenset(x for x in text.split("|") if x)


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------
def generate_demo_data(config: Config) -> ExperimentData:
    """Generate a small but nontrivial benchmark that exercises all code paths.

    Deterministic given ``config.seed`` so demo results are reproducible.
    """
    rng = np.random.default_rng(config.seed)
    organs = list(config.organs)
    n_cases = config.n_demo_cases

    datasets = np.array(["Pancreas", "LiTS", "FLARE"])
    probs = np.array([0.28, 0.22, 0.50])
    dataset = rng.choice(datasets, size=n_cases, p=probs)
    splits = rng.choice(["train", "val", "test"], size=n_cases, p=[0.55, 0.20, 0.25])
    case_ids = np.array([f"C{i:04d}" for i in range(n_cases)])

    available = []
    for ds in dataset:
        if ds == "Pancreas":
            available.append({"pancreas"})
        elif ds == "LiTS":
            available.append({"liver"})
        else:
            available.append(set(organs))

    cases = pd.DataFrame({
        "case_id": case_ids,
        "dataset": dataset,
        "split": splits,
        "available_organs": [organs_to_text(x) for x in available],
    })

    feature_specs = FEATURE_SPECS + [("cross_organ_distribution", "binary", set(organs))]

    rows_ref, rows_pred = [], []
    latent: dict[str, dict[str, float]] = {}
    for cid, case_organs in zip(case_ids, available):
        burden = max(0.0, rng.gamma(2.0, 15.0))
        multiplicity = int(max(1, rng.poisson(1.5)))
        liver = int("liver" in case_organs and rng.random() < 0.45)
        pancreas = int("pancreas" in case_organs and rng.random() < 0.35)
        kidney = int(bool({"left_kidney", "right_kidney"} & case_organs) and rng.random() < 0.18)
        values = {
            "tumor_burden": burden,
            "lesion_multiplicity": multiplicity,
            "liver_tumor_present": liver,
            "pancreas_tumor_present": pancreas,
            "kidney_tumor_present": kidney,
            "liver_containment": int(liver and rng.random() < 0.85),
            "pancreas_containment": int(pancreas and rng.random() < 0.85),
            "cross_organ_distribution": int(sum([liver, pancreas, kidney]) >= 2),
        }
        latent[cid] = values

        for feature, ftype, support in feature_specs:
            value = values[feature]
            rows_ref.append({
                "case_id": cid,
                "feature": feature,
                "value": float(value),
                "feature_type": ftype,
                "support_organs": organs_to_text(support),
            })
            if ftype == "numeric":
                pred = max(0.0, float(value) + rng.normal(0, 0.08 * max(1.0, float(value))))
            else:
                flip = rng.random() < 0.06
                pred = float(1 - int(value)) if flip else float(value)
            rows_pred.append({
                "case_id": cid,
                "feature": feature,
                "value": pred,
                "feature_type": ftype,
                "support_organs": organs_to_text(support),
            })

    phen_ref = pd.DataFrame(rows_ref)
    phen_pred = pd.DataFrame(rows_pred)

    test_cases = cases.loc[cases["split"] == "test", "case_id"].tolist()
    if len(test_cases) < 12:
        test_cases = case_ids[-max(12, n_cases // 4):].tolist()

    query_features = [
        "liver_tumor_present",
        "pancreas_tumor_present",
        "tumor_burden",
        "lesion_multiplicity",
        "cross_organ_distribution",
    ]
    query_rows, rel_rows = [], []
    qid = 0
    for qcase in test_cases:
        vals = latent[qcase]
        chosen = rng.choice(query_features, size=2, replace=False).tolist()
        primary, secondary = [], []
        for idx, feature in enumerate(chosen):
            support = next(s for f, _, s in feature_specs if f == feature)
            if feature == "tumor_burden":
                threshold = float(np.quantile([latent[c]["tumor_burden"] for c in case_ids], 0.55))
                pred = {"feature": feature, "op": ">=", "value": threshold, "support_organs": list(support)}
            elif feature == "lesion_multiplicity":
                pred = {"feature": feature, "op": ">=", "value": 2, "support_organs": list(support)}
            else:
                pred = {"feature": feature, "op": "==", "value": int(vals[feature]), "support_organs": list(support)}
            (primary if idx == 0 else secondary).append(pred)

        query_id = f"Q{qid:04d}"
        qid += 1
        query_rows.append({
            "query_id": query_id,
            "query_case_id": qcase,
            "primary_predicates": json.dumps(primary),
            "secondary_predicates": json.dumps(secondary),
        })

        for cand in case_ids:
            if cand == qcase:
                continue
            cand_vals = latent[cand]

            def holds(p, cand_vals=cand_vals):
                x = cand_vals[p["feature"]]
                if p["op"] == "==":
                    return x == p["value"]
                if p["op"] == ">=":
                    return x >= p["value"]
                if p["op"] == "<=":
                    return x <= p["value"]
                raise ValueError(p["op"])

            p_ok = all(holds(p) for p in primary)
            s_count = sum(holds(p) for p in secondary)
            binary = int(p_ok and s_count == len(secondary))
            if not p_ok:
                grade = 0.0
            elif not secondary:
                grade = 3.0
            else:
                grade = 1.0 + 2.0 * s_count / len(secondary)
            rel_rows.append({
                "query_id": query_id,
                "candidate_id": cand,
                "binary_relevance": binary,
                "graded_relevance": grade,
            })

    return ExperimentData(
        cases=cases,
        phenotypes_ref=phen_ref,
        phenotypes_pred=phen_pred,
        queries=pd.DataFrame(query_rows),
        relevance=pd.DataFrame(rel_rows),
    )


def load_experiment_data(data_dir: Path) -> ExperimentData:
    """Load a real benchmark from the five required CSVs under ``data_dir``."""
    data_dir = Path(data_dir)
    required = {
        "cases": data_dir / "cases.csv",
        "phenotypes_ref": data_dir / "phenotypes_ref.csv",
        "phenotypes_pred": data_dir / "phenotypes_pred.csv",
        "queries": data_dir / "queries.csv",
        "relevance": data_dir / "relevance.csv",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    return ExperimentData(**{name: pd.read_csv(path) for name, path in required.items()})


def load_data(config: Config) -> ExperimentData:
    """Dispatch to demo or real data based on ``config.use_demo_data``."""
    if config.use_demo_data:
        return generate_demo_data(config)
    return load_experiment_data(config.data_dir)


# ---------------------------------------------------------------------------
# Validation and schema
# ---------------------------------------------------------------------------
def validate_data(data: ExperimentData) -> None:
    case_ids = set(data.cases["case_id"])
    for name, frame in {
        "phenotypes_ref": data.phenotypes_ref,
        "phenotypes_pred": data.phenotypes_pred,
    }.items():
        unknown = set(frame["case_id"]) - case_ids
        if unknown:
            raise ValueError(f"{name} contains unknown cases: {sorted(unknown)[:5]}")

    qids = set(data.queries["query_id"])
    if set(data.relevance["query_id"]) - qids:
        raise ValueError("Relevance table contains unknown queries.")
    if set(data.relevance["candidate_id"]) - case_ids:
        raise ValueError("Relevance table contains unknown candidate cases.")
    if data.relevance.duplicated(["query_id", "candidate_id"]).any():
        raise ValueError("Duplicate query/candidate relevance rows detected.")


def phenotype_schema(phenotypes: pd.DataFrame) -> pd.DataFrame:
    schema = (
        phenotypes[["feature", "feature_type", "support_organs"]]
        .drop_duplicates()
        .sort_values("feature")
        .reset_index(drop=True)
    )
    if schema["feature"].duplicated().any():
        raise ValueError("A feature has inconsistent type or support metadata.")
    return schema


# ---------------------------------------------------------------------------
# Fitted corpus state (replaces the notebook's module-level globals)
# ---------------------------------------------------------------------------
@dataclass
class Corpus:
    """Feature matrices, schema, and training statistics derived from data.

    Built once via :meth:`build`; passed explicitly to every scoring function.
    """

    schema: pd.DataFrame
    features: list[str]
    feature_index: dict[str, int]
    feature_support: dict[str, frozenset[str]]
    feature_type: dict[str, str]
    case_order: list[str]
    case_to_row: dict[str, int]
    x_ref_full: np.ndarray
    m_ref_full: np.ndarray
    x_pred_full: np.ndarray
    m_pred_full: np.ndarray
    feature_ranges: np.ndarray
    train_rows: np.ndarray
    feature_means: np.ndarray
    feature_modes: np.ndarray
    query_case_ids: set[str]

    @classmethod
    def build(cls, data: ExperimentData, config: Config) -> "Corpus":
        schema = phenotype_schema(data.phenotypes_ref)
        features = schema["feature"].tolist()
        feature_index = {f: i for i, f in enumerate(features)}
        feature_support = {
            row.feature: parse_organs(row.support_organs)
            for row in schema.itertuples(index=False)
        }
        feature_type = dict(zip(schema["feature"], schema["feature_type"]))

        x_ref, m_ref, case_order = _feature_matrix(data.cases, data.phenotypes_ref, features)
        x_pred, m_pred, _ = _feature_matrix(data.cases, data.phenotypes_pred, features)
        case_to_row = {c: i for i, c in enumerate(case_order)}

        with np.errstate(all="ignore"):
            ranges = np.nanmax(x_ref, axis=0) - np.nanmin(x_ref, axis=0)
        ranges = np.where((~np.isfinite(ranges)) | (ranges == 0), 1.0, ranges)

        train_rows = data.cases["split"].eq("train").to_numpy()
        means, modes = _fit_imputation_statistics(x_ref, m_ref, train_rows, features)

        return cls(
            schema=schema,
            features=features,
            feature_index=feature_index,
            feature_support=feature_support,
            feature_type=feature_type,
            case_order=case_order,
            case_to_row=case_to_row,
            x_ref_full=x_ref,
            m_ref_full=m_ref,
            x_pred_full=x_pred,
            m_pred_full=m_pred,
            feature_ranges=ranges,
            train_rows=train_rows,
            feature_means=means,
            feature_modes=modes,
            query_case_ids=set(data.queries["query_case_id"]),
        )


def _feature_matrix(
    cases: pd.DataFrame,
    phenotypes: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    case_order = cases["case_id"].tolist()
    pivot = phenotypes.pivot(index="case_id", columns="feature", values="value")
    pivot = pivot.reindex(index=case_order, columns=features)
    X = pivot.to_numpy(dtype=float)
    M = ~np.isnan(X)
    return X, M, case_order


def _fit_imputation_statistics(
    X: np.ndarray,
    M: np.ndarray,
    train_rows: np.ndarray,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train = X[train_rows]
    train_m = M[train_rows]
    means = np.zeros(X.shape[1], dtype=float)
    modes = np.zeros(X.shape[1], dtype=float)
    for j in range(len(features)):
        vals = train[train_m[:, j], j]
        means[j] = np.mean(vals) if len(vals) else 0.0
        if len(vals):
            unique, counts = np.unique(vals, return_counts=True)
            modes[j] = unique[np.argmax(counts)]
        else:
            modes[j] = 0.0
    return means, modes
