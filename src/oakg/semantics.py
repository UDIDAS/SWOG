"""Structured-query semantics: closed-world, open-world, and OAKG three-valued (Section 10.2)."""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .data import Corpus, ExperimentData

T, F, U = "T", "F", "U"


def predicate_holds(value: float, op: str, target: float) -> bool:
    if op == "==":
        return value == target
    if op == ">=":
        return value >= target
    if op == "<=":
        return value <= target
    if op == ">":
        return value > target
    if op == "<":
        return value < target
    raise ValueError(op)


def evaluate_predicate(
    case_id: str,
    predicate: Mapping[str, Any],
    X: np.ndarray,
    M: np.ndarray,
    semantics: str,
    corpus: Corpus,
) -> str:
    idx = corpus.case_to_row[case_id]
    j = corpus.feature_index[predicate["feature"]]
    observed = bool(M[idx, j])

    if semantics == "closed_world":
        if not observed:
            return F
    elif semantics in {"open_world", "oakg"}:
        if not observed:
            return U
    else:
        raise ValueError(semantics)

    return T if predicate_holds(X[idx, j], predicate["op"], predicate["value"]) else F


def strong_kleene_and(values: Sequence[str]) -> str:
    if F in values:
        return F
    if all(v == T for v in values):
        return T
    return U


def evaluate_query_on_case(
    case_id: str,
    primary: Sequence[Mapping[str, Any]],
    secondary: Sequence[Mapping[str, Any]],
    X: np.ndarray,
    M: np.ndarray,
    semantics: str,
    corpus: Corpus,
) -> str:
    vals = [
        evaluate_predicate(case_id, p, X, M, semantics, corpus)
        for p in [*primary, *secondary]
    ]
    if semantics in {"open_world", "oakg"}:
        return strong_kleene_and(vals)
    return T if all(v == T for v in vals) else F


def structured_query_evaluation(
    X: np.ndarray,
    M: np.ndarray,
    track: str,
    data: ExperimentData,
    corpus: Corpus,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for query in data.queries.itertuples(index=False):
        primary = json.loads(query.primary_predicates)
        secondary = json.loads(query.secondary_predicates)
        rel = data.relevance[data.relevance["query_id"].eq(query.query_id)]
        truth_lookup = dict(
            zip(rel["candidate_id"].astype(str), rel["binary_relevance"].astype(int))
        )
        for candidate, binary_truth in truth_lookup.items():
            truth = T if binary_truth else F
            for semantics in ["closed_world", "open_world", "oakg"]:
                prediction = evaluate_query_on_case(
                    candidate, primary, secondary, X, M, semantics, corpus
                )
                rows.append({
                    "query_id": query.query_id,
                    "case_id": candidate,
                    "track": track,
                    "semantics": semantics,
                    "truth": truth,
                    "prediction": prediction,
                })
    detail = pd.DataFrame(rows)

    summary_rows = []
    labels = [T, F, U]
    for semantics, group in detail.groupby("semantics"):
        y_true = group["truth"].tolist()
        y_pred = group["prediction"].tolist()
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        summary_rows.append({
            "track": track,
            "semantics": semantics,
            "precision_T": p[0], "recall_T": r[0], "f1_T": f[0],
            "precision_F": p[1], "recall_F": r[1], "f1_F": f[1],
            "precision_U": p[2], "recall_U": r[2], "f1_U": f[2],
            "macro_F1": np.mean(f),
            "indeterminate_rate": np.mean(group["prediction"].eq(U)),
            "unsupported_negative_rate": np.mean(
                group["prediction"].eq(F) & group["truth"].eq(T)
            ),
        })
    return detail, pd.DataFrame(summary_rows)
