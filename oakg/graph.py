"""Weisfeiler-Lehman graph-kernel baseline (Section 8.1).

The case-centered graph uses exactly the nodes, relation labels, and evidence
permitted by the masking condition. WL relabeling uses a stable hash so
embeddings are reproducible across processes (Python's built-in ``hash`` is
salted per-run and must not be used here).
"""
from __future__ import annotations

import hashlib

import networkx as nx
import numpy as np

from .data import Corpus


def _stable_label(signature: str) -> str:
    return hashlib.md5(signature.encode("utf-8")).hexdigest()[:16]


def case_graph(case_id: str, X: np.ndarray, M: np.ndarray, corpus: Corpus) -> nx.Graph:
    idx = corpus.case_to_row[case_id]
    graph = nx.Graph()
    graph.add_node(f"case:{case_id}", label="CTScan")

    for feature, col in corpus.feature_index.items():
        if not M[idx, col]:
            continue
        value = X[idx, col]
        support = corpus.feature_support[feature]
        fnode = f"feature:{feature}:{value:.3f}"
        graph.add_node(fnode, label=f"{feature}={round(float(value), 2)}")
        graph.add_edge(f"case:{case_id}", fnode, relation="hasPhenotype")
        for organ in support:
            onode = f"organ:{organ}"
            graph.add_node(onode, label=f"Organ:{organ}")
            graph.add_edge(fnode, onode, relation="supportedBy")
    return graph


def wl_feature_dict(graph: nx.Graph, h: int = 2) -> dict[str, float]:
    labels = {n: str(graph.nodes[n].get("label", "")) for n in graph.nodes}
    counts: dict[str, float] = {}
    for label in labels.values():
        counts[f"0:{label}"] = counts.get(f"0:{label}", 0.0) + 1.0

    for iteration in range(1, h + 1):
        new_labels = {}
        for node in graph.nodes:
            neighbors = sorted(labels[nbr] for nbr in graph.neighbors(node))
            signature = labels[node] + "||" + "||".join(neighbors)
            new_label = _stable_label(signature)
            new_labels[node] = new_label
            key = f"{iteration}:{new_label}"
            counts[key] = counts.get(key, 0.0) + 1.0
        labels = new_labels
    return counts


def wl_embeddings(
    X: np.ndarray,
    M: np.ndarray,
    corpus: Corpus,
    h: int = 2,
) -> tuple[np.ndarray, list[str]]:
    feature_dicts = [wl_feature_dict(case_graph(cid, X, M, corpus), h=h) for cid in corpus.case_order]
    vocab = sorted(set().union(*(d.keys() for d in feature_dicts)))
    index = {k: i for i, k in enumerate(vocab)}
    emb = np.zeros((len(corpus.case_order), len(vocab)), dtype=float)
    for row, d in enumerate(feature_dicts):
        for key, value in d.items():
            emb[row, index[key]] = value
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.where(norms == 0, 1.0, norms)
    return emb, vocab
