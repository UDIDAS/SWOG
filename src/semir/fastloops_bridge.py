"""
Bridge between fastloops Rust crate and the SEMIR pipeline.

Converts fastloops output (node_feats, edge_index, edge_feats, labels)
into the format expected by SEMIR's GINE training, VKG builder, etc.

The Rust crate uses seed-based flood fill (comparing each voxel to the
region's canonical voxel) instead of connected components, preventing
transitive intensity drift that plagued the Python implementation.
"""

import numpy as np
import time

try:
    import fastloops
except ImportError:
    fastloops = None


def _layout_3d(C):
    """Feature layout matching the Rust crate's 3D output."""
    return dict(
        area=0, s=[1, 2, 3],
        cov=[(4, 0, 0), (5, 1, 1), (6, 2, 2), (7, 0, 1), (8, 0, 2), (9, 1, 2)],
        chan0=10, boundary=10 + C + 6, D=3,
    )


def node_invariants(node_feats, C=1, eps=1e-6):
    """
    Extract scale- and orientation-invariant node features from the raw
    Rust output. Returns dict of arrays indexed by supernode ID.
    """
    f = node_feats.astype(np.float64)
    L = _layout_3d(C)
    D = L["D"]
    N = f.shape[0]

    V = f[:, L["area"]]
    Vsafe = np.maximum(V, 1.0)

    # Centroid
    mean_coord = np.stack([f[:, c] for c in L["s"]], axis=1) / Vsafe[:, None]

    # Covariance matrix
    cov = np.zeros((N, D, D))
    for col, i, j in L["cov"]:
        cij = f[:, col] / Vsafe - mean_coord[:, i] * mean_coord[:, j]
        cov[:, i, j] = cij
        cov[:, j, i] = cij

    # Eigendecomposition
    w = np.linalg.eigvalsh(cov)
    w = np.clip(w, 0.0, None)
    _, vec = np.linalg.eigh(cov)
    principal = vec[..., -1]
    trace = w.sum(axis=1)
    degenerate = trace < eps

    # Shape descriptors (linearity, planarity, sphericity)
    denom = w[:, 2] + eps
    shape = np.stack([
        (w[:, 2] - w[:, 1]) / denom,
        (w[:, 1] - w[:, 0]) / denom,
        w[:, 0] / denom,
    ], axis=1)
    shape[degenerate] = 0.0

    line_like = np.where(degenerate, 0.0, shape[:, 0])

    # Mean channel intensity (normalized to [0, 1])
    chan = f[:, L["chan0"]:L["chan0"] + C] / Vsafe[:, None] / 255.0

    # Compactness: surface / volume^((D-1)/D)
    boundary = f[:, L["boundary"]]
    compactness = boundary / np.power(Vsafe, (D - 1.0) / D)

    # Elongation from eigenvalues
    elongation = np.where(w[:, 0] > eps, w[:, 2] / (w[:, 0] + eps), 1.0)
    elongation = np.clip(elongation, 1.0, 100.0)

    return dict(
        volume=V,
        boundary=boundary,
        centroid=mean_coord,
        compactness=compactness,
        elongation=elongation,
        mean_intensity=chan[:, 0] if C == 1 else chan,
        intensity_std=np.zeros(N),  # not available from raw Rust output
        principal=principal,
        line_like=line_like,
        shape=shape,
        eig=w,
        log_size=np.log(Vsafe),
    )


def edge_invariants(node_feats, edge_index, edge_feats, C=1, eps=1e-6):
    """Extract scale-invariant edge features."""
    inv = node_invariants(node_feats, C, eps)
    a = edge_index[0].astype(np.int64)
    b = edge_index[1].astype(np.int64)
    ef = edge_feats.astype(np.float64)
    blsafe = np.maximum(ef[:, 0], 1.0)

    size_contrast = np.abs(inv["volume"][a] - inv["volume"][b]) / (inv["volume"][a] + inv["volume"][b] + eps)
    bfrac_a = ef[:, 0] / (inv["boundary"][a] + eps)
    bfrac_b = ef[:, 0] / (inv["boundary"][b] + eps)
    mean_contrast = np.abs(inv["mean_intensity"][a] - inv["mean_intensity"][b])
    shape_dissim = np.abs(inv["shape"][a] - inv["shape"][b])
    axis_align = (np.abs(np.sum(inv["principal"][a] * inv["principal"][b], axis=1))
                  * np.minimum(inv["line_like"][a], inv["line_like"][b]))
    bcontrast = (ef[:, 1] / blsafe) / 255.0
    cut_frac = ef[:, 3] / blsafe

    cols = [
        size_contrast[:, None],
        bfrac_a[:, None],
        bfrac_b[:, None],
        mean_contrast[:, None] if mean_contrast.ndim == 1 else np.abs(mean_contrast),
        shape_dissim,
        axis_align[:, None],
        bcontrast[:, None],
        cut_frac[:, None],
    ]
    return np.concatenate(cols, axis=1).astype(np.float32)


def build_graph_minor_rust(volume_u8: np.ndarray,
                           merge_distance: int = 30,
                           cut_distance: int = 90,
                           delete_small: int = 10,
                           delete_large: int = None,
                           connectivity: str = "faces"):
    """
    Build graph minor using the Rust fastloops crate.

    Parameters
    ----------
    volume_u8 : uint8 array, shape (D, H, W, C) — channel-last, C-contiguous
    merge_distance : max intensity diff for merging (uint8 scale)
    cut_distance : min intensity diff for edge deletion (uint8 scale)
    delete_small : delete supernodes with fewer voxels than this
    delete_large : delete supernodes with more voxels than this (default: n_voxels)
    connectivity : 'faces' (6-conn), 'faces_edges', or 'faces_corners'

    Returns
    -------
    dict matching build_graph_minor() output format:
        labels, n_supernodes, adjacency, full_adjacency, stats,
        plus rust_node_feats, rust_edge_index, rust_edge_feats for direct use
    """
    if fastloops is None:
        raise ImportError("fastloops Rust crate not installed")

    D, H, W, C = volume_u8.shape
    n_voxels = D * H * W
    if delete_large is None:
        delete_large = n_voxels

    t0 = time.time()
    raw_nf, raw_ei, raw_ef, raw_labels, raw_adj = fastloops.merge_and_cut(
        volume_u8,
        merge_distance=merge_distance,
        cut_distance=cut_distance,
        delete_small_node_max_size=delete_small,
        delete_large_node_min_size=delete_large,
        connectivity=connectivity,
    )
    t_rust = time.time() - t0

    labels_np = np.asarray(raw_labels)  # (D, H, W), int64, -1 = deleted
    n_supernodes = raw_nf.shape[0]
    n_edges = raw_ei.shape[1]

    # Convert labels: -1 (deleted) -> 0, supernode IDs from 0-based -> 1-based
    labels_out = labels_np.copy()
    labels_out[labels_out >= 0] += 1  # shift to 1-based
    labels_out[labels_np < 0] = 0     # deleted -> 0

    # Build adjacency dict matching Python format
    ei = raw_ei  # shape (2, n_edges)
    adjacency = {}
    full_adjacency = {}
    if n_edges > 0:
        a_ids = ei[0].astype(int) + 1  # 1-based
        b_ids = ei[1].astype(int) + 1
        ef_np = raw_ef.astype(np.float64)
        mean_contrasts = ef_np[:, 1] / np.maximum(ef_np[:, 0], 1.0) / 255.0

        for idx in range(n_edges):
            key = (int(min(a_ids[idx], b_ids[idx])),
                   int(max(a_ids[idx], b_ids[idx])))
            full_adjacency[key] = float(mean_contrasts[idx])
            # alpha-filtered: only keep edges below cut threshold
            cut_frac = float(ef_np[idx, 3] / max(ef_np[idx, 0], 1))
            if cut_frac < 0.5:  # less than half the boundary is "cut"
                adjacency[key] = float(mean_contrasts[idx])

    stats = {
        "n_voxels": n_voxels,
        "n_supernodes_after_contraction": n_supernodes,
        "n_supernodes_after_deletion": n_supernodes,
        "n_edges": len(adjacency),
        "n_edges_full": len(full_adjacency),
        "compression_ratio": n_voxels / max(n_supernodes, 1),
        "time_total_s": round(t_rust, 2),
        "merge_distance": merge_distance,
        "cut_distance": cut_distance,
        "delete_small": delete_small,
        "backend": "rust_fastloops",
    }

    return {
        "labels": labels_out,
        "n_supernodes": n_supernodes,
        "adjacency": adjacency,
        "full_adjacency": full_adjacency,
        "stats": stats,
        # Raw Rust outputs for direct feature extraction
        "rust_node_feats": raw_nf,
        "rust_edge_index": raw_ei,
        "rust_edge_feats": raw_ef,
    }


def prepare_volume_u8(ct_raw: np.ndarray, hu_min: int = 20, hu_max: int = 180):
    """Convert raw CT (float32 HU) to uint8 channel-last format for fastloops."""
    ct_u8 = np.clip(ct_raw, hu_min, hu_max)
    ct_u8 = ((ct_u8 - hu_min) / (hu_max - hu_min) * 255).round().astype(np.uint8)
    ct_u8 = np.ascontiguousarray(ct_u8[..., np.newaxis])  # (D, H, W, 1)
    return ct_u8
