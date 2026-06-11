"""
Graph minor construction from 3D medical volumes.

Implements SEMIR's three parameterized operations:
  1. Edge contraction  (ψ) — merge adjacent supernodes with similar intensity
  2. Node deletion     (β) — remove supernodes outside size/intensity bounds
  3. Edge deletion     (α) — sever connections across strong gradients

Contraction strategy: adaptive two-pass watershed with intensity refinement.
Falls back to C flood-fill (paper's Algorithm 3) if semir_minor.so is present.

Output: a reduced graph of supernodes from ~10^7 voxels, plus a
bijective voxel→supernode map for exact lifting.
"""

import numpy as np
from scipy.ndimage import label as nd_label, find_objects, distance_transform_edt


# ---------------------------------------------------------------------------
# Step 1: Edge contraction — adaptive hybrid (default)
# ---------------------------------------------------------------------------

def _contraction_adaptive(volume: np.ndarray, psi: float,
                          grid_stride: int = 20) -> np.ndarray:
    """
    Adaptive two-pass watershed with intensity refinement.

    Pass 1 (coarse): Watershed at grid_stride → ~2K supervoxels (fast).
    Detect:  Flag supervoxels with intensity std > psi*0.4 (heterogeneous
             = likely contains a tissue boundary or small tumor).
    Pass 2 (fine):  Re-run watershed at stride/3 ONLY inside flagged
             supervoxels, then split by intensity quantization.

    The rest of the volume keeps the coarse partition.
    This is adaptive mesh refinement — fine where anomalous, coarse elsewhere.
    """
    from skimage.segmentation import watershed
    from skimage.filters import sobel

    H, W, D = volume.shape
    vol = volume.astype(np.float64)
    gradient = sobel(vol)

    # ---- Pass 1: Coarse watershed ----
    markers_coarse = np.zeros((H, W, D), dtype=np.int64)
    mid = 0
    for x in range(0, H, grid_stride):
        for y in range(0, W, grid_stride):
            for z in range(0, D, grid_stride):
                mid += 1
                markers_coarse[x, y, z] = mid

    coarse_labels = watershed(gradient, markers=markers_coarse).astype(np.int64)

    # ---- Detect anomalous supervoxels ----
    flat_coarse = coarse_labels.ravel()
    flat_vol = vol.ravel()
    max_c = int(flat_coarse.max())

    counts = np.bincount(flat_coarse, minlength=max_c + 1)
    sums = np.bincount(flat_coarse, weights=flat_vol, minlength=max_c + 1)
    sq_sums = np.bincount(flat_coarse, weights=flat_vol ** 2, minlength=max_c + 1)
    safe = np.where(counts > 0, counts, 1)
    means = sums / safe
    stds = np.sqrt(np.maximum(sq_sums / safe - means ** 2, 0))

    # Flag: high std AND enough voxels to be worth refining
    threshold = psi * 0.4
    anomalous = (stds > threshold) & (counts > 30)
    n_anomalous = anomalous.sum()

    if n_anomalous == 0:
        return coarse_labels

    # Build anomalous mask (vectorized via LUT)
    anom_lut = anomalous.astype(np.bool_)
    anom_mask = anom_lut[coarse_labels]

    # ---- Pass 2: Fine watershed inside anomalous regions ----
    fine_stride = max(3, grid_stride // 3)
    markers_fine = np.zeros((H, W, D), dtype=np.int64)
    fine_id = max_c + 1
    for x in range(0, H, fine_stride):
        for y in range(0, W, fine_stride):
            for z in range(0, D, fine_stride):
                if anom_mask[x, y, z]:
                    markers_fine[x, y, z] = fine_id
                    fine_id += 1

    if fine_id == max_c + 1:
        return coarse_labels

    # Set markers: coarse labels for non-anomalous, fine markers for anomalous
    combined_markers = coarse_labels.copy()
    combined_markers[anom_mask] = 0
    fine_marker_mask = markers_fine > 0
    combined_markers[fine_marker_mask] = markers_fine[fine_marker_mask]

    refined_labels = watershed(gradient, markers=combined_markers).astype(np.int64)

    # ---- Stage 3: Intensity split within refined labels ----
    n_bins = max(2, int(np.ceil(1.0 / max(psi, 1e-6))))
    quantized = np.clip(np.floor(vol * n_bins).astype(np.int64), 0, n_bins - 1)

    # Compound labels only for anomalous voxels
    final_labels = refined_labels.copy()
    max_refined = int(refined_labels.max())

    compound_anom = refined_labels[anom_mask] * n_bins + quantized[anom_mask]
    compound_anom = compound_anom + max_refined * n_bins

    final_labels_flat = final_labels.ravel()
    anom_flat = anom_mask.ravel()
    final_labels_flat[anom_flat] = compound_anom.ravel()
    final_labels = final_labels_flat.reshape(H, W, D)

    # Connected components for spatial coherence
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = H * W * D
    idx = np.arange(n, dtype=np.int64).reshape(H, W, D)
    flat_final = final_labels.ravel()

    all_src, all_dst = [], []
    for axis in range(3):
        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        sl_lo[axis] = slice(0, -1)
        sl_hi[axis] = slice(1, None)
        i1 = idx[tuple(sl_lo)].ravel()
        i2 = idx[tuple(sl_hi)].ravel()
        same = flat_final[i1] == flat_final[i2]
        all_src.append(i1[same])
        all_dst.append(i2[same])

    src = np.concatenate(all_src)
    dst = np.concatenate(all_dst)
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(len(rows), dtype=np.int8)
    adj = coo_matrix((data, (rows, cols)), shape=(n, n))

    _, cc_labels = connected_components(adj, directed=False)
    return (cc_labels + 1).reshape(H, W, D).astype(np.int64)


# ---------------------------------------------------------------------------
# Step 1 (alternative): C flood-fill — paper's Algorithm 3
# ---------------------------------------------------------------------------

def _edge_contraction_c(volume: np.ndarray, psi: float, alpha: float,
                        beta_min: int, beta_max: int,
                        m_min: float, m_max: float) -> np.ndarray:
    """
    SEMIR edge contraction via C flood-fill with canonical-voxel anchoring.

    This implements the paper's Algorithm 3 (FloodFillContract):
      - Each supernode grows from a seed voxel
      - Neighbors are compared against the SEED intensity, not their
        immediate neighbor — prevents transitive intensity drift
      - Coprime traversal order avoids directional bias
      - Node deletion (beta/m bounds) applied during construction

    Returns label volume (0 = deleted, 1..N = supernode IDs).
    """
    import ctypes, os

    H, W, D = volume.shape
    vol = np.ascontiguousarray(volume, dtype=np.float64)
    labels = np.zeros((H, W, D), dtype=np.int32)

    so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "semir_minor.so")
    lib = ctypes.CDLL(so_path)

    lib.semir_minor_construct.restype = ctypes.c_int
    lib.semir_minor_construct.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double,
        ctypes.c_void_p,
    ]

    lib.semir_minor_construct(
        vol.ctypes.data,
        H, W, D,
        psi, alpha,
        beta_min, beta_max,
        m_min, m_max,
        labels.ctypes.data,
    )

    return labels.astype(np.int64)


# ---------------------------------------------------------------------------
# Step 2: Node deletion
# ---------------------------------------------------------------------------

def _node_deletion(labels: np.ndarray, volume: np.ndarray,
                   beta_min: int, beta_max: int,
                   m_min: float, m_max: float) -> np.ndarray:
    """
    Remove supernodes whose size or mean intensity falls outside bounds.
    Deleted supernodes get label 0 (background).
    Fully vectorised — no Python loops over supernodes.
    """
    flat_labels = labels.ravel()
    flat_vol = volume.ravel().astype(np.float64)
    max_id = int(flat_labels.max())

    if max_id == 0:
        return labels.copy()

    counts = np.bincount(flat_labels, minlength=max_id + 1)
    intensity_sums = np.bincount(flat_labels, weights=flat_vol, minlength=max_id + 1)
    safe_counts = np.where(counts > 0, counts, 1)
    mean_int = intensity_sums / safe_counts

    keep = np.ones(max_id + 1, dtype=bool)
    keep[0] = False
    keep &= (counts >= beta_min) & (counts <= beta_max)
    keep &= (mean_int >= m_min) & (mean_int <= m_max)

    lut = np.zeros(max_id + 1, dtype=np.int64)
    new_ids = np.cumsum(keep)
    lut[keep] = new_ids[keep]

    out = lut[flat_labels].reshape(labels.shape)
    return out


# ---------------------------------------------------------------------------
# Step 3: Edge deletion
# ---------------------------------------------------------------------------

def _edge_deletion(labels: np.ndarray, volume: np.ndarray,
                   alpha: float):
    """
    Build adjacency between supernodes. Delete edges where the
    mean-intensity difference exceeds α.
    Fully vectorised — no Python loops over boundary voxels.

    Returns:
        adj : dict  {(id_a, id_b): mean_diff}  for surviving edges
    """
    flat_labels = labels.ravel()
    flat_vol = volume.ravel().astype(np.float64)
    max_id = int(flat_labels.max())

    if max_id == 0:
        return {}

    counts = np.bincount(flat_labels, minlength=max_id + 1)
    sums = np.bincount(flat_labels, weights=flat_vol, minlength=max_id + 1)
    safe_counts = np.where(counts > 0, counts, 1)
    mean_int_arr = sums / safe_counts

    all_a = []
    all_b = []
    for axis in range(3):
        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        sl_lo[axis] = slice(0, -1)
        sl_hi[axis] = slice(1, None)

        a = labels[tuple(sl_lo)].ravel()
        b = labels[tuple(sl_hi)].ravel()
        boundary = (a != b) & (a > 0) & (b > 0)
        all_a.append(a[boundary])
        all_b.append(b[boundary])

    if not all_a:
        return {}

    a_arr = np.concatenate(all_a)
    b_arr = np.concatenate(all_b)

    lo = np.minimum(a_arr, b_arr)
    hi = np.maximum(a_arr, b_arr)

    pair_keys = lo.astype(np.int64) * (max_id + 1) + hi.astype(np.int64)
    unique_keys = np.unique(pair_keys)

    u_lo = unique_keys // (max_id + 1)
    u_hi = unique_keys % (max_id + 1)

    diffs = np.abs(mean_int_arr[u_lo] - mean_int_arr[u_hi])
    keep = diffs <= alpha

    surviving = {}
    for i_lo, i_hi, d in zip(u_lo[keep], u_hi[keep], diffs[keep]):
        surviving[(int(i_lo), int(i_hi))] = float(d)

    return surviving


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph_minor(volume: np.ndarray,
                      psi: float,
                      alpha: float,
                      beta_min: int = 10,
                      beta_max: int = 500000,
                      m_min: float = -1e9,
                      m_max: float = 1e9,
                      fast: bool = True,
                      method: str = "auto"):
    """
    Full SEMIR graph minor construction.

    Parameters
    ----------
    volume   : 3D array (H, W, D), float, typically HU-windowed CT
    psi      : edge contraction threshold (intensity difference)
    alpha    : edge deletion threshold (supernode mean-intensity diff)
    beta_min : minimum supernode size (voxels)
    beta_max : maximum supernode size (voxels)
    m_min    : minimum mean intensity for supernode retention
    m_max    : maximum mean intensity for supernode retention
    fast     : ignored (kept for API compatibility)
    method   : "auto" = C flood-fill if available, else adaptive hybrid.
               "adaptive" = force adaptive two-pass watershed.
               "c" = force C flood-fill (fails if .so missing).

    Returns
    -------
    dict with keys:
        labels        : (H,W,D) int array, supernode ID per voxel (0=deleted)
        n_supernodes  : int
        adjacency     : dict {(i,j): intensity_diff}
        full_adjacency: dict — all edges (no α filter), for GNN
        stats         : dict with compression ratio, timings, etc.
    """
    import time

    t0 = time.time()

    # Resolve method
    resolved = method
    if resolved == "auto":
        import os
        so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "semir_minor.so")
        resolved = "c" if os.path.exists(so_path) else "adaptive"

    if resolved == "c":
        # C does contraction + node deletion in one pass
        labels = _edge_contraction_c(volume, psi, alpha,
                                     beta_min, beta_max, m_min, m_max)
        n_after_contraction = len(np.unique(labels))
        t1 = time.time()
        # Also apply Python node deletion for consistency
        # (C may use slightly different criteria)
        labels = _node_deletion(labels, volume, beta_min, beta_max, m_min, m_max)
        n_after_deletion = len(np.unique(labels[labels > 0]))
        t2 = time.time()
    else:
        # Adaptive two-pass watershed
        labels = _contraction_adaptive(volume, psi)
        n_after_contraction = len(np.unique(labels))
        t1 = time.time()
        labels = _node_deletion(labels, volume, beta_min, beta_max, m_min, m_max)
        n_after_deletion = len(np.unique(labels[labels > 0]))
        t2 = time.time()

    # Step 3: Edge deletion
    adjacency = _edge_deletion(labels, volume, alpha)
    t3 = time.time()

    n_voxels = int(np.prod(volume.shape))
    n_supernodes = int(n_after_deletion)

    stats = {
        "n_voxels": n_voxels,
        "n_supernodes_after_contraction": int(n_after_contraction),
        "n_supernodes_after_deletion": n_supernodes,
        "n_edges": len(adjacency),
        "compression_ratio": n_voxels / max(n_supernodes, 1),
        "time_contraction_s": round(t1 - t0, 2),
        "time_deletion_s": round(t2 - t1, 2),
        "time_edge_deletion_s": round(t3 - t2, 2),
        "time_total_s": round(t3 - t0, 2),
        "psi": psi,
        "alpha": alpha,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "method": resolved,
    }

    # Full adjacency (no α filtering) for GNN use
    full_adjacency = _edge_deletion(labels, volume, alpha=1e9)

    return {
        "labels": labels,
        "n_supernodes": n_supernodes,
        "adjacency": adjacency,
        "full_adjacency": full_adjacency,
        "stats": stats,
    }
