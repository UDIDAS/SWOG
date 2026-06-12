"""
Supernode and edge feature extraction for SEMIR graph minors.

Node features (7):  volume, boundary_length, compactness, elongation,
                    dominant_axis, mean_intensity, intensity_std
Edge features (4):  log_volume_ratio, intensity_diff_norm,
                    distance_norm, orientation_cos
"""

import numpy as np
from scipy.ndimage import find_objects, binary_erosion, generate_binary_structure


def extract_node_features(labels: np.ndarray,
                          volume: np.ndarray,
                          voxel_spacing: tuple = (1.0, 1.0, 1.0)) -> dict:
    """
    Compute per-supernode features from the label map and intensity volume.
    Uses find_objects for per-supernode bounding boxes to avoid full-volume scans.

    Returns dict  {supernode_id: feature_dict}
    """
    max_id = int(labels.max())
    if max_id == 0:
        return {}

    flat_labels = labels.ravel()
    flat_vol = volume.ravel().astype(np.float64)

    # Vectorised basic stats via bincount
    counts = np.bincount(flat_labels, minlength=max_id + 1)
    int_sums = np.bincount(flat_labels, weights=flat_vol, minlength=max_id + 1)
    int_sq_sums = np.bincount(flat_labels, weights=flat_vol ** 2, minlength=max_id + 1)
    safe_counts = np.where(counts > 0, counts, 1).astype(np.float64)
    means = int_sums / safe_counts
    variances = int_sq_sums / safe_counts - means ** 2
    stds = np.sqrt(np.maximum(variances, 0))

    # Per-supernode bounding boxes for localised operations
    slices = find_objects(labels)  # list of tuple-of-slices, indexed by (id-1)

    # Boundary: erode the full label volume once, then count per-supernode
    struct = generate_binary_structure(3, 1)
    fg = labels > 0
    eroded = binary_erosion(fg, structure=struct)
    boundary_mask = fg & ~eroded
    boundary_counts = np.bincount(flat_labels * boundary_mask.ravel().astype(np.int64),
                                  minlength=max_id + 1)

    # Centroid via weighted coordinate sums
    zz, yy, xx = np.meshgrid(
        np.arange(labels.shape[0], dtype=np.float64),
        np.arange(labels.shape[1], dtype=np.float64),
        np.arange(labels.shape[2], dtype=np.float64),
        indexing="ij",
    )
    z_sums = np.bincount(flat_labels, weights=zz.ravel(), minlength=max_id + 1)
    y_sums = np.bincount(flat_labels, weights=yy.ravel(), minlength=max_id + 1)
    x_sums = np.bincount(flat_labels, weights=xx.ravel(), minlength=max_id + 1)
    cz = z_sums / safe_counts
    cy = y_sums / safe_counts
    cx = x_sums / safe_counts

    features = {}
    active_ids = np.where(counts > 0)[0]
    active_ids = active_ids[active_ids > 0]

    for sid in active_ids:
        a_u = int(counts[sid])
        b_u = int(boundary_counts[sid])
        # 3D isoperimetric ratio: 36πV²/A³ (=1 for perfect sphere).
        # a_u = volume in voxels, b_u = boundary voxel count (proxy for surface area).
        # This is the 3D formulation; the 2D version would be 4πA/P².
        compactness = (36.0 * np.pi * a_u ** 2) / (b_u ** 3 + 1e-8)

        centroid = [float(cz[sid]), float(cy[sid]), float(cx[sid])]

        # Elongation + dominant axis vector via PCA in physical (mm) coordinates.
        # Voxel coords are scaled by spacing to avoid bias from anisotropic
        # acquisition (e.g. 0.64×0.64×2.5mm pancreas CT).
        elongation = 1.0
        dominant_axis = 0.0
        dominant_axis_vec = [1.0, 0.0, 0.0]  # default for tiny supernodes
        if a_u >= 20 and slices is not None and sid - 1 < len(slices) and slices[sid - 1] is not None:
            sl = slices[sid - 1]
            local_mask = labels[sl] == sid
            coords = np.argwhere(local_mask).astype(np.float64)
            if len(coords) >= 3:
                # Scale voxel indices to mm using spacing
                spacing_arr = np.array(voxel_spacing, dtype=np.float64)
                coords_mm = coords * spacing_arr
                centered = coords_mm - coords_mm.mean(axis=0)
                try:
                    cov = np.cov(centered.T)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    order = np.argsort(eigvals)[::-1]
                    eigvals = eigvals[order]
                    eigvecs = eigvecs[:, order]
                    elongation = min(float(eigvals[0] / (eigvals[-1] + 1e-8)), 100.0)
                    dominant_axis = float(eigvals[0])
                    dominant_axis_vec = eigvecs[:, 0].tolist()
                except np.linalg.LinAlgError:
                    pass

        features[int(sid)] = {
            "id": int(sid),
            "volume": a_u,
            "boundary_length": b_u,
            "compactness": round(float(compactness), 6),
            "elongation": round(float(elongation), 4),
            "dominant_axis": round(dominant_axis, 4),
            "dominant_axis_vec": dominant_axis_vec,
            "mean_intensity": round(float(means[sid]), 4),
            "intensity_std": round(float(stds[sid]), 4),
            "centroid": [round(c, 2) for c in centroid],
        }

    return features


def _compute_shared_boundary(labels: np.ndarray, id_a: int, id_b: int) -> int:
    """Count boundary voxels shared between two adjacent supernodes."""
    # Vectorised: count faces where label transitions from id_a to id_b
    count = 0
    for axis in range(3):
        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        sl_lo[axis] = slice(0, -1)
        sl_hi[axis] = slice(1, None)
        a = labels[tuple(sl_lo)]
        b = labels[tuple(sl_hi)]
        count += int(((a == id_a) & (b == id_b)).sum())
        count += int(((a == id_b) & (b == id_a)).sum())
    return count


def _precompute_shared_boundaries(labels: np.ndarray, adjacency: dict) -> dict:
    """Precompute shared boundary length for all adjacent pairs. Vectorised."""
    flat = labels.ravel()
    H, W, D = labels.shape
    idx = np.arange(H * W * D, dtype=np.int64).reshape(H, W, D)

    # Collect all boundary face pairs
    all_a, all_b = [], []
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

    # Canonicalise: (min, max)
    lo = np.minimum(a_arr, b_arr)
    hi = np.maximum(a_arr, b_arr)
    max_id = int(labels.max())
    pair_keys = lo.astype(np.int64) * (max_id + 1) + hi.astype(np.int64)

    # Count occurrences per pair
    unique_keys, counts = np.unique(pair_keys, return_counts=True)
    result = {}
    for k, c in zip(unique_keys, counts):
        id_lo = int(k // (max_id + 1))
        id_hi = int(k % (max_id + 1))
        result[(id_lo, id_hi)] = int(c)
    return result


def extract_edge_features(labels: np.ndarray,
                          volume: np.ndarray,
                          adjacency: dict,
                          node_features: dict) -> dict:
    """
    Compute per-edge features for supernode pairs.

    Features (12 per edge — Luke's expanded set):
      Original 4: log_volume_ratio, intensity_diff_norm, distance_norm, orientation_cos
      New ratios: compactness_ratio, elongation_ratio, boundary_ratio, intensity_std_ratio
      New relational: shared_boundary, shared_boundary_frac_a, shared_boundary_frac_b,
                      intensity_gradient (diff / distance)

    Returns dict  {(id_a, id_b): feature_dict}
    """
    edge_feats = {}
    max_dim = max(labels.shape)

    # Precompute shared boundaries vectorised
    shared_bounds = _precompute_shared_boundaries(labels, adjacency)

    for (i, j), _ in adjacency.items():
        fi = node_features.get(i)
        fj = node_features.get(j)
        if fi is None or fj is None:
            continue

        # --- Original 4 features ---
        log_vol_ratio = float(np.log(fi["volume"] / (fj["volume"] + 1e-8) + 1e-8))

        int_range = max(abs(fi["mean_intensity"]) + abs(fj["mean_intensity"]), 1e-8)
        int_diff_norm = abs(fi["mean_intensity"] - fj["mean_intensity"]) / int_range

        ci = np.array(fi["centroid"])
        cj = np.array(fj["centroid"])
        diff = cj - ci
        dist = float(np.linalg.norm(diff))
        dist_norm = dist / max_dim

        di = np.array(fi.get("dominant_axis_vec", [1, 0, 0]), dtype=np.float64)
        dj = np.array(fj.get("dominant_axis_vec", [1, 0, 0]), dtype=np.float64)
        di_norm = np.linalg.norm(di) + 1e-8
        dj_norm = np.linalg.norm(dj) + 1e-8
        cos_theta = float(abs(np.dot(di / di_norm, dj / dj_norm)))

        # --- Luke's additional ratios ---
        # Ratios between all geometric properties (scale-invariant)
        compact_ratio = fi["compactness"] / (fj["compactness"] + 1e-8)
        log_compact_ratio = float(np.log(compact_ratio + 1e-8))

        elong_ratio = fi["elongation"] / (fj["elongation"] + 1e-8)
        log_elong_ratio = float(np.log(elong_ratio + 1e-8))

        boundary_ratio = fi["boundary_length"] / (fj["boundary_length"] + 1e-8)
        log_boundary_ratio = float(np.log(boundary_ratio + 1e-8))

        std_ratio = fi["intensity_std"] / (fj["intensity_std"] + 1e-8)
        log_std_ratio = float(np.log(std_ratio + 1e-8))

        # --- Shared boundary (structural) ---
        key = (min(i, j), max(i, j))
        shared = shared_bounds.get(key, 0)
        # Fraction of each supernode's boundary that is shared
        shared_frac_a = shared / (fi["boundary_length"] + 1e-8)
        shared_frac_b = shared / (fj["boundary_length"] + 1e-8)

        # Intensity gradient: how fast intensity changes per unit distance
        int_gradient = abs(fi["mean_intensity"] - fj["mean_intensity"]) / (dist + 1e-8)

        edge_feats[(i, j)] = {
            "log_volume_ratio": round(log_vol_ratio, 4),
            "intensity_diff_norm": round(float(int_diff_norm), 4),
            "distance_norm": round(float(dist_norm), 4),
            "orientation_cos": round(cos_theta, 4),
            "log_compactness_ratio": round(log_compact_ratio, 4),
            "log_elongation_ratio": round(log_elong_ratio, 4),
            "log_boundary_ratio": round(log_boundary_ratio, 4),
            "log_std_ratio": round(log_std_ratio, 4),
            "shared_boundary": round(float(shared), 4),
            "shared_boundary_frac_a": round(float(shared_frac_a), 4),
            "shared_boundary_frac_b": round(float(shared_frac_b), 4),
            "intensity_gradient": round(float(int_gradient), 6),
        }

    return edge_feats


def build_pyg_graph(node_features: dict,
                    edge_features: dict,
                    labels_vol: np.ndarray,
                    gt_seg: np.ndarray = None):
    """
    Build a PyTorch Geometric Data object from SEMIR features.

    If gt_seg is provided, assigns binary labels:
      1 = supernode overlaps with tumor (label==2 in gt_seg)
      0 = background
    """
    import torch
    from torch_geometric.data import Data

    sids = sorted(node_features.keys())
    sid_to_idx = {s: i for i, s in enumerate(sids)}

    node_feat_names = ["volume", "boundary_length", "compactness",
                       "elongation", "dominant_axis",
                       "mean_intensity", "intensity_std"]
    x = np.array([[node_features[s][f] for f in node_feat_names] for s in sids],
                 dtype=np.float32)

    for col in range(x.shape[1]):
        mn, mx = x[:, col].min(), x[:, col].max()
        if mx - mn > 1e-8:
            x[:, col] = (x[:, col] - mn) / (mx - mn)

    edge_idx = []
    edge_attr = []
    edge_feat_names = ["log_volume_ratio", "intensity_diff_norm",
                       "distance_norm", "orientation_cos",
                       "log_compactness_ratio", "log_elongation_ratio",
                       "log_boundary_ratio", "log_std_ratio",
                       "shared_boundary", "shared_boundary_frac_a",
                       "shared_boundary_frac_b", "intensity_gradient"]

    for (i, j), ef in edge_features.items():
        if i in sid_to_idx and j in sid_to_idx:
            edge_idx.append([sid_to_idx[i], sid_to_idx[j]])
            edge_idx.append([sid_to_idx[j], sid_to_idx[i]])
            feat = [ef[f] for f in edge_feat_names]
            edge_attr.append(feat)
            edge_attr.append(feat)

    if len(edge_idx) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_t = torch.zeros((0, len(edge_feat_names)), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_idx, dtype=torch.long).t().contiguous()
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr_t,
    )

    # Vectorised GT label assignment
    if gt_seg is not None:
        flat_labels = labels_vol.ravel()
        flat_gt = (gt_seg.ravel() == 2).astype(np.int64)
        max_id = int(flat_labels.max())
        tumor_counts = np.bincount(flat_labels, weights=flat_gt.astype(np.float64),
                                   minlength=max_id + 1)
        total_counts = np.bincount(flat_labels, minlength=max_id + 1)
        safe_total = np.where(total_counts > 0, total_counts, 1)
        overlap_frac = tumor_counts / safe_total

        y = np.array([int(overlap_frac[s] > 0.1) for s in sids], dtype=np.int64)
        data.y = torch.tensor(y, dtype=torch.long)

    # Store mapping externally (not as Data attrs — breaks PyG collation)
    mapping = {"sid_to_idx": sid_to_idx,
               "idx_to_sid": {i: s for s, i in sid_to_idx.items()}}

    return data, mapping
