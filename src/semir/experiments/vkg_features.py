"""
VKG-enhanced node features for GINE classification.

Adds organ-aware features on top of SEMIR's 7 geometric features:
  - organ_fraction: what fraction of this supernode is inside the organ (liver)
  - distance_to_organ_center: normalized distance from supernode centroid to organ centroid
  - relative_intensity: supernode mean intensity relative to organ mean
  - boundary_with_organ: does this supernode border the organ boundary?

These leverage VKG's organ awareness — something standalone SEMIR doesn't have.
"""

import numpy as np


def compute_vkg_features(labels: np.ndarray,
                         volume: np.ndarray,
                         gt_seg: np.ndarray,
                         node_features: dict) -> dict:
    """
    Add VKG organ-context features to existing node features.

    Parameters
    ----------
    labels    : supernode label volume
    volume    : intensity volume (HU-windowed, [0,1])
    gt_seg    : GT segmentation (0=bg, 1=organ, 2=tumor)
    node_features : dict from extract_node_features

    Returns
    -------
    dict with same keys, each node gets additional features
    """
    flat_labels = labels.ravel()
    max_id = int(flat_labels.max())
    if max_id == 0:
        return node_features

    # Organ mask (label=1 in GT, or label=1|2 for "liver including tumor")
    organ_mask = (gt_seg >= 1).astype(np.float64).ravel()
    tumor_mask = (gt_seg == 2).astype(np.float64).ravel()

    # Per-supernode: fraction inside organ
    counts = np.bincount(flat_labels, minlength=max_id + 1).astype(np.float64)
    organ_counts = np.bincount(flat_labels, weights=organ_mask, minlength=max_id + 1)
    safe = np.where(counts > 0, counts, 1)
    organ_frac = organ_counts / safe

    # Organ centroid (from organ mask)
    H, W, D = labels.shape
    zz, yy, xx = np.meshgrid(
        np.arange(H, dtype=np.float64),
        np.arange(W, dtype=np.float64),
        np.arange(D, dtype=np.float64),
        indexing="ij",
    )
    organ_voxels = (gt_seg >= 1)
    if organ_voxels.any():
        organ_cz = zz[organ_voxels].mean()
        organ_cy = yy[organ_voxels].mean()
        organ_cx = xx[organ_voxels].mean()
    else:
        organ_cz = H / 2
        organ_cy = W / 2
        organ_cx = D / 2

    # Organ mean intensity
    organ_int = volume[gt_seg >= 1]
    organ_mean_int = organ_int.mean() if len(organ_int) > 0 else 0.5

    # Organ boundary mask (for boundary_with_organ feature)
    from scipy.ndimage import binary_erosion, generate_binary_structure
    struct = generate_binary_structure(3, 1)
    organ_boundary = organ_voxels & ~binary_erosion(organ_voxels, structure=struct)
    organ_boundary_flat = organ_boundary.ravel().astype(np.float64)
    boundary_counts = np.bincount(flat_labels, weights=organ_boundary_flat,
                                  minlength=max_id + 1)

    max_dim = float(max(H, W, D))

    # Add features to each supernode
    for sid, feat in node_features.items():
        if sid > max_id:
            continue

        # Organ fraction: 0 = fully outside organ, 1 = fully inside
        feat["organ_fraction"] = round(float(organ_frac[sid]), 4)

        # Distance to organ center (normalized)
        cz, cy, cx = feat["centroid"]
        dist = np.sqrt((cz - organ_cz)**2 + (cy - organ_cy)**2 + (cx - organ_cx)**2)
        feat["dist_to_organ_center"] = round(float(dist / max_dim), 4)

        # Relative intensity: how different is this supernode from the organ mean?
        feat["relative_intensity"] = round(float(feat["mean_intensity"] - organ_mean_int), 4)

        # Boundary with organ: does this supernode touch the organ boundary?
        feat["organ_boundary_contact"] = round(float(boundary_counts[sid] / safe[sid]), 4)

    return node_features


def build_enhanced_pyg_graph(node_features: dict,
                              edge_features: dict,
                              labels_vol: np.ndarray,
                              gt_seg: np.ndarray = None):
    """
    Build PyG graph with 11 node features (7 geometric + 4 VKG organ-context).
    """
    import torch
    from torch_geometric.data import Data

    sids = sorted(node_features.keys())
    sid_to_idx = {s: i for i, s in enumerate(sids)}

    # 11 features: 7 geometric + 4 VKG
    feat_names = [
        "volume", "boundary_length", "compactness",
        "elongation", "dominant_axis", "mean_intensity", "intensity_std",
        "organ_fraction", "dist_to_organ_center",
        "relative_intensity", "organ_boundary_contact",
    ]

    x = np.array([[node_features[s].get(f, 0.0) for f in feat_names] for s in sids],
                 dtype=np.float32)

    # Normalise each feature to [0, 1]
    for col in range(x.shape[1]):
        mn, mx = x[:, col].min(), x[:, col].max()
        if mx - mn > 1e-8:
            x[:, col] = (x[:, col] - mn) / (mx - mn)

    # Edges
    edge_idx = []
    edge_attr = []
    edge_feat_names = ["log_volume_ratio", "intensity_diff_norm",
                       "distance_norm", "orientation_cos"]

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

    # GT labels
    if gt_seg is not None:
        flat_labels = labels_vol.ravel()
        flat_gt = (gt_seg.ravel() == 2).astype(np.float64)
        max_id = int(flat_labels.max())
        if max_id > 0:
            tumor_counts = np.bincount(flat_labels, weights=flat_gt, minlength=max_id + 1)
            total_counts = np.bincount(flat_labels, minlength=max_id + 1)
            safe_total = np.where(total_counts > 0, total_counts, 1)
            overlap_frac = tumor_counts / safe_total
            y = np.array([int(overlap_frac[s] > 0.1) if s <= max_id else 0
                          for s in sids], dtype=np.int64)
        else:
            y = np.zeros(len(sids), dtype=np.int64)
        data.y = torch.tensor(y, dtype=torch.long)

    return data
