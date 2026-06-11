"""
Group adjacent tumor supernodes into physical tumors.

A single clinical tumor spans many supernodes in the graph minor.
This module finds connected components among tumor-labeled supernodes
(using the spatial adjacency from the label volume) and aggregates
their features into per-tumor phenotypes — matching the granularity
of the CSV ground truth (871 tumors in LiTS).
"""

import numpy as np
from scipy.ndimage import label as nd_label, generate_binary_structure


def group_tumor_supernodes(labels_vol: np.ndarray,
                           gt_seg: np.ndarray,
                           node_features: dict,
                           voxel_spacing: tuple = (1.0, 1.0, 1.0),
                           overlap_thresh: float = 0.1):
    """
    Group tumor supernodes into physical tumors via connected components
    on the tumor mask derived from the supernode labels.

    Returns list of dicts, one per grouped tumor, with aggregated features.
    """
    flat_labels = labels_vol.ravel()
    max_label = int(labels_vol.max())
    if max_label == 0:
        return []

    # Identify tumor supernodes (same logic as build_pyg_graph)
    flat_gt = (gt_seg.ravel() == 2).astype(np.float64)
    tumor_counts = np.bincount(flat_labels, weights=flat_gt, minlength=max_label + 1)
    total_counts = np.bincount(flat_labels, minlength=max_label + 1)
    safe = np.where(total_counts > 0, total_counts, 1)
    overlap = tumor_counts / safe
    tumor_sids = set(np.where((overlap > overlap_thresh) & (np.arange(max_label + 1) > 0))[0])

    if not tumor_sids:
        return []

    # Build binary tumor mask from tumor supernodes (LUT)
    tumor_lut = np.zeros(max_label + 1, dtype=bool)
    for s in tumor_sids:
        tumor_lut[s] = True
    tumor_mask = tumor_lut[labels_vol]

    # Connected components on the tumor mask = physical tumors
    struct = generate_binary_structure(3, 1)  # 6-connected
    tumor_cc, n_tumors = nd_label(tumor_mask, structure=struct)

    if n_tumors == 0:
        return []

    # Pre-compute organ distance map
    organ_mask = gt_seg == 1
    dist_map = None
    if organ_mask.any():
        from scipy.ndimage import distance_transform_edt
        dist_map = distance_transform_edt(~organ_mask, sampling=voxel_spacing)

    voxel_vol_mm3 = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]

    # Aggregate features per grouped tumor
    grouped_tumors = []
    for tid in range(1, n_tumors + 1):
        cc_mask = tumor_cc == tid
        cc_voxels = int(cc_mask.sum())
        if cc_voxels == 0:
            continue

        # Which supernodes belong to this tumor?
        sids_in_tumor = set(np.unique(labels_vol[cc_mask])) & tumor_sids

        # Aggregate supernode features
        volumes = []
        boundaries = []
        intensities_mean = []
        intensities_std = []
        compactnesses = []
        elongations = []
        centroids = []

        for sid in sids_in_tumor:
            f = node_features.get(int(sid))
            if f is None:
                continue
            volumes.append(f["volume"])
            boundaries.append(f["boundary_length"])
            intensities_mean.append(f["mean_intensity"])
            intensities_std.append(f["intensity_std"])
            compactnesses.append(f["compactness"])
            elongations.append(f["elongation"])
            centroids.append(f["centroid"])

        if not volumes:
            continue

        # Aggregated phenotypes
        total_volume_voxels = sum(volumes)
        volume_cm3 = total_volume_voxels * voxel_vol_mm3 / 1000.0
        diameter_mm = 2.0 * (3.0 * volume_cm3 * 1000.0 / (4.0 * np.pi)) ** (1.0 / 3.0)

        centroid = np.mean(centroids, axis=0).tolist()

        # Weighted mean intensity (by supernode volume)
        total_v = sum(volumes)
        mean_intensity = sum(m * v for m, v in zip(intensities_mean, volumes)) / max(total_v, 1)

        # Pooled std (combine within-supernode and between-supernode variance)
        pooled_var = sum(
            (s ** 2 + (m - mean_intensity) ** 2) * v
            for s, m, v in zip(intensities_std, intensities_mean, volumes)
        ) / max(total_v, 1)
        intensity_std = float(np.sqrt(max(pooled_var, 0)))

        # Compactness of the grouped tumor (from its voxel mask)
        boundary_voxels = _count_boundary(cc_mask)
        compactness = (36.0 * np.pi * cc_voxels ** 2) / (boundary_voxels ** 3 + 1e-8)
        compactness = min(compactness, 1.0)

        # Elongation via PCA on tumor voxel coordinates
        coords = np.argwhere(cc_mask).astype(np.float64)
        if len(coords) >= 3:
            centered = coords - coords.mean(axis=0)
            try:
                cov = np.cov(centered.T)
                eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
                elongation = min(float(eigvals[0] / (eigvals[-1] + 1e-8)), 100.0)
            except np.linalg.LinAlgError:
                elongation = 1.0
        else:
            elongation = 1.0

        # Distance to organ
        if dist_map is not None:
            dist_to_organ = float(dist_map[cc_mask].min())
        else:
            dist_to_organ = 0.0

        # Morphology
        if compactness > 0.6:
            morphology = "spherical"
        elif elongation > 5.0:
            morphology = "irregular"
        else:
            morphology = "ovoid"

        # T-stage
        if diameter_mm <= 20:
            t_stage = "T1"
        elif diameter_mm <= 40:
            t_stage = "T2"
        elif volume_cm3 <= 50:
            t_stage = "T3"
        else:
            t_stage = "T4"

        # Subregion from centroid
        mid_x = labels_vol.shape[2] / 2.0
        subregion = "right_lobe" if centroid[2] < mid_x else "left_lobe"

        grouped_tumors.append({
            "tumor_id": tid,
            "n_supernodes": len(sids_in_tumor),
            "supernode_ids": sorted(sids_in_tumor),
            "volume_voxels": total_volume_voxels,
            "volume_cm3": round(volume_cm3, 4),
            "diameter_mm": round(diameter_mm, 2),
            "centroid": [round(c, 2) for c in centroid],
            "mean_intensity": round(float(mean_intensity), 4),
            "intensity_std": round(intensity_std, 4),
            "compactness": round(float(compactness), 6),
            "elongation": round(float(elongation), 4),
            "boundary_voxels": boundary_voxels,
            "morphology": morphology,
            "t_stage": t_stage,
            "subregion": subregion,
            "distance_to_organ_mm": round(dist_to_organ, 2),
            "source": "semir_grouped",
        })

    return grouped_tumors


def _count_boundary(mask):
    from scipy.ndimage import binary_erosion, generate_binary_structure
    struct = generate_binary_structure(3, 1)
    eroded = binary_erosion(mask, structure=struct)
    return int((mask & ~eroded).sum())
