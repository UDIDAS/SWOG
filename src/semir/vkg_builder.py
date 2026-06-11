"""
Build Visual Knowledge Graph (VKG) directly from SEMIR supernodes.

Replaces vkg_from_csv.py: instead of reading pre-computed CSV columns,
this module derives all node attributes (volume, diameter, coverage,
subregion, morphology, T-stage) from the learned graph minor representation.
"""

import numpy as np
import json


# ---- Schema-compatible node/edge helpers ----

def _make_id(prefix, name):
    slug = name.lower().replace(" ", "_").replace(",", "").replace(".", "")
    return f"{prefix}_{slug}"


def _estimate_t_stage(diameter_mm, volume_cm3):
    if diameter_mm <= 20:
        return "T1"
    elif diameter_mm <= 40:
        return "T2"
    elif volume_cm3 <= 50:
        return "T3"
    else:
        return "T4"


def _morphology_label(compactness, elongation):
    if compactness > 0.6:
        return "spherical"
    elif elongation > 5.0:
        return "irregular"
    else:
        return "ovoid"


# ---- Main builder ----

def build_vkg_from_semir(case_id: str,
                         dataset: str,
                         labels_vol: np.ndarray,
                         volume: np.ndarray,
                         node_features: dict,
                         edge_features: dict,
                         gt_seg: np.ndarray = None,
                         predictions: np.ndarray = None,
                         voxel_spacing: tuple = (1.0, 1.0, 1.0)):
    """
    Construct VKG nodes and edges from SEMIR supernode features.

    Parameters
    ----------
    case_id        : e.g. "lits_volume_0"
    dataset        : e.g. "LiTS"
    labels_vol     : supernode label volume from graph minor
    volume         : original intensity volume
    node_features  : dict from features.extract_node_features()
    edge_features  : dict from features.extract_edge_features()
    gt_seg         : ground truth segmentation (optional, for validation)
    predictions    : predicted supernode labels from GINE (optional)
    voxel_spacing  : (sz, sy, sx) in mm

    Returns
    -------
    dict with "nodes", "edges", "phenotypes"
    """
    nodes = []
    edges = []
    phenotypes = []

    # Dataset-specific mappings
    organ_name = _dataset_organ(dataset)
    disease_name = _dataset_disease(dataset)

    # Identify tumor supernodes
    tumor_sids = set()
    if predictions is not None:
        tumor_sids = set(predictions)
    elif gt_seg is not None:
        for sid, feat in node_features.items():
            mask = labels_vol == sid
            if (gt_seg[mask] == 2).sum() / max(mask.sum(), 1) > 0.1:
                tumor_sids.add(sid)

    # Pre-compute per-supernode min organ distance via LUT (O(N) not O(K×N))
    _min_dist_lut = None
    max_label = int(labels_vol.max()) if labels_vol.size > 0 else 0
    if gt_seg is not None and max_label > 0:
        organ_mask = gt_seg == 1
        if organ_mask.any():
            from scipy.ndimage import distance_transform_edt
            dist_map = distance_transform_edt(~organ_mask, sampling=voxel_spacing)
            flat_labels = labels_vol.ravel()
            flat_dist = dist_map.ravel().astype(np.float64)
            _min_dist_lut = np.full(max_label + 1, np.inf)
            np.minimum.at(_min_dist_lut, flat_labels, flat_dist)

    # Shared nodes (deduped per case)
    organ_id = _make_id("organ", organ_name)
    disease_id = _make_id("disease", disease_name)
    nodes.append({"id": organ_id, "type": "Organ", "name": organ_name, "modality": "CT"})
    nodes.append({"id": disease_id, "type": "Disease", "name": disease_name, "modality": "CT"})

    # Process each tumor supernode
    for t_idx, sid in enumerate(sorted(tumor_sids)):
        feat = node_features.get(sid)
        if feat is None:
            continue

        # ---- Derive phenotypes directly from supernode features ----
        # Volume: voxels → cm³
        voxel_vol_mm3 = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
        volume_cm3 = feat["volume"] * voxel_vol_mm3 / 1000.0

        # Diameter from volume (sphere approximation)
        diameter_mm = 2.0 * (3.0 * volume_cm3 * 1000.0 / (4.0 * np.pi)) ** (1.0 / 3.0)

        # Coverage: tumor volume / organ volume
        organ_voxels = sum(
            f["volume"] for s, f in node_features.items() if s not in tumor_sids
        )
        coverage_pct = feat["volume"] / max(organ_voxels + feat["volume"], 1) * 100.0

        # Subregion from centroid position
        subregion = _subregion_from_centroid(feat["centroid"], labels_vol.shape, dataset)

        # Morphology from compactness + elongation
        morphology = _morphology_label(feat["compactness"], feat["elongation"])

        # T-stage
        t_stage = _estimate_t_stage(diameter_mm, volume_cm3)

        # Distance to organ boundary via pre-computed LUT
        if _min_dist_lut is not None and sid < len(_min_dist_lut):
            dist_to_boundary = float(_min_dist_lut[sid])
            if np.isinf(dist_to_boundary):
                dist_to_boundary = 0.0
        else:
            dist_to_boundary = 0.0

        # ---- Build graph nodes & edges ----
        tumor_id = f"semir_{case_id}_tumor_{t_idx}"
        anatomy_id = _make_id("anatomy", subregion)
        stage_id = _make_id("stage", t_stage)

        # Tumor node
        nodes.append({
            "id": tumor_id,
            "type": "Tumor",
            "name": f"tumor_{t_idx}",
            "modality": "CT",
            "dataset": dataset,
            "case_id": case_id,
            "supernode_id": int(sid),
            # SEMIR-derived features embedded directly
            "semir_compactness": feat["compactness"],
            "semir_elongation": feat["elongation"],
            "semir_mean_intensity": feat["mean_intensity"],
            "semir_intensity_std": feat["intensity_std"],
        })
        nodes.append({"id": anatomy_id, "type": "Anatomy", "name": subregion, "modality": "CT"})
        nodes.append({"id": stage_id, "type": "Stage", "name": t_stage, "modality": "CT"})

        # Core edges
        edges.append({"source": tumor_id, "target": organ_id, "relation": "part_of"})
        edges.append({"source": tumor_id, "target": disease_id, "relation": "suggestive_of"})
        edges.append({"source": tumor_id, "target": anatomy_id, "relation": "located_in"})
        edges.append({"source": tumor_id, "target": stage_id, "relation": "has_stage"})

        # Feature nodes
        feat_map = {
            "volume_cm3": round(volume_cm3, 4),
            "diameter_mm": round(diameter_mm, 2),
            "coverage_pct": round(coverage_pct, 4),
            "compactness": feat["compactness"],
            "elongation": feat["elongation"],
            "mean_intensity": feat["mean_intensity"],
            "intensity_std": feat["intensity_std"],
            "distance_to_organ_mm": round(dist_to_boundary, 2),
        }
        for fname, fval in feat_map.items():
            fid = f"{tumor_id}_feat_{fname}"
            nodes.append({
                "id": fid,
                "type": "Feature",
                "name": fname,
                "value": fval,
                "modality": "CT",
            })
            edges.append({"source": tumor_id, "target": fid, "relation": "has_feature"})

        # Morphology observation
        morph_id = _make_id("obs", morphology)
        nodes.append({"id": morph_id, "type": "Observation", "name": morphology,
                       "term": morphology, "modality": "CT"})
        edges.append({"source": tumor_id, "target": morph_id, "relation": "has_morphology"})

        # Phenotype record
        phenotypes.append({
            "case_id": case_id,
            "tumor_idx": t_idx,
            "supernode_id": int(sid),
            "volume_cm3": round(volume_cm3, 4),
            "diameter_mm": round(diameter_mm, 2),
            "coverage_pct": round(coverage_pct, 4),
            "compactness": feat["compactness"],
            "elongation": feat["elongation"],
            "subregion": subregion,
            "morphology": morphology,
            "t_stage": t_stage,
            "distance_to_organ_mm": round(dist_to_boundary, 2),
            "mean_intensity": feat["mean_intensity"],
            "intensity_std": feat["intensity_std"],
            "source": "semir_graph_minor",
        })

    # Deduplicate nodes by ID
    seen = set()
    deduped_nodes = []
    for n in nodes:
        if n["id"] not in seen:
            seen.add(n["id"])
            deduped_nodes.append(n)

    return {
        "nodes": deduped_nodes,
        "edges": edges,
        "phenotypes": phenotypes,
    }


# ---- Dataset-specific helpers ----

def _dataset_organ(dataset):
    return {"LiTS": "liver", "MSD_Task07_Pancreas": "pancreas",
            "FLARE2023": "multi_organ"}.get(dataset, "organ")


def _dataset_disease(dataset):
    return {"LiTS": "hepatocellular_carcinoma",
            "MSD_Task07_Pancreas": "pancreatic_adenocarcinoma",
            "FLARE2023": "abdominal_malignancy"}.get(dataset, "malignancy")


def _subregion_from_centroid(centroid, vol_shape, dataset):
    """Classify tumor subregion from centroid position."""
    if dataset == "LiTS":
        mid_x = vol_shape[2] / 2.0
        return "right_lobe" if centroid[2] < mid_x else "left_lobe"
    elif dataset in ("MSD_Task07_Pancreas", "pancreas"):
        # head/body/tail by x position relative to volume
        rel = centroid[2] / vol_shape[2]
        if rel < 0.33:
            return "pancreatic_tail"
        elif rel < 0.66:
            return "pancreatic_body"
        else:
            return "pancreatic_head"
    return "abdominal_organ"


def _approx_organ_distance(sid, labels_vol, gt_seg, spacing):
    """Approximate distance from tumor supernode to organ boundary."""
    if gt_seg is None:
        return 0.0
    from scipy.ndimage import distance_transform_edt
    organ_mask = gt_seg == 1
    if organ_mask.sum() == 0:
        return 0.0
    dist_map = distance_transform_edt(~organ_mask, sampling=spacing)
    tumor_mask = labels_vol == sid
    if tumor_mask.sum() == 0:
        return 0.0
    return float(dist_map[tumor_mask].min())
