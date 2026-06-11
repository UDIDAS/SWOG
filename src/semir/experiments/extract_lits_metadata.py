"""
Extract per-tumor metadata from LiTS dataset (.npy volumes).

LiTS labels: 0=background, 1=liver, 2=tumor.
Multiple tumors per patient — each gets its own row.

Usage:
    conda run -n llmft python -m semir.extract_lits_metadata
"""

import os
import sys
import re
import csv
import numpy as np
from scipy.ndimage import distance_transform_edt, center_of_mass, label as nd_label

DATA_ROOT = "/scratch/ud3d4/acm_data/Data"
OUT_CSV = "/home/ud3d4/Desktop/SWOG/results/semir_pancreas/lits_gt_metadata.csv"


def extract_lits_phenotypes(vol_id):
    """Extract all tumor phenotypes from a single LiTS case."""
    ct_path = os.path.join(DATA_ROOT, "ct", f"volume-{vol_id}.npy")
    seg_path = os.path.join(DATA_ROOT, "seg", f"segmentation-{vol_id}.npy")

    if not os.path.exists(ct_path) or not os.path.exists(seg_path):
        return []

    ct = np.load(ct_path).astype(np.float32)
    seg = np.load(seg_path).astype(np.int32)

    # LiTS .npy files don't store spacing — assume 1mm isotropic
    # (actual spacing varies but is unavailable without the original NIfTI)
    spacing = (1.0, 1.0, 1.0)
    voxel_vol_mm3 = 1.0

    liver_mask = seg == 1
    tumor_mask = seg == 2
    liver_voxels = int(liver_mask.sum())
    total_tumor_voxels = int(tumor_mask.sum())

    if total_tumor_voxels == 0:
        return []

    # Split individual tumors via connected components
    tumor_cc, n_tumors = nd_label(tumor_mask)

    # Distance transform from liver boundary (computed once)
    if liver_mask.any():
        dist_map = distance_transform_edt(~liver_mask, sampling=spacing)
    else:
        dist_map = np.zeros_like(seg, dtype=np.float64)

    rows = []
    for tid in range(1, n_tumors + 1):
        single_tumor = tumor_cc == tid
        t_voxels = int(single_tumor.sum())
        if t_voxels == 0:
            continue

        vol_cm3 = t_voxels * voxel_vol_mm3 / 1000.0
        diameter_mm = 2.0 * (3.0 * vol_cm3 * 1000.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
        coverage_pct = t_voxels / max(liver_voxels, 1) * 100.0

        centroid = center_of_mass(single_tumor)
        cx, cy, cz = centroid[0] * spacing[0], centroid[1] * spacing[1], centroid[2] * spacing[2]

        min_dist = float(dist_map[single_tumor].min())
        mean_dist = float(dist_map[single_tumor].mean())

        coords = np.argwhere(single_tumor)
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)
        bbox_size = (bbox_max - bbox_min + 1) * np.array(spacing)

        # Subregion from centroid relative to liver center
        if liver_mask.any():
            liver_centroid = center_of_mass(liver_mask)
            mid_x = liver_centroid[2]
            subregion = "right_lobe" if cz < mid_x else "left_lobe"
        else:
            subregion = "unknown"

        # HU stats
        tumor_hu = ct[single_tumor]
        mean_hu = float(tumor_hu.mean())
        std_hu = float(tumor_hu.std())

        # T-stage
        if diameter_mm <= 20:
            t_stage = "T1"
        elif diameter_mm <= 40:
            t_stage = "T2"
        elif vol_cm3 > 50:
            t_stage = "T4"
        else:
            t_stage = "T3"

        rows.append({
            "case_id": f"volume-{vol_id}",
            "tumor_id": tid,
            "total_tumor_count": n_tumors,
            "tumor_volume_voxels": t_voxels,
            "tumor_volume_cm3": round(vol_cm3, 4),
            "liver_volume_voxels": liver_voxels,
            "liver_volume_cm3": round(liver_voxels / 1000.0, 4),
            "tumor_coverage_pct": round(coverage_pct, 4),
            "tumor_diameter_mm": round(diameter_mm, 2),
            "tumor_centroid_x_mm": round(cx, 2),
            "tumor_centroid_y_mm": round(cy, 2),
            "tumor_centroid_z_mm": round(cz, 2),
            "distance_to_liver_min_mm": round(min_dist, 2),
            "distance_to_liver_mean_mm": round(mean_dist, 2),
            "bbox_x_mm": round(bbox_size[0], 2),
            "bbox_y_mm": round(bbox_size[1], 2),
            "bbox_z_mm": round(bbox_size[2], 2),
            "subregion": subregion,
            "t_stage": t_stage,
            "mean_hu": round(mean_hu, 2),
            "std_hu": round(std_hu, 2),
            "connected_organ": "liver",
        })

    return rows


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    ct_dir = os.path.join(DATA_ROOT, "ct")
    seg_dir = os.path.join(DATA_ROOT, "seg")

    # Discover available volume IDs
    vol_ids = []
    for f in sorted(os.listdir(ct_dir)):
        m = re.match(r"volume-(\d+)\.npy", f)
        if m:
            vid = int(m.group(1))
            if os.path.exists(os.path.join(seg_dir, f"segmentation-{vid}.npy")):
                vol_ids.append(vid)

    print(f"Found {len(vol_ids)} LiTS volumes with segmentations", flush=True)

    all_rows = []
    for i, vid in enumerate(sorted(vol_ids)):
        rows = extract_lits_phenotypes(vid)
        all_rows.extend(rows)
        n_tumors = len(rows)
        if n_tumors > 0:
            total_vol = sum(r["tumor_volume_cm3"] for r in rows)
            print(f"  [{i+1}/{len(vol_ids)}] volume-{vid}: {n_tumors} tumors, "
                  f"total vol={total_vol:.2f}cm3", flush=True)
        else:
            print(f"  [{i+1}/{len(vol_ids)}] volume-{vid}: no tumors", flush=True)

    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} tumor rows across {len(vol_ids)} volumes to {OUT_CSV}",
          flush=True)


if __name__ == "__main__":
    main()
