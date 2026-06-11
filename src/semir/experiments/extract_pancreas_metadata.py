"""
Extract tumor and organ metadata from MSD Task07 Pancreas GT segmentation masks.

Generates a CSV with per-patient phenotypes: volume, diameter, coverage,
centroid (x,y,z), distance to organ, and subregion — computed directly
from the segmentation masks, not from any prior experiment output.

Usage:
    conda run -n llmft python -m semir.extract_pancreas_metadata
"""

import os
import sys
import re
import csv
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, center_of_mass

DATA_ROOT = "/scratch/ud3d4/acm_data/Pancreas"
OUT_CSV = "/home/ud3d4/Desktop/SWOG/results/semir_pancreas/pancreas_gt_metadata.csv"


def extract_phenotypes(ct_path, seg_path):
    """Extract all phenotypes from a single case."""
    ct_nii = nib.load(ct_path)
    seg_nii = nib.load(seg_path)

    ct = ct_nii.get_fdata().astype(np.float32)
    seg = seg_nii.get_fdata().astype(np.int32)
    spacing = tuple(float(s) for s in ct_nii.header.get_zooms()[:3])
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]

    tumor_mask = seg == 2
    pancreas_mask = seg == 1
    tumor_voxels = int(tumor_mask.sum())
    pancreas_voxels = int(pancreas_mask.sum())

    if tumor_voxels == 0:
        return None

    # Volume
    tumor_vol_cm3 = tumor_voxels * voxel_vol_mm3 / 1000.0
    pancreas_vol_cm3 = pancreas_voxels * voxel_vol_mm3 / 1000.0

    # Diameter (sphere-equivalent)
    diameter_mm = 2.0 * (3.0 * tumor_vol_cm3 * 1000.0 / (4.0 * np.pi)) ** (1.0 / 3.0)

    # Coverage
    coverage_pct = tumor_voxels / max(pancreas_voxels, 1) * 100.0

    # Centroids in mm (voxel coords * spacing)
    tumor_centroid_vox = center_of_mass(tumor_mask)
    tumor_x = tumor_centroid_vox[0] * spacing[0]
    tumor_y = tumor_centroid_vox[1] * spacing[1]
    tumor_z = tumor_centroid_vox[2] * spacing[2]

    pancreas_centroid_vox = center_of_mass(pancreas_mask) if pancreas_voxels > 0 else (0, 0, 0)
    pancreas_x = pancreas_centroid_vox[0] * spacing[0]
    pancreas_y = pancreas_centroid_vox[1] * spacing[1]
    pancreas_z = pancreas_centroid_vox[2] * spacing[2]

    # Distance to organ boundary (mm)
    if pancreas_mask.any():
        dist_map = distance_transform_edt(~pancreas_mask, sampling=spacing)
        min_dist = float(dist_map[tumor_mask].min())
        mean_dist = float(dist_map[tumor_mask].mean())
    else:
        min_dist = 0.0
        mean_dist = 0.0

    # Tumor bounding box dimensions (mm)
    coords = np.argwhere(tumor_mask)
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    bbox_size = (bbox_max - bbox_min + 1) * np.array(spacing)

    # Subregion from relative position of tumor centroid to pancreas centroid
    if pancreas_voxels > 0:
        rel_z = (tumor_centroid_vox[2] - pancreas_centroid_vox[2]) / max(1, abs(pancreas_centroid_vox[2]))
        if rel_z < -0.3:
            subregion = "tail"
        elif rel_z > 0.3:
            subregion = "head"
        else:
            subregion = "body"
    else:
        subregion = "unknown"

    # Tumor HU statistics (from raw CT, not windowed)
    tumor_hu = ct[tumor_mask]
    mean_hu = float(tumor_hu.mean())
    std_hu = float(tumor_hu.std())
    min_hu = float(tumor_hu.min())
    max_hu = float(tumor_hu.max())

    # T-stage estimate
    if diameter_mm <= 20:
        t_stage = "T1"
    elif diameter_mm <= 40:
        t_stage = "T2"
    elif tumor_vol_cm3 > 50:
        t_stage = "T4"
    else:
        t_stage = "T3"

    return {
        "tumor_volume_voxels": tumor_voxels,
        "tumor_volume_cm3": round(tumor_vol_cm3, 4),
        "pancreas_volume_voxels": pancreas_voxels,
        "pancreas_volume_cm3": round(pancreas_vol_cm3, 4),
        "tumor_coverage_pct": round(coverage_pct, 4),
        "tumor_diameter_mm": round(diameter_mm, 2),
        "tumor_centroid_x_mm": round(tumor_x, 2),
        "tumor_centroid_y_mm": round(tumor_y, 2),
        "tumor_centroid_z_mm": round(tumor_z, 2),
        "pancreas_centroid_x_mm": round(pancreas_x, 2),
        "pancreas_centroid_y_mm": round(pancreas_y, 2),
        "pancreas_centroid_z_mm": round(pancreas_z, 2),
        "distance_to_organ_min_mm": round(min_dist, 2),
        "distance_to_organ_mean_mm": round(mean_dist, 2),
        "bbox_x_mm": round(bbox_size[0], 2),
        "bbox_y_mm": round(bbox_size[1], 2),
        "bbox_z_mm": round(bbox_size[2], 2),
        "subregion": subregion,
        "t_stage": t_stage,
        "mean_hu": round(mean_hu, 2),
        "std_hu": round(std_hu, 2),
        "min_hu": round(min_hu, 2),
        "max_hu": round(max_hu, 2),
        "spacing_x_mm": round(spacing[0], 4),
        "spacing_y_mm": round(spacing[1], 4),
        "spacing_z_mm": round(spacing[2], 4),
    }


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    img_dir = os.path.join(DATA_ROOT, "imagesTr")
    seg_dir = os.path.join(DATA_ROOT, "labelsTr")

    cases = []
    for f in sorted(os.listdir(img_dir)):
        m = re.match(r"(pancreas_\d+)\.nii\.gz", f)
        if m:
            name = m.group(1)
            if os.path.exists(os.path.join(seg_dir, f"{name}.nii.gz")):
                cases.append(name)

    print(f"Found {len(cases)} pancreas cases", flush=True)

    fieldnames = ["case_id"] + list(extract_phenotypes.__code__.co_varnames)
    # Determine fieldnames from first case
    ct_path = os.path.join(img_dir, f"{cases[0]}.nii.gz")
    seg_path = os.path.join(seg_dir, f"{cases[0]}.nii.gz")
    sample = extract_phenotypes(ct_path, seg_path)
    fieldnames = ["case_id"] + list(sample.keys())

    rows = []
    for i, name in enumerate(cases):
        ct_path = os.path.join(img_dir, f"{name}.nii.gz")
        seg_path = os.path.join(seg_dir, f"{name}.nii.gz")

        pheno = extract_phenotypes(ct_path, seg_path)
        if pheno is None:
            print(f"  [{i+1}/{len(cases)}] {name}: no tumor found, skipping", flush=True)
            continue

        row = {"case_id": name}
        row.update(pheno)
        rows.append(row)

        print(f"  [{i+1}/{len(cases)}] {name}: vol={pheno['tumor_volume_cm3']:.2f}cm3, "
              f"diam={pheno['tumor_diameter_mm']:.1f}mm, "
              f"centroid=({pheno['tumor_centroid_x_mm']:.1f}, "
              f"{pheno['tumor_centroid_y_mm']:.1f}, "
              f"{pheno['tumor_centroid_z_mm']:.1f})mm, "
              f"stage={pheno['t_stage']}, region={pheno['subregion']}",
              flush=True)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} cases to {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
