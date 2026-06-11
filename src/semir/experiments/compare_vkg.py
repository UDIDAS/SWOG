"""
Side-by-side comparison: SEMIR VKG vs GT CSV (IPKG) for a few patients.

Usage:
    ~/.conda/envs/llmft/bin/python3 src/semir/compare_vkg.py
"""
import numpy as np
import nibabel as nib
import csv
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semir.graph_minor import build_graph_minor
from semir.features import extract_node_features, extract_edge_features
from semir.vkg_builder import build_vkg_from_semir

DATA = "/scratch/ud3d4/acm_data/Pancreas"
CSV_PATH = "/home/ud3d4/Desktop/Projects/acm_mmkg/data/Pancreas_Tumor_Analysis.csv"

CASES = ["pancreas_001", "pancreas_004", "pancreas_005", "pancreas_006", "pancreas_010"]

# SEMIR parameters (from user's pipeline_pancreas.py)
PSI = 0.12
ALPHA = 0.12
BETA_MIN = 100


def load_csv():
    data = {}
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            name = row["CT Scan"].replace(".nii", "").strip()
            data[name] = row
    return data


def run_semir(name):
    ct_nii = nib.load(f"{DATA}/imagesTr/{name}.nii.gz")
    seg_nii = nib.load(f"{DATA}/labelsTr/{name}.nii.gz")
    spacing = tuple(float(s) for s in ct_nii.header.get_zooms()[:3])
    ct = ct_nii.get_fdata().astype(np.float32)
    seg = seg_nii.get_fdata().astype(np.int32)

    # Narrow HU window
    vol = np.clip(ct, 20, 180)
    vol = (vol - 20) / 160.0

    # Graph minor
    gm = build_graph_minor(vol, psi=PSI, alpha=ALPHA, beta_min=BETA_MIN,
                           beta_max=500000, m_min=0.0, m_max=1.0, fast=True)

    # Features
    nf = extract_node_features(gm["labels"], vol)
    adj = gm.get("full_adjacency", gm["adjacency"])
    ef = extract_edge_features(gm["labels"], vol, adj, nf)

    # Identify tumor supernodes via GT overlap
    labels = gm["labels"]
    max_label = int(labels.max())
    flat_labels = labels.ravel()
    flat_gt = (seg.ravel() == 2).astype(np.float64)
    tumor_counts = np.bincount(flat_labels, weights=flat_gt, minlength=max_label + 1)
    total_counts = np.bincount(flat_labels, minlength=max_label + 1)
    safe = np.where(total_counts > 0, total_counts, 1)
    overlap = tumor_counts / safe
    gt_tumor_sids = {int(s) for s in nf.keys() if s <= max_label and overlap[s] > 0.1}

    # Build VKG
    vkg = build_vkg_from_semir(
        case_id=name, dataset="MSD_Task07_Pancreas",
        labels_vol=gm["labels"], volume=vol,
        node_features=nf, edge_features=ef,
        gt_seg=seg, predictions=gt_tumor_sids,
        voxel_spacing=spacing,
    )

    return vkg, gm["stats"], spacing, seg, gt_tumor_sids


def main():
    csv_data = load_csv()

    print("=" * 105)
    print("  SIDE-BY-SIDE: GT CSV (IPKG)  vs  SEMIR-derived VKG")
    print("=" * 105)

    for name in CASES:
        vkg, stats, spacing, seg, tumor_sids = run_semir(name)
        csv_row = csv_data.get(name, {})
        phenos = vkg["phenotypes"]

        # Aggregate SEMIR phenotypes
        if phenos:
            semir_vol = sum(p["volume_cm3"] for p in phenos)
            semir_diam = max(p["diameter_mm"] for p in phenos)
            semir_cov = sum(p["coverage_pct"] for p in phenos)
            semir_dist = min(p["distance_to_organ_mm"] for p in phenos)
            semir_comp = np.mean([p["compactness"] for p in phenos])
            semir_elong = np.mean([p["elongation"] for p in phenos])
            semir_morph = phenos[0]["morphology"]
            semir_stage = phenos[0]["t_stage"]
            semir_sub = phenos[0]["subregion"]
        else:
            semir_vol = semir_diam = semir_cov = semir_dist = 0
            semir_comp = semir_elong = 0
            semir_morph = semir_stage = semir_sub = "N/A"

        # CSV values
        csv_vol_vox = int(csv_row.get("Tumor Volume", 0))
        csv_pan_vol = int(csv_row.get("Pancreas Volume", 0))
        csv_cov = float(csv_row.get("Tumor Coverage (%)", 0))
        csv_dist = float(csv_row.get("Distance (voxels)", 0))
        voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
        csv_vol_cm3 = csv_vol_vox * voxel_vol_mm3 / 1000.0
        csv_diam = 2.0 * (3.0 * max(csv_vol_cm3, 0.001) * 1000.0 / (4.0 * np.pi)) ** (1/3)
        csv_dist_mm = csv_dist * spacing[0]

        # Node/edge counts
        node_types = {}
        for n in vkg["nodes"]:
            node_types[n["type"]] = node_types.get(n["type"], 0) + 1
        edge_types = {}
        for e in vkg["edges"]:
            edge_types[e["relation"]] = edge_types.get(e["relation"], 0) + 1

        # Print
        print(f"\n{'─' * 105}")
        print(f"  {name}  |  shape={seg.shape}  spacing=({spacing[0]:.2f}, {spacing[1]:.2f}, {spacing[2]:.1f}) mm")
        print(f"  SEMIR: {stats['n_supernodes_after_deletion']:,} supernodes, "
              f"{len(tumor_sids)} tumor SN  |  "
              f"VKG: {len(vkg['nodes'])} nodes, {len(vkg['edges'])} edges")
        print(f"{'─' * 105}")
        print(f"  {'Feature':<30s} {'GT CSV (IPKG)':<22s} {'SEMIR VKG':<22s} {'Delta':<20s}")
        print(f"  {'─' * 100}")

        def row(feat, csv_val, semir_val, unit=""):
            if csv_val > 0:
                err = abs(semir_val - csv_val) / csv_val * 100
                delta = f"{err:.1f}% err"
            else:
                delta = "---"
            print(f"  {feat:<30s} {csv_val:<22.2f} {semir_val:<22.4f} {delta:<20s}")

        row("Tumor Volume (cm3)", csv_vol_cm3, semir_vol)
        row("Diameter (mm)", csv_diam, semir_diam)
        row("Coverage (%)", csv_cov, semir_cov)
        row("Distance to organ (mm)", csv_dist_mm, semir_dist)

        print(f"  {'─' * 100}")
        print(f"  {'Compactness':<30s} {'---':<22s} {semir_comp:<22.4f} {'SEMIR-only':<20s}")
        print(f"  {'Elongation':<30s} {'---':<22s} {semir_elong:<22.4f} {'SEMIR-only':<20s}")
        print(f"  {'Morphology':<30s} {'---':<22s} {semir_morph:<22s} {'SEMIR-only':<20s}")
        print(f"  {'T-Stage':<30s} {'---':<22s} {semir_stage:<22s} {'SEMIR-only':<20s}")
        print(f"  {'Subregion':<30s} {'---':<22s} {semir_sub:<22s} {'SEMIR-only':<20s}")

        print(f"  {'─' * 100}")
        print(f"  VKG Nodes: {node_types}")
        print(f"  VKG Edges: {edge_types}")

    print(f"\n{'=' * 105}")
    print(f"  SEMIR replaces all GT CSV columns + adds 5 geometric features.")
    print(f"  Tumor capture limited by beta_min={BETA_MIN} — small tumors lost.")
    print(f"{'=' * 105}")


if __name__ == "__main__":
    main()
