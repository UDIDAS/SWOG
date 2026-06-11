"""Standalone grouping: rebuild graph minors, group tumor supernodes, compare with CSV."""
import os, sys, json, re, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semir.graph_minor import build_graph_minor
from semir.features import extract_node_features
from semir.grouping import group_tumor_supernodes

DATA_ROOT = "/scratch/ud3d4/acm_data/Data"
RESULTS_DIR = "/home/ud3d4/Desktop/SWOG/results/semir_lits"
CSV_PATH = "/home/ud3d4/Desktop/Projects/acm_mmkg/data/LiTS_Liver_Tumor_Analysis_with_GT.csv"

def hu_window(v):
    v = np.clip(v, -150, 250)
    return (v + 150) / 400.0

def pr(msg=""):
    print(msg, flush=True)

# Discover volumes
ct_dir = os.path.join(DATA_ROOT, "ct")
vol_ids = sorted([
    int(m.group(1)) for f in os.listdir(ct_dir)
    if (m := re.match(r"volume-(\d+)\.npy", f))
    and os.path.exists(os.path.join(DATA_ROOT, "seg", f"segmentation-{m.group(1)}.npy"))
])
pr(f"Found {len(vol_ids)} volumes")

all_grouped = []
for i, vid in enumerate(vol_ids):
    ct = np.load(os.path.join(DATA_ROOT, "ct", f"volume-{vid}.npy")).astype(np.float32)
    seg = np.load(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")).astype(np.int32)
    ct = hu_window(ct)

    if (seg == 2).sum() == 0:
        pr(f"  vol-{vid}: no tumors, skip  ({i+1}/{len(vol_ids)})")
        continue

    gm = build_graph_minor(ct, psi=0.05, alpha=0.02, beta_min=10, beta_max=500000,
                           m_min=0.0, m_max=1.0, fast=True)
    nf = extract_node_features(gm["labels"], ct)
    grouped = group_tumor_supernodes(gm["labels"], seg, nf)
    for g in grouped:
        g["case_id"] = f"lits_volume_{vid}"
    all_grouped.extend(grouped)
    pr(f"  vol-{vid}: {len(grouped)} physical tumors  ({i+1}/{len(vol_ids)})")

# Save
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR, "grouped_tumors.json"), "w") as f:
    json.dump(all_grouped, f, indent=2,
              default=lambda o: int(o) if hasattr(o, "item") else str(o))

pr(f"\n{'='*60}")
pr(f"  SEMIR grouped tumors: {len(all_grouped)}")

# Compare with CSV
import csv
csv_tumors = []
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        csv_tumors.append({
            "volume_voxels": int(row["Tumor Volume"]),
            "coverage_pct": float(row["Tumor Coverage (%)"]),
            "distance": float(row["Connected Organ Distance (voxels)"]),
        })

pr(f"  CSV tumors:           {len(csv_tumors)}")
ratio = len(all_grouped) / max(len(csv_tumors), 1)
pr(f"  Ratio:                {ratio:.2f}x")

hdr = f"  {'Feature':<25s} {'CSV (mean)':<15s} {'SEMIR (mean)':<15s}"
sep = f"  {'-'*55}"
pr(hdr)
pr(sep)

csv_vol = np.mean([t["volume_voxels"] for t in csv_tumors])
sem_vol = np.mean([g["volume_voxels"] for g in all_grouped])
csv_dist = np.mean([t["distance"] for t in csv_tumors])
sem_dist = np.mean([g["distance_to_organ_mm"] for g in all_grouped])
sem_diam = np.mean([g["diameter_mm"] for g in all_grouped])
sem_comp = np.mean([g["compactness"] for g in all_grouped])
sem_elong = np.mean([g["elongation"] for g in all_grouped])

pr(f"  {'Volume (voxels)':<25s} {csv_vol:<15.1f} {sem_vol:<15.1f}")
pr(f"  {'Distance':<25s} {csv_dist:<15.2f} {sem_dist:<15.2f}")
pr(f"  {'Diameter (mm)':<25s} {'N/A':<15s} {sem_diam:<15.2f}")
pr(f"  {'Compactness':<25s} {'N/A':<15s} {sem_comp:<15.6f}")
pr(f"  {'Elongation':<25s} {'N/A':<15s} {sem_elong:<15.4f}")

from collections import Counter
stages = Counter(g["t_stage"] for g in all_grouped)
morphs = Counter(g["morphology"] for g in all_grouped)
pr(f"\n  T-stage:    {dict(stages)}")
pr(f"  Morphology: {dict(morphs)}")

pr(f"\n  File saved: {os.path.join(RESULTS_DIR, 'grouped_tumors.json')}")
sz = os.path.getsize(os.path.join(RESULTS_DIR, "grouped_tumors.json"))
pr(f"  File size:  {sz/1024:.1f} KB")
