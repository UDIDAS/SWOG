"""
Build and visualize a VKG from SEMIR supernodes — like slide 4.

Uses GT segmentation to identify tumor/organ supernodes (bypassing GINE),
then produces a 3D scatter plot with organ nodes, volumes, and distance edges.
"""

import os
import sys
import numpy as np
import nibabel as nib
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semir.graph_minor import build_graph_minor
from semir.features import extract_node_features

DATA_ROOT = "/scratch/ud3d4/acm_data/Pancreas"
RESULTS_DIR = "/home/ud3d4/Desktop/SWOG/results/semir_pancreas"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_case(name):
    ct_nii = nib.load(os.path.join(DATA_ROOT, "imagesTr", f"{name}.nii.gz"))
    seg_nii = nib.load(os.path.join(DATA_ROOT, "labelsTr", f"{name}.nii.gz"))
    ct = ct_nii.get_fdata().astype(np.float32)
    seg = seg_nii.get_fdata().astype(np.int32)
    spacing = tuple(float(s) for s in ct_nii.header.get_zooms()[:3])
    return ct, seg, spacing


def build_patient_vkg(name):
    """Build a per-patient VKG with organ + tumor nodes from SEMIR supernodes."""
    print(f"\n  Processing {name}...", flush=True)
    ct, seg, spacing = load_case(name)

    # HU window
    ct_w = np.clip(ct, 20, 180).astype(np.float64)
    ct_w = (ct_w - 20) / 160.0

    # Graph minor (optimized params from few-shot search)
    gm = build_graph_minor(ct_w, psi=0.04, alpha=0.04, beta_min=2,
                           beta_max=500000, m_min=0.0, m_max=1.0, fast=True)
    labels = gm["labels"]
    n_sn = gm["stats"]["n_supernodes_after_deletion"]
    print(f"    {n_sn} supernodes", flush=True)

    # Extract features
    nf = extract_node_features(labels, ct_w)

    # Classify supernodes using GT
    flat_labels = labels.ravel()
    flat_gt = seg.ravel()
    max_label = int(labels.max())

    # For each supernode, compute overlap with each GT class
    counts = np.bincount(flat_labels, minlength=max_label + 1)
    pancreas_counts = np.bincount(flat_labels,
                                  weights=(flat_gt == 1).astype(float),
                                  minlength=max_label + 1)
    tumor_counts = np.bincount(flat_labels,
                               weights=(flat_gt == 2).astype(float),
                               minlength=max_label + 1)
    safe = np.where(counts > 0, counts, 1)

    # Aggregate per-class supernodes
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]

    nodes = []
    # --- Pancreas organ node ---
    pancreas_sids = [s for s in nf if s <= max_label and
                     pancreas_counts[s] / safe[s] > 0.3]
    if pancreas_sids:
        p_vol_vox = sum(int(counts[s]) for s in pancreas_sids)
        p_vol_cc = p_vol_vox * voxel_vol_mm3 / 1000.0
        # Weighted centroid (weight by supernode size)
        cx = sum(nf[s]["centroid"][0] * counts[s] for s in pancreas_sids) / max(p_vol_vox, 1)
        cy = sum(nf[s]["centroid"][1] * counts[s] for s in pancreas_sids) / max(p_vol_vox, 1)
        cz = sum(nf[s]["centroid"][2] * counts[s] for s in pancreas_sids) / max(p_vol_vox, 1)
        # Convert voxel coords to mm
        centroid_mm = (cx * spacing[0], cy * spacing[1], cz * spacing[2])
        nodes.append({
            "name": "Pancreas",
            "type": "Organ",
            "volume_cc": round(p_vol_cc, 1),
            "centroid_mm": [round(c, 1) for c in centroid_mm],
            "color": "green",
            "n_supernodes": len(pancreas_sids),
        })

    # --- Tumor node ---
    tumor_sids = [s for s in nf if s <= max_label and
                  tumor_counts[s] / safe[s] > 0.1]
    if tumor_sids:
        t_vol_vox = sum(int(counts[s]) for s in tumor_sids)
        t_vol_cc = t_vol_vox * voxel_vol_mm3 / 1000.0
        cx = sum(nf[s]["centroid"][0] * counts[s] for s in tumor_sids) / max(t_vol_vox, 1)
        cy = sum(nf[s]["centroid"][1] * counts[s] for s in tumor_sids) / max(t_vol_vox, 1)
        cz = sum(nf[s]["centroid"][2] * counts[s] for s in tumor_sids) / max(t_vol_vox, 1)
        centroid_mm = (cx * spacing[0], cy * spacing[1], cz * spacing[2])

        # Morphology from supernodes
        comp = np.mean([nf[s]["compactness"] for s in tumor_sids])
        elong = np.mean([nf[s]["elongation"] for s in tumor_sids])
        if comp > 0.6:
            morph = "spherical"
        elif elong > 5.0:
            morph = "irregular"
        else:
            morph = "ovoid"

        diam_mm = 2.0 * (3.0 * t_vol_cc * 1000.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
        if diam_mm <= 20:
            t_stage = "T1"
        elif diam_mm <= 40:
            t_stage = "T2"
        elif t_vol_cc <= 50:
            t_stage = "T3"
        else:
            t_stage = "T4"

        nodes.append({
            "name": "Tumor",
            "type": "Tumor",
            "volume_cc": round(t_vol_cc, 1),
            "centroid_mm": [round(c, 1) for c in centroid_mm],
            "color": "red",
            "n_supernodes": len(tumor_sids),
            "diameter_mm": round(diam_mm, 1),
            "morphology": morph,
            "t_stage": t_stage,
            "compactness": round(comp, 3),
            "elongation": round(elong, 2),
        })

    # --- Edges: distances between nodes ---
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            ci = np.array(nodes[i]["centroid_mm"])
            cj = np.array(nodes[j]["centroid_mm"])
            dist = float(np.linalg.norm(ci - cj))
            edges.append({
                "source": nodes[i]["name"],
                "target": nodes[j]["name"],
                "distance_mm": round(dist, 2),
            })

    # Coverage
    if pancreas_sids and tumor_sids:
        p_vol = sum(int(counts[s]) for s in pancreas_sids)
        t_vol = sum(int(counts[s]) for s in tumor_sids)
        coverage = t_vol / max(p_vol + t_vol, 1) * 100
    else:
        coverage = 0

    return {
        "case_id": name,
        "spacing_mm": list(spacing),
        "nodes": nodes,
        "edges": edges,
        "coverage_pct": round(coverage, 2),
        "n_supernodes": n_sn,
    }


def plot_vkg_2d(vkg_data, out_path):
    """Clean 2D VKG: nodes as labeled circles, edges with distances."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor("#f8f9fa")

    nodes = vkg_data["nodes"]
    edges = vkg_data["edges"]

    color_map = {"Pancreas": "#27ae60", "Tumor": "#c0392b"}
    edge_color_map = {"Pancreas": "#2ecc71", "Tumor": "#e74c3c"}

    # Fixed layout: pancreas left, tumor right, equal spacing
    spacing = 150  # fixed horizontal gap between nodes
    cy = 0  # shared y
    fixed_pos = {"Pancreas": (-spacing / 2, cy), "Tumor": (spacing / 2, cy)}
    positions = {n["name"]: fixed_pos.get(n["name"], (0, 0)) for n in nodes}

    # Draw edges first (behind nodes)
    for edge in edges:
        if edge["source"] in positions and edge["target"] in positions:
            p1 = positions[edge["source"]]
            p2 = positions[edge["target"]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color="#7f8c8d", linewidth=2.5, linestyle="--",
                    alpha=0.6, zorder=1)
            mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            ax.annotate(f"{edge['distance_mm']:.1f} mm",
                        xy=mid, fontsize=12, color="#555",
                        ha="center", va="bottom", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white", edgecolor="#bbb",
                                  alpha=0.9),
                        zorder=3)

    # Draw nodes
    for node in nodes:
        pos = positions[node["name"]]
        vol = node["volume_cc"]
        color = color_map.get(node["name"], "#3498db")
        edge_c = edge_color_map.get(node["name"], "#2980b9")

        radius = max(18, min(40, 10 + vol * 0.8))
        circle = plt.Circle(pos, radius, facecolor=color, edgecolor="white",
                             linewidth=3, alpha=0.85, zorder=4)
        ax.add_patch(circle)

        # Volume inside circle
        ax.text(pos[0], pos[1], f"{vol:.1f}cc",
                fontsize=12, color="white", fontweight="bold",
                ha="center", va="center", zorder=5)

        # Name + details label above
        label_lines = [f"{node['name']}"]
        if node["type"] == "Tumor":
            label_lines.append(f"{node.get('t_stage', '')} | "
                               f"{node.get('morphology', '')} | "
                               f"d={node.get('diameter_mm', 0):.0f}mm")
        label_text = "\n".join(label_lines)

        offset_y = radius + 12
        ax.annotate(label_text, xy=pos, xytext=(pos[0], pos[1] + offset_y),
                    fontsize=11, fontweight="bold", ha="center", va="bottom",
                    color="#2c3e50",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor=edge_c, linewidth=2, alpha=0.95),
                    zorder=6)

    # Title and metadata
    case_id = vkg_data["case_id"]
    coverage = vkg_data["coverage_pct"]
    n_sn = vkg_data["n_supernodes"]
    ax.set_title(f"SEMIR VKG: {case_id}\n"
                 f"Coverage: {coverage:.1f}%  |  Supernodes: {n_sn}",
                 fontsize=14, fontweight="bold", pad=15)

    ax.set_aspect("equal")
    ax.axis("off")

    pad = 70
    ax.set_xlim(-spacing / 2 - pad, spacing / 2 + pad)
    ax.set_ylim(-pad, pad + 30)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, facecolor="white")
    plt.close()
    print(f"    Saved: {out_path}", flush=True)


def plot_multi_patient_2d(all_vkgs, out_path):
    """Grid of per-patient VKG cards, each with fixed-spacing layout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import math

    n = len(all_vkgs)
    cols = min(4, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    fig.set_facecolor("white")
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    color_map = {"Pancreas": "#27ae60", "Tumor": "#c0392b"}

    for idx, vkg in enumerate(all_vkgs):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.set_facecolor("#f8f9fa")

        gap = 80
        fixed_pos = {"Pancreas": (-gap / 2, 0), "Tumor": (gap / 2, 0)}
        positions = {nd["name"]: fixed_pos.get(nd["name"], (0, 0))
                     for nd in vkg["nodes"]}

        # Edge
        for edge in vkg["edges"]:
            if edge["source"] in positions and edge["target"] in positions:
                p1, p2 = positions[edge["source"]], positions[edge["target"]]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        color="#7f8c8d", linewidth=1.5, linestyle="--",
                        alpha=0.5, zorder=1)
                mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                ax.text(mid[0], mid[1] - 6, f"{edge['distance_mm']:.0f}mm",
                        fontsize=7, color="#555", ha="center", va="top",
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor="white", edgecolor="#ccc",
                                  alpha=0.85),
                        zorder=3)

        # Nodes
        for nd in vkg["nodes"]:
            pos = positions[nd["name"]]
            vol = nd["volume_cc"]
            color = color_map.get(nd["name"], "#3498db")
            radius = max(10, min(25, 8 + vol * 0.5))
            circle = plt.Circle(pos, radius, facecolor=color,
                                edgecolor="white", linewidth=2,
                                alpha=0.85, zorder=4)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], f"{vol:.1f}",
                    fontsize=8, color="white", fontweight="bold",
                    ha="center", va="center", zorder=5)
            # Name above
            label = nd["name"]
            if nd["type"] == "Tumor":
                label += f"\n{nd.get('t_stage', '')}"
            ax.text(pos[0], pos[1] + radius + 5, label,
                    fontsize=7, ha="center", va="bottom",
                    fontweight="bold", color="#2c3e50", zorder=6)

        ax.set_title(f"{vkg['case_id']}  ({vkg['coverage_pct']:.0f}%)",
                     fontsize=9, fontweight="bold", pad=4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-gap, gap)
        ax.set_ylim(-45, 50)

    # Hide empty subplots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.suptitle(f"SEMIR VKG — {n} Pancreas Patients\n"
                 f"Green = Pancreas (cc)  |  Red = Tumor (cc)",
                 fontsize=13, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, facecolor="white")
    plt.close()
    print(f"\n  Multi-patient VKG saved: {out_path}", flush=True)


def main():
    cases = sorted([f.replace(".nii.gz", "")
                    for f in os.listdir(os.path.join(DATA_ROOT, "imagesTr"))
                    if f.endswith(".nii.gz") and not f.startswith("._")
                    and os.path.exists(os.path.join(DATA_ROOT, "labelsTr", f))])[:10]

    print(f"Building VKGs for {len(cases)} pancreas cases...", flush=True)

    all_vkgs = []
    for name in cases:
        vkg = build_patient_vkg(name)
        all_vkgs.append(vkg)

        # Per-patient 2D plot
        plot_vkg_2d(vkg, os.path.join(RESULTS_DIR, f"vkg_{name}.png"))

        # Print summary
        for node in vkg["nodes"]:
            extra = ""
            if node["type"] == "Tumor":
                extra = f"  {node['t_stage']} {node['morphology']} d={node['diameter_mm']}mm"
            print(f"    {node['name']:>10s}: {node['volume_cc']:>7.1f}cc  "
                  f"centroid={node['centroid_mm']}{extra}", flush=True)
        for edge in vkg["edges"]:
            print(f"    {edge['source']} <-> {edge['target']}: "
                  f"{edge['distance_mm']:.1f}mm", flush=True)

    # Multi-patient 2D plot
    plot_multi_patient_2d(all_vkgs, os.path.join(RESULTS_DIR, "vkg_all_patients.png"))

    # Save JSON
    with open(os.path.join(RESULTS_DIR, "vkg_pancreas.json"), "w") as f:
        json.dump(all_vkgs, f, indent=2)
    print(f"\n  VKG JSON saved: {RESULTS_DIR}/vkg_pancreas.json", flush=True)


if __name__ == "__main__":
    main()
