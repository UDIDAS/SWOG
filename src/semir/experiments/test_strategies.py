"""
Compare all 5 contraction strategies on 3 representative pancreas cases.

Usage:
    conda run -n llmft python -m semir.test_strategies
"""
import os, sys, time, json
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semir.graph_minor import build_graph_minor

DATA_ROOT = "/scratch/ud3d4/acm_data/Task07_Pancreas"
RESULTS_DIR = "/home/ud3d4/Desktop/SWOG/results/strategy_comparison"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Representative cases spanning the coverage range
TEST_CASES = [
    ("pancreas_318", 0.6),    # tiny tumor
    ("pancreas_135", 7.6),    # median
    ("pancreas_415", 541.7),  # tumor >> organ
]

STRATEGIES = [
    # (method, psi, alpha, beta_min, beta_max, label)
    ("quantized",          0.12, 0.12, 100, 500000, "A0: Quantized (baseline)"),
    ("flood_fill",         0.12, 0.12, 100, 500000, "A: Flood-fill seed-anchored"),
    ("watershed",          0.12, 0.12, 100, 500000, "B: Grid-seeded watershed"),
    ("flood_fill_capped",  0.12, 0.12, 100, 5000,   "C: Flood-fill capped (5K)"),
    ("felzenszwalb",       0.12, 0.12, 100, 500000, "D: Felzenszwalb 3D"),
    ("kmeans",             0.12, 0.12, 100, 500000, "E: K-means supervoxels"),
]


def evaluate(gm, seg, beta_min):
    """Compute tumor capture metrics for a graph minor result."""
    labels = gm["labels"]
    max_label = int(labels.max())
    flat_labels = labels.ravel()
    flat_gt = (seg.ravel() == 2).astype(np.float64)

    if max_label == 0:
        return {"n_sn": 0, "n_tumor_sn": 0, "capture_pct": 0.0, "full_edges": 0}

    tumor_counts = np.bincount(flat_labels, weights=flat_gt, minlength=max_label + 1)
    total_counts = np.bincount(flat_labels, minlength=max_label + 1)
    safe = np.where(total_counts > 0, total_counts, 1)
    overlap = tumor_counts / safe

    tumor_sids = [i for i in range(max_label + 1)
                  if total_counts[i] >= beta_min and overlap[i] > 0.1]
    captured = sum(int(tumor_counts[i]) for i in tumor_sids)
    gt_total = int((seg == 2).sum())

    full_edges = len(gm.get("full_adjacency", {}))

    return {
        "n_sn": gm["stats"]["n_supernodes_after_deletion"],
        "n_tumor_sn": len(tumor_sids),
        "capture_pct": round(captured / max(gt_total, 1) * 100, 1),
        "full_edges": full_edges,
        "gt_tumor": gt_total,
        "captured_tumor": captured,
    }


def main():
    print("=" * 100, flush=True)
    print("  SEMIR Contraction Strategy Comparison", flush=True)
    print("=" * 100, flush=True)

    # Load test cases
    cases = {}
    for name, cov in TEST_CASES:
        ct = nib.load(os.path.join(DATA_ROOT, "imagesTr", f"{name}.nii.gz")).get_fdata().astype(np.float32)
        seg = nib.load(os.path.join(DATA_ROOT, "labelsTr", f"{name}.nii.gz")).get_fdata().astype(np.int32)
        # Narrow pancreas window
        ct = np.clip(ct, 20, 180)
        ct = (ct - 20.0) / 160.0
        cases[name] = (ct, seg, cov)
        print(f"  Loaded {name}: shape={ct.shape} tumor={int((seg==2).sum()):,} "
              f"organ={int((seg==1).sum()):,} cov={cov}%", flush=True)

    all_results = []

    for method, psi, alpha, beta_min, beta_max, label in STRATEGIES:
        print(f"\n{'='*80}", flush=True)
        print(f"  Strategy: {label}", flush=True)
        print(f"  method={method} psi={psi} alpha={alpha} beta_min={beta_min} beta_max={beta_max}",
              flush=True)
        print(f"{'='*80}", flush=True)

        for name, (ct, seg, cov) in cases.items():
            t0 = time.time()
            try:
                gm = build_graph_minor(
                    ct, psi=psi, alpha=alpha,
                    beta_min=beta_min, beta_max=beta_max,
                    m_min=0.0, m_max=1.0,
                    method=method,
                )
                dt = time.time() - t0
                metrics = evaluate(gm, seg, beta_min)
                metrics["time_s"] = round(dt, 1)
                metrics["strategy"] = label
                metrics["case"] = name
                metrics["coverage"] = cov
                all_results.append(metrics)

                print(f"  {name} (cov={cov:>5.1f}%): "
                      f"{metrics['n_sn']:>6,} SN  "
                      f"{metrics['n_tumor_sn']:>4d} tumor SN  "
                      f"capture={metrics['capture_pct']:>5.1f}%  "
                      f"edges={metrics['full_edges']:>6,}  "
                      f"time={dt:>5.1f}s", flush=True)

            except Exception as e:
                dt = time.time() - t0
                print(f"  {name} (cov={cov:>5.1f}%): FAILED after {dt:.1f}s — {e}",
                      flush=True)
                all_results.append({
                    "strategy": label, "case": name, "coverage": cov,
                    "n_sn": 0, "n_tumor_sn": 0, "capture_pct": 0.0,
                    "full_edges": 0, "time_s": round(dt, 1), "error": str(e),
                })

    # Summary table
    print(f"\n\n{'='*100}", flush=True)
    print(f"  SUMMARY TABLE", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"  {'Strategy':<35s} {'Case':<18s} {'Cov%':>6s} {'SN':>7s} "
          f"{'TuSN':>5s} {'Capt%':>7s} {'Edges':>7s} {'Time':>6s}", flush=True)
    print(f"  {'-'*95}", flush=True)

    for r in all_results:
        err = r.get("error", "")
        if err:
            print(f"  {r['strategy']:<35s} {r['case']:<18s} {r['coverage']:>5.1f}% "
                  f"{'FAIL':>7s} {'':>5s} {'':>7s} {'':>7s} {r['time_s']:>5.1f}s", flush=True)
        else:
            print(f"  {r['strategy']:<35s} {r['case']:<18s} {r['coverage']:>5.1f}% "
                  f"{r['n_sn']:>7,} {r['n_tumor_sn']:>5d} "
                  f"{r['capture_pct']:>6.1f}% {r['full_edges']:>7,} "
                  f"{r['time_s']:>5.1f}s", flush=True)

    # Save results
    with open(os.path.join(RESULTS_DIR, "comparison.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_DIR}/comparison.json", flush=True)


if __name__ == "__main__":
    main()
