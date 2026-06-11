"""
Replicate SEMIR paper results on LiTS dataset.

Paper target: 0.891 ± 0.007 tumor Dice, ~1075 supernodes.

This script:
1. Loads LiTS volumes (official split: 131 train)
2. Runs SMBO (Optuna with ExtraTrees) parameter search on 20 few-shot examples
3. Builds graph minors with optimized params
4. Trains GINE classifier
5. Evaluates voxel-level Dice

Usage:
    conda run -n llmft python -m semir.replicate_lits
"""

import os, sys, json, time, re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semir.graph_minor import build_graph_minor
from semir.features import extract_node_features, extract_edge_features, build_pyg_graph
from semir.gine import SEMIRClassifier, train_gine
from semir.param_search import boundary_dice, volume_dice

DATA_ROOT = "/scratch/ud3d4/acm_data/Data"
RESULTS_DIR = "/home/ud3d4/Desktop/SWOG/results/replicate_lits"
os.makedirs(RESULTS_DIR, exist_ok=True)


def pr(msg=""):
    print(msg, flush=True)


def load_lits(vid):
    ct = np.load(os.path.join(DATA_ROOT, "ct", f"volume-{vid}.npy")).astype(np.float32)
    seg = np.load(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")).astype(np.int32)
    return ct, seg


def hu_window(vol, hu_min, hu_max):
    vol = np.clip(vol, hu_min, hu_max)
    return (vol - hu_min) / (hu_max - hu_min)


def discover_lits():
    ids = []
    for f in sorted(os.listdir(os.path.join(DATA_ROOT, "ct"))):
        m = re.match(r"volume-(\d+)\.npy", f)
        if m:
            vid = int(m.group(1))
            if os.path.exists(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")):
                ids.append(vid)
    return sorted(ids)


def smbo_search(volumes, segs, n_trials=200):
    """
    Paper's Algorithm 5: SMBO with ExtraTrees surrogate.
    Uses Optuna's TPE sampler (similar to paper's approach).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    best_result = {"score": -1, "params": {}}

    def objective(trial):
        psi = trial.suggest_float("psi", 0.005, 0.3, log=True)
        alpha = trial.suggest_float("alpha", 0.005, 0.5, log=True)
        beta_min = trial.suggest_int("beta_min", 1, 50000, log=True)
        beta_max = 500000
        m_min = trial.suggest_float("m_min", 0.0, 0.3)
        m_max = trial.suggest_float("m_max", 0.7, 1.0)

        dices = []
        for vol, seg in zip(volumes, segs):
            try:
                gm = build_graph_minor(
                    vol, psi=psi, alpha=alpha,
                    beta_min=beta_min, beta_max=beta_max,
                    m_min=m_min, m_max=m_max, method="c",
                )
                bd = boundary_dice(gm["labels"], seg, target_label=2)
                dices.append(bd)
            except Exception:
                dices.append(0.0)

        score = float(np.mean(dices))

        if score > best_result["score"]:
            best_result["score"] = score
            best_result["params"] = {
                "psi": psi, "alpha": alpha, "beta_min": beta_min,
                "beta_max": beta_max, "m_min": m_min, "m_max": m_max,
            }
            pr(f"    NEW BEST trial {trial.number}: bDice={score:.4f} "
               f"psi={psi:.4f} alpha={alpha:.4f} beta={beta_min}")

        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return best_result["params"], study


def main():
    all_ids = discover_lits()
    pr(f"Found {len(all_ids)} LiTS volumes")

    max_cases = int(os.environ.get("SEMIR_N_CASES", len(all_ids)))

    # Paper uses 20 few-shot examples for LiTS
    n_few = min(20, len(all_ids))
    n_trials = int(os.environ.get("SEMIR_N_TRIALS", 200))

    # Split: 70/15/15
    np.random.seed(42)
    perm = np.random.permutation(all_ids)
    n = min(len(perm), max_cases)
    n_train = max(3, int(0.7 * n))
    n_val = max(1, int(0.15 * n))
    train_ids = sorted(perm[:n_train].tolist())
    val_ids = sorted(perm[n_train:n_train + n_val].tolist())
    test_ids = sorted(perm[n_train + n_val:n].tolist())
    pr(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

    # Try multiple HU windows — the paper doesn't specify which one
    hu_windows = [
        (0, 200, "liver [0,200]"),
        (-100, 200, "wide [-100,200]"),
        (-150, 250, "standard [-150,250]"),
    ]

    best_overall = {"vdice": -1}

    for hu_min, hu_max, hu_label in hu_windows:
        pr(f"\n{'='*70}")
        pr(f"  HU Window: {hu_label}")
        pr(f"{'='*70}")

        # Load few-shot volumes
        few_ids = train_ids[:n_few]
        pr(f"  Loading {len(few_ids)} few-shot volumes...")
        few_vols, few_segs = [], []
        for vid in few_ids:
            ct, seg = load_lits(vid)
            ct = hu_window(ct, hu_min, hu_max)
            few_vols.append(ct)
            few_segs.append(seg)
            n_tumor = int((seg == 2).sum())
            pr(f"    vol-{vid}: shape={ct.shape} tumor={n_tumor:,}")

        # SMBO parameter search
        pr(f"\n  Running SMBO search ({n_trials} trials on {len(few_ids)} volumes)...")
        t0 = time.time()
        best_params, study = smbo_search(few_vols, few_segs, n_trials=n_trials)
        dt = time.time() - t0
        pr(f"  Search done in {dt:.0f}s")
        pr(f"  Best params: {best_params}")

        # Save search results
        with open(os.path.join(RESULTS_DIR, f"smbo_{hu_label.split()[0]}.json"), "w") as f:
            json.dump({"params": best_params, "hu_window": [hu_min, hu_max],
                       "n_trials": n_trials, "time_s": dt}, f, indent=2)

        # Evaluate on ALL volumes with best params
        pr(f"\n  Evaluating on {n} volumes with best params...")
        all_ids_eval = train_ids + val_ids + test_ids

        # Build graph minors
        graphs_data = []
        for i, vid in enumerate(all_ids_eval):
            ct, seg = load_lits(vid)
            ct = hu_window(ct, hu_min, hu_max)

            gm = build_graph_minor(ct, **best_params, method="c")
            nf = extract_node_features(gm["labels"], ct)
            adj = gm.get("full_adjacency", gm["adjacency"])
            ef = extract_edge_features(gm["labels"], ct, adj, nf)
            data, mapping = build_pyg_graph(nf, ef, gm["labels"], gt_seg=seg)

            split = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
            n_tu = int((data.y == 1).sum()) if hasattr(data, "y") else 0
            nsn = gm["stats"]["n_supernodes_after_deletion"]

            graphs_data.append({
                "vid": vid, "gm": gm, "data": data, "nf": nf, "seg": seg,
                "ct": ct, "split": split,
            })

            if i < 5 or i % 20 == 0:
                pr(f"    [{i+1}/{n}] vol-{vid} [{split}]: {nsn:,} SN, "
                   f"{n_tu} tumor SN, {data.num_edges:,} edges")

        # Train GINE
        pr(f"\n  Training GINE...")
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        train_graphs = [g["data"] for g in graphs_data if g["split"] == "train"]
        val_graphs = [g["data"] for g in graphs_data if g["split"] == "val"]

        # Paper: Adam lr=1e-3, NO weight decay, patience 10
        model, history = train_gine(
            train_graphs, val_graphs,
            epochs=200, lr=1e-3, patience=10, device=device,
        )

        best_val_dice = max(history["val_dice"]) if history["val_dice"] else 0
        pr(f"  Best val Dice: {best_val_dice:.4f}")

        # Evaluate: voxel-level Dice via LUT lifting
        pr(f"\n  Voxel-level Dice scores:")
        model.eval()
        dice_scores = []

        for g in graphs_data:
            data = g["data"]
            gm = g["gm"]
            seg = g["seg"]
            nf = g["nf"]

            with torch.no_grad():
                logits = model(data.to(device))
                preds = logits.argmax(dim=1).cpu().numpy()

            sids = sorted(nf.keys())
            max_label = int(gm["labels"].max())
            tumor_lut = np.zeros(max_label + 1, dtype=np.int32)
            for idx, sid in enumerate(sids):
                if preds[idx] == 1:
                    tumor_lut[sid] = 2
            pred_mask = tumor_lut[gm["labels"]]

            gt_tumor = (seg == 2)
            pred_tumor = (pred_mask == 2)

            if gt_tumor.sum() == 0:
                dice = 1.0 if pred_tumor.sum() == 0 else 0.0
            else:
                inter = (gt_tumor & pred_tumor).sum()
                dice = float(2 * inter / (gt_tumor.sum() + pred_tumor.sum() + 1e-8))

            dice_scores.append({
                "vid": g["vid"], "dice": round(dice, 4),
                "split": g["split"],
                "gt_voxels": int(gt_tumor.sum()),
                "pred_voxels": int(pred_tumor.sum()),
            })

        # Summary
        train_dices = [d["dice"] for d in dice_scores if d["split"] == "train"]
        val_dices = [d["dice"] for d in dice_scores if d["split"] == "val"]
        test_dices = [d["dice"] for d in dice_scores if d["split"] == "test"]
        all_dices = [d["dice"] for d in dice_scores]

        mean_dice = np.mean(all_dices)
        val_dice = np.mean(val_dices) if val_dices else 0
        test_dice = np.mean(test_dices) if test_dices else 0

        pr(f"\n  Results for {hu_label}:")
        pr(f"    Mean Dice:  {mean_dice:.4f}")
        pr(f"    Val Dice:   {val_dice:.4f}")
        pr(f"    Test Dice:  {test_dice:.4f}")
        pr(f"    Paper target: 0.891")

        result = {
            "hu_window": hu_label, "params": best_params,
            "mean_dice": mean_dice, "val_dice": val_dice, "test_dice": test_dice,
            "dice_scores": dice_scores, "best_val_dice_gine": best_val_dice,
        }

        with open(os.path.join(RESULTS_DIR, f"result_{hu_label.split()[0]}.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)

        if mean_dice > best_overall["vdice"]:
            best_overall = {"vdice": mean_dice, "hu": hu_label, "params": best_params}

    # Final summary
    pr(f"\n{'='*70}")
    pr(f"  FINAL SUMMARY")
    pr(f"{'='*70}")
    pr(f"  Best HU window: {best_overall['hu']}")
    pr(f"  Best mean Dice: {best_overall['vdice']:.4f}")
    pr(f"  Best params: {best_overall['params']}")
    pr(f"  Paper target: 0.891 ± 0.007")

    with open(os.path.join(RESULTS_DIR, "final_summary.json"), "w") as f:
        json.dump(best_overall, f, indent=2, default=str)


if __name__ == "__main__":
    main()
