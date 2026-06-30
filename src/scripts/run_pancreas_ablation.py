#!/usr/bin/env python3
"""
Pancreas ablation: isolate augmentation vs box prompt contribution.

Already have:
  - Baseline (no aug, no box): organ 0.793, tumor 0.795
  - Both (aug + box):          organ 0.846, tumor 0.917

This script runs the two missing configs:
  - Aug only  (aug=True,  box=False)
  - Box only  (aug=False, box=True)

Same case-level split, same training strategy as run_pancreas_nifti.py.
"""
import sys, os
sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")

import numpy as np
from sklearn.model_selection import train_test_split
from run_pancreas_nifti import (
    get_case_ids, extract_slices_from_volumes,
    PANCREAS_DIR, OUTPUT_DIR
)
from run_flare import (
    generate_coordinates, run_curriculum, run_traditional_increments,
    evaluate_test
)

ABLATION_DIR = "/scratch/ud3d4/acm_data/Pancreas/ablation"
os.makedirs(ABLATION_DIR, exist_ok=True)

CONFIGS = [
    {"name": "aug_only",  "augment": True,  "use_boxes": False},
    {"name": "box_only",  "augment": False, "use_boxes": True},
]


def run_ablation():
    case_ids = get_case_ids()
    train_cases, test_cases = train_test_split(case_ids, test_size=0.2, random_state=42)
    train_cases, val_cases = train_test_split(train_cases, test_size=0.125, random_state=42)
    print(f"Cases: train={len(train_cases)}, val={len(val_cases)}, test={len(test_cases)}")

    for target, label_value, label_name, train_fn in [
        ("organ", 1, "organ", "curriculum"),
        ("tumor", 2, "tumor", "traditional"),
    ]:
        print(f"\n{'='*70}")
        print(f"Target: {label_name} (label={label_value})")
        print(f"{'='*70}")

        # Extract slices (reuse across configs)
        print(f"Extracting {label_name} slices...")
        x_train, y_train, _ = extract_slices_from_volumes(train_cases, label_value)
        x_val, y_val, _ = extract_slices_from_volumes(val_cases, label_value)
        x_test, y_test, _ = extract_slices_from_volumes(test_cases, label_value)
        print(f"Slices: train={len(x_train)}, val={len(x_val)}, test={len(x_test)}")

        # Coords (reuse from main training if they exist, otherwise generate)
        coords_path = os.path.join(OUTPUT_DIR, f"train_coords_{label_name}.npy")
        if os.path.exists(coords_path):
            train_coords = np.load(coords_path, allow_pickle=True)
        else:
            train_coords = generate_coordinates(y_train)
            np.save(coords_path, train_coords)

        val_coords_path = os.path.join(OUTPUT_DIR, f"val_coords_{label_name}.npy")
        if os.path.exists(val_coords_path):
            val_coords = np.load(val_coords_path, allow_pickle=True)
        else:
            val_coords = generate_coordinates(y_val)
            np.save(val_coords_path, val_coords)

        test_coords = generate_coordinates(y_test)

        for config in CONFIGS:
            tag = f"{label_name}_{config['name']}"
            model_path = os.path.join(ABLATION_DIR, f"{tag}.pth")

            print(f"\n--- {tag}: augment={config['augment']}, use_boxes={config['use_boxes']} ---")

            if not os.path.exists(model_path):
                cfg = {
                    "model_save_path": model_path,
                    "epochs": 1000, "batch_size": 5, "lr": 1e-5,
                    "augment": config["augment"],
                    "use_boxes": config["use_boxes"],
                }

                if train_fn == "curriculum":
                    cfg["patience"] = 20
                    cfg["data_increment_patience"] = 3
                    run_curriculum(cfg, x_train, y_train, x_val, y_val,
                                   train_coords, val_coords, order="h2e")
                else:
                    organ_path = os.path.join(ABLATION_DIR, f"organ_{config['name']}.pth")
                    if os.path.exists(organ_path):
                        cfg["pretrained_path"] = organ_path
                        print(f"  Transfer from: {organ_path}")
                    run_traditional_increments(cfg, x_train, y_train, x_val, y_val,
                                               train_coords, val_coords)

            evaluate_test(model_path, x_test, y_test, test_coords,
                         viz_tag=tag, use_boxes=config["use_boxes"])

    print("\n" + "=" * 70)
    print("ABLATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    run_ablation()
