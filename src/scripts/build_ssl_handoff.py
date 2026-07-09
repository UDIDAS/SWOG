#!/usr/bin/env python3
"""
Build the SSL segmentation hand-off bundle for the friend's KG platform.

Produces, for both Pancreas (MSD Task07) and LiTS:
  - ssl_handoff_<dataset>.json  (filled per the v1.1 template)
  - ssl_predictions/<dataset>/  (our SAM3 v3 predictions, multi-label 0/1/2)
  - ground_truth/<dataset>/     (GT masks, for side-by-side)
  - ssl_models/*.sha1           (checkpoint hashes; large .pt referenced, not bundled)
  - validate_ssl_handoff.py     (their validator, copied in)

Predictions/GT come from our existing SAM3 NIfTI deliveries. Per-case stats and
derived phenotypes are computed here. Honesty caveats (oracle box prompts, LiTS
liver copied from GT) are written into model.notes.
"""
import os, sys, json, csv, shutil, hashlib
sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
import numpy as np
import nibabel as nib
from scipy import ndimage
from sklearn.model_selection import train_test_split

PANCREAS_DELIV = "/scratch/ud3d4/acm_data/Pancreas/sam3_delivery"
PANCREAS_SRC = "/scratch/ud3d4/acm_data/Pancreas"
LITS_DELIV = "/scratch/ud3d4/acm_data/LiTS/sam3_delivery"
SAM3_PANC_DIR = "/scratch/ud3d4/acm_data/Pancreas/sam3"
SAM3_LITS_DIR = "/scratch/ud3d4/acm_data/LiTS/sam3"
OUT = "/scratch/ud3d4/acm_data/ssl_handoff_ours"
VALIDATOR_SRC = ("/dev/shm/claude-100469397/-home-ud3d4-Desktop-SWOG/"
                 "0bec8eda-3322-4186-a3f5-fea171d418a5/scratchpad/ssl_handoff_v1/validate_ssl_handoff.py")

CAVEATS = (
    "Predictions produced by SAM3 (ViT-L) fine-tuned with the v3 partial-freeze recipe "
    "(freeze backbone blocks 0-19 + patch/pos embeds; train blocks 20-31 + FPN + decoder; "
    "discriminative LR enc 1e-5/dec 1e-4; warmup + cosine annealing; 0.7*Dice + 0.3*Focal). "
    "IMPORTANT CAVEATS: (1) SAM3 is prompt-based and every prediction used a GT-derived "
    "bounding box prompt (padded +/-3px) around each structure -- i.e. predictions are "
    "SEMI-ORACLE (the model was told roughly where to look), not fully autonomous. This is "
    "the gap our CRISP-SAM prompt-generator work aims to close. (2) Model is 2D slice-based "
    "(256x256), not 3D; predictions are stacked back to volume. (3) LiTS LIVER (label 1) is "
    "COPIED FROM GROUND TRUTH -- only the tumor (label 2) is predicted by SAM3; liver will "
    "therefore appear identical in GT vs prediction. Pancreas predicts both organ and tumor. "
    "HU windows: pancreas [-100,300], LiTS [-100,400]. No connected-component or morphological "
    "postprocessing (per-slice sigmoid > 0.5)."
)


def sha1_file(path, first_mb_only=False):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        if first_mb_only:
            h.update(f.read(1024 * 1024))
        else:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def label_stats(mask, spacing):
    voxels = int(mask.sum())
    if voxels == 0:
        return {"voxels": 0, "volume_cm3": 0.0, "max_diameter_mm": 0.0,
                "extent_mm": [0.0, 0.0, 0.0], "centroid_mm": [None, None, None]}
    vox_vol = float(np.prod(spacing))
    coords = np.argwhere(mask).astype(np.float64) * np.array(spacing, dtype=np.float64)
    extent = (coords.max(0) - coords.min(0))
    centroid = coords.mean(0)
    return {
        "voxels": voxels,
        "volume_cm3": round(voxels * vox_vol / 1000.0, 2),
        "max_diameter_mm": round(float(extent.max()), 1),
        "extent_mm": [round(float(e), 1) for e in extent],
        "centroid_mm": [round(float(c), 1) for c in centroid],
    }


def anatomic_location(seg):
    """Head/Body/Tail from tumor centroid X-offset vs pancreas, per spec heuristic."""
    panc = np.argwhere(seg == 1)
    tum = np.argwhere(seg == 2)
    if len(tum) == 0 or len(panc) == 0:
        return "Unknown"
    px, tx = panc[:, 0], tum[:, 0]
    ext = px.max() - px.min()
    if ext == 0:
        return "Unknown"
    offset = (tx.mean() - px.mean()) / ext
    if offset > 0.12:
        return "PancreaticHead"
    if offset < -0.12:
        return "PancreaticTail"
    return "PancreaticBody"


def dice(pred_mask, gt_mask):
    inter = float((pred_mask & gt_mask).sum())
    s = float(pred_mask.sum() + gt_mask.sum())
    return round((2 * inter + 1e-6) / (s + 1e-6), 4) if s > 0 else None


def model_block(dataset, ckpt_rel, ckpt_sha):
    return {
        "architecture": "SAM3 (Segment Anything 3, ViT-L) - v3 partial-freeze fine-tune",
        "ssl_pretrain_method": "None medical SSL; SAM3 backbone pretrained on SA-1B (natural images)",
        "ssl_pretrain_data": [{"name": "SA-1B (via SAM3 backbone, natural images)", "num_volumes": 0, "labeled": False}],
        "ssl_pretrain_epochs": 0,
        "finetune_method": "Partial encoder freeze (blocks 0-19) + discriminative LR + warmup/cosine",
        "finetune_loss": "0.7*Dice + 0.3*Focal",
        "finetune_epochs": 45,
        "input_patch_size_xyz": [256, 256, 1],
        "sliding_window_overlap": 0.0,
        "inference_window_level_hu": (100 if dataset == "pancreas" else 150),
        "inference_window_width_hu": 400,
        "postprocessing": "None (per-slice sigmoid > 0.5, box-prompted)",
        "framework": "PyTorch 2.5.1 + HuggingFace transformers (facebook/sam3)",
        "training_date_iso": "2026-07-08",
        "checkpoint_relative_path": ckpt_rel,
        "checkpoint_sha1": ckpt_sha,
        "training_gpu": "2x NVIDIA L40S (48GB)",
        "training_runtime_hours": None,
        "notes": CAVEATS,
    }


def build_pancreas():
    ds_out = os.path.join(OUT, "ssl_predictions", "pancreas")
    gt_out = os.path.join(OUT, "ground_truth", "pancreas")
    os.makedirs(ds_out, exist_ok=True); os.makedirs(gt_out, exist_ok=True)

    case_ids = sorted(d for d in os.listdir(PANCREAS_DELIV) if d.startswith("pancreas_"))
    train, test = train_test_split(case_ids, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.125, random_state=42)
    split_of = {c: s for s, lst in [("train", train), ("val", val), ("test", test)] for c in lst}

    cases = {}
    test_dice = {"1": [], "2": []}
    for cid in case_ids:
        pred_nii = nib.load(os.path.join(PANCREAS_DELIV, cid, "pred.nii.gz"))
        gt_nii = nib.load(os.path.join(PANCREAS_DELIV, cid, "gt.nii.gz"))
        pred = pred_nii.get_fdata().astype(np.uint8)
        gt = gt_nii.get_fdata().astype(np.uint8)
        spacing = [float(z) for z in gt_nii.header.get_zooms()[:3]]
        shape_xyz = [int(s) for s in pred.shape]

        shutil.copy(os.path.join(PANCREAS_DELIV, cid, "pred.nii.gz"), os.path.join(ds_out, f"{cid}.nii.gz"))
        shutil.copy(os.path.join(PANCREAS_DELIV, cid, "gt.nii.gz"), os.path.join(gt_out, f"{cid}.nii.gz"))

        d1 = dice(pred == 1, gt == 1); d2 = dice(pred == 2, gt == 2)
        if split_of[cid] == "test":
            if d1 is not None: test_dice["1"].append(d1)
            if d2 is not None: test_dice["2"].append(d2)

        loc = anatomic_location(pred)
        cases[cid] = {
            "split": split_of[cid], "has_ground_truth": True,
            "ct": {"relative_path": f"Task07_Pancreas/imagesTr/{cid}.nii.gz", "dtype": "int16",
                   "shape_xyz": shape_xyz, "voxel_spacing_mm": spacing,
                   "voxel_volume_mm3": round(float(np.prod(spacing)), 4),
                   "physical_extent_mm_xyz": [round(shape_xyz[i] * spacing[i], 1) for i in range(3)],
                   "orientation_axcodes": list(nib.aff2axcodes(gt_nii.affine)),
                   "ct_intensity_min_hu": None, "ct_intensity_max_hu": None,
                   "ct_intensity_mean_hu": None, "ct_intensity_std_hu": None, "sha1_first_mb": None},
            "prediction": {"relative_path": f"ssl_predictions/pancreas/{cid}.nii.gz", "dtype": "uint8",
                           "shape_xyz": shape_xyz, "affine_matches_ct": True,
                           "generated_at_iso": "2026-07-08", "sha1": sha1_file(os.path.join(ds_out, f"{cid}.nii.gz")),
                           "softmax_confidence_relative_path": None},
            "ground_truth": {"relative_path": f"ground_truth/pancreas/{cid}.nii.gz", "dtype": "uint8",
                             "shape_xyz": shape_xyz, "sha1": sha1_file(os.path.join(gt_out, f"{cid}.nii.gz"))},
            "per_label_stats_pred": {"1": label_stats(pred == 1, spacing), "2": label_stats(pred == 2, spacing)},
            "per_label_stats_gt": {"1": label_stats(gt == 1, spacing), "2": label_stats(gt == 2, spacing)},
            "per_case_metrics": {"label_1": {"dice": d1, "hd95_mm": None, "nsd_2mm": None},
                                 "label_2": {"dice": d2, "hd95_mm": None, "nsd_2mm": None}},
            "uncertainty": {"mean_voxel_entropy": None, "tumor_label_mean_uncertainty": None, "calibrated": False},
            "derived_phenotype_pancreas": {
                "has_pancreas": bool((pred == 1).any()), "has_tumor": bool((pred == 2).any()),
                "pancreas_volume_cm3": label_stats(pred == 1, spacing)["volume_cm3"],
                "tumor_volume_cm3": label_stats(pred == 2, spacing)["volume_cm3"],
                "tumor_diameter_mm": label_stats(pred == 2, spacing)["max_diameter_mm"],
                "anatomic_location": loc, "anatomic_location_confidence": "low (heuristic centroid offset)"},
        }

    ckpt = os.path.join(SAM3_PANC_DIR, "sam3_v3_tumor.pth")
    ckpt_sha = sha1_file(ckpt) if os.path.exists(ckpt) else None
    H = {
        "schema_version": "1.1",
        "dataset": {"name": "MSD Task07 Pancreas", "short_id": "pancreas", "modality": "CT",
                    "anatomy_focus": "Pancreas + pancreatic tumor (PDAC)", "source_url": "http://medicaldecathlon.com/",
                    "license": "CC-BY-SA 4.0 (Memorial Sloan Kettering)", "num_cases_total": len(case_ids),
                    "voxel_spacing_convention_mm": "native NIfTI header (variable)", "coordinate_system": "LPS"},
        "labels": {"0": {"name": "background", "rgb": [0, 0, 0], "hex": "#000000", "is_tumor": False, "parent_organ": None},
                   "1": {"name": "Pancreas", "rgb": [230, 175, 45], "hex": "#E6AF2D", "is_tumor": False, "parent_organ": "Pancreas"},
                   "2": {"name": "Pancreatic Tumor", "rgb": [220, 50, 50], "hex": "#DC3232", "is_tumor": True, "parent_organ": "Pancreas"}},
        "model": model_block("pancreas", "ssl_models/sam3_pancreas_tumor_v3.pt", ckpt_sha),
        "splits": {"train": train, "val": val, "test": test, "external_unlabeled_for_ssl_pretrain": [],
                   "split_strategy": "case-level (patient-held-out), 70/10/20", "random_seed": 42},
        "overall_metrics": {
            "test_dice_mean": {"label_1": round(float(np.mean(test_dice["1"])), 4) if test_dice["1"] else None,
                               "label_2": round(float(np.mean(test_dice["2"])), 4) if test_dice["2"] else None},
            "test_dice_std": {"label_1": round(float(np.std(test_dice["1"])), 4) if test_dice["1"] else None,
                              "label_2": round(float(np.std(test_dice["2"])), 4) if test_dice["2"] else None},
            "test_hd95_mm_mean": {"label_1": None, "label_2": None},
            "test_nsd_2mm_mean": {"label_1": None, "label_2": None},
            "test_sensitivity": {"label_1": None, "label_2": None},
            "test_specificity": {"label_1": None, "label_2": None}},
        "cases": cases,
    }
    json.dump(H, open(os.path.join(OUT, "ssl_handoff_pancreas.json"), "w"), indent=2)
    print(f"pancreas: {len(case_ids)} cases, test Dice organ={H['overall_metrics']['test_dice_mean']['label_1']} "
          f"tumor={H['overall_metrics']['test_dice_mean']['label_2']}")


def build_lits():
    ds_out = os.path.join(OUT, "ssl_predictions", "lits")
    gt_out = os.path.join(OUT, "ground_truth", "lits")
    os.makedirs(ds_out, exist_ok=True); os.makedirs(gt_out, exist_ok=True)

    vol_dirs = sorted((d for d in os.listdir(LITS_DELIV) if d.startswith("volume-")),
                      key=lambda d: int(d.split("-")[1]))
    vids = [int(d.split("-")[1]) for d in vol_dirs]
    train, test = train_test_split(vids, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.125, random_state=42)
    split_of = {v: s for s, lst in [("train", train), ("val", val), ("test", test)] for v in lst}
    cid_of = lambda v: f"LiTs-{v:03d}"

    spacing = [1.0, 1.0, 1.0]
    cases = {}
    test_dice = {"1": [], "2": []}
    for vid in vids:
        cid = cid_of(vid)
        pred = nib.load(os.path.join(LITS_DELIV, f"volume-{vid}", "pred.nii.gz")).get_fdata().astype(np.uint8)
        gt = nib.load(os.path.join(LITS_DELIV, f"volume-{vid}", "gt.nii.gz")).get_fdata().astype(np.uint8)
        shape_xyz = [int(s) for s in pred.shape]
        shutil.copy(os.path.join(LITS_DELIV, f"volume-{vid}", "pred.nii.gz"), os.path.join(ds_out, f"{cid}.nii.gz"))
        shutil.copy(os.path.join(LITS_DELIV, f"volume-{vid}", "gt.nii.gz"), os.path.join(gt_out, f"{cid}.nii.gz"))

        d1 = dice(pred == 1, gt == 1); d2 = dice(pred == 2, gt == 2)
        if split_of[vid] == "test":
            if d1 is not None: test_dice["1"].append(d1)
            if d2 is not None: test_dice["2"].append(d2)
        lesion_count = int(ndimage.label(pred == 2)[1])

        cases[cid] = {
            "split": split_of[vid], "has_ground_truth": True,
            "ct": {"relative_path": f"LiTs/ct/volume-{vid}.npy", "dtype": "float32", "shape_xyz": shape_xyz,
                   "voxel_spacing_mm": spacing, "voxel_volume_mm3": 1.0,
                   "physical_extent_mm_xyz": [float(s) for s in shape_xyz],
                   "orientation_axcodes": ["L", "A", "S"], "ct_intensity_min_hu": None, "ct_intensity_max_hu": None,
                   "ct_intensity_mean_hu": None, "ct_intensity_std_hu": None, "sha1_first_mb": None},
            "prediction": {"relative_path": f"ssl_predictions/lits/{cid}.nii.gz", "dtype": "uint8",
                           "shape_xyz": shape_xyz, "affine_matches_ct": True, "generated_at_iso": "2026-07-08",
                           "sha1": sha1_file(os.path.join(ds_out, f"{cid}.nii.gz")), "softmax_confidence_relative_path": None},
            "ground_truth": {"relative_path": f"ground_truth/lits/{cid}.nii.gz", "dtype": "uint8",
                             "shape_xyz": shape_xyz, "sha1": sha1_file(os.path.join(gt_out, f"{cid}.nii.gz"))},
            "per_label_stats_pred": {"1": label_stats(pred == 1, spacing), "2": label_stats(pred == 2, spacing)},
            "per_label_stats_gt": {"1": label_stats(gt == 1, spacing), "2": label_stats(gt == 2, spacing)},
            "per_case_metrics": {"label_1": {"dice": d1, "hd95_mm": None, "nsd_2mm": None},
                                 "label_2": {"dice": d2, "hd95_mm": None, "nsd_2mm": None}},
            "uncertainty": {"mean_voxel_entropy": None, "tumor_label_mean_uncertainty": None, "calibrated": False},
            "derived_phenotype_lits": {
                "has_liver": bool((pred == 1).any()), "has_tumor": bool((pred == 2).any()),
                "liver_volume_cm3": label_stats(pred == 1, spacing)["volume_cm3"],
                "tumor_volume_cm3": label_stats(pred == 2, spacing)["volume_cm3"],
                "tumor_max_diameter_mm": label_stats(pred == 2, spacing)["max_diameter_mm"],
                "tumor_lesion_count": lesion_count},
        }

    ckpt = os.path.join(SAM3_LITS_DIR, "lits_sam3_v3_caselevel_tumor.pth")
    ckpt_sha = sha1_file(ckpt) if os.path.exists(ckpt) else None
    H = {
        "schema_version": "1.1",
        "dataset": {"name": "LiTS17 (Liver Tumor Segmentation Challenge)", "short_id": "lits", "modality": "CT",
                    "anatomy_focus": "Liver + liver tumor (HCC + metastases)",
                    "source_url": "https://competitions.codalab.org/competitions/17094",
                    "license": "LiTS Challenge terms (Bilic et al., MedIA 2023)", "num_cases_total": len(vids),
                    "voxel_spacing_convention_mm": "cached .npy pre-resampled to isotropic 1x1x1", "coordinate_system": "LPS"},
        "labels": {"0": {"name": "background", "rgb": [0, 0, 0], "hex": "#000000", "is_tumor": False, "parent_organ": None},
                   "1": {"name": "Liver", "rgb": [180, 90, 90], "hex": "#B45A5A", "is_tumor": False, "parent_organ": "Liver"},
                   "2": {"name": "Liver Tumor", "rgb": [240, 60, 60], "hex": "#F03C3C", "is_tumor": True, "parent_organ": "Liver"}},
        "model": model_block("lits", "ssl_models/sam3_lits_tumor_v3.pt", ckpt_sha),
        "splits": {"train": [cid_of(v) for v in train], "val": [cid_of(v) for v in val], "test": [cid_of(v) for v in test],
                   "external_unlabeled_for_ssl_pretrain": [], "split_strategy": "case-level (patient-held-out), 70/10/20", "random_seed": 42},
        "overall_metrics": {
            "test_dice_mean": {"label_1": round(float(np.mean(test_dice["1"])), 4) if test_dice["1"] else None,
                               "label_2": round(float(np.mean(test_dice["2"])), 4) if test_dice["2"] else None},
            "test_dice_std": {"label_1": round(float(np.std(test_dice["1"])), 4) if test_dice["1"] else None,
                              "label_2": round(float(np.std(test_dice["2"])), 4) if test_dice["2"] else None},
            "test_hd95_mm_mean": {"label_1": None, "label_2": None},
            "test_nsd_2mm_mean": {"label_1": None, "label_2": None},
            "test_sensitivity": {"label_1": None, "label_2": None},
            "test_specificity": {"label_1": None, "label_2": None}},
        "cases": cases,
    }
    json.dump(H, open(os.path.join(OUT, "ssl_handoff_lits.json"), "w"), indent=2)
    print(f"lits: {len(vids)} cases, test Dice liver={H['overall_metrics']['test_dice_mean']['label_1']} "
          f"tumor={H['overall_metrics']['test_dice_mean']['label_2']}")


def write_checkpoint_hashes():
    md = os.path.join(OUT, "ssl_models")
    os.makedirs(md, exist_ok=True)
    entries = [
        ("sam3_pancreas_organ_v3.pt", os.path.join(SAM3_PANC_DIR, "sam3_v3_organ.pth")),
        ("sam3_pancreas_tumor_v3.pt", os.path.join(SAM3_PANC_DIR, "sam3_v3_tumor.pth")),
        ("sam3_lits_tumor_v3.pt", os.path.join(SAM3_LITS_DIR, "lits_sam3_v3_caselevel_tumor.pth")),
    ]
    lines = ["# SAM3 checkpoints (SHA1)\n",
             "Large (~3.3GB each) so referenced, not bundled. Available on request / Google Drive.\n\n"]
    for name, path in entries:
        if os.path.exists(path):
            s = sha1_file(path)
            open(os.path.join(md, name + ".sha1"), "w").write(f"{s}  {name}\n")
            lines.append(f"- {name}: `{s}` ({os.path.getsize(path)/1e9:.2f} GB)\n")
    open(os.path.join(md, "CHECKPOINTS.md"), "w").writelines(lines)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(VALIDATOR_SRC):
        shutil.copy(VALIDATOR_SRC, os.path.join(OUT, "validate_ssl_handoff.py"))
    build_pancreas()
    build_lits()
    write_checkpoint_hashes()
    print("BUNDLE:", OUT)
