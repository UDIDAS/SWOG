#!/usr/bin/env python3
"""
FLARE SAM Fine-tuning Pipeline
-------------------------------
End-to-end pipeline for fine-tuning SAM (Segment Anything Model) on the
FLARE dataset using DBSCAN-generated point prompts.

Based on the SAM-DBSCAN_FLARE notebooks.

Usage:
    conda run -n llmft python flare_sam_finetune.py --class_id 7 --epochs 50

Requires: torch, transformers, monai, scikit-learn, scikit-image, scipy
"""

import argparse
import csv
import os
import socket
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from monai.losses import DiceLoss
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from skimage.measure import label, regionprops
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import SamModel, SamProcessor

torch.set_num_threads(1)

# ── Data paths ──────────────────────────────────────────────────────────────
FLARE_DATA_DIR = "/scratch/ud3d4/acm_data/FLARE"
OUTPUT_DIR = "/scratch/ud3d4/acm_data/FLARE/runs"


# ── Metrics ─────────────────────────────────────────────────────────────────
def compute_dice(pred, gt):
    intersection = torch.sum(pred * gt)
    return (2 * intersection + 1e-6) / (torch.sum(pred) + torch.sum(gt) + 1e-6)


def compute_iou(pred, gt):
    inter = torch.logical_and(pred, gt).sum().float()
    union = torch.logical_or(pred, gt).sum().float()
    return inter / (union + 1e-6)


def compute_accuracy(pred, gt):
    return (pred == gt).sum().item() / gt.numel()


def compute_precision(pred, gt):
    tp = torch.sum((pred == 1) & (gt == 1)).float()
    fp = torch.sum((pred == 1) & (gt == 0)).float()
    return tp / (tp + fp + 1e-6)


def compute_sensitivity(pred, gt):
    tp = torch.sum((pred == 1) & (gt == 1)).float()
    fn = torch.sum((pred == 0) & (gt == 1)).float()
    return tp / (tp + fn + 1e-6)


def compute_specificity(pred, gt):
    tn = torch.sum((pred == 0) & (gt == 0)).float()
    fp = torch.sum((pred == 1) & (gt == 0)).float()
    return tn / (tn + fp + 1e-6)


# ── DBSCAN coordinate generation ───────────────────────────────────────────
def apply_dbscan(region_coords, eps=5, min_samples=10):
    """Cluster region pixels with DBSCAN, return cluster centers."""
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(region_coords)
    labels = db.labels_
    centers = []
    for k in set(labels):
        if k == -1:
            continue
        pts = region_coords[labels == k]
        centers.append(pts.mean(axis=0))
    return np.array(centers) if centers else np.array([])


def generate_coordinates(masks, eps=5, min_samples=10):
    """For each mask, extract point prompts via connected-component + DBSCAN.

    Returns array of (slice_idx, row, col) tuples.
    """
    coordinates = []
    for i, mask in enumerate(masks):
        labeled_mask = label(mask)
        regions = regionprops(labeled_mask)
        for region in regions:
            region_coords = np.array(region.coords)
            cluster_centers = apply_dbscan(region_coords, eps, min_samples)
            if cluster_centers.size == 0:
                centroid = np.round(region.centroid).astype(int)
                coordinates.append((i, centroid[0], centroid[1]))
            else:
                for center in cluster_centers:
                    coordinates.append((i, int(center[0]), int(center[1])))
    return np.array(coordinates)


# ── Dataset ─────────────────────────────────────────────────────────────────
class FLAREDataset(Dataset):
    def __init__(self, images, labels, coordinates):
        self.images = images
        self.labels = labels
        self.coordinates = coordinates

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # (H, W, 3) uint8
        label = self.labels[idx]  # (H, W) binary
        # Get point coordinates for this slice: list of (col, row) = (x, y)
        coords = [(pt[2], pt[1]) for pt in self.coordinates if pt[0] == idx]
        return image, label, coords


def collate_fn(batch):
    images, labels, coords = zip(*batch)
    images = list(images)
    labels = list(labels)
    coords = list(coords)
    return images, labels, coords


# ── Data loading & preprocessing ────────────────────────────────────────────
def load_and_split(class_id, test_size=0.2, val_size=0.5, seed=42):
    """Load a single FLARE class, invert labels, split into train/val/test."""
    img_path = os.path.join(FLARE_DATA_DIR, f"class_{class_id}_images.npy")
    lbl_path = os.path.join(FLARE_DATA_DIR, f"class_{class_id}_labels.npy")

    print(f"Loading class {class_id} from {img_path}")
    images = np.load(img_path)
    labels = np.load(lbl_path)
    print(f"  images: {images.shape}, labels: {labels.shape}")

    # Ensure binary labels: organ=1, background=0
    labels = (labels > 0).astype(np.uint8)

    # Split: first 80/20, then split the 20% into val/test (50/50 = 10%/10%)
    x_train, x_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=test_size, random_state=seed
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=val_size, random_state=seed
    )

    print(f"  train: {x_train.shape}, val: {x_val.shape}, test: {x_test.shape}")
    return x_train, y_train, x_val, y_val, x_test, y_test


def get_or_generate_coords(labels, split_name, class_id, eps=5, min_samples=10):
    """Load cached coordinates or generate them with DBSCAN."""
    cache_path = os.path.join(
        OUTPUT_DIR, f"{split_name}_coordinates_class{class_id}.npy"
    )
    if os.path.exists(cache_path):
        print(f"  Loading cached {split_name} coordinates from {cache_path}")
        return np.load(cache_path, allow_pickle=True)

    print(f"  Generating {split_name} coordinates with DBSCAN (eps={eps}, min_samples={min_samples})...")
    coords = generate_coordinates(labels, eps=eps, min_samples=min_samples)
    np.save(cache_path, coords)
    print(f"  Saved {len(coords)} coordinates to {cache_path}")
    return coords


# ── DDP training ────────────────────────────────────────────────────────────
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def setup_ddp(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def train_worker(
    rank,
    world_size,
    port,
    args,
    x_train,
    y_train,
    x_val,
    y_val,
    train_coords,
    val_coords,
):
    setup_ddp(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")
    print(f"[Rank {rank}] Initializing on {device}", flush=True)

    # Load model
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam = SamModel.from_pretrained("facebook/sam-vit-base").to(device)

    # Optionally load a pretrained checkpoint for transfer learning
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"[Rank {rank}] Loading pretrained weights from {args.pretrained_path}")
        state_dict = torch.load(args.pretrained_path, map_location=device)
        # Strip "module." prefix if from DDP
        cleaned = {
            k.replace("module.", ""): v for k, v in state_dict.items()
        }
        sam.load_state_dict(cleaned, strict=False)

    sam = DDP(sam, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    loss_fn = DiceLoss(to_onehot_y=False, sigmoid=False)
    optimizer = Adam(sam.parameters(), lr=args.lr)
    scaler = GradScaler()

    train_dataset = FLAREDataset(x_train, y_train, train_coords)
    val_dataset = FLAREDataset(x_val, y_val, val_coords)

    best_val_loss = float("inf")
    no_improve = 0
    all_train_losses = []
    all_val_losses = []

    model_save_path = os.path.join(
        OUTPUT_DIR, f"sam_flare_class{args.class_id}.pth"
    )
    csv_path = os.path.join(
        OUTPUT_DIR, f"metrics_class{args.class_id}.csv"
    )

    if rank == 0:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Epoch", "Train Loss", "Val Loss",
                "Train Acc", "Val Acc",
                "Train Dice", "Val Dice",
                "Train IoU", "Val IoU",
            ])

    start_time = time.time()

    for epoch in range(args.epochs):
        # ── Train ───────────────────────────────────────────────────────
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=0,
        )

        sam.train()
        epoch_train = {"loss": [], "dice": [], "iou": [], "acc": []}

        pbar = tqdm(
            train_loader,
            desc=f"[R{rank}] Train E{epoch+1}",
            disable=(rank != 0),
        )
        for images_batch, labels_batch, coords_batch in pbar:
            # Prepare images: ensure (N, 3, H, W) float
            imgs_processed = []
            for img in images_batch:
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img).float()
                if img.ndim == 2:
                    img = img.unsqueeze(0).repeat(3, 1, 1)
                elif img.ndim == 3 and img.shape[-1] == 3:
                    img = img.permute(2, 0, 1)  # HWC -> CHW
                imgs_processed.append(img)

            # Process through SAM processor (handles resizing to 1024x1024)
            inputs = processor(
                images=[img.numpy().transpose(1, 2, 0) if isinstance(img, torch.Tensor) else img
                        for img in imgs_processed],
                return_tensors="pt",
                do_rescale=False,
            )
            pixel_values = inputs["pixel_values"].to(device)

            # Prepare point prompts: (batch, num_points, 2) and labels (batch, num_points)
            # SAM expects points in (x, y) format, scaled to 1024x1024
            # Original images are 256x256, SAM resizes to 1024x1024 → scale factor 4
            scale = 1024.0 / 256.0
            input_points_list = []
            input_labels_list = []
            for coords in coords_batch:
                if len(coords) > 0:
                    pts = torch.tensor(coords, dtype=torch.float32) * scale  # (N, 2) already (x, y)
                    input_points_list.append(pts.unsqueeze(0))  # (1, N, 2)
                    input_labels_list.append(torch.ones(1, len(coords), dtype=torch.long))
                else:
                    # No points: use center as fallback
                    input_points_list.append(torch.tensor([[[512.0, 512.0]]]))
                    input_labels_list.append(torch.ones(1, 1, dtype=torch.long))

            # Pad to same number of points per batch
            max_pts = max(p.shape[1] for p in input_points_list)
            padded_points = []
            padded_labels = []
            for pts, lbs in zip(input_points_list, input_labels_list):
                n = pts.shape[1]
                if n < max_pts:
                    # Pad with zeros and label -1 (ignored)
                    pad_pts = torch.zeros(1, max_pts - n, 2)
                    pad_lbs = -torch.ones(1, max_pts - n, dtype=torch.long)
                    pts = torch.cat([pts, pad_pts], dim=1)
                    lbs = torch.cat([lbs, pad_lbs], dim=1)
                padded_points.append(pts)
                padded_labels.append(lbs)

            input_points = torch.cat(padded_points, dim=0).unsqueeze(1).to(device)  # (B, 1, max_pts, 2)
            input_labels = torch.cat(padded_labels, dim=0).unsqueeze(1).to(device)  # (B, 1, max_pts)

            # Prepare ground truth labels: resize to SAM output size (256x256)
            gt_masks = []
            for lbl in labels_batch:
                if isinstance(lbl, np.ndarray):
                    lbl = torch.from_numpy(lbl).float()
                gt_masks.append(lbl.unsqueeze(0))  # (1, H, W)
            gt_masks = torch.stack(gt_masks).to(device)  # (B, 1, 256, 256)

            optimizer.zero_grad()
            with autocast():
                outputs = sam(
                    pixel_values=pixel_values,
                    input_points=input_points,
                    input_labels=input_labels,
                    multimask_output=False,
                )
                # pred_masks: (B, 1, 256, 256)
                pred_probs = torch.sigmoid(outputs.pred_masks.squeeze(1))

                # Resize gt to match pred if needed
                if pred_probs.shape != gt_masks.shape:
                    gt_masks = torch.nn.functional.interpolate(
                        gt_masks, size=pred_probs.shape[-2:], mode="nearest"
                    )

                loss = loss_fn(pred_probs, gt_masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pred_binary = (pred_probs > 0.5).float()
            epoch_train["loss"].append(loss.item())
            epoch_train["dice"].append(compute_dice(pred_binary, gt_masks).item())
            epoch_train["iou"].append(compute_iou(pred_binary, gt_masks.bool()).item())
            epoch_train["acc"].append(compute_accuracy(pred_binary, gt_masks.bool()))

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        dist.barrier()

        # ── Validate ────────────────────────────────────────────────────
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=0,
        )

        sam.eval()
        epoch_val = {"loss": [], "dice": [], "iou": [], "acc": []}

        with torch.no_grad():
            for images_batch, labels_batch, coords_batch in val_loader:
                imgs_processed = []
                for img in images_batch:
                    if isinstance(img, np.ndarray):
                        img = torch.from_numpy(img).float()
                    if img.ndim == 2:
                        img = img.unsqueeze(0).repeat(3, 1, 1)
                    elif img.ndim == 3 and img.shape[-1] == 3:
                        img = img.permute(2, 0, 1)
                    imgs_processed.append(img)

                inputs = processor(
                    images=[img.numpy().transpose(1, 2, 0) if isinstance(img, torch.Tensor) else img
                            for img in imgs_processed],
                    return_tensors="pt",
                    do_rescale=False,
                )
                pixel_values = inputs["pixel_values"].to(device)

                scale = 1024.0 / 256.0
                input_points_list = []
                input_labels_list = []
                for coords in coords_batch:
                    if len(coords) > 0:
                        pts = torch.tensor(coords, dtype=torch.float32) * scale
                        input_points_list.append(pts.unsqueeze(0))
                        input_labels_list.append(torch.ones(1, len(coords), dtype=torch.long))
                    else:
                        input_points_list.append(torch.tensor([[[512.0, 512.0]]]))
                        input_labels_list.append(torch.ones(1, 1, dtype=torch.long))

                max_pts = max(p.shape[1] for p in input_points_list)
                padded_points = []
                padded_labels = []
                for pts, lbs in zip(input_points_list, input_labels_list):
                    n = pts.shape[1]
                    if n < max_pts:
                        pad_pts = torch.zeros(1, max_pts - n, 2)
                        pad_lbs = -torch.ones(1, max_pts - n, dtype=torch.long)
                        pts = torch.cat([pts, pad_pts], dim=1)
                        lbs = torch.cat([lbs, pad_lbs], dim=1)
                    padded_points.append(pts)
                    padded_labels.append(lbs)

                input_points = torch.cat(padded_points, dim=0).unsqueeze(1).to(device)
                input_labels = torch.cat(padded_labels, dim=0).unsqueeze(1).to(device)

                gt_masks = []
                for lbl in labels_batch:
                    if isinstance(lbl, np.ndarray):
                        lbl = torch.from_numpy(lbl).float()
                    gt_masks.append(lbl.unsqueeze(0))
                gt_masks = torch.stack(gt_masks).to(device)

                with autocast():
                    outputs = sam(
                        pixel_values=pixel_values,
                        input_points=input_points,
                        input_labels=input_labels,
                        multimask_output=False,
                    )
                    pred_probs = torch.sigmoid(outputs.pred_masks.squeeze(1))
                    if pred_probs.shape != gt_masks.shape:
                        gt_masks = torch.nn.functional.interpolate(
                            gt_masks, size=pred_probs.shape[-2:], mode="nearest"
                        )
                    val_loss = loss_fn(pred_probs, gt_masks)

                pred_binary = (pred_probs > 0.5).float()
                epoch_val["loss"].append(val_loss.item())
                epoch_val["dice"].append(compute_dice(pred_binary, gt_masks).item())
                epoch_val["iou"].append(compute_iou(pred_binary, gt_masks.bool()).item())
                epoch_val["acc"].append(compute_accuracy(pred_binary, gt_masks.bool()))

        # ── Epoch summary ───────────────────────────────────────────────
        avg = {
            k: {m: np.mean(v) for m, v in d.items()}
            for k, d in [("train", epoch_train), ("val", epoch_val)]
        }

        all_train_losses.append(avg["train"]["loss"])
        all_val_losses.append(avg["val"]["loss"])

        if rank == 0:
            print(
                f"  E{epoch+1}: "
                f"TrLoss={avg['train']['loss']:.4f} TrDice={avg['train']['dice']:.4f} "
                f"TrIoU={avg['train']['iou']:.4f} TrAcc={avg['train']['acc']:.4f} | "
                f"VaLoss={avg['val']['loss']:.4f} VaDice={avg['val']['dice']:.4f} "
                f"VaIoU={avg['val']['iou']:.4f} VaAcc={avg['val']['acc']:.4f}",
                flush=True,
            )
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    avg["train"]["loss"], avg["val"]["loss"],
                    avg["train"]["acc"], avg["val"]["acc"],
                    avg["train"]["dice"], avg["val"]["dice"],
                    avg["train"]["iou"], avg["val"]["iou"],
                ])

        # Early stopping
        if avg["val"]["loss"] < best_val_loss:
            best_val_loss = avg["val"]["loss"]
            no_improve = 0
            if rank == 0:
                torch.save(sam.state_dict(), model_save_path)
                print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1
            if rank == 0:
                print(f"  -> No improve ({no_improve}/{args.patience})")

        if no_improve >= args.patience:
            if rank == 0:
                print(f"  Early stopping at epoch {epoch+1}")
            break

        dist.barrier()

    if rank == 0:
        elapsed = time.time() - start_time
        h, r = divmod(elapsed, 3600)
        m, s = divmod(r, 60)
        print(f"Training complete in {int(h)}h {int(m)}m {int(s)}s")

    dist.destroy_process_group()


# ── Test evaluation ─────────────────────────────────────────────────────────
def evaluate_test(args, x_test, y_test, test_coords):
    """Run inference on test set using the best saved model."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam = SamModel.from_pretrained("facebook/sam-vit-base")

    model_path = os.path.join(OUTPUT_DIR, f"sam_flare_class{args.class_id}.pth")
    if not os.path.exists(model_path):
        print(f"ERROR: No model checkpoint found at {model_path}")
        print("Train a model first before running --test_only")
        return

    state_dict = torch.load(model_path, map_location=device)
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    sam.load_state_dict(cleaned, strict=False)
    print(f"Loaded best model from {model_path}")

    sam.to(device).eval()

    test_dataset = FLAREDataset(x_test, y_test, test_coords)
    test_loader = DataLoader(
        test_dataset, batch_size=1, collate_fn=collate_fn, num_workers=0
    )

    all_metrics = {"dice": [], "iou": [], "acc": [], "precision": [], "sensitivity": [], "specificity": []}
    scale = 1024.0 / 256.0

    with torch.no_grad():
        for images_batch, labels_batch, coords_batch in tqdm(test_loader, desc="Testing"):
            img = images_batch[0]
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)

            inputs = processor(
                images=[img.numpy().transpose(1, 2, 0)],
                return_tensors="pt",
                do_rescale=False,
            )
            pixel_values = inputs["pixel_values"].to(device)

            coords = coords_batch[0]
            if len(coords) > 0:
                pts = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).unsqueeze(0) * scale  # (1, 1, N, 2)
                lbs = torch.ones(1, 1, len(coords), dtype=torch.long)
            else:
                pts = torch.tensor([[[[512.0, 512.0]]]])  # (1, 1, 1, 2)
                lbs = torch.ones(1, 1, 1, dtype=torch.long)

            pts = pts.to(device)
            lbs = lbs.to(device)

            outputs = sam(
                pixel_values=pixel_values,
                input_points=pts,
                input_labels=lbs,
                multimask_output=False,
            )
            pred_probs = torch.sigmoid(outputs.pred_masks.squeeze())
            pred_binary = (pred_probs > 0.5).float()

            lbl = labels_batch[0]
            if isinstance(lbl, np.ndarray):
                lbl = torch.from_numpy(lbl).float()
            gt = lbl.to(device)

            # Resize if needed
            if pred_binary.shape != gt.shape:
                gt = torch.nn.functional.interpolate(
                    gt.unsqueeze(0).unsqueeze(0),
                    size=pred_binary.shape[-2:],
                    mode="nearest",
                ).squeeze()

            all_metrics["dice"].append(compute_dice(pred_binary, gt).item())
            all_metrics["iou"].append(compute_iou(pred_binary, gt.bool()).item())
            all_metrics["acc"].append(compute_accuracy(pred_binary, gt.bool()))
            all_metrics["precision"].append(compute_precision(pred_binary, gt).item())
            all_metrics["sensitivity"].append(compute_sensitivity(pred_binary, gt).item())
            all_metrics["specificity"].append(compute_specificity(pred_binary, gt).item())

    print("\n=== Test Results ===")
    for metric, values in all_metrics.items():
        print(f"  {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    # Save test metrics
    csv_path = os.path.join(OUTPUT_DIR, f"test_metrics_class{args.class_id}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(all_metrics.keys()))
        for i in range(len(all_metrics["dice"])):
            writer.writerow([all_metrics[k][i] for k in all_metrics])
    print(f"Test metrics saved to {csv_path}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FLARE SAM Fine-tuning")
    parser.add_argument("--class_id", type=int, default=7, help="FLARE class ID (0-14)")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Path to pretrained model for transfer learning")
    parser.add_argument("--test_only", action="store_true", help="Only run test evaluation")
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs (default: all available)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and split data
    x_train, y_train, x_val, y_val, x_test, y_test = load_and_split(args.class_id)

    # Generate DBSCAN coordinates
    print("Generating/loading DBSCAN coordinates...")
    train_coords = get_or_generate_coords(y_train, "train", args.class_id)
    val_coords = get_or_generate_coords(y_val, "val", args.class_id)
    test_coords = get_or_generate_coords(y_test, "test", args.class_id)
    print(f"  Coordinates: train={len(train_coords)}, val={len(val_coords)}, test={len(test_coords)}")

    if args.test_only:
        evaluate_test(args, x_test, y_test, test_coords)
        return

    # Train with DDP
    world_size = args.gpus or torch.cuda.device_count()
    print(f"\nStarting DDP training on {world_size} GPUs...")
    port = find_free_port()

    mp.spawn(
        train_worker,
        args=(
            world_size, port, args,
            x_train, y_train, x_val, y_val,
            train_coords, val_coords,
        ),
        nprocs=world_size,
        join=True,
    )

    # Evaluate on test set
    print("\nRunning test evaluation...")
    evaluate_test(args, x_test, y_test, test_coords)


if __name__ == "__main__":
    main()
