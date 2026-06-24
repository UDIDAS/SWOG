#!/usr/bin/env python3
"""
Runner script for FLARE AUSAM notebook — executes all sections sequentially.
Run with: conda run -n llmft python run_flare.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Setup (copied from notebook)
# ═══════════════════════════════════════════════════════════════════════════════
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import csv, socket, time, random

from monai.losses import DiceLoss
from scipy.stats import entropy as scipy_entropy
from scipy import ndimage
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from skimage.measure import label, regionprops
from skimage import exposure
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import SamModel, SamProcessor

torch.set_num_threads(1)

FLARE_DATA_DIR = "/scratch/ud3d4/acm_data/FLARE"
OUTPUT_DIR = "/scratch/ud3d4/acm_data/FLARE/runs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
WORLD_SIZE = torch.cuda.device_count()
SCALE = 1024.0 / 256.0

print(f"PyTorch {torch.__version__}, GPUs: {WORLD_SIZE}")

# ── Metrics ──
def compute_dice(pred, gt):
    intersection = torch.sum(pred * gt)
    return (2 * intersection + 1e-6) / (torch.sum(pred) + torch.sum(gt) + 1e-6)

def compute_iou(pred, gt):
    return torch.logical_and(pred, gt).sum().float() / (torch.logical_or(pred, gt).sum().float() + 1e-6)

def compute_accuracy(pred, gt):
    return (pred.view(-1) == gt.view(-1)).sum().float() / gt.numel()

def compute_precision(pred, gt):
    tp = torch.sum((pred == 1) & (gt == 1)).float()
    return tp / (tp + torch.sum((pred == 1) & (gt == 0)).float() + 1e-6)

def compute_sensitivity(pred, gt):
    tp = torch.sum((pred == 1) & (gt == 1)).float()
    return tp / (tp + torch.sum((pred == 0) & (gt == 1)).float() + 1e-6)

def compute_specificity(pred, gt):
    tn = torch.sum((pred == 0) & (gt == 0)).float()
    return tn / (tn + torch.sum((pred == 1) & (gt == 0)).float() + 1e-6)

def all_metrics(pred_binary, gt):
    return {
        "dice": compute_dice(pred_binary.float(), gt.float()).item(),
        "iou": compute_iou(pred_binary, gt.bool()).item(),
        "acc": compute_accuracy(pred_binary, gt.bool()).item(),
        "precision": compute_precision(pred_binary, gt).item(),
        "sensitivity": compute_sensitivity(pred_binary, gt).item(),
        "specificity": compute_specificity(pred_binary, gt).item(),
    }

# ── DBSCAN ──
def apply_dbscan(region_coords, eps=5, min_samples=10):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(region_coords)
    centers = []
    for k in set(db.labels_):
        if k == -1: continue
        centers.append(region_coords[db.labels_ == k].mean(axis=0))
    return np.array(centers) if centers else np.array([])

def generate_coordinates(masks, eps=5, min_samples=10):
    coordinates = []
    for i, mask in enumerate(masks):
        for region in regionprops(label(mask)):
            centers = apply_dbscan(np.array(region.coords), eps, min_samples)
            if centers.size == 0:
                c = np.round(region.centroid).astype(int)
                coordinates.append((i, c[0], c[1]))
            else:
                for center in centers:
                    coordinates.append((i, int(center[0]), int(center[1])))
    return np.array(coordinates)

# ── Dataset ──
class FLAREDataset(Dataset):
    def __init__(self, images, labels, coordinates):
        self.images, self.labels, self.coordinates = images, labels, coordinates
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], [(pt[2], pt[1]) for pt in self.coordinates if pt[0] == idx]

def collate_fn(batch):
    images, labels, coords = zip(*batch)
    return list(images), list(labels), list(coords)

def load_and_split(class_id, test_size=0.2, val_size=0.5, seed=42):
    images = np.load(os.path.join(FLARE_DATA_DIR, f"class_{class_id}_images.npy"))
    labels = np.load(os.path.join(FLARE_DATA_DIR, f"class_{class_id}_labels.npy"))
    labels = (labels > 0).astype(np.uint8)
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(images, labels, test_size=test_size, random_state=seed)
    x_va, x_te, y_va, y_te = train_test_split(x_tmp, y_tmp, test_size=val_size, random_state=seed)
    print(f"Class {class_id}: train={x_tr.shape[0]}, val={x_va.shape[0]}, test={x_te.shape[0]}")
    return x_tr, y_tr, x_va, y_va, x_te, y_te

def get_or_generate_coords(labels, split_name, class_id):
    cache = os.path.join(OUTPUT_DIR, f"{split_name}_coords_class{class_id}.npy")
    if os.path.exists(cache):
        return np.load(cache, allow_pickle=True)
    print(f"  Generating {split_name} coordinates...")
    coords = generate_coordinates(labels)
    np.save(cache, coords)
    return coords

# ── Training Helpers ──
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0)); return s.getsockname()[1]

def setup_ddp(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def prepare_batch(images_batch, labels_batch, coords_batch, processor, device):
    imgs = [img.astype(np.float32) if isinstance(img, np.ndarray) else img for img in images_batch]
    imgs = [img if img.ndim == 3 and img.shape[-1] == 3 else np.stack([img]*3, axis=-1) for img in imgs]
    inputs = processor(images=imgs, return_tensors="pt", do_rescale=False)
    pixel_values = inputs["pixel_values"].to(device)

    pts_list, lbs_list = [], []
    for coords in coords_batch:
        if len(coords) > 0:
            pts_list.append(torch.tensor(coords, dtype=torch.float32).unsqueeze(0) * SCALE)
            lbs_list.append(torch.ones(1, len(coords), dtype=torch.long))
        else:
            pts_list.append(torch.tensor([[[512.0, 512.0]]]))
            lbs_list.append(torch.ones(1, 1, dtype=torch.long))

    max_pts = max(p.shape[1] for p in pts_list)
    pp, pl = [], []
    for pts, lbs in zip(pts_list, lbs_list):
        n = pts.shape[1]
        if n < max_pts:
            pts = torch.cat([pts, torch.zeros(1, max_pts - n, 2)], dim=1)
            lbs = torch.cat([lbs, -torch.ones(1, max_pts - n, dtype=torch.long)], dim=1)
        pp.append(pts); pl.append(lbs)

    input_points = torch.cat(pp, dim=0).unsqueeze(1).to(device)
    input_labels = torch.cat(pl, dim=0).unsqueeze(1).to(device)

    gt_masks = torch.stack([torch.from_numpy(l).float().unsqueeze(0) if isinstance(l, np.ndarray)
                            else l.unsqueeze(0) for l in labels_batch]).to(device)
    return pixel_values, input_points, input_labels, gt_masks

def forward_sam(sam, pv, ip, il, gt, loss_fn):
    out = sam(pixel_values=pv, input_points=ip, input_labels=il, multimask_output=False)
    pred = torch.sigmoid(out.pred_masks.squeeze(1))
    if pred.shape != gt.shape:
        gt = torch.nn.functional.interpolate(gt, size=pred.shape[-2:], mode="nearest")
    return loss_fn(pred, gt), (pred > 0.5).float(), gt

# ── Generic trainer ──
def train_worker(rank, world_size, port, cfg, x_tr, y_tr, x_va, y_va, tr_c, va_c, subset_idx=None):
    setup_ddp(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    if cfg.get("pretrained_path") and os.path.exists(cfg["pretrained_path"]):
        state = torch.load(cfg["pretrained_path"], map_location=device)
        sam.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
        if rank == 0: print(f"  Loaded pretrained from {cfg['pretrained_path']}")
    sam = DDP(sam, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    loss_fn = DiceLoss(to_onehot_y=False, sigmoid=False)
    optimizer = Adam(sam.parameters(), lr=cfg.get("lr", 1e-5))
    scaler = GradScaler("cuda")
    train_ds = FLAREDataset(x_tr, y_tr, tr_c)
    val_ds = FLAREDataset(x_va, y_va, va_c)
    best, no_imp = float("inf"), 0
    bs, pat = cfg.get("batch_size", 5), cfg.get("patience", 10)
    save_path = cfg["model_save_path"]
    csv_path = save_path.replace(".pth", "_metrics.csv")
    if rank == 0:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["Epoch","TrLoss","VaLoss","TrDice","VaDice","TrIoU","VaIoU"])
    for epoch in range(cfg.get("epochs", 250)):
        sub = Subset(train_ds, subset_idx) if subset_idx is not None else train_ds
        ts = DistributedSampler(sub, num_replicas=world_size, rank=rank, shuffle=True); ts.set_epoch(epoch)
        tl = DataLoader(sub, batch_size=bs, sampler=ts, collate_fn=collate_fn)
        sam.train(); ep = {"loss":[], "dice":[], "iou":[]}
        for ib, lb, cb in tl:
            pv,ip,il,gt = prepare_batch(ib,lb,cb,processor,device)
            optimizer.zero_grad()
            with autocast("cuda"): loss,pb,gt = forward_sam(sam,pv,ip,il,gt,loss_fn)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            ep["loss"].append(loss.item()); ep["dice"].append(compute_dice(pb,gt).item()); ep["iou"].append(compute_iou(pb,gt.bool()).item())
        dist.barrier()
        vs = DistributedSampler(val_ds, num_replicas=world_size, rank=rank)
        vl = DataLoader(val_ds, batch_size=bs, sampler=vs, collate_fn=collate_fn)
        sam.eval(); ev = {"loss":[], "dice":[], "iou":[]}
        with torch.no_grad():
            for ib,lb,cb in vl:
                pv,ip,il,gt = prepare_batch(ib,lb,cb,processor,device)
                with autocast("cuda"): loss,pb,gt = forward_sam(sam,pv,ip,il,gt,loss_fn)
                ev["loss"].append(loss.item()); ev["dice"].append(compute_dice(pb,gt).item()); ev["iou"].append(compute_iou(pb,gt.bool()).item())
        at = {k:np.mean(v) for k,v in ep.items()}; av = {k:np.mean(v) for k,v in ev.items()}
        if rank == 0:
            print(f"  E{epoch+1}: TrDice={at['dice']:.4f} VaDice={av['dice']:.4f} VaLoss={av['loss']:.4f}")
            with open(csv_path,"a",newline="") as f:
                csv.writer(f).writerow([epoch+1,at["loss"],av["loss"],at["dice"],av["dice"],at["iou"],av["iou"]])
        if av["loss"] < best:
            best = av["loss"]; no_imp = 0
            if rank == 0: torch.save(sam.state_dict(), save_path)
        else:
            no_imp += 1
            if no_imp >= pat:
                if rank == 0: print(f"  Early stopping at epoch {epoch+1}")
                break
        dist.barrier()
    dist.destroy_process_group()

def run_training(cfg, x_tr, y_tr, x_va, y_va, tr_c, va_c, subset_idx=None):
    port = find_free_port()
    mp.spawn(train_worker, args=(WORLD_SIZE,port,cfg,x_tr,y_tr,x_va,y_va,tr_c,va_c,subset_idx), nprocs=WORLD_SIZE, join=True)

# ── Traditional With Increments (exact Tumor notebook strategy) ──
def traditional_increments_worker(rank, world_size, port, cfg, x_tr, y_tr, x_va, y_va, tr_c, va_c, percentages):
    """Train with fixed percentage steps. When val loss plateaus at one level, move to next.
    This is the strategy that produced the best Tumor result (Dice 0.839) in the original notebook.
    """
    setup_ddp(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    if cfg.get("pretrained_path") and os.path.exists(cfg["pretrained_path"]):
        state = torch.load(cfg["pretrained_path"], map_location=device)
        sam.load_state_dict({k.replace("module.",""): v for k,v in state.items()}, strict=False)
    sam = DDP(sam, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    loss_fn = DiceLoss(to_onehot_y=False, sigmoid=False)
    optimizer = Adam(sam.parameters(), lr=cfg.get("lr", 1e-5))
    scaler = GradScaler("cuda")
    train_ds = FLAREDataset(x_tr, y_tr, tr_c)
    val_ds = FLAREDataset(x_va, y_va, va_c)
    n_train = len(train_ds)

    early_stopping_patience = 10
    percentage_switch_patience = 5
    pct_idx = 0
    current_frac = percentages[pct_idx]
    best_val_loss = float("inf")
    no_imp = 0
    no_imp_for_pct = 0
    bs = cfg.get("batch_size", 5)
    save_path = cfg["model_save_path"]
    csv_path = save_path.replace(".pth", "_metrics.csv")

    if rank == 0:
        print(f"  Traditional increments: {len(percentages)} steps, starting at {current_frac*100:.1f}%")
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["Epoch","TrLoss","VaLoss","TrDice","VaDice","TrIoU","VaIoU","PctData"])

    for epoch in range(cfg.get("epochs", 1000)):
        # Select random subset at current percentage
        num_samples = max(1, int(n_train * current_frac))
        indices = np.random.RandomState(epoch).choice(n_train, num_samples, replace=False).tolist()
        sub = Subset(train_ds, indices)

        ts = DistributedSampler(sub, num_replicas=world_size, rank=rank, shuffle=True); ts.set_epoch(epoch)
        tl = DataLoader(sub, batch_size=bs, sampler=ts, collate_fn=collate_fn)
        sam.train(); ep = {"loss":[],"dice":[],"iou":[]}
        for ib,lb,cb in tl:
            pv,ip,il,gt = prepare_batch(ib,lb,cb,processor,device)
            optimizer.zero_grad()
            with autocast("cuda"): loss,pb,gt = forward_sam(sam,pv,ip,il,gt,loss_fn)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(sam.parameters(), max_norm=2.0)
            scaler.step(optimizer); scaler.update()
            ep["loss"].append(loss.item()); ep["dice"].append(compute_dice(pb,gt).item()); ep["iou"].append(compute_iou(pb,gt.bool()).item())
        dist.barrier()

        vs = DistributedSampler(val_ds, num_replicas=world_size, rank=rank); vs.set_epoch(epoch)
        vl = DataLoader(val_ds, batch_size=bs, sampler=vs, collate_fn=collate_fn)
        sam.eval(); ev = {"loss":[],"dice":[],"iou":[]}
        with torch.no_grad():
            for ib,lb,cb in vl:
                pv,ip,il,gt = prepare_batch(ib,lb,cb,processor,device)
                with autocast("cuda"): loss,pb,gt = forward_sam(sam,pv,ip,il,gt,loss_fn)
                ev["loss"].append(loss.item()); ev["dice"].append(compute_dice(pb,gt).item()); ev["iou"].append(compute_iou(pb,gt.bool()).item())

        at = {m:np.mean(v) for m,v in ep.items()}; av = {m:np.mean(v) for m,v in ev.items()}

        # Broadcast improved from rank 0
        improved_t = torch.tensor([int(av["loss"] < best_val_loss)], device=device)
        dist.broadcast(improved_t, src=0)
        improved = bool(improved_t.item())

        if rank == 0:
            print(f"  E{epoch+1}: TrDice={at['dice']:.4f} VaDice={av['dice']:.4f} VaLoss={av['loss']:.4f} "
                  f"Data={current_frac*100:.1f}% (step {pct_idx+1}/{len(percentages)})")
            with open(csv_path,"a",newline="") as f:
                csv.writer(f).writerow([epoch+1,at["loss"],av["loss"],at["dice"],av["dice"],at["iou"],av["iou"],current_frac*100])

        if improved:
            best_val_loss = av["loss"]; no_imp = 0; no_imp_for_pct = 0
            if rank == 0: torch.save(sam.state_dict(), save_path)
        else:
            no_imp += 1; no_imp_for_pct += 1
            if no_imp == early_stopping_patience:
                if rank == 0: print(f"  Early stopping at epoch {epoch+1}")
                break
            # Switch to next percentage when stuck
            if no_imp_for_pct == percentage_switch_patience:
                if pct_idx < len(percentages) - 1:
                    pct_idx += 1
                    current_frac = percentages[pct_idx]
                    no_imp_for_pct = 0
                    if rank == 0:
                        print(f"  >> Stepping to {current_frac*100:.1f}% (step {pct_idx+1}/{len(percentages)})")
        dist.barrier()
    dist.destroy_process_group()

def run_traditional_increments(cfg, x_tr, y_tr, x_va, y_va, tr_c, va_c, percentages=None):
    """Launch Traditional With Increments training (exact Tumor notebook strategy)."""
    if percentages is None:
        # Default percentages from the original Tumor notebook
        percentages = [14.80, 30.30, 40.44, 48.46, 55.50, 62.01,
                       68.34, 74.50, 80.60, 86.60, 92.57, 98.44, 100.0]
        percentages = [p/100.0 for p in percentages]
    port = find_free_port()
    mp.spawn(traditional_increments_worker,
             args=(WORLD_SIZE, port, cfg, x_tr, y_tr, x_va, y_va, tr_c, va_c, percentages),
             nprocs=WORLD_SIZE, join=True)

def visualize_predictions(images, gts, preds, dices, coords_list, save_path, title, n=3):
    """Save figure showing best-n and worst-n predictions side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sorted_idx = np.argsort(dices)
    worst = sorted_idx[:n]
    best = sorted_idx[-n:][::-1]

    fig, axes = plt.subplots(2 * n, 4, figsize=(20, 5 * n))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    for row, (label, indices) in enumerate([("Best", best), ("Worst", worst)]):
        for j, idx in enumerate(indices):
            r = row * n + j
            img = images[idx]
            gt = gts[idx]
            pred = preds[idx]
            pts = coords_list[idx]

            # Image with points
            axes[r, 0].imshow(img)
            if len(pts) > 0:
                axes[r, 0].scatter([p[0] for p in pts], [p[1] for p in pts], c="red", marker="x", s=40)
            axes[r, 0].set_title(f"{label} #{j+1} — Image + Points")
            axes[r, 0].axis("off")

            # Ground truth
            axes[r, 1].imshow(gt, cmap="gray")
            axes[r, 1].set_title("Ground Truth")
            axes[r, 1].axis("off")

            # Prediction
            axes[r, 2].imshow(pred, cmap="gray")
            axes[r, 2].set_title(f"Prediction (Dice={dices[idx]:.3f})")
            axes[r, 2].axis("off")

            # Overlay: GT in green, Pred in red, overlap in yellow
            overlay = np.zeros((*gt.shape, 3))
            overlay[..., 1] = gt       # green = GT
            overlay[..., 0] = pred     # red = pred
            axes[r, 3].imshow(overlay)
            axes[r, 3].set_title("Overlay (G=GT, R=Pred)")
            axes[r, 3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Visualization saved to {save_path}")


def plot_training_curve(csv_path, save_path):
    """Plot training/validation Dice and loss curves from metrics CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(os.path.basename(csv_path).replace("_metrics.csv", ""), fontsize=14)

    ax1.plot(df["Epoch"], df["TrLoss"], label="Train Loss", color="blue")
    ax1.plot(df["Epoch"], df["VaLoss"], label="Val Loss", color="red")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.set_title("Loss")

    ax2.plot(df["Epoch"], df["TrDice"], label="Train Dice", color="blue")
    ax2.plot(df["Epoch"], df["VaDice"], label="Val Dice", color="red")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Dice"); ax2.legend(); ax2.set_title("Dice")
    ax2.set_ylim(0, 1)

    # If PctData column exists (curriculum), add it as secondary axis
    if "PctData" in df.columns:
        ax3 = ax2.twinx()
        ax3.fill_between(df["Epoch"], 0, df["PctData"], alpha=0.1, color="green")
        ax3.set_ylabel("% Data Used", color="green")
        ax3.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curve saved to {save_path}")


def evaluate_test(model_path, x_te, y_te, te_c, viz_tag=None):
    """Test evaluation with optional visualization. viz_tag names the output files."""
    device = torch.device("cuda:0")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam = SamModel.from_pretrained("facebook/sam-vit-base")
    assert os.path.exists(model_path), f"No checkpoint at {model_path}"
    state = torch.load(model_path, map_location=device)
    sam.load_state_dict({k.replace("module.",""): v for k,v in state.items()}, strict=False)
    sam.to(device).eval()
    loader = DataLoader(FLAREDataset(x_te, y_te, te_c), batch_size=1, collate_fn=collate_fn)
    results = {k:[] for k in ["dice","iou","acc","precision","sensitivity","specificity"]}

    # Store predictions for visualization
    all_images, all_gts, all_preds, all_coords = [], [], [], []

    with torch.no_grad():
        for ib,lb,cb in tqdm(loader, desc="Testing"):
            pv,ip,il,gt = prepare_batch(ib,lb,cb,processor,device)
            out = sam(pixel_values=pv, input_points=ip, input_labels=il, multimask_output=False)
            pred = torch.sigmoid(out.pred_masks.squeeze(1))
            if pred.shape != gt.shape: gt = torch.nn.functional.interpolate(gt, size=pred.shape[-2:], mode="nearest")
            pred_bin = (pred>0.5).float()
            m = all_metrics(pred_bin, gt)
            for k in results: results[k].append(m[k])

            # Store for viz (keep on CPU, numpy)
            all_images.append(ib[0] if isinstance(ib[0], np.ndarray) else ib[0].numpy())
            all_gts.append(gt.squeeze().cpu().numpy())
            all_preds.append(pred_bin.squeeze().cpu().numpy())
            all_coords.append(cb[0])

    print(f"\n=== Test Results ({os.path.basename(model_path)}) ===")
    for k,v in results.items(): print(f"  {k}: {np.mean(v):.4f} +/- {np.std(v):.4f}")

    # Visualization
    tag = viz_tag or os.path.basename(model_path).replace(".pth", "")
    viz_dir = os.path.join(OUTPUT_DIR, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    visualize_predictions(
        all_images, all_gts, all_preds, results["dice"], all_coords,
        save_path=os.path.join(viz_dir, f"{tag}_predictions.png"),
        title=f"{tag} — Best & Worst Predictions (Test Dice={np.mean(results['dice']):.4f})",
    )

    # Training curve
    csv_path = model_path.replace(".pth", "_metrics.csv")
    plot_training_curve(csv_path, os.path.join(viz_dir, f"{tag}_training_curve.png"))

    return {k: (np.mean(v), np.std(v)) for k,v in results.items()}

# ── Curriculum helpers ──
def calculate_entropy(image):
    flat = image.flatten()
    hist, _ = np.histogram(flat, bins=np.arange(flat.max() + 2))
    hist = hist / hist.sum()
    return scipy_entropy(hist)

def compute_entropies(images):
    return np.array([calculate_entropy(img) for img in images])

def determine_initial_training_data(images, entropies, k=1):
    """Exact reproduction of notebook's determine_initial_training_data_percentage.
    Returns (initial_pct, entropy_derivative, threshold, mean_d, std_d, original_image_indices).
    original_image_indices: indices where entropy derivative crosses the threshold.
    """
    entropy_deriv = np.diff(entropies)
    mean_d = np.mean(entropy_deriv)
    std_d = np.std(entropy_deriv)
    threshold = mean_d - k * std_d
    significant_change_indices = np.where(entropy_deriv < threshold)[0]
    original_image_indices = significant_change_indices + 1
    initial_pct = len(original_image_indices) / len(images) * 100
    initial_pct = max(10.0, initial_pct)  # ensure at least 10%
    return initial_pct, entropy_deriv, threshold, mean_d, std_d, original_image_indices

def adjust_training_data_percentage(current_pct, rate_of_change, max_pct=100):
    """Exact reproduction of notebook's adjust_training_data_percentage."""
    return min(current_pct + rate_of_change * 10, max_pct)

def curriculum_train_worker(rank, world_size, port, cfg, x_tr, y_tr, x_va, y_va,
                            tr_c, va_c, entropies):
    """Curriculum worker matching the original notebook exactly.
    Uses entropy derivative thresholds (not sorted ranking) for data selection.
    Hyperparams: epochs=1000, patience=10, data_increment_patience=5, batch=5, lr=1e-5.
    """
    setup_ddp(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    if cfg.get("pretrained_path") and os.path.exists(cfg["pretrained_path"]):
        state = torch.load(cfg["pretrained_path"], map_location=device)
        sam.load_state_dict({k.replace("module.",""): v for k,v in state.items()}, strict=False)
    sam = DDP(sam, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    loss_fn = DiceLoss(to_onehot_y=False, sigmoid=False)
    optimizer = Adam(sam.parameters(), lr=cfg.get("lr", 1e-5))
    scaler = GradScaler("cuda")
    train_ds = FLAREDataset(x_tr, y_tr, tr_c)
    val_ds = FLAREDataset(x_va, y_va, va_c)

    # ── Curriculum state (matching notebook exactly) ──
    pct, entropy_deriv, threshold, mean_d, std_d, orig_indices = \
        determine_initial_training_data(x_tr, entropies)
    k_val = 1.0
    early_stopping_patience = cfg.get("patience", 10)
    data_increment_patience = cfg.get("data_increment_patience", 5)
    best_val_loss = float("inf")
    no_imp = 0
    bs = cfg.get("batch_size", 5)  # notebook: 5
    save_path = cfg["model_save_path"]
    csv_path = save_path.replace(".pth", "_metrics.csv")

    if rank == 0:
        print(f"  Initial data: {pct:.1f}% ({len(orig_indices)} indices from entropy threshold)")
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["Epoch","TrLoss","VaLoss","TrDice","VaDice","TrIoU","VaIoU","PctData","k"])

    for epoch in range(cfg.get("epochs", 1000)):
        # Select training subset using entropy-derived indices (notebook L208-210)
        num_samples = max(1, int(len(orig_indices) * (pct / 100.0)))
        subset_indices = orig_indices[:num_samples].tolist()
        sub = Subset(train_ds, subset_indices)

        ts = DistributedSampler(sub, num_replicas=world_size, rank=rank, shuffle=True); ts.set_epoch(epoch)
        tl = DataLoader(sub, batch_size=bs, sampler=ts, collate_fn=collate_fn, drop_last=False)

        sam.train(); ep = {"loss":[],"dice":[],"iou":[]}
        for ib,lb,cb in tl:
            pv,ip,il,gt = prepare_batch(ib,lb,cb,processor,device)
            optimizer.zero_grad()
            with autocast("cuda"): loss,pb,gt = forward_sam(sam,pv,ip,il,gt,loss_fn)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(sam.parameters(), max_norm=2.0)
            scaler.step(optimizer); scaler.update()
            ep["loss"].append(loss.item()); ep["dice"].append(compute_dice(pb,gt).item()); ep["iou"].append(compute_iou(pb,gt.bool()).item())
        dist.barrier()

        vs = DistributedSampler(val_ds, num_replicas=world_size, rank=rank); vs.set_epoch(epoch)
        vl = DataLoader(val_ds, batch_size=bs, sampler=vs, collate_fn=collate_fn, drop_last=False)
        sam.eval(); ev = {"loss":[],"dice":[],"iou":[]}
        with torch.no_grad():
            for ib,lb,cb in vl:
                pv,ip,il,gt = prepare_batch(ib,lb,cb,processor,device)
                with autocast("cuda"): loss,pb,gt = forward_sam(sam,pv,ip,il,gt,loss_fn)
                ev["loss"].append(loss.item()); ev["dice"].append(compute_dice(pb,gt).item()); ev["iou"].append(compute_iou(pb,gt.bool()).item())

        at = {m:np.mean(v) for m,v in ep.items()}; av = {m:np.mean(v) for m,v in ev.items()}
        if rank == 0:
            print(f"  E{epoch+1}: TrDice={at['dice']:.4f} VaDice={av['dice']:.4f} VaLoss={av['loss']:.4f} "
                  f"Data={pct:.1f}% ({num_samples}/{len(orig_indices)} samples) k={k_val:.3f}")
            with open(csv_path,"a",newline="") as f:
                csv.writer(f).writerow([epoch+1,at["loss"],av["loss"],at["dice"],av["dice"],at["iou"],av["iou"],pct,k_val])

        # ── Early stopping & data expansion (matching notebook L446-514) ──
        # Broadcast improved decision from rank 0 to keep all ranks in sync
        improved_t = torch.tensor([int(av["loss"] < best_val_loss)], device=device)
        dist.broadcast(improved_t, src=0)
        improved = bool(improved_t.item())

        if improved:
            best_val_loss = av["loss"]; no_imp = 0
            if rank == 0: torch.save(sam.state_dict(), save_path)
        else:
            no_imp += 1
            if no_imp == early_stopping_patience:
                if rank == 0: print(f"  Early stopping at epoch {epoch+1}")
                break
            # Data expansion (notebook L470-514) — runs identically on all ranks
            if no_imp % data_increment_patience == 0:
                current_labels = np.concatenate([x_tr[orig_indices[i]] for i in range(num_samples)])
                if current_labels.size > 0:
                    current_entropy = calculate_entropy(current_labels)
                    entropy_roc = abs((current_entropy - np.mean(entropies)) / (np.std(entropies) + 1e-8))
                    new_pct = adjust_training_data_percentage(pct, entropy_roc)
                    old_pct = pct
                    pct = new_pct
                    k_val /= 2
                    threshold = mean_d - k_val * std_d
                    significant = np.where(entropy_deriv < threshold)[0]
                    orig_indices = significant + 1
                    if rank == 0:
                        print(f"  >> Expanding: {old_pct:.1f}% -> {pct:.1f}% "
                              f"(k={k_val:.3f}, {len(orig_indices)} candidate indices)")
        dist.barrier()
    dist.destroy_process_group()

def run_curriculum(cfg, x_tr, y_tr, x_va, y_va, tr_c, va_c, order="e2h"):
    """Launch curriculum training. order param kept for API compat but not used
    (the notebook uses entropy derivative thresholds, not sorted ordering)."""
    entropies = compute_entropies(x_tr)
    port = find_free_port()
    mp.spawn(curriculum_train_worker, args=(WORLD_SIZE,port,cfg,x_tr,y_tr,x_va,y_va,tr_c,va_c,entropies), nprocs=WORLD_SIZE, join=True)


def main():
    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Baseline (reuse existing model or train)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("SECTION 2: Single-Stage Baseline — Class 12 (Duodenum)")
    print("="*70)

    x_tr12, y_tr12, x_va12, y_va12, x_te12, y_te12 = load_and_split(12)
    tr_c12 = get_or_generate_coords(y_tr12, "train", 12)
    va_c12 = get_or_generate_coords(y_va12, "val", 12)
    te_c12 = get_or_generate_coords(y_te12, "test", 12)

    s2_path = os.path.join(OUTPUT_DIR, "s2_baseline_class12.pth")
    existing = os.path.join(OUTPUT_DIR, "sam_flare_class12.pth")
    if os.path.exists(existing) and not os.path.exists(s2_path):
        import shutil
        shutil.copy2(existing, s2_path)
        print(f"  Reusing existing model from {existing}")

    if not os.path.exists(s2_path):
        cfg_s2 = {"model_save_path": s2_path, "epochs": 250, "batch_size": 5, "lr": 1e-5, "patience": 10}
        run_training(cfg_s2, x_tr12, y_tr12, x_va12, y_va12, tr_c12, va_c12)

    results_s2 = evaluate_test(s2_path, x_te12, y_te12, te_c12)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Curriculum Learning: E2H & H2E on Class 4 (Pancreas)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("SECTION 3: Curriculum Learning — Class 4 (Pancreas)")
    print("="*70)

    x_tr4, y_tr4, x_va4, y_va4, x_te4, y_te4 = load_and_split(4)
    tr_c4 = get_or_generate_coords(y_tr4, "train", 4)
    va_c4 = get_or_generate_coords(y_va4, "val", 4)
    te_c4 = get_or_generate_coords(y_te4, "test", 4)

    s3e_path = os.path.join(OUTPUT_DIR, "s3_e2h_class4.pth")
    if not os.path.exists(s3e_path):
        print("\n--- E2H (Easy-to-Hard) ---")
        cfg_e2h = {"model_save_path": s3e_path, "epochs": 1000, "batch_size": 5, "lr": 1e-5, "patience": 10}
        run_curriculum(cfg_e2h, x_tr4, y_tr4, x_va4, y_va4, tr_c4, va_c4, order="e2h")
    results_s3_e2h = evaluate_test(s3e_path, x_te4, y_te4, te_c4)

    s3h_path = os.path.join(OUTPUT_DIR, "s3_h2e_class4.pth")
    if not os.path.exists(s3h_path):
        print("\n--- H2E (Hard-to-Easy) ---")
        cfg_h2e = {"model_save_path": s3h_path, "epochs": 1000, "batch_size": 5, "lr": 1e-5, "patience": 10}
        run_curriculum(cfg_h2e, x_tr4, y_tr4, x_va4, y_va4, tr_c4, va_c4, order="h2e")
    results_s3_h2e = evaluate_test(s3h_path, x_te4, y_te4, te_c4)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — Transfer Learning: Liver → Duodenum
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("SECTION 5: Transfer Learning — Class 1 (Liver) -> Class 12 (Duodenum)")
    print("="*70)

    x_tr1, y_tr1, x_va1, y_va1, x_te1, y_te1 = load_and_split(1)
    tr_c1 = get_or_generate_coords(y_tr1, "train", 1)
    va_c1 = get_or_generate_coords(y_va1, "val", 1)
    te_c1 = get_or_generate_coords(y_te1, "test", 1)

    s5_liver = os.path.join(OUTPUT_DIR, "s5_liver_class1.pth")
    if not os.path.exists(s5_liver):
        print("\n--- Step 1: Train Liver ---")
        run_training({"model_save_path": s5_liver, "epochs": 250, "batch_size": 5, "lr": 1e-5, "patience": 10},
                     x_tr1, y_tr1, x_va1, y_va1, tr_c1, va_c1)

    s5_transfer = os.path.join(OUTPUT_DIR, "s5_transfer_class12.pth")
    if not os.path.exists(s5_transfer):
        print("\n--- Step 2: Transfer to Duodenum ---")
        run_training({"model_save_path": s5_transfer, "pretrained_path": s5_liver,
                      "epochs": 250, "batch_size": 5, "lr": 1e-5, "patience": 10},
                     x_tr12, y_tr12, x_va12, y_va12, tr_c12, va_c12)

    results_s5 = evaluate_test(s5_transfer, x_te12, y_te12, te_c12)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — Tumor: H2E on Class 14
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("SECTION 6: Tumor Segmentation — Class 14")
    print("="*70)

    x_tr14, y_tr14, x_va14, y_va14, x_te14, y_te14 = load_and_split(14)
    tr_c14 = get_or_generate_coords(y_tr14, "train", 14)
    va_c14 = get_or_generate_coords(y_va14, "val", 14)
    te_c14 = get_or_generate_coords(y_te14, "test", 14)

    s6_path = os.path.join(OUTPUT_DIR, "s6_h2e_class14.pth")
    if not os.path.exists(s6_path):
        cfg_tumor = {"model_save_path": s6_path, "epochs": 1000, "batch_size": 5, "lr": 1e-5, "patience": 10}
        run_curriculum(cfg_tumor, x_tr14, y_tr14, x_va14, y_va14, tr_c14, va_c14, order="h2e")
    results_s6 = evaluate_test(s6_path, x_te14, y_te14, te_c14)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 9 — Summary
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("FLARE AUSAM — RESULTS SUMMARY")
    print("="*80)
    def fmt(r, m="dice"):
        if r and m in r: return f"{r[m][0]:.4f} +/- {r[m][1]:.4f}"
        return "N/A"

    print(f"{'Section':<10} {'Method':<30} {'Class':<15} {'Test Dice':<22} {'Target'}")
    print("-"*85)
    for sec, method, cls, res, tgt in [
        ("S2",     "Single-stage baseline",     "12 (Duodenum)", results_s2,      "—"),
        ("S3-E2H", "E2H curriculum",            "4 (Pancreas)",  results_s3_e2h,  "0.822"),
        ("S3-H2E", "H2E curriculum",            "4 (Pancreas)",  results_s3_h2e,  "0.822"),
        ("S5",     "Transfer Liver->Duodenum",  "12 (Duodenum)", results_s5,      "—"),
        ("S6",     "H2E tumor",                 "14 (Tumor)",    results_s6,      "0.839"),
    ]:
        print(f"{sec:<10} {method:<30} {cls:<15} {fmt(res):<22} {tgt}")
    print("="*80)
    print("\nAll metrics saved to:", OUTPUT_DIR)

if __name__ == '__main__':
    main()
