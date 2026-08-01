#!/usr/bin/env python3
"""
Pancreas SAM3 experiment — compare SAM3 vs SAM1 on the same test split.

SAM3 is a DETR-based instance segmentation model with text+box prompts
(no point prompts). This script tests:
  1. Zero-shot SAM3 with text prompts ("pancreas", "tumor")
  2. Zero-shot SAM3 with GT-derived box prompts
  3. Fine-tuned SAM3 with box prompts (same protocol as SAM1)

Requires: HF_TOKEN with access to facebook/sam3 (gated model).
"""
import sys, os, csv
sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")

import numpy as np
from scipy.ndimage import rotate as nd_rotate, gaussian_filter, map_coordinates
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, Subset, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from monai.losses import DiceLoss, FocalLoss
from transformers import Sam3Model, Sam3Processor

from run_pancreas_nifti import (
    get_case_ids, extract_slices_from_volumes,
    PANCREAS_DIR, OUTPUT_DIR, hu_to_rgb
)
from run_flare import (
    compute_dice, compute_iou, all_metrics,
    find_free_port, WORLD_SIZE
)

SAM3_DIR = "/scratch/ud3d4/acm_data/Pancreas/sam3"
os.makedirs(SAM3_DIR, exist_ok=True)

SAM3_MODEL_ID = "facebook/sam3"
SAM3_IMAGE_SIZE = 1008
def _load_hf_token():
    """Token from env, else the standard HF cache file. Keeps secrets out of source."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    for p in (os.path.expanduser("~/.cache/huggingface/token"),
              os.path.expanduser("~/.huggingface/token")):
        if os.path.exists(p):
            with open(p) as f:
                t = f.read().strip()
            if t:
                return t
    return None

HF_TOKEN = _load_hf_token()


# ── Augmentation ──
def elastic_deform(img, label, alpha=15, sigma=3):
    """Apply random elastic deformation to image and label jointly."""
    shape = img.shape[:2]
    dx = gaussian_filter(np.random.randn(*shape) * alpha, sigma)
    dy = gaussian_filter(np.random.randn(*shape) * alpha, sigma)
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    coords = [np.clip(y + dy, 0, shape[0] - 1), np.clip(x + dx, 0, shape[1] - 1)]
    if img.ndim == 3:
        out_img = np.stack([map_coordinates(img[..., c], coords, order=1) for c in range(img.shape[2])], axis=-1)
    else:
        out_img = map_coordinates(img, coords, order=1)
    out_label = map_coordinates(label, coords, order=0)
    return out_img.astype(img.dtype), out_label.astype(label.dtype)


def augment_strong(img, label):
    """Stronger augmentation: flip + rotation + brightness + noise + elastic."""
    if np.random.rand() < 0.5:
        img = np.ascontiguousarray(img[:, ::-1])
        label = np.ascontiguousarray(label[:, ::-1])
    if np.random.rand() < 0.5:
        angle = np.random.uniform(-15, 15)
        img = nd_rotate(img, angle, axes=(0, 1), reshape=False, order=1, mode='nearest')
        label = nd_rotate(label, angle, axes=(0, 1), reshape=False, order=0, mode='nearest')
    if isinstance(img, np.ndarray) and img.dtype == np.uint8:
        shift = int(np.random.randint(-20, 21))
        img = np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)
    if np.random.rand() < 0.3:
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if np.random.rand() < 0.3:
        img, label = elastic_deform(img, label)
    return np.ascontiguousarray(img), np.ascontiguousarray(label)


# ── Dataset ──
class Sam3Dataset(Dataset):
    """Dataset that returns images and GT masks for SAM3."""
    def __init__(self, images, labels, augment=False, strong_augment=False):
        self.images = images
        self.labels = labels
        self.augment = augment
        self.strong_augment = strong_augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.strong_augment:
            img, label = augment_strong(img, label)
        elif self.augment:
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                label = np.ascontiguousarray(label[:, ::-1])
            if isinstance(img, np.ndarray) and img.dtype == np.uint8:
                shift = int(np.random.randint(-15, 16))
                img = np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)
        return img, label


def bbox_from_mask(mask_2d, pad=3):
    """Return absolute (x1, y1, x2, y2) bbox from a binary mask.
    SAM3 processor expects xyxy in pixel coords; it normalizes + converts to cxcywh internally."""
    ys, xs = np.where(mask_2d > 0)
    if len(xs) == 0:
        return None
    H, W = mask_2d.shape
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(W - 1, int(xs.max()) + pad)
    y2 = min(H - 1, int(ys.max()) + pad)
    return [x1, y1, x2, y2]


def prepare_sam3_batch(images, labels, processor, device, text_prompt="visual", use_boxes=True):
    """Prepare a batch for SAM3 inference/training."""
    imgs = [img if isinstance(img, np.ndarray) else np.array(img) for img in images]

    boxes = None
    boxes_labels = None
    if use_boxes:
        boxes = []
        boxes_labels = []
        for lbl in labels:
            arr = lbl if isinstance(lbl, np.ndarray) else lbl.numpy()
            box = bbox_from_mask(arr, pad=3)
            if box is None:
                H, W = arr.shape
                box = [0, 0, W - 1, H - 1]
            boxes.append([box])
            boxes_labels.append([1])

    text = [text_prompt] * len(imgs) if isinstance(text_prompt, str) else text_prompt
    inputs = processor(
        images=imgs,
        text=text,
        input_boxes=boxes,
        input_boxes_labels=boxes_labels,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(device)
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")
    input_boxes = inputs.get("input_boxes")
    input_boxes_labels = inputs.get("input_boxes_labels")

    if input_ids is not None:
        input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    if input_boxes is not None:
        input_boxes = input_boxes.to(device)
    if input_boxes_labels is not None:
        input_boxes_labels = input_boxes_labels.to(device)

    gt_masks = torch.stack([
        torch.from_numpy(l).float().unsqueeze(0) if isinstance(l, np.ndarray)
        else l.unsqueeze(0) for l in labels
    ]).to(device)

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "input_boxes": input_boxes,
        "input_boxes_labels": input_boxes_labels,
    }, gt_masks


def extract_best_mask(outputs, processor, target_size=(256, 256)):
    """Extract the best predicted mask from SAM3 output for each image in batch."""
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.0,
        mask_threshold=0.5,
        target_sizes=[target_size] * outputs.pred_masks.shape[0],
    )

    masks = []
    for r in results:
        if len(r["masks"]) > 0:
            best_idx = r["scores"].argmax()
            masks.append(r["masks"][best_idx].float())
        else:
            masks.append(torch.zeros(target_size[0], target_size[1]))
    return torch.stack(masks).unsqueeze(1)


def extract_best_mask_soft(pred_masks, pred_logits):
    """Extract best mask from raw SAM3 output (differentiable w.r.t. mask)."""
    B = pred_masks.shape[0]
    masks = []
    for i in range(B):
        if pred_logits is not None:
            scores = pred_logits[i].sigmoid()  # [num_queries]
        else:
            scores = pred_masks[i].flatten(1).abs().sum(dim=-1)
        best_idx = scores.argmax()
        masks.append(pred_masks[i, best_idx:best_idx+1])
    return torch.stack(masks)


# ── Zero-shot evaluation ──
def evaluate_zero_shot(x_test, y_test, text_prompt, use_boxes=True, tag="zeroshot"):
    """Evaluate SAM3 zero-shot on test data."""
    device = torch.device("cuda:0")
    processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    sam3 = Sam3Model.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    sam3.to(device).eval()

    results = {k: [] for k in ["dice", "iou", "acc", "precision", "sensitivity", "specificity"]}

    print(f"\n=== Zero-shot SAM3: text='{text_prompt}', boxes={use_boxes} ===", flush=True)
    for i in tqdm(range(len(x_test)), desc=f"Testing ({tag})", mininterval=30):
        img = x_test[i]
        gt = y_test[i]

        inputs, gt_mask = prepare_sam3_batch(
            [img], [gt], processor, device,
            text_prompt=text_prompt, use_boxes=use_boxes
        )

        with torch.no_grad():
            with autocast("cuda"):
                outputs = sam3(**inputs)

            pred_mask_soft = extract_best_mask_soft(
                outputs.pred_masks, outputs.pred_logits
            )
            H, W = gt.shape
            pred_resized = torch.nn.functional.interpolate(
                pred_mask_soft.float(), size=(H, W), mode="bilinear", align_corners=False
            )
            pred_bin = (pred_resized.sigmoid() > 0.5).float()

        m = all_metrics(pred_bin.squeeze(), gt_mask.squeeze())
        for k in results:
            results[k].append(m[k])

    print(f"\n=== {tag} Results ===")
    for k, v in results.items():
        print(f"  {k}: {np.mean(v):.4f} +/- {np.std(v):.4f}")
    sys.stdout.flush()

    return {k: (np.mean(v), np.std(v)) for k, v in results.items()}


# ── Fine-tuning ──
def collate_sam3(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)


def train_worker(rank, world_size, port, cfg, x_tr, y_tr, x_va, y_va):
    """DDP training worker for SAM3 fine-tuning."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}")
    processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    sam3 = Sam3Model.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN).to(device)

    if cfg.get("pretrained_path") and os.path.exists(cfg["pretrained_path"]):
        state = torch.load(cfg["pretrained_path"], map_location=device)
        sam3.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
        if rank == 0:
            print(f"  Loaded pretrained from {cfg['pretrained_path']}")

    if cfg.get("freeze_encoder", False):
        frozen = 0
        for name, param in sam3.named_parameters():
            if "vision_encoder" in name:
                param.requires_grad = False
                frozen += 1
        trainable = sum(p.numel() for p in sam3.parameters() if p.requires_grad)
        total = sum(p.numel() for p in sam3.parameters())
        if rank == 0:
            print(f"  Frozen {frozen} vision encoder params, trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M", flush=True)

    sam3 = DDP(sam3, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    loss_fn = DiceLoss(to_onehot_y=False, sigmoid=True)
    trainable_params = [p for p in sam3.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=cfg.get("lr", 1e-5))
    scaler = GradScaler("cuda")

    augment = cfg.get("augment", False)
    strong_augment = cfg.get("strong_augment", False)
    text_prompt = cfg.get("text_prompt", "visual")
    train_ds = Sam3Dataset(x_tr, y_tr, augment=augment, strong_augment=strong_augment)
    val_ds = Sam3Dataset(x_va, y_va, augment=False)

    best, no_imp = float("inf"), 0
    bs = cfg.get("batch_size", 2)
    pat = cfg.get("patience", 15)
    save_path = cfg["model_save_path"]
    csv_path = save_path.replace(".pth", "_metrics.csv")

    if rank == 0:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["Epoch", "TrLoss", "VaLoss", "TrDice", "VaDice"])

    for epoch in range(cfg.get("epochs", 250)):
        ts = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        ts.set_epoch(epoch)
        tl = DataLoader(train_ds, batch_size=bs, sampler=ts, collate_fn=collate_sam3)
        sam3.train()
        ep = {"loss": [], "dice": []}

        for images_batch, labels_batch in tl:
            inputs, gt_masks = prepare_sam3_batch(
                images_batch, labels_batch, processor, device,
                text_prompt=text_prompt, use_boxes=cfg.get("use_boxes", True)
            )
            optimizer.zero_grad()
            with autocast("cuda"):
                outputs = sam3(**inputs)
                pred_mask = extract_best_mask_soft(outputs.pred_masks, outputs.pred_logits)
                gt_resized = torch.nn.functional.interpolate(
                    gt_masks, size=pred_mask.shape[-2:], mode="nearest"
                )
                loss = loss_fn(pred_mask, gt_resized)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pred_bin = (pred_mask.sigmoid() > 0.5).float()
            ep["loss"].append(loss.item())
            ep["dice"].append(compute_dice(pred_bin, gt_resized).item())

        dist.barrier()

        # Validation
        vs = DistributedSampler(val_ds, num_replicas=world_size, rank=rank)
        vl = DataLoader(val_ds, batch_size=bs, sampler=vs, collate_fn=collate_sam3)
        sam3.eval()
        ev = {"loss": [], "dice": []}

        with torch.no_grad():
            for images_batch, labels_batch in vl:
                inputs, gt_masks = prepare_sam3_batch(
                    images_batch, labels_batch, processor, device,
                    text_prompt=text_prompt, use_boxes=cfg.get("use_boxes", True)
                )
                with autocast("cuda"):
                    outputs = sam3(**inputs)
                    pred_mask = extract_best_mask_soft(outputs.pred_masks, outputs.pred_logits)
                    gt_resized = torch.nn.functional.interpolate(
                        gt_masks, size=pred_mask.shape[-2:], mode="nearest"
                    )
                    loss = loss_fn(pred_mask, gt_resized)

                pred_bin = (pred_mask.sigmoid() > 0.5).float()
                ev["loss"].append(loss.item())
                ev["dice"].append(compute_dice(pred_bin, gt_resized).item())

        at = {k: np.mean(v) for k, v in ep.items()}
        av = {k: np.mean(v) for k, v in ev.items()}

        improved = torch.tensor(1 if av["loss"] < best else 0, device=device)
        dist.broadcast(improved, src=0)

        if rank == 0:
            print(f"  E{epoch+1}: TrDice={at['dice']:.4f} VaDice={av['dice']:.4f} VaLoss={av['loss']:.4f}", flush=True)
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch + 1, at["loss"], av["loss"], at["dice"], av["dice"]])

        if improved.item():
            best = av["loss"]
            no_imp = 0
            if rank == 0:
                torch.save(sam3.state_dict(), save_path)
        else:
            no_imp += 1
            if no_imp >= pat:
                if rank == 0:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

        dist.barrier()

    dist.destroy_process_group()


def evaluate_finetuned(model_path, x_test, y_test, tag="finetuned"):
    """Evaluate fine-tuned SAM3 on test data."""
    device = torch.device("cuda:0")
    processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    sam3 = Sam3Model.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)

    assert os.path.exists(model_path), f"No checkpoint at {model_path}"
    state = torch.load(model_path, map_location=device)
    sam3.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
    sam3.to(device).eval()

    results = {k: [] for k in ["dice", "iou", "acc", "precision", "sensitivity", "specificity"]}

    print(f"\n=== Fine-tuned SAM3: {tag} ===")
    for i in tqdm(range(len(x_test)), desc=f"Testing ({tag})"):
        img = x_test[i]
        gt = y_test[i]

        inputs, gt_mask = prepare_sam3_batch(
            [img], [gt], processor, device,
            text_prompt="visual", use_boxes=True
        )

        with torch.no_grad():
            with autocast("cuda"):
                outputs = sam3(**inputs)

            pred_mask_soft = extract_best_mask_soft(
                outputs.pred_masks, outputs.pred_logits
            )
            H, W = gt.shape
            pred_resized = torch.nn.functional.interpolate(
                pred_mask_soft.float(), size=(H, W), mode="bilinear", align_corners=False
            )
            pred_bin = (pred_resized.sigmoid() > 0.5).float()

        m = all_metrics(pred_bin.squeeze(), gt_mask.squeeze())
        for k in results:
            results[k].append(m[k])

    print(f"\n=== {tag} Results ===")
    for k, v in results.items():
        print(f"  {k}: {np.mean(v):.4f} +/- {np.std(v):.4f}")
    sys.stdout.flush()

    return {k: (np.mean(v), np.std(v)) for k, v in results.items()}


def train_worker_v3(rank, world_size, port, cfg, x_tr, y_tr, x_va, y_va):
    """V3 training: partial encoder freeze + discriminative LR + cosine schedule + dice+focal loss."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}")
    processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    sam3 = Sam3Model.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN).to(device)

    if cfg.get("pretrained_path") and os.path.exists(cfg["pretrained_path"]):
        state = torch.load(cfg["pretrained_path"], map_location=device)
        sam3.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
        if rank == 0:
            print(f"  Loaded pretrained from {cfg['pretrained_path']}")

    freeze_blocks = cfg.get("freeze_blocks", 20)
    encoder_lr = cfg.get("encoder_lr", 1e-5)
    decoder_lr = cfg.get("decoder_lr", 1e-4)

    frozen_count = 0
    encoder_params = []
    decoder_params = []
    for name, param in sam3.named_parameters():
        if "vision_encoder" in name:
            block_idx = None
            if "backbone.layers." in name:
                parts = name.split("backbone.layers.")
                if len(parts) > 1:
                    try:
                        block_idx = int(parts[1].split(".")[0])
                    except ValueError:
                        pass
            if block_idx is not None and block_idx < freeze_blocks:
                param.requires_grad = False
                frozen_count += 1
            elif "patch_embed" in name or "position_embed" in name:
                param.requires_grad = False
                frozen_count += 1
            else:
                encoder_params.append(param)
        else:
            decoder_params.append(param)

    trainable = sum(p.numel() for p in sam3.parameters() if p.requires_grad)
    total = sum(p.numel() for p in sam3.parameters())
    if rank == 0:
        print(f"  Partial freeze: {frozen_count} params frozen (blocks 0-{freeze_blocks-1} + patch/pos embed)")
        print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M")
        print(f"  LR: encoder={encoder_lr}, decoder={decoder_lr}")

    sam3 = DDP(sam3, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    dice_loss_fn = DiceLoss(to_onehot_y=False, sigmoid=True)
    focal_loss_fn = FocalLoss(to_onehot_y=False, use_softmax=False, gamma=2.0)
    dice_weight = cfg.get("dice_weight", 0.7)
    focal_weight = cfg.get("focal_weight", 0.3)

    optimizer = AdamW([
        {"params": encoder_params, "lr": encoder_lr, "weight_decay": 1e-2},
        {"params": decoder_params, "lr": decoder_lr, "weight_decay": 1e-4},
    ])

    warmup_epochs = cfg.get("warmup_epochs", 5)
    T_0 = cfg.get("cosine_T0", 30)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=2)
    scaler = GradScaler("cuda")

    strong_augment = cfg.get("strong_augment", True)
    text_prompt = cfg.get("text_prompt", "visual")
    train_ds = Sam3Dataset(x_tr, y_tr, augment=False, strong_augment=strong_augment)
    val_ds = Sam3Dataset(x_va, y_va, augment=False)

    best, no_imp = float("inf"), 0
    bs = cfg.get("batch_size", 2)
    pat = cfg.get("patience", 25)
    save_path = cfg["model_save_path"]
    csv_path = save_path.replace(".pth", "_metrics.csv")

    if rank == 0:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["Epoch", "TrLoss", "VaLoss", "TrDice", "VaDice", "LR_enc", "LR_dec"])

    for epoch in range(cfg.get("epochs", 250)):
        # Linear warmup
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                if pg is optimizer.param_groups[0]:
                    pg["lr"] = encoder_lr * warmup_factor
                else:
                    pg["lr"] = decoder_lr * warmup_factor
        else:
            scheduler.step(epoch - warmup_epochs)

        ts = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        ts.set_epoch(epoch)
        tl = DataLoader(train_ds, batch_size=bs, sampler=ts, collate_fn=collate_sam3)
        sam3.train()
        ep = {"loss": [], "dice": []}

        for images_batch, labels_batch in tl:
            inputs, gt_masks = prepare_sam3_batch(
                images_batch, labels_batch, processor, device,
                text_prompt=text_prompt, use_boxes=cfg.get("use_boxes", True)
            )
            optimizer.zero_grad()
            with autocast("cuda"):
                outputs = sam3(**inputs)
                pred_mask = extract_best_mask_soft(outputs.pred_masks, outputs.pred_logits)
                gt_resized = torch.nn.functional.interpolate(
                    gt_masks, size=pred_mask.shape[-2:], mode="nearest"
                )
                d_loss = dice_loss_fn(pred_mask, gt_resized)
                f_loss = focal_loss_fn(pred_mask, gt_resized)
                loss = dice_weight * d_loss + focal_weight * f_loss

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in sam3.parameters() if p.requires_grad], max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()

            pred_bin = (pred_mask.sigmoid() > 0.5).float()
            ep["loss"].append(loss.item())
            ep["dice"].append(compute_dice(pred_bin, gt_resized).item())

        dist.barrier()

        vs = DistributedSampler(val_ds, num_replicas=world_size, rank=rank)
        vl = DataLoader(val_ds, batch_size=bs, sampler=vs, collate_fn=collate_sam3)
        sam3.eval()
        ev = {"loss": [], "dice": []}

        with torch.no_grad():
            for images_batch, labels_batch in vl:
                inputs, gt_masks = prepare_sam3_batch(
                    images_batch, labels_batch, processor, device,
                    text_prompt=text_prompt, use_boxes=cfg.get("use_boxes", True)
                )
                with autocast("cuda"):
                    outputs = sam3(**inputs)
                    pred_mask = extract_best_mask_soft(outputs.pred_masks, outputs.pred_logits)
                    gt_resized = torch.nn.functional.interpolate(
                        gt_masks, size=pred_mask.shape[-2:], mode="nearest"
                    )
                    d_loss = dice_loss_fn(pred_mask, gt_resized)
                    f_loss = focal_loss_fn(pred_mask, gt_resized)
                    loss = dice_weight * d_loss + focal_weight * f_loss

                pred_bin = (pred_mask.sigmoid() > 0.5).float()
                ev["loss"].append(loss.item())
                ev["dice"].append(compute_dice(pred_bin, gt_resized).item())

        at = {k: np.mean(v) for k, v in ep.items()}
        av = {k: np.mean(v) for k, v in ev.items()}

        improved = torch.tensor(1 if av["loss"] < best else 0, device=device)
        dist.broadcast(improved, src=0)

        if rank == 0:
            lr_enc = optimizer.param_groups[0]["lr"]
            lr_dec = optimizer.param_groups[1]["lr"]
            print(f"  E{epoch+1}: TrDice={at['dice']:.4f} VaDice={av['dice']:.4f} VaLoss={av['loss']:.4f} lr_e={lr_enc:.2e} lr_d={lr_dec:.2e}", flush=True)
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch + 1, at["loss"], av["loss"], at["dice"], av["dice"], lr_enc, lr_dec])

        if improved.item():
            best = av["loss"]
            no_imp = 0
            if rank == 0:
                torch.save(sam3.state_dict(), save_path)
        else:
            no_imp += 1
            if no_imp >= pat:
                if rank == 0:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

        dist.barrier()

    dist.destroy_process_group()


# ── Main ──
def run():
    case_ids = get_case_ids()
    train_cases, test_cases = train_test_split(case_ids, test_size=0.2, random_state=42)
    train_cases, val_cases = train_test_split(train_cases, test_size=0.125, random_state=42)
    print(f"Cases: train={len(train_cases)}, val={len(val_cases)}, test={len(test_cases)}", flush=True)

    all_results = {}

    for label_value, label_name in [(1, "organ"), (2, "tumor")]:
        print(f"\n{'='*70}")
        print(f"Target: {label_name} (label={label_value})")
        print(f"{'='*70}")

        print(f"Extracting {label_name} slices...")
        x_train, y_train, _ = extract_slices_from_volumes(train_cases, label_value)
        x_val, y_val, _ = extract_slices_from_volumes(val_cases, label_value)
        x_test, y_test, _ = extract_slices_from_volumes(test_cases, label_value)
        print(f"Slices: train={len(x_train)}, val={len(x_val)}, test={len(x_test)}", flush=True)

        # ── Phase 1: Zero-shot with text ──
        text = "pancreas" if label_name == "organ" else "pancreatic tumor"
        zs_text = evaluate_zero_shot(
            x_test, y_test, text_prompt=text, use_boxes=False,
            tag=f"{label_name}_zeroshot_text"
        )
        all_results[f"{label_name}_zeroshot_text"] = zs_text

        # ── Phase 2: Zero-shot with GT box ──
        zs_box = evaluate_zero_shot(
            x_test, y_test, text_prompt="visual", use_boxes=True,
            tag=f"{label_name}_zeroshot_box"
        )
        all_results[f"{label_name}_zeroshot_box"] = zs_box

        # ── Phase 3: Fine-tune with box prompts ──
        model_path = os.path.join(SAM3_DIR, f"sam3_{label_name}.pth")
        if not os.path.exists(model_path):
            print(f"\n--- Fine-tuning SAM3 for {label_name} ---")
            cfg = {
                "model_save_path": model_path,
                "epochs": 250,
                "batch_size": 2,
                "lr": 1e-5,
                "patience": 15,
                "augment": True,
                "text_prompt": "visual",
            }
            if label_name == "tumor":
                organ_path = os.path.join(SAM3_DIR, "sam3_organ.pth")
                if os.path.exists(organ_path):
                    cfg["pretrained_path"] = organ_path
                    print(f"  Transfer from: {organ_path}")

            port = find_free_port()
            mp.spawn(
                train_worker,
                args=(WORLD_SIZE, port, cfg, x_train, y_train, x_val, y_val),
                nprocs=WORLD_SIZE, join=True,
            )

        ft = evaluate_finetuned(
            model_path, x_test, y_test, tag=f"{label_name}_finetuned"
        )
        all_results[f"{label_name}_finetuned"] = ft

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SAM3 vs SAM1 COMPARISON")
    print("=" * 70)
    print(f"{'Config':<30s} {'Organ Dice':>12s} {'Tumor Dice':>12s}")
    print("-" * 54)

    # SAM1 reference (from ablation evaluate_test)
    print(f"{'SAM1 box-only (reference)':<30s} {'0.828':>12s} {'0.914':>12s}")

    for key in ["zeroshot_text", "zeroshot_box", "finetuned"]:
        organ = all_results.get(f"organ_{key}", {}).get("dice", (0, 0))
        tumor = all_results.get(f"tumor_{key}", {}).get("dice", (0, 0))
        print(f"SAM3 {key:<25s} {organ[0]:>12.4f} {tumor[0]:>12.4f}")

    print("=" * 70)
    print("DONE")


def run_v2():
    """V2: frozen encoder + stronger augmentation."""
    case_ids = get_case_ids()
    train_cases, test_cases = train_test_split(case_ids, test_size=0.2, random_state=42)
    train_cases, val_cases = train_test_split(train_cases, test_size=0.125, random_state=42)
    print(f"Cases: train={len(train_cases)}, val={len(val_cases)}, test={len(test_cases)}", flush=True)

    all_results = {}

    for label_value, label_name in [(1, "organ"), (2, "tumor")]:
        print(f"\n{'='*70}")
        print(f"V2 Target: {label_name} (label={label_value})")
        print(f"{'='*70}", flush=True)

        print(f"Extracting {label_name} slices...")
        x_train, y_train, _ = extract_slices_from_volumes(train_cases, label_value)
        x_val, y_val, _ = extract_slices_from_volumes(val_cases, label_value)
        x_test, y_test, _ = extract_slices_from_volumes(test_cases, label_value)
        print(f"Slices: train={len(x_train)}, val={len(x_val)}, test={len(x_test)}", flush=True)

        model_path = os.path.join(SAM3_DIR, f"sam3_v2_{label_name}.pth")
        if not os.path.exists(model_path):
            print(f"\n--- V2 Fine-tuning SAM3 for {label_name} (frozen encoder + strong aug) ---", flush=True)
            cfg = {
                "model_save_path": model_path,
                "epochs": 250,
                "batch_size": 4,
                "lr": 1e-4,
                "patience": 20,
                "augment": False,
                "strong_augment": True,
                "freeze_encoder": True,
                "text_prompt": "visual",
            }
            if label_name == "tumor":
                organ_path = os.path.join(SAM3_DIR, "sam3_v2_organ.pth")
                if os.path.exists(organ_path):
                    cfg["pretrained_path"] = organ_path
                    print(f"  Transfer from: {organ_path}", flush=True)

            port = find_free_port()
            mp.spawn(
                train_worker,
                args=(WORLD_SIZE, port, cfg, x_train, y_train, x_val, y_val),
                nprocs=WORLD_SIZE, join=True,
            )

        ft = evaluate_finetuned(
            model_path, x_test, y_test, tag=f"{label_name}_v2_finetuned"
        )
        all_results[f"{label_name}_v2"] = ft

    print("\n" + "=" * 70)
    print("SAM3 V2 (FROZEN ENCODER + STRONG AUG) vs V1 vs SAM1")
    print("=" * 70)
    print(f"{'Config':<35s} {'Organ Dice':>12s} {'Tumor Dice':>12s}")
    print("-" * 59)
    print(f"{'SAM1 box+points (reference)':<35s} {'0.828':>12s} {'0.914':>12s}")
    print(f"{'SAM3 v1 full finetune':<35s} {'0.861':>12s} {'0.888':>12s}")
    organ = all_results.get("organ_v2", {}).get("dice", (0, 0))
    tumor = all_results.get("tumor_v2", {}).get("dice", (0, 0))
    print(f"{'SAM3 v2 frozen+strong_aug':<35s} {organ[0]:>12.4f} {tumor[0]:>12.4f}")
    print("=" * 70)
    print("DONE")


def run_v3():
    """V3: partial encoder freeze + discriminative LR + cosine schedule + dice+focal loss."""
    case_ids = get_case_ids()
    train_cases, test_cases = train_test_split(case_ids, test_size=0.2, random_state=42)
    train_cases, val_cases = train_test_split(train_cases, test_size=0.125, random_state=42)
    print(f"Cases: train={len(train_cases)}, val={len(val_cases)}, test={len(test_cases)}", flush=True)

    all_results = {}

    for label_value, label_name in [(1, "organ"), (2, "tumor")]:
        print(f"\n{'='*70}")
        print(f"V3 Target: {label_name} (label={label_value})")
        print(f"{'='*70}", flush=True)

        print(f"Extracting {label_name} slices...")
        x_train, y_train, _ = extract_slices_from_volumes(train_cases, label_value)
        x_val, y_val, _ = extract_slices_from_volumes(val_cases, label_value)
        x_test, y_test, _ = extract_slices_from_volumes(test_cases, label_value)
        print(f"Slices: train={len(x_train)}, val={len(x_val)}, test={len(x_test)}", flush=True)

        model_path = os.path.join(SAM3_DIR, f"sam3_v3_{label_name}.pth")
        if not os.path.exists(model_path):
            print(f"\n--- V3 Fine-tuning SAM3 for {label_name} ---", flush=True)
            cfg = {
                "model_save_path": model_path,
                "epochs": 250,
                "batch_size": 2,
                "patience": 25,
                "strong_augment": True,
                "text_prompt": "visual",
                "freeze_blocks": 20,
                "encoder_lr": 1e-5,
                "decoder_lr": 1e-4,
                "warmup_epochs": 5,
                "cosine_T0": 30,
                "dice_weight": 0.7,
                "focal_weight": 0.3,
            }
            if label_name == "tumor":
                organ_path = os.path.join(SAM3_DIR, "sam3_v3_organ.pth")
                if os.path.exists(organ_path):
                    cfg["pretrained_path"] = organ_path
                    print(f"  Transfer from: {organ_path}", flush=True)

            port = find_free_port()
            mp.spawn(
                train_worker_v3,
                args=(WORLD_SIZE, port, cfg, x_train, y_train, x_val, y_val),
                nprocs=WORLD_SIZE, join=True,
            )

        ft = evaluate_finetuned(
            model_path, x_test, y_test, tag=f"{label_name}_v3_finetuned"
        )
        all_results[f"{label_name}_v3"] = ft

    print("\n" + "=" * 70)
    print("SAM3 V3 (PARTIAL FREEZE + DISC LR + COSINE + DICE+FOCAL) vs V1 vs SAM1")
    print("=" * 70)
    print(f"{'Config':<45s} {'Organ Dice':>12s} {'Tumor Dice':>12s}")
    print("-" * 69)
    print(f"{'SAM1 box+points (reference)':<45s} {'0.828':>12s} {'0.914':>12s}")
    print(f"{'SAM3 v1 full finetune':<45s} {'0.861':>12s} {'0.888':>12s}")
    print(f"{'SAM3 v2 frozen+strong_aug':<45s} {'0.826':>12s} {'0.853':>12s}")
    organ = all_results.get("organ_v3", {}).get("dice", (0, 0))
    tumor = all_results.get("tumor_v3", {}).get("dice", (0, 0))
    print(f"{'SAM3 v3 partial_freeze+disc_lr+cosine+focal':<45s} {organ[0]:>12.4f} {tumor[0]:>12.4f}")
    print("=" * 70)
    print("DONE")


import nibabel as nib
from skimage.transform import resize as sk_resize

SAM3_DELIVERY_DIR = "/scratch/ud3d4/acm_data/Pancreas/sam3_delivery"


def _sam3_infer_slice(model, processor, rgb, mask256, device, text="visual"):
    """Run SAM3 on one 256x256 RGB slice with a GT-derived box prompt.
    Returns a 256x256 float probability map, or None if the mask is empty."""
    box = bbox_from_mask(mask256, pad=3)  # [x1,y1,x2,y2] in 256-px space
    if box is None:
        return None
    inputs = processor(
        images=[rgb], text=[text],
        input_boxes=[[box]], input_boxes_labels=[[1]],
        return_tensors="pt",
    )
    pv = inputs["pixel_values"].to(device)
    kw = {"pixel_values": pv}
    for k in ("input_ids", "attention_mask", "input_boxes", "input_boxes_labels"):
        if inputs.get(k) is not None:
            kw[k] = inputs[k].to(device)
    with torch.no_grad():
        with autocast("cuda"):
            outputs = model(**kw)
        pred = extract_best_mask_soft(outputs.pred_masks, outputs.pred_logits)  # [1,1,h,w]
        pred = torch.nn.functional.interpolate(
            pred.float(), size=(256, 256), mode="bilinear", align_corners=False
        )
        prob = pred.sigmoid().squeeze().cpu().numpy()
    return prob


def _load_sam3_ckpt(ckpt_path, device):
    model = Sam3Model.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
    model.to(device).eval()
    return model


def deliver_worker(rank, world_size):
    """Generate per-case NIfTI deliverables for a shard of cases on GPU `rank`."""
    device = torch.device(f"cuda:{rank}")
    processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)

    organ_model = _load_sam3_ckpt(os.path.join(SAM3_DIR, "sam3_v3_organ.pth"), device)
    tumor_model = _load_sam3_ckpt(os.path.join(SAM3_DIR, "sam3_v3_tumor.pth"), device)
    if rank == 0:
        print(f"  [GPU{rank}] loaded organ + tumor SAM3 models", flush=True)

    all_cases = get_case_ids()
    shard = all_cases[rank::world_size]
    manifest = []

    for ci, case_id in enumerate(shard):
        case_dir = os.path.join(SAM3_DELIVERY_DIR, case_id)
        os.makedirs(case_dir, exist_ok=True)

        img_nii = nib.load(f"{PANCREAS_DIR}/imagesTr/{case_id}.nii.gz")
        lbl_nii = nib.load(f"{PANCREAS_DIR}/labelsTr/{case_id}.nii.gz")
        ct_data = img_nii.get_fdata()
        gt_data = lbl_nii.get_fdata().astype(np.uint8)

        ct_dest = os.path.join(case_dir, "ct.nii.gz")
        if not os.path.exists(ct_dest):
            os.symlink(f"{PANCREAS_DIR}/imagesTr/{case_id}.nii.gz", ct_dest)
        gt_dest = os.path.join(case_dir, "gt.nii.gz")
        if not os.path.exists(gt_dest):
            nib.save(nib.Nifti1Image(gt_data, img_nii.affine, img_nii.header), gt_dest)

        pred_combined = np.zeros(ct_data.shape, dtype=np.uint8)

        for z in range(ct_data.shape[2]):
            gt_slice = gt_data[:, :, z]
            if gt_slice.max() == 0:
                continue
            orig_h, orig_w = gt_slice.shape
            ct_resized = sk_resize(ct_data[:, :, z], (256, 256), preserve_range=True, anti_aliasing=True)
            rgb = hu_to_rgb(ct_resized).astype(np.uint8)

            organ_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            organ256 = sk_resize((gt_slice == 1).astype(np.uint8), (256, 256), order=0, preserve_range=True).astype(np.uint8)
            if organ256.sum() > 0:
                prob = _sam3_infer_slice(organ_model, processor, rgb, organ256, device)
                if prob is not None:
                    organ_mask = (sk_resize(prob, (orig_h, orig_w), order=1, preserve_range=True) > 0.5).astype(np.uint8)

            tumor_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            tumor256 = sk_resize((gt_slice == 2).astype(np.uint8), (256, 256), order=0, preserve_range=True).astype(np.uint8)
            if tumor256.sum() > 0:
                prob = _sam3_infer_slice(tumor_model, processor, rgb, tumor256, device)
                if prob is not None:
                    tumor_mask = (sk_resize(prob, (orig_h, orig_w), order=1, preserve_range=True) > 0.5).astype(np.uint8)

            combined = organ_mask.copy()
            combined[tumor_mask > 0] = 2
            pred_combined[:, :, z] = combined

        nib.save(nib.Nifti1Image(pred_combined, img_nii.affine, img_nii.header),
                 os.path.join(case_dir, "pred.nii.gz"))

        spacing = img_nii.header.get_zooms()
        voxel_vol_cm3 = float(np.prod(spacing)) / 1000
        organ_dice = compute_dice(
            torch.from_numpy((pred_combined == 1).astype(float)),
            torch.from_numpy((gt_data == 1).astype(float))).item()
        tumor_dice = (compute_dice(
            torch.from_numpy((pred_combined == 2).astype(float)),
            torch.from_numpy((gt_data == 2).astype(float))).item()
            if 2 in gt_data else float("nan"))

        manifest.append({
            "case_id": case_id, "dataset": "pancreas",
            "shape": f"{ct_data.shape[0]},{ct_data.shape[1]},{ct_data.shape[2]}",
            "spacing_mm": f"{spacing[0]:.4f},{spacing[1]:.4f},{spacing[2]:.4f}",
            "organ_dice": f"{organ_dice:.4f}",
            "tumor_dice": f"{tumor_dice:.4f}" if not np.isnan(tumor_dice) else "N/A",
            "gt_organ_cm3": f"{np.sum(gt_data == 1) * voxel_vol_cm3:.2f}",
            "gt_tumor_cm3": f"{np.sum(gt_data == 2) * voxel_vol_cm3:.2f}",
            "pred_organ_cm3": f"{np.sum(pred_combined == 1) * voxel_vol_cm3:.2f}",
            "pred_tumor_cm3": f"{np.sum(pred_combined == 2) * voxel_vol_cm3:.2f}",
        })
        if rank == 0:
            print(f"  [GPU{rank}] {ci+1}/{len(shard)} {case_id} organ={organ_dice:.3f} tumor={manifest[-1]['tumor_dice']}", flush=True)

    part_path = os.path.join(SAM3_DELIVERY_DIR, f"manifest_rank{rank}.csv")
    with open(part_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    print(f"  [GPU{rank}] done: {len(manifest)} cases -> {part_path}", flush=True)


def generate_deliverables_v3():
    """Multi-GPU: split 281 cases across all GPUs, generate SAM3 NIfTI deliverables."""
    os.makedirs(SAM3_DELIVERY_DIR, exist_ok=True)
    for name in ("sam3_v3_organ.pth", "sam3_v3_tumor.pth"):
        p = os.path.join(SAM3_DIR, name)
        assert os.path.exists(p), f"Missing checkpoint: {p}"
    print(f"Generating SAM3 deliverables across {WORLD_SIZE} GPUs -> {SAM3_DELIVERY_DIR}", flush=True)

    mp.spawn(deliver_worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)

    # Merge per-rank manifests
    rows = []
    for rank in range(WORLD_SIZE):
        part = os.path.join(SAM3_DELIVERY_DIR, f"manifest_rank{rank}.csv")
        if os.path.exists(part):
            rows.extend(list(csv.DictReader(open(part))))
    rows.sort(key=lambda r: r["case_id"])
    manifest_path = os.path.join(SAM3_DELIVERY_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    organ_dices = [float(r["organ_dice"]) for r in rows]
    tumor_dices = [float(r["tumor_dice"]) for r in rows if r["tumor_dice"] != "N/A"]
    print("\n" + "=" * 60)
    print(f"SAM3 DELIVERABLES: {len(rows)} cases -> {SAM3_DELIVERY_DIR}")
    print(f"Manifest: {manifest_path}")
    print(f"Organ Dice (all cases): {np.mean(organ_dices):.4f} +/- {np.std(organ_dices):.4f}")
    if tumor_dices:
        print(f"Tumor Dice (all cases): {np.mean(tumor_dices):.4f} +/- {np.std(tumor_dices):.4f}")
    print("=" * 60)
    print("DONE")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", action="store_true", help="Run v2: frozen encoder + strong augmentation")
    parser.add_argument("--v3", action="store_true", help="Run v3: partial freeze + discriminative LR + cosine + focal")
    parser.add_argument("--deliver", action="store_true", help="Generate NIfTI deliverables (multi-GPU) from v3 checkpoints")
    args = parser.parse_args()
    if args.deliver:
        generate_deliverables_v3()
    elif args.v3:
        run_v3()
    elif args.v2:
        run_v2()
    else:
        run()
