"""
SEMIR-JEPA: 3D graph-minor pooling + DINO features + IG-JEPA objective

Bridges SWOG's 3D fastloops crate with IG-JEPA's self-supervised framework.

Pipeline:
  1. Build 3D graph minors via fastloops (SWOG Rust crate)
  2. Extract DINO ViT features per CT slice, map to 3D supernodes via centroid
  3. IG-JEPA pre-training (topology-aware masking + VICReg)
  4. MLP probe for tumor classification
  5. Lift to voxels + Dice evaluation

Side project while awaiting SEMIR author's LiTS code.
"""

import os, sys, copy, time, json, re, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import fastloops

DATA_ROOT = "/scratch/ud3d4/acm_data/Data"
RESULTS_DIR = "/home/ud3d4/Desktop/SWOG/results/semir_jepa"
os.makedirs(RESULTS_DIR, exist_ok=True)

HU_MIN, HU_MAX = -50, 250
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


def pr(msg=""):
    print(msg, flush=True)


def section(title):
    pr(f"\n{'='*70}")
    pr(f"  {title}")
    pr(f"{'='*70}")


# ============================================================
# 1. Data loading + 3D graph construction
# ============================================================

def discover_volumes():
    ct_dir = os.path.join(DATA_ROOT, "ct")
    vids = []
    for f in sorted(os.listdir(ct_dir)):
        m = re.match(r"volume-(\d+)\.npy", f)
        if m:
            vid = int(m.group(1))
            seg_path = os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")
            if os.path.exists(seg_path):
                seg = np.load(seg_path)
                if (seg == 2).sum() > 0:
                    vids.append(vid)
    return sorted(vids)


def load_volume(vid):
    ct = np.load(os.path.join(DATA_ROOT, "ct", f"volume-{vid}.npy")).astype(np.float32)
    seg = np.load(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")).astype(np.int32)
    return ct, seg


def build_3d_graph(ct, seg, psi=3, alpha=15):
    """Build 3D graph minor using SWOG fastloops crate."""
    ct_u8 = np.clip(ct, HU_MIN, HU_MAX)
    ct_u8 = ((ct_u8 - HU_MIN) / (HU_MAX - HU_MIN) * 255).round().astype(np.uint8)
    ct_u8 = np.ascontiguousarray(ct_u8[..., np.newaxis])
    n_vox = ct.size

    raw_nf, raw_ei, raw_ef, raw_labels, raw_adj = fastloops.merge_and_cut(
        ct_u8, merge_distance=psi, cut_distance=alpha,
        delete_small_node_max_size=0,
        delete_large_node_min_size=n_vox + 1,
        delete_value_min=0, delete_value_max=255,
        connectivity="faces",
    )
    return np.asarray(raw_nf), np.asarray(raw_ei), np.asarray(raw_ef), np.asarray(raw_labels), ct_u8


# ============================================================
# 2. DINO feature extraction
# ============================================================

def load_dino_model(device):
    """Load pretrained DINO ViT-S/16."""
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_dino_features_for_volume(dino, ct_u8_volume, labels_np, device, batch_size=32):
    """
    Extract DINO features for each 3D supernode.

    For each supernode, find its centroid slice (z), extract DINO patch features
    from that slice, and map the (x,y) centroid to the 14x14 patch grid.

    Returns: (N_supernodes, 384) float32 array
    """
    flat = labels_np.ravel()
    valid = flat >= 0
    if not valid.any():
        return np.zeros((0, 384), dtype=np.float32)

    max_id = int(flat[valid].max())
    n_sn = max_id + 1
    D, H, W = labels_np.shape

    # Compute supernode centroids (z, y, x) via bincount
    coords_z = np.zeros(flat.shape, dtype=np.float64)
    coords_y = np.zeros(flat.shape, dtype=np.float64)
    coords_x = np.zeros(flat.shape, dtype=np.float64)
    for z in range(D):
        for y in range(H):
            start = z * H * W + y * W
            coords_z[start:start + W] = z
            coords_y[start:start + W] = y
            coords_x[start:start + W] = np.arange(W)

    counts = np.bincount(flat[valid], minlength=n_sn).astype(np.float64).clip(min=1)
    sum_z = np.bincount(flat[valid], weights=coords_z[valid], minlength=n_sn)
    sum_y = np.bincount(flat[valid], weights=coords_y[valid], minlength=n_sn)
    sum_x = np.bincount(flat[valid], weights=coords_x[valid], minlength=n_sn)
    centroid_z = (sum_z / counts).astype(np.int32).clip(0, D - 1)
    centroid_y = (sum_y / counts / H).clip(0, 1)  # normalized [0,1]
    centroid_x = (sum_x / counts / W).clip(0, 1)

    # Find unique slices needed
    unique_slices = np.unique(centroid_z)

    # Extract DINO features per slice (batched)
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    slice_features = {}  # z -> (14, 14, 384) patch features
    for batch_start in range(0, len(unique_slices), batch_size):
        batch_z = unique_slices[batch_start:batch_start + batch_size]
        imgs = []
        for z in batch_z:
            # Convert CT slice to 3-channel image, resize to 224x224
            ct_slice = ct_u8_volume[z, :, :, 0]  # (H, W) uint8
            img_rgb = np.stack([ct_slice, ct_slice, ct_slice], axis=-1)  # (H, W, 3)
            # Resize to 224x224
            from PIL import Image
            pil_img = Image.fromarray(img_rgb).resize((224, 224), Image.BILINEAR)
            imgs.append(transform(pil_img))

        batch_tensor = torch.stack(imgs).to(device)
        tokens = dino.get_intermediate_layers(batch_tensor, n=1)[0]
        # tokens shape: (B, 197, 384) — 196 patches + 1 CLS
        patch_tokens = tokens[:, 1:, :].reshape(-1, 14, 14, 384).cpu().numpy()

        for i, z in enumerate(batch_z):
            slice_features[z] = patch_tokens[i]

    # Map each supernode to its DINO feature via centroid
    dino_feats = np.zeros((n_sn, 384), dtype=np.float32)
    for sid in range(n_sn):
        z = centroid_z[sid]
        if z in slice_features:
            py = int(np.clip(centroid_y[sid] * 14, 0, 13))
            px = int(np.clip(centroid_x[sid] * 14, 0, 13))
            dino_feats[sid] = slice_features[z][py, px]

    return dino_feats


# ============================================================
# 3. Build enriched PyG graph
# ============================================================

def compute_geometric_features(raw_nf, labels_np, ct_u8, C=1):
    """Extract 9 geometric features (same as SEMIR v7)."""
    f = raw_nf.astype(np.float64)
    N = f.shape[0]
    area = 0; s = [1, 2, 3]; cov_idx = [(4,0,0),(5,1,1),(6,2,2),(7,0,1),(8,0,2),(9,1,2)]
    chan0 = 10; boundary = 10 + C + 6

    V = f[:, area]; Vsafe = np.maximum(V, 1.0)
    mean_coord = np.stack([f[:, c] for c in s], axis=1) / Vsafe[:, None]
    cov = np.zeros((N, 3, 3))
    for col, i, j in cov_idx:
        cij = f[:, col] / Vsafe - mean_coord[:, i] * mean_coord[:, j]
        cov[:, i, j] = cij; cov[:, j, i] = cij
    w = np.linalg.eigvalsh(cov); w = np.clip(w, 0, None)
    _, vec = np.linalg.eigh(cov); principal = vec[..., -1]
    degenerate = w.sum(axis=1) < 1e-6
    denom = w[:, 2] + 1e-6
    shape = np.stack([(w[:,2]-w[:,1])/denom, (w[:,1]-w[:,0])/denom, w[:,0]/denom], axis=1)
    shape[degenerate] = 0
    chan = f[:, chan0:chan0+C] / Vsafe[:, None] / 255.0
    compactness = f[:, boundary] / np.power(Vsafe, 2.0/3.0)
    elongation = np.where(w[:,0] > 1e-6, w[:,2]/(w[:,0]+1e-6), 1.0)
    elongation = np.clip(elongation, 1.0, 100.0)

    # Sign-canonicalize principal axis
    if len(principal):
        mc = np.argmax(np.abs(principal), axis=1)
        signs = np.sign(principal[np.arange(N), mc]); signs[signs==0] = 1
        principal = principal * signs[:, None]

    # Intensity std
    flat = labels_np.ravel(); valid = flat >= 0
    max_id = int(flat[valid].max()) if valid.any() else -1
    int_std = np.zeros(N, dtype=np.float32)
    if max_id >= 0:
        vals = ct_u8[..., 0].ravel().astype(np.float64) / 255.0
        counts = np.bincount(flat[valid], minlength=max_id+1).astype(np.float64).clip(min=1)
        sums = np.bincount(flat[valid], weights=vals[valid], minlength=max_id+1)
        sq_sums = np.bincount(flat[valid], weights=vals[valid]**2, minlength=max_id+1)
        var = sq_sums / counts - (sums / counts)**2
        int_std[:min(N, len(var))] = np.sqrt(np.maximum(var[:N], 0)).astype(np.float32)

    x = np.column_stack([
        np.log1p(V), np.log1p(f[:, boundary]),
        compactness, elongation,
        principal[:, 0], principal[:, 1], principal[:, 2],
        chan[:, 0], int_std,
    ]).astype(np.float32)
    return x


def build_enriched_graph(raw_nf, raw_ei, raw_ef, labels_np, seg, ct_u8,
                         dino_feats, overlap_th=0.10):
    """Build PyG graph with geometric (9) + DINO (384) = 393-dim features."""
    n_sn = raw_nf.shape[0]
    geo_feats = compute_geometric_features(raw_nf, labels_np, ct_u8)

    # Reduce DINO 384 → 32 via PCA to fit in GPU memory
    # (2M nodes × 384 floats = 3GB per graph — OOMs on 49GB GPU)
    from sklearn.decomposition import PCA
    DINO_DIM = 32
    if dino_feats.shape[0] >= n_sn:
        dino_raw = dino_feats[:n_sn]
    else:
        dino_raw = np.zeros((n_sn, 384), dtype=np.float32)
        dino_raw[:dino_feats.shape[0]] = dino_feats
    # Fit PCA on this volume's DINO features
    pca = PCA(n_components=DINO_DIM, random_state=42)
    dino_part = pca.fit_transform(dino_raw).astype(np.float32)

    # Combine: 9 geometric + 32 DINO-PCA = 41
    x = np.concatenate([geo_feats, dino_part], axis=1).astype(np.float32)

    # Z-score geometric features (first 9), leave DINO features as-is (already normalized)
    for col in range(9):
        mu, sigma = float(x[:, col].mean()), float(x[:, col].std())
        if sigma > 1e-8:
            x[:, col] = (x[:, col] - mu) / sigma
        else:
            x[:, col] = 0.0

    # Edges: filter out cut edges (cut_count = raw_ef[:, 3] > 0) to keep graph manageable.
    # Luke's crate keeps cut edges with cut_frac feature, but 20M edges OOMs on GPU
    # with TransformerConv. Non-cut edges carry the actual graph structure.
    if raw_ei.shape[1] > 0:
        ef = raw_ef.astype(np.float64)
        boundary_count = ef[:, 0].clip(min=1)
        cut_frac = ef[:, 3] / boundary_count
        non_cut = cut_frac < 0.5  # keep edges where less than half the boundary is cut
        ei_filtered = raw_ei[:, non_cut]
        if ei_filtered.shape[1] > 0:
            ei_fwd = torch.tensor(ei_filtered, dtype=torch.long)
            ei_rev = torch.stack([ei_fwd[1], ei_fwd[0]])
            edge_index = torch.cat([ei_fwd, ei_rev], dim=1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Labels
    flat = labels_np.ravel(); valid = flat >= 0
    gt = (seg.ravel() == 2).astype(np.float64)
    max_id = int(flat[valid].max()) if valid.any() else -1
    y = np.zeros(n_sn, dtype=np.int64)
    if max_id >= 0:
        tc = np.bincount(flat[valid], weights=gt[valid], minlength=max_id+1)
        total_c = np.bincount(flat[valid], minlength=max_id+1)
        overlap = tc / np.maximum(total_c, 1)
        y[:min(n_sn, len(overlap))] = (overlap[:n_sn] >= overlap_th).astype(np.int64)

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
    )


# ============================================================
# 4. IG-JEPA model (from IG-JEPA paper)
# ============================================================

class GINEncoder(nn.Module):
    """GIN encoder — lighter than TransformerConv, handles large graphs."""
    def __init__(self, in_dim, hid, layers=3):
        super().__init__()
        from torch_geometric.nn import GINConv
        self.proj = nn.Linear(in_dim, hid)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, hid))
            self.convs.append(GINConv(mlp))
            self.norms.append(nn.LayerNorm(hid))
        self.drop = nn.Dropout(0.1)

    def forward(self, x, edge_index):
        x = self.proj(x)
        for conv, norm in zip(self.convs, self.norms):
            x = x + self.drop(F.gelu(norm(conv(x, edge_index))))
        return x


class IGJEPA(nn.Module):
    def __init__(self, in_dim, hid=256, layers=3, heads=4, mask_ratio=0.4, mom=0.996):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mom = mom
        self.hid = hid
        self.enc = GINEncoder(in_dim, hid, layers)
        self.tgt = copy.deepcopy(self.enc)
        for p in self.tgt.parameters():
            p.requires_grad = False
        self.pred = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, hid))
        self.lam_var = 25.0
        self.lam_cov = 1.0

    @torch.no_grad()
    def ema_update(self):
        for p, t in zip(self.enc.parameters(), self.tgt.parameters()):
            t.data.mul_(self.mom).add_(p.data, alpha=1 - self.mom)

    def subgraph_mask(self, ei, N):
        """Fast random masking (no BFS — BFS is O(E) in Python, unusable at 7M edges)."""
        nm = max(1, int(N * self.mask_ratio))
        perm = torch.randperm(N, device=ei.device)
        mi = perm[:nm].sort().values
        mask_set = torch.zeros(N, dtype=torch.bool, device=ei.device)
        mask_set[mi] = True
        # Context edges: only between unmasked nodes
        ctx_ei = ei[:, ~mask_set[ei[0]] & ~mask_set[ei[1]]]
        # Inbound edges: context → masked
        inbound_ei = ei[:, ~mask_set[ei[0]] & mask_set[ei[1]]]
        return mi, ctx_ei, inbound_ei, mask_set

    def forward(self, x, edge_index):
        N = x.size(0)
        mi, ctx_ei, inbound_ei, mask_set = self.subgraph_mask(edge_index, N)

        # Teacher: sees full graph (no masking)
        with torch.no_grad():
            tz = self.tgt(x, edge_index)
            tgt = F.layer_norm(tz[mi], [self.hid])

        # Student: encode context nodes with context-only edges
        xc = x.clone()
        xc[mi] = 0  # zero masked features
        # First pass: encode context nodes (ctx_ei only)
        cz = self.enc(xc, ctx_ei)

        # Aggregate context into masked positions via inbound edges
        # For each masked node, mean-pool its context neighbors' embeddings
        if inbound_ei.size(1) > 0:
            src_emb = cz[inbound_ei[0]]  # context node embeddings
            dst_idx = inbound_ei[1]       # masked node indices
            # Scatter mean: aggregate context embeddings per masked node
            agg = torch.zeros(N, self.hid, device=x.device)
            counts = torch.zeros(N, 1, device=x.device)
            agg.index_add_(0, dst_idx, src_emb)
            counts.index_add_(0, dst_idx, torch.ones_like(dst_idx, dtype=torch.float32).unsqueeze(1))
            counts = counts.clamp(min=1)
            masked_ctx = agg[mi] / counts[mi]
        else:
            masked_ctx = torch.zeros(len(mi), self.hid, device=x.device)

        # Predict masked embeddings from aggregated context
        pred = F.layer_norm(self.pred(masked_ctx), [self.hid])

        loss_pred = F.smooth_l1_loss(pred, tgt)

        # VICReg on context embeddings
        ctx_emb = cz[~mask_set]
        Nc = ctx_emb.size(0)
        std = ctx_emb.std(dim=0)
        loss_var = F.relu(1.0 - std).mean()
        z_c = ctx_emb - ctx_emb.mean(dim=0)
        cov_mat = (z_c.T @ z_c) / max(Nc - 1, 1)
        loss_cov = (cov_mat.fill_diagonal_(0) ** 2).sum() / self.hid

        return loss_pred + self.lam_var * loss_var + self.lam_cov * loss_cov, {"std": std.mean().item()}

    def encode(self, x, edge_index):
        return self.enc(x, edge_index)


# ============================================================
# MAIN
# ============================================================

def main():
    section("SEMIR-JEPA: 3D graph minor + DINO + IG-JEPA")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pr(f"Device: {device}")

    # ---- Discover and split ----
    all_vids = discover_volumes()
    pr(f"Found {len(all_vids)} LiTS volumes with tumor")

    np.random.seed(42)
    perm = np.random.permutation(len(all_vids))
    n_train = int(0.7 * len(all_vids))
    n_val = int(0.15 * len(all_vids))
    train_ids = sorted([all_vids[i] for i in perm[:n_train]])
    val_ids = sorted([all_vids[i] for i in perm[n_train:n_train + n_val]])
    test_ids = sorted([all_vids[i] for i in perm[n_train + n_val:]])
    pr(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

    # Use a subset for initial testing
    MAX_VOLS = 20  # start small
    MAX_NODES = 2_000_000  # skip volumes with more nodes than this
    train_subset = train_ids[:min(MAX_VOLS, len(train_ids))]
    val_subset = val_ids[:min(5, len(val_ids))]
    pr(f"Using subset: {len(train_subset)} train / {len(val_subset)} val")

    # ---- Load DINO ----
    section("Loading DINO ViT-S/16")
    dino = load_dino_model(device)
    pr("DINO loaded")

    # ---- Build enriched 3D graphs ----
    section("Building 3D graphs with DINO features")
    PSI, ALPHA = 3, 15
    graphs = {}
    labels_map = {}
    segs_map = {}

    for vid in train_subset + val_subset:
        t0 = time.time()
        ct, seg = load_volume(vid)
        raw_nf, raw_ei, raw_ef, labels_np, ct_u8 = build_3d_graph(ct, seg, psi=PSI, alpha=ALPHA)

        # DINO features
        dino_feats = extract_dino_features_for_volume(dino, ct_u8, labels_np, device)

        # Build enriched graph
        g = build_enriched_graph(raw_nf, raw_ei, raw_ef, labels_np, seg, ct_u8,
                                 dino_feats, overlap_th=0.10)
        if g.num_nodes > MAX_NODES:
            pr(f"  vol-{vid}: SKIPPED ({g.num_nodes:,} nodes > {MAX_NODES:,})")
            continue
        graphs[vid] = g
        labels_map[vid] = labels_np
        segs_map[vid] = seg

        n_tu = int((g.y == 1).sum())
        dt = time.time() - t0
        pr(f"  vol-{vid}: {g.num_nodes:,} nodes ({n_tu} tumor), "
           f"{g.num_edges:,} edges, feats={g.x.shape[1]}  [{dt:.1f}s]")

    # Free DINO from GPU
    del dino
    torch.cuda.empty_cache()

    # ---- Feature discrimination check ----
    section("Feature discrimination check")
    g0 = graphs[train_subset[0]]
    x = g0.x.numpy()
    y = g0.y.numpy()
    if (y == 1).sum() > 0:
        for name, cols in [("geometric (0-8)", slice(0, 9)), ("DINO mean", slice(9, 393))]:
            bg = x[y == 0, cols]
            tu = x[y == 1, cols]
            if bg.ndim == 1:
                bg = bg[:, None]
                tu = tu[:, None]
            ds = []
            for c in range(bg.shape[1]):
                d = abs(tu[:, c].mean() - bg[:, c].mean()) / (np.sqrt((bg[:, c].std()**2 + tu[:, c].std()**2) / 2) + 1e-8)
                ds.append(d)
            pr(f"  {name}: max Cohen's d = {max(ds):.4f}, mean = {np.mean(ds):.4f}")

    # ---- JEPA pre-training ----
    section("IG-JEPA pre-training")
    in_dim = graphs[train_subset[0]].x.shape[1]
    HID = 128
    LAYERS = 2
    PRETRAIN_EPOCHS = 30
    pr(f"in_dim={in_dim}, hid={HID}, layers={LAYERS}, epochs={PRETRAIN_EPOCHS}")

    model = IGJEPA(in_dim, HID, LAYERS).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-4, weight_decay=0.01)
    sch = CosineAnnealingLR(opt, T_max=PRETRAIN_EPOCHS, eta_min=1e-6)

    # Per-graph training (full-graph, no batching needed for small subset)
    train_graph_list = [graphs[v] for v in train_subset if v in graphs]

    for ep in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        total_loss = 0
        processed = 0
        for g in np.random.permutation(len(train_graph_list)):
            g_data = train_graph_list[int(g)].to(device)
            try:
                opt.zero_grad()
                loss, info = model(g_data.x, g_data.edge_index)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                model.ema_update()
                total_loss += loss.item()
                processed += 1
                del g_data
            except torch.cuda.OutOfMemoryError:
                try:
                    del g_data
                except:
                    pass
                torch.cuda.empty_cache()
                continue
            torch.cuda.empty_cache()
        sch.step()
        if ep % 10 == 0 or ep <= 3:
            pr(f"  Ep {ep:3d} | Loss: {total_loss/max(processed,1):.4f} | "
               f"std: {info.get('std', 0):.4f} | {processed} vols")

    # ---- MLP probe for tumor classification ----
    section("MLP probe for tumor classification")
    model.eval()
    for p in model.enc.parameters():
        p.requires_grad = False

    probe = nn.Sequential(
        nn.Linear(HID, HID), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(HID, HID // 2), nn.GELU(),
        nn.Linear(HID // 2, 2),
    ).to(device)

    # Class weights from training data
    total_pos = sum(int((graphs[v].y == 1).sum()) for v in train_subset if v in graphs)
    total_neg = sum(int((graphs[v].y == 0).sum()) for v in train_subset if v in graphs)
    ratio = total_neg / max(total_pos, 1)
    eff = min(np.sqrt(ratio), 30.0)
    class_weight = torch.tensor([1.0, eff], dtype=torch.float32).to(device)
    pr(f"Class weight: [1.0, {eff:.1f}] (ratio: {ratio:.0f}:1)")

    probe_opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    PROBE_EPOCHS = 100

    best_dice = -1.0
    best_state = None

    for ep in range(1, PROBE_EPOCHS + 1):
        probe.train()
        total_loss = 0
        processed = 0
        for g_data in [graphs[v].to(device) for v in train_subset if v in graphs]:
            try:
                with torch.no_grad():
                    emb = model.encode(g_data.x, g_data.edge_index)
                logits = probe(emb)
                loss = F.cross_entropy(logits, g_data.y, weight=class_weight)
                probe_opt.zero_grad()
                loss.backward()
                probe_opt.step()
                total_loss += loss.item()
                processed += 1
                del g_data, emb, logits
            except torch.cuda.OutOfMemoryError:
                try:
                    del g_data
                except:
                    pass
                torch.cuda.empty_cache()
                continue
            torch.cuda.empty_cache()

        if ep % 20 == 0 or ep <= 3:
            # Validate with lifted voxel Dice
            probe.eval()
            tp = fp = fn = 0
            fg_rates = []
            with torch.no_grad():
                for vid in [v for v in val_subset if v in graphs]:
                    g = graphs[vid].to(device)
                    try:
                        emb = model.encode(g.x, g.edge_index)
                        preds = probe(emb).argmax(dim=1)
                        fg_rates.append(float((preds == 1).float().mean().cpu()))
                        preds = preds.cpu().numpy()
                    except:
                        torch.cuda.empty_cache()
                        continue
                    del g
                    torch.cuda.empty_cache()

                    lnp = labels_map[vid]
                    seg = segs_map[vid]
                    flat = lnp.ravel()
                    valid = flat >= 0
                    if not valid.any():
                        continue
                    max_id = int(flat[valid].max())
                    lut = np.zeros(max_id + 1, dtype=np.int8)
                    lut[:min(len(preds), max_id + 1)] = preds[:min(len(preds), max_id + 1)]
                    pred_mask = np.where(valid, lut[flat], 0).reshape(lnp.shape).astype(bool)
                    gt_mask = seg == 2
                    inter = int((pred_mask & gt_mask).sum())
                    tp += inter
                    fp += int(pred_mask.sum()) - inter
                    fn += int(gt_mask.sum()) - inter

            val_dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
            mean_fg = np.mean(fg_rates) if fg_rates else 0
            marker = ""
            if val_dice > best_dice:
                best_dice = val_dice
                best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
                marker = " *"
            pr(f"  Ep {ep:3d} | Loss: {total_loss/max(processed,1):.4f} | "
               f"val_dice={val_dice:.4f} | fg={mean_fg:.4f}{marker}")

    pr(f"\nBest val Dice: {best_dice:.4f}")

    # Save
    torch.save({
        "model": model.state_dict(),
        "probe": best_state or probe.state_dict(),
        "best_dice": best_dice,
    }, os.path.join(RESULTS_DIR, "checkpoint.pt"))

    summary = {
        "params": {"psi": PSI, "alpha": ALPHA, "hu": [HU_MIN, HU_MAX],
                    "hid": HID, "layers": LAYERS, "in_dim": in_dim},
        "best_val_dice": float(best_dice),
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "probe_epochs": PROBE_EPOCHS,
        "n_train": len(train_subset),
        "n_val": len(val_subset),
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    pr(f"\nSaved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
