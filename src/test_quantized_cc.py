"""Test quantized connected components with node deletion sweep."""
import numpy as np
import sys
from scipy.ndimage import label as nd_label
sys.path.insert(0, '.')
from semir.graph_minor import _node_deletion

ct = np.load('/scratch/ud3d4/acm_data/Data/ct/volume-17.npy').astype(np.float32)
seg = np.load('/scratch/ud3d4/acm_data/Data/seg/segmentation-17.npy').astype(np.int32)
ct_norm = np.clip(ct, 0, 200).astype(np.float64) / 200.0
HWD = int(np.prod(ct_norm.shape))


def oracle(labels, seg):
    flat_l = labels.ravel()
    flat_gt = (seg.ravel() == 2).astype(np.float64)
    max_id = int(flat_l.max())
    if max_id == 0:
        return 0.0, 0
    tc = np.bincount(flat_l, weights=flat_gt, minlength=max_id + 1)
    ttc = np.bincount(flat_l, minlength=max_id + 1)
    safe = np.where(ttc > 0, ttc, 1)
    ovlp = tc / safe
    tlut = (ovlp > 0.1).astype(np.int32)
    tlut[0] = 0
    pred = tlut[flat_l]
    gt_m = flat_gt.astype(np.int32)
    tp = int((pred & gt_m).sum())
    fp = int(pred.sum()) - tp
    fn = int(gt_m.sum()) - tp
    n_tu = int((ovlp[1:] > 0.1).sum())
    return 2 * tp / (2 * tp + fp + fn + 1e-8), n_tu


# Quantized CC with 16 bins
n_bins = 16
quantized = np.clip(np.floor(ct_norm * n_bins).astype(np.int32), 0, n_bins - 1)
labels = np.zeros_like(quantized, dtype=np.int64)
offset = 0
for b in range(n_bins):
    mask = quantized == b
    if mask.sum() == 0:
        continue
    cc, n_cc = nd_label(mask)
    labels[mask] = cc[mask] + offset
    offset += n_cc

print(f"Raw: {int(labels.max()):,} SN, oracle={oracle(labels, seg)[0]:.4f}", flush=True)

# Tumor supernode size distribution
flat_l = labels.ravel()
flat_gt = (seg.ravel() == 2).astype(np.float64)
max_id = int(flat_l.max())
tc = np.bincount(flat_l, weights=flat_gt, minlength=max_id + 1)
ttc = np.bincount(flat_l, minlength=max_id + 1)
safe = np.where(ttc > 0, ttc, 1)
ovlp = tc / safe
tumor_sids = np.where((ovlp > 0.1) & (np.arange(max_id + 1) > 0))[0]
tumor_sizes = ttc[tumor_sids]
print(f"Tumor SN sizes: min={tumor_sizes.min()} median={np.median(tumor_sizes):.0f} "
      f"mean={tumor_sizes.mean():.1f} max={tumor_sizes.max():,}", flush=True)

# Node deletion sweep
header = f"{'bmin':>6s} {'#SN':>8s} {'oracle':>8s} {'tSN':>6s} {'delT%':>6s}"
print(f"\n{header}", flush=True)
for bmin in [1, 5, 10, 50, 100, 500, 1000, 5000]:
    labels_d = _node_deletion(labels.copy(), ct_norm,
                              beta_min=bmin, beta_max=HWD // 3,
                              m_min=0.0, m_max=1.0)
    n_sn = len(np.unique(labels_d[labels_d > 0]))
    o, n_tu = oracle(labels_d, seg)
    dt = int(((labels_d == 0) & (seg == 2)).sum())
    gt_t = int((seg == 2).sum())
    m = " ***" if 500 <= n_sn <= 5000 and o > 0.7 else ""
    print(f"{bmin:6d} {n_sn:>8,} {o:8.4f} {n_tu:>6d} {dt / max(gt_t, 1) * 100:5.1f}%{m}", flush=True)
