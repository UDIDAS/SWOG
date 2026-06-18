"""Sweep protection tightness: std_mult x dilate on 5 volumes."""
import numpy as np, os, fastloops_band
from scipy.ndimage import binary_dilation

DATA_ROOT = '/scratch/ud3d4/acm_data/Data'
HU_MIN, HU_MAX = -50, 250
BB_AREA = 0; BB_NFG = 21; BB_NBG = 22

band_of = np.array([0]*37+[37]*18+[55]*2+[57]*8+[65]*6+[71]*5+[76]*3+[79]*3+[82]*3+[85]*3+[88,89,90,90,90,93,94,95,95,97,98,99,100,101,101,103,103,105,106,107,108,108,110,111,112,113,114,115,116,117,117,119,120,121,122,123,124,125,126,127,128,129,130,131,131,133,134,134,136,136,138,139,139,141,141,141,144,145,145,145,145,149,149,149,149,149,154,154,156,156,156,156,156,156,162,162,162,162,162,167,167,167,167,167,167,173,173,173,173,173,173,179,179,179,179,179,179,179,179,187,187,187,187,187,187,187,187,187]+[196]*60, dtype=np.uint8)
deleted_arr = np.zeros(256, dtype=np.uint8); deleted_arr[196:] = 1

def bbox_from_mask(mask, margin=32):
    coords = np.argwhere(mask)
    lo = np.maximum(coords.min(0)-32, 0)
    hi = np.minimum(coords.max(0)+33, mask.shape)
    return tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))

vids = [0, 27, 48, 76, 104]

# Cache band_build results (expensive part)
cache = {}
for vid in vids:
    ct = np.load(os.path.join(DATA_ROOT, 'ct', f'volume-{vid}.npy')).astype(np.float32)
    seg = np.load(os.path.join(DATA_ROOT, 'seg', f'segmentation-{vid}.npy')).astype(np.int32)
    organ_mask = (seg == 1) | (seg == 2)
    slc = bbox_from_mask(organ_mask)
    ct_crop = ct[slc]; seg_crop = seg[slc]; organ_crop = organ_mask[slc]
    ct_u8 = np.clip(ct_crop, HU_MIN, HU_MAX)
    ct_u8 = ((ct_u8 - HU_MIN) / (HU_MAX - HU_MIN) * 255).round().astype(np.uint8)
    ct_u8_4d = np.ascontiguousarray(ct_u8[..., np.newaxis])
    mask = (seg_crop == 2).astype(np.uint8)
    nf, ei, ef, labels, adj = fastloops_band.band_build(ct_u8_4d, mask, band_of, deleted_arr)
    nf = np.asarray(nf); labels = np.asarray(labels)
    liver_vals = ct_u8[organ_crop]
    cache[vid] = dict(nf=nf, labels=labels, ct_u8=ct_u8, organ_crop=organ_crop,
                      liver_mean=liver_vals.mean(), liver_std=liver_vals.std())
    print(f"  cached vol-{vid}: {len(nf):,} nodes", flush=True)

print(flush=True)
print(f" std  dil |", end="")
for vid in vids:
    print(f"  vol-{vid:>3d}         ", end="")
print("| mean_nodes  mean_oracle", flush=True)
print("-" * 110, flush=True)

for std_mult in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
    for dilate in [2, 1, 0]:
        row_nodes, row_oracles = [], []
        for vid in vids:
            c = cache[vid]
            thr = c["liver_mean"] - std_mult * c["liver_std"]
            cand = c["organ_crop"] & (c["ct_u8"] < thr)
            if dilate > 0:
                prot = binary_dilation(cand, iterations=dilate)
            else:
                prot = cand
            flat = c["labels"].ravel(); fprot = prot.ravel(); valid = flat >= 0
            N = len(c["nf"])
            node_prot = np.zeros(N, dtype=bool)
            if valid.any():
                mid = int(flat[valid].max())
                pc = np.bincount(flat[valid & fprot], minlength=mid + 1)
                node_prot[:min(N, len(pc))] = pc[:N] > 0
            pids = np.where(node_prot)[0]
            n_prot = len(pids)
            fg = c["nf"][pids, BB_NFG]; bg = c["nf"][pids, BB_NBG]; gt = fg.sum()
            maj = fg > bg
            oracle = float(2*fg[maj].sum()/(2*fg[maj].sum()+bg[maj].sum()+(gt-fg[maj].sum())+1e-8)) if gt > 0 else 0
            row_nodes.append(n_prot)
            row_oracles.append(oracle)
        print(f"{std_mult:4.1f}  {dilate:3d} |", end="")
        for i in range(len(vids)):
            print(f"  {row_nodes[i]:>7,} {row_oracles[i]:.3f}", end="")
        print(f" | {np.mean(row_nodes):>9,.0f}  {np.mean(row_oracles):.4f}", flush=True)
