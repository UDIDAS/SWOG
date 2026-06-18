"""
Test Stage 2 contraction: merge band-flood supernodes with canonical-intensity Union-Find.

band_build → ~1.6M tiny supernodes
contraction → merge adjacent supernodes where |canonical_a - canonical_b| ≤ ψ
            → ~1K-10K large supernodes (paper range)

Key: canonical intensity = one representative voxel's intensity per supernode.
After merge, the larger supernode's canonical is kept — prevents transitive drift.
"""
import numpy as np, os, time
import fastloops_band

DATA_ROOT = '/scratch/ud3d4/acm_data/Data'
HU_MIN, HU_MAX = -50, 250
BB_AREA = 0; BB_CHAN0 = 10; BB_NFG = 21; BB_NBG = 22

band_of = np.array([0]*37+[37]*18+[55]*2+[57]*8+[65]*6+[71]*5+[76]*3+[79]*3+[82]*3+[85]*3+[88,89,90,90,90,93,94,95,95,97,98,99,100,101,101,103,103,105,106,107,108,108,110,111,112,113,114,115,116,117,117,119,120,121,122,123,124,125,126,127,128,129,130,131,131,133,134,134,136,136,138,139,139,141,141,141,144,145,145,145,145,149,149,149,149,149,154,154,156,156,156,156,156,156,162,162,162,162,162,167,167,167,167,167,167,173,173,173,173,173,173,179,179,179,179,179,179,179,179,187,187,187,187,187,187,187,187,187]+[196]*60, dtype=np.uint8)
deleted_arr = np.zeros(256, dtype=np.uint8); deleted_arr[196:] = 1

def bbox_from_mask(mask, margin=32):
    coords = np.argwhere(mask)
    lo = np.maximum(coords.min(0)-32, 0)
    hi = np.minimum(coords.max(0)+33, mask.shape)
    return tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))


def contract_supernodes(labels, nf, psi_u8):
    """
    Merge adjacent supernodes whose canonical intensities differ by ≤ psi_u8.
    Uses Union-Find with canonical-intensity tracking (larger supernode keeps its canonical).

    Returns (new_labels, new_nf, merge_map).
    """
    N = len(nf)
    if N == 0:
        return labels, nf, np.arange(0)

    # Canonical intensity per supernode: mean intensity in uint8 space
    Vsafe = np.maximum(nf[:, BB_AREA], 1.0)
    canonical = (nf[:, BB_CHAN0] / Vsafe).astype(np.float64)  # mean channel value (0-255)

    # Build supernode adjacency from label volume (vectorized)
    t0 = time.time()
    all_a, all_b = [], []
    for axis in range(3):
        sl_lo = [slice(None)] * 3; sl_hi = [slice(None)] * 3
        sl_lo[axis] = slice(0, -1); sl_hi[axis] = slice(1, None)
        a = labels[tuple(sl_lo)].ravel()
        b = labels[tuple(sl_hi)].ravel()
        boundary = (a != b) & (a >= 0) & (b >= 0)
        all_a.append(a[boundary]); all_b.append(b[boundary])

    if not all_a:
        return labels, nf, np.arange(N)

    a_arr = np.concatenate(all_a).astype(np.int64)
    b_arr = np.concatenate(all_b).astype(np.int64)

    # Deduplicate pairs
    lo = np.minimum(a_arr, b_arr); hi = np.maximum(a_arr, b_arr)
    pair_keys = lo * N + hi
    unique_keys = np.unique(pair_keys)
    u_a = unique_keys // N; u_b = unique_keys % N
    t_adj = time.time() - t0

    # Compute canonical diffs and sort (Kruskal-like)
    diffs = np.abs(canonical[u_a] - canonical[u_b])
    order = np.argsort(diffs)
    u_a = u_a[order]; u_b = u_b[order]; diffs = diffs[order]

    # Union-Find with canonical tracking
    t0 = time.time()
    parent = np.arange(N, dtype=np.int64)
    size = nf[:, BB_AREA].copy().astype(np.int64)
    canon = canonical.copy()

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    n_merges = 0
    for i in range(len(u_a)):
        if diffs[i] > psi_u8:
            break  # sorted, no more valid merges
        ra = find(int(u_a[i]))
        rb = find(int(u_b[i]))
        if ra == rb:
            continue
        # Re-check canonical diff (may have changed due to prior merges)
        if abs(canon[ra] - canon[rb]) > psi_u8:
            continue
        # Merge: larger keeps its canonical
        if size[ra] >= size[rb]:
            parent[rb] = ra
            size[ra] += size[rb]
            # canon[ra] stays (larger supernode's canonical)
        else:
            parent[ra] = rb
            size[rb] += size[ra]
            # canon[rb] stays
        n_merges += 1
    t_merge = time.time() - t0

    # Compress all paths
    for i in range(N):
        find(i)

    # Relabel: root IDs → contiguous IDs
    roots = parent.copy()
    unique_roots, inv = np.unique(roots, return_inverse=True)
    N_new = len(unique_roots)

    # Relabel volume
    lut = np.full(N, -1, dtype=np.int64)
    lut[np.arange(N)] = inv
    flat = labels.ravel()
    new_flat = np.full_like(flat, -1)
    valid = flat >= 0
    new_flat[valid] = lut[flat[valid]]
    new_labels = new_flat.reshape(labels.shape)

    # Accumulate features by new label
    new_nf = np.zeros((N_new, nf.shape[1]), dtype=np.float64)
    np.add.at(new_nf, inv, nf.astype(np.float64))

    print(f"    contraction: {N:,} → {N_new:,} supernodes ({n_merges:,} merges)  "
          f"adj={t_adj:.1f}s  merge={t_merge:.1f}s", flush=True)

    return new_labels, new_nf.astype(np.float32), inv


# Test on several volumes with different psi
vids = [0, 27, 48, 76, 104]

for vid in vids:
    ct = np.load(os.path.join(DATA_ROOT, 'ct', f'volume-{vid}.npy')).astype(np.float32)
    seg = np.load(os.path.join(DATA_ROOT, 'seg', f'segmentation-{vid}.npy')).astype(np.int32)
    organ_mask = (seg == 1) | (seg == 2)
    slc = bbox_from_mask(organ_mask)
    ct_crop = ct[slc]; seg_crop = seg[slc]
    ct_u8 = np.clip(ct_crop, HU_MIN, HU_MAX)
    ct_u8 = ((ct_u8 - HU_MIN) / (HU_MAX - HU_MIN) * 255).round().astype(np.uint8)
    ct_u8_4d = np.ascontiguousarray(ct_u8[..., np.newaxis])
    mask = (seg_crop == 2).astype(np.uint8)

    t0 = time.time()
    nf, ei, ef, labels, adj = fastloops_band.band_build(ct_u8_4d, mask, band_of, deleted_arr)
    nf = np.asarray(nf); labels = np.asarray(labels)
    t_build = time.time() - t0
    N_orig = len(nf)
    gt_tumor = int((seg_crop == 2).sum())

    print(f"\nvol-{vid}: {N_orig:,} supernodes, {gt_tumor:,} tumor voxels (build={t_build:.1f}s)", flush=True)

    for psi in [5, 10, 15, 20, 30]:
        new_labels, new_nf, merge_map = contract_supernodes(labels.copy(), nf, psi)
        N_new = len(new_nf)

        # Check tumor recall + oracle
        flat = new_labels.ravel(); gt_flat = (seg_crop == 2).ravel()
        valid = flat >= 0
        fg = np.zeros(N_new, dtype=np.float64)
        bg = np.zeros(N_new, dtype=np.float64)
        if valid.any():
            mid = int(flat[valid].max())
            fg_bc = np.bincount(flat[valid], weights=gt_flat[valid].astype(np.float64), minlength=mid+1)
            tot_bc = np.bincount(flat[valid], minlength=mid+1)
            fg[:min(N_new, len(fg_bc))] = fg_bc[:N_new]
            bg[:min(N_new, len(tot_bc))] = (tot_bc - fg_bc)[:N_new]
        gt_t = fg.sum()
        maj = fg > bg
        oracle = float(2*fg[maj].sum()/(2*fg[maj].sum()+bg[maj].sum()+(gt_t-fg[maj].sum())+1e-8)) if gt_t > 0 else 0

        # Tumor recall: how many GT tumor voxels are in valid supernodes?
        tumor_vox_valid = int((valid.reshape(seg_crop.shape) & (seg_crop == 2)).sum())
        tumor_recall = tumor_vox_valid / max(gt_tumor, 1)

        print(f"  psi={psi:>2d}: {N_new:>8,} nodes  oracle={oracle:.4f}  "
              f"tumor_recall={tumor_recall:.4f} ({tumor_vox_valid:,}/{gt_tumor:,})", flush=True)
