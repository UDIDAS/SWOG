"""Test merge_and_cut (binary tensor) with different merge_distance + voxel reassignment."""
import numpy as np, os, time, fastloops
from scipy.ndimage import distance_transform_edt

DATA_ROOT = '/scratch/ud3d4/acm_data/Data'

def bbox_from_mask(mask, margin=32):
    coords = np.argwhere(mask)
    lo = np.maximum(coords.min(0)-32, 0)
    hi = np.minimum(coords.max(0)+33, mask.shape)
    return tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))

def test_vol(vid, md, cut_d=100):
    ct = np.load(os.path.join(DATA_ROOT, 'ct', f'volume-{vid}.npy')).astype(np.float32)
    seg = np.load(os.path.join(DATA_ROOT, 'seg', f'segmentation-{vid}.npy')).astype(np.int32)
    organ_mask = (seg == 1) | (seg == 2)
    slc = bbox_from_mask(organ_mask)
    ct_crop = ct[slc]; seg_crop = seg[slc]
    ct_u8 = np.clip(ct_crop, -50, 250)
    ct_u8 = ((ct_u8 + 50) / 300 * 255).round().astype(np.uint8)
    ct_u8_4d = np.ascontiguousarray(ct_u8[..., np.newaxis])

    t0 = time.time()
    nf, ei, ef, labels, adj = fastloops.merge_and_cut(ct_u8_4d, merge_distance=md, cut_distance=cut_d)
    dt = time.time() - t0
    nf = np.asarray(nf); labels = np.asarray(labels)
    N = nf.shape[0]

    # Reassign deleted voxels via distance_transform_edt
    survivor_mask = (labels >= 0)
    deleted_pct = (~survivor_mask).sum() / labels.size * 100

    if survivor_mask.any():
        _, nearest_idx = distance_transform_edt(~survivor_mask, return_indices=True)
        reassigned = labels[tuple(nearest_idx)]
    else:
        reassigned = labels.copy()

    # Oracle + tumor recall on reassigned labels
    flat_r = reassigned.ravel().astype(np.int64)
    gt = (seg_crop == 2).ravel().astype(np.float64)
    mid = int(flat_r.max()) if flat_r.max() >= 0 else 0
    fg = np.bincount(flat_r, weights=gt, minlength=mid + 1)[:N]
    tot = np.bincount(flat_r, minlength=mid + 1)[:N]
    bg = tot - fg
    gt_t = fg.sum()
    maj = fg > bg
    oracle = float(2*fg[maj].sum()/(2*fg[maj].sum()+bg[maj].sum()+(gt_t-fg[maj].sum())+1e-8)) if gt_t > 0 else 0
    tumor_total = int((seg_crop == 2).sum())
    tumor_recall = float(gt_t) / max(tumor_total, 1)

    # Boundary Dice: how well do supernode boundaries align with GT boundaries?
    from scipy.ndimage import binary_erosion
    gt_mask = seg_crop == 2
    gt_boundary = gt_mask ^ binary_erosion(gt_mask)
    sn_boundary = np.zeros_like(labels, dtype=bool)
    for axis in range(3):
        sl_lo = [slice(None)] * 3; sl_hi = [slice(None)] * 3
        sl_lo[axis] = slice(0, -1); sl_hi[axis] = slice(1, None)
        diff = reassigned[tuple(sl_lo)] != reassigned[tuple(sl_hi)]
        sn_boundary[tuple(sl_lo)] |= diff
        sn_boundary[tuple(sl_hi)] |= diff
    inter = int((gt_boundary & sn_boundary).sum())
    bdice = 2 * inter / (gt_boundary.sum() + sn_boundary.sum() + 1e-8)

    return N, oracle, tumor_recall, deleted_pct, bdice, dt

print("  vid  md |    nodes   oracle  t_recall  del%  bdice   time")
print("-" * 65)
for vid in [0, 27, 48, 76, 104]:
    for md in [5, 10, 15, 20, 30, 50]:
        N, oracle, recall, del_pct, bdice, dt = test_vol(vid, md)
        print(f"  {vid:>3d}  {md:>2d} | {N:>8,}  {oracle:.4f}  {recall:.4f}  {del_pct:4.1f}%  {bdice:.4f}  {dt:.1f}s",
              flush=True)
    print(flush=True)
