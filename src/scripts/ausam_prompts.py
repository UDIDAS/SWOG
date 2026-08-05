#!/usr/bin/env python3
"""AUSAM prompting, adapted to SAM3. The original AUSAM (SAM1) ran DBSCAN over each GT connected component
and emitted one *point* per cluster. SAM3 has no point interface (text + box only), so we keep the identical
DBSCAN spatial decomposition but emit one *bounding box* per cluster — a 2-kidney slice yields 2 boxes, a
liver yields 1. Faithful to the method (DBSCAN-driven multi-prompt from the mask), on the SAM3 backbone so the
eventual AUSAM-vs-KG comparison shares one backbone.
"""
import numpy as np
from skimage.measure import label, regionprops
from sklearn.cluster import DBSCAN


def dbscan_boxes(mask, eps=5, min_samples=10, pad=3):
    """GT mask -> list of [x1,y1,x2,y2] boxes: regionprops components, DBSCAN within each, one bbox per cluster.
    Mirrors run_flare.generate_coordinates (eps=5, min_samples=10) but returns boxes, not points."""
    mask = np.asarray(mask) > 0
    H, W = mask.shape
    boxes = []
    for region in regionprops(label(mask.astype(np.uint8))):
        coords = region.coords                                   # (N, 2) as (row, col)
        clusters = []
        if len(coords) >= min_samples:
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
            clusters = [coords[db.labels_ == k] for k in set(db.labels_) if k != -1]
        if not clusters:
            clusters = [coords]                                  # fallback: the whole component
        for c in clusters:
            y1, x1 = c.min(0); y2, x2 = c.max(0)
            boxes.append([max(0, int(x1) - pad), max(0, int(y1) - pad),
                          min(W - 1, int(x2) + pad), min(H - 1, int(y2) + pad)])
    if not boxes:                                                # empty mask -> whole-image box
        boxes = [[0, 0, W - 1, H - 1]]
    return boxes
