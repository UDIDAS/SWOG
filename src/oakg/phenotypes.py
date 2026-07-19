"""Derive imaging phenotypes from segmentation masks (multi-organ aware).

Maps labeled NIfTI masks to the phenotype observations in the imaging KG schema:
organ presence/volume, tumor presence, tumor burden, lesion multiplicity, and
tumor-in-organ containment. A dataset may cover one organ (Pancreas, LiTS) or
many (FLARE); tumor observations exist only where the source segments tumor.
Pure array functions here; :mod:`oakg.build_benchmark` walks the datasets.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

# A tumor connected component must exceed this many voxels to count as a lesion
# (suppresses single-voxel segmentation noise inflating multiplicity).
MIN_LESION_VOXELS = 10


@dataclass(frozen=True)
class DatasetSpec:
    """How a source dataset lays out its masks and which organs/tumors it labels.

    layout="combined": one multi-label mask file per case; ``organ_labels`` and
        ``tumor_labels`` give the integer for each organ/tumor.
    layout="per_organ": one binary mask file per organ per case (label 1); the
        organ set is ``organs`` and tumor is unavailable.
    """
    name: str
    organs: tuple[str, ...]
    layout: str = "combined"
    organ_labels: dict[str, int] = field(default_factory=dict)
    tumor_labels: dict[str, int] = field(default_factory=dict)

    @property
    def has_tumor(self) -> bool:
        return bool(self.tumor_labels)


def voxel_volume_cm3(spacing: tuple[float, float, float]) -> float:
    """mm^3 per voxel -> cm^3 (1 cm^3 = 1000 mm^3)."""
    return float(np.prod(spacing)) / 1000.0


def _bbox(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return coords.min(axis=0), coords.max(axis=0)


def bbox_contained(inner: np.ndarray, organ: np.ndarray, margin: int = 2) -> bool:
    """True if the ``inner`` voxel set's bbox lies within ``organ``'s bbox (± margin)."""
    if inner.size == 0 or organ.size == 0:
        return False
    imin, imax = _bbox(inner)
    omin, omax = _bbox(organ)
    return bool(np.all(imin >= omin - margin) and np.all(imax <= omax + margin))


def organ_phenotypes(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    organ_label: int,
    tumor_label: int | None = None,
    min_lesion_voxels: int = MIN_LESION_VOXELS,
) -> dict[str, float | None]:
    """Phenotype observations for ONE organ within a mask.

    ``None`` marks an observation the mask cannot define (e.g. volume when the
    organ is absent, or any tumor field when the source segments no tumor) and
    becomes a missing value downstream. Tumor fields are omitted entirely
    (returned as None) when ``tumor_label`` is None.
    """
    vox_cm3 = voxel_volume_cm3(spacing)
    organ_coords = np.argwhere(mask == organ_label)
    organ_present = organ_coords.shape[0] > 0

    out: dict[str, float | None] = {
        "present": float(organ_present),
        "volume_cm3": float(organ_coords.shape[0] * vox_cm3) if organ_present else None,
        "tumor_present": None,
        "tumor_burden_cm3": None,
        "lesion_multiplicity": None,
        "tumor_containment": None,
    }
    if tumor_label is None:
        return out

    tumor_mask = mask == tumor_label
    tumor_coords = np.argwhere(tumor_mask)
    tumor_present = tumor_coords.shape[0] > 0
    out["tumor_present"] = float(tumor_present)
    out["tumor_burden_cm3"] = float(tumor_coords.shape[0] * vox_cm3)

    if tumor_present:
        labeled, n = ndimage.label(tumor_mask)
        sizes = np.bincount(labeled.ravel())[1:] if n else np.array([])
        out["lesion_multiplicity"] = float(max(int(np.sum(sizes >= min_lesion_voxels)), 1))
    else:
        out["lesion_multiplicity"] = 0.0

    if tumor_present and organ_present:
        out["tumor_containment"] = float(bbox_contained(tumor_coords, organ_coords))
    return out
