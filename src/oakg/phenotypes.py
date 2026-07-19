"""Derive imaging phenotypes from segmentation masks.

Maps a labeled NIfTI mask (organ + tumor labels) to the phenotype observations
in the imaging KG schema: organ presence/volume, tumor presence, tumor burden,
lesion multiplicity, and tumor-in-organ containment. Pure array functions here;
:mod:`oakg.build_benchmark` walks the datasets and writes the benchmark CSVs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# A tumor connected component must exceed this many voxels to count as a lesion
# (suppresses single-voxel segmentation noise inflating multiplicity).
MIN_LESION_VOXELS = 10


@dataclass(frozen=True)
class DatasetSpec:
    """How a source dataset labels its masks and which organ it covers."""
    name: str          # dataset name, e.g. "Pancreas"
    organ: str         # covered organ, e.g. "pancreas"
    organ_label: int   # integer label for the organ in the mask
    tumor_label: int   # integer label for the tumor in the mask


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


def extract_phenotypes(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    spec: DatasetSpec,
    min_lesion_voxels: int = MIN_LESION_VOXELS,
) -> dict[str, float | None]:
    """Compute phenotype observations for one case's mask.

    Returns a dict keyed by phenotype name. ``None`` marks an observation that
    cannot be derived from this mask (e.g. containment when no organ was
    segmented) and becomes a missing value downstream.
    """
    vox_cm3 = voxel_volume_cm3(spacing)
    organ_coords = np.argwhere(mask == spec.organ_label)
    tumor_mask = mask == spec.tumor_label
    tumor_coords = np.argwhere(tumor_mask)

    organ_present = organ_coords.shape[0] > 0
    tumor_present = tumor_coords.shape[0] > 0

    tumor_burden_cm3 = float(tumor_coords.shape[0] * vox_cm3)
    organ_volume_cm3 = float(organ_coords.shape[0] * vox_cm3) if organ_present else 0.0

    # Lesion multiplicity: connected components of the tumor label above noise floor.
    if tumor_present:
        labeled, n = ndimage.label(tumor_mask)
        if n:
            sizes = np.bincount(labeled.ravel())[1:]
            multiplicity = int(np.sum(sizes >= min_lesion_voxels))
            multiplicity = max(multiplicity, 1)  # tumor exists -> at least one lesion
        else:
            multiplicity = 0
    else:
        multiplicity = 0

    # Containment: tumor bbox within organ bbox. Undefined without both.
    if tumor_present and organ_present:
        containment: float | None = float(bbox_contained(tumor_coords, organ_coords))
    else:
        containment = None

    return {
        "organ_present": float(organ_present),
        "organ_volume_cm3": organ_volume_cm3 if organ_present else None,
        "tumor_present": float(tumor_present),
        "tumor_burden_cm3": tumor_burden_cm3,
        "lesion_multiplicity": float(multiplicity),
        "tumor_containment": containment,
    }
