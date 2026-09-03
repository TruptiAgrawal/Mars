"""Agent 1: thresholding + connected-component segmentation with per-voxel confidence, mocking a learned segmentation model."""

import numpy as np
from scipy import ndimage

from mars.models import SegmentationResult


def segment(volume: np.ndarray, threshold: float = 120.0) -> SegmentationResult:
    above = volume > threshold
    labeled, num_components = ndimage.label(above)

    mask = np.zeros_like(volume, dtype=np.uint8)
    confidence = np.zeros_like(volume, dtype=np.float32)

    if num_components > 0:
        sizes = ndimage.sum(above, labeled, index=range(1, num_components + 1))
        largest_label = int(np.argmax(sizes)) + 1
        mask = (labeled == largest_label).astype(np.uint8)

        voxel_confidence = np.clip((volume - threshold) / threshold, 0.0, 1.0)
        confidence = np.where(mask == 1, voxel_confidence, 0.0).astype(np.float32)

    aggregate = float(confidence[mask == 1].mean()) if mask.sum() > 0 else 0.0

    return SegmentationResult(
        mask=mask.tolist(),
        per_voxel_confidence=confidence.tolist(),
        aggregate_confidence=aggregate,
    )
