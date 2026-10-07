"""Agent 2: checks a predicted mask against clinical metadata for anatomical and size plausibility."""

import numpy as np

from mars.data.synthetic import voxel_centroid
from mars.models import ClinicalMetadata, ValidationResult

SITES: dict[str, tuple[float, float]] = {
    "anterior": (0.0, 0.5),
    "posterior": (0.5, 1.0),
}


def validate(mask: np.ndarray, metadata: ClinicalMetadata) -> ValidationResult:
    conflicts: list[str] = []
    voxel_count = int(mask.sum())

    if voxel_count == 0:
        return ValidationResult(consistency_score=0.0, conflicts=["no lesion detected in mask"])

    z_centroid, _, _ = voxel_centroid(mask)
    z_depth = mask.shape[0]
    z_fraction = z_centroid / max(z_depth - 1, 1)

    site_range = SITES.get(metadata.expected_site)
    site_ok = site_range is not None and site_range[0] <= z_fraction <= site_range[1]
    if not site_ok:
        conflicts.append(
            f"mask centroid at z-fraction {z_fraction:.2f} is outside expected site '{metadata.expected_site}'"
        )

    lo, hi = metadata.expected_size_range_mm3
    size_ok = lo <= voxel_count <= hi
    if not size_ok:
        conflicts.append(
            f"mask size {voxel_count} voxels is outside expected range ({lo}, {hi})"
        )

    checks_passed = sum([site_ok, size_ok])
    score = checks_passed / 2.0

    return ValidationResult(consistency_score=score, conflicts=conflicts)
