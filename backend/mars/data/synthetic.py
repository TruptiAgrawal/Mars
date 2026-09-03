"""Generates synthetic 3D CT-like volumes with an embedded Gaussian-blob lesion, for mock pipeline testing."""

import numpy as np


def generate_volume(
    shape: tuple[int, int, int] = (32, 64, 64),
    center: tuple[float, float, float] | None = None,
    radius: float = 6.0,
    intensity: float = 200.0,
    background: float = 50.0,
    noise_std: float = 10.0,
    seed: int = 0,
) -> np.ndarray:
    if center is None:
        center = tuple(s / 2 for s in shape)

    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    dist = np.sqrt(
        (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    )
    blob = intensity * np.exp(-(dist**2) / (2 * radius**2))
    volume = background + blob
    if noise_std > 0:
        volume = volume + rng.normal(0.0, noise_std, size=shape)
    return volume.astype(np.float32)


def voxel_centroid(mask: np.ndarray) -> tuple[float, float, float]:
    coords = np.argwhere(mask > 0)
    mean = coords.mean(axis=0)
    return tuple(float(v) for v in mean)
