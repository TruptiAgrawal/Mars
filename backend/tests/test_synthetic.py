import numpy as np
from mars.data.synthetic import generate_volume, voxel_centroid


def test_generate_volume_shape_and_determinism():
    v1 = generate_volume(shape=(16, 32, 32), center=(8, 16, 16), radius=4.0, seed=42)
    v2 = generate_volume(shape=(16, 32, 32), center=(8, 16, 16), radius=4.0, seed=42)
    assert v1.shape == (16, 32, 32)
    assert np.array_equal(v1, v2)


def test_generate_volume_lesion_brighter_than_background():
    v = generate_volume(
        shape=(16, 32, 32), center=(8, 16, 16), radius=4.0,
        intensity=200.0, background=50.0, noise_std=0.0, seed=1,
    )
    assert v[8, 16, 16] > v[0, 0, 0]


def test_voxel_centroid_of_simple_mask():
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    centroid = voxel_centroid(mask)
    assert centroid == (1.5, 1.5, 1.5)
