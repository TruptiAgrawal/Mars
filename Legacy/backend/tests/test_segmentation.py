import numpy as np
from mars.agents.segmentation import segment
from mars.data.synthetic import generate_volume


def test_segment_detects_lesion_blob():
    volume = generate_volume(
        shape=(16, 32, 32), center=(8, 16, 16), radius=4.0,
        intensity=200.0, background=50.0, noise_std=0.0, seed=1,
    )
    result = segment(volume, threshold=120.0)
    mask = np.array(result.mask)
    assert mask.shape == volume.shape
    assert mask[8, 16, 16] == 1
    assert mask[0, 0, 0] == 0
    assert 0.0 <= result.aggregate_confidence <= 1.0


def test_segment_low_contrast_blob_has_lower_confidence_than_high_contrast():
    high_contrast = generate_volume(
        shape=(16, 32, 32), center=(8, 16, 16), radius=5.0,
        intensity=200.0, background=50.0, noise_std=0.0, seed=1,
    )
    low_contrast = generate_volume(
        shape=(16, 32, 32), center=(8, 16, 16), radius=5.0,
        intensity=130.0, background=50.0, noise_std=0.0, seed=1,
    )
    high_result = segment(high_contrast, threshold=120.0)
    low_result = segment(low_contrast, threshold=120.0)
    assert low_result.aggregate_confidence < high_result.aggregate_confidence


def test_segment_no_lesion_returns_empty_mask():
    flat = np.full((8, 8, 8), 50.0, dtype=np.float32)
    result = segment(flat, threshold=120.0)
    mask = np.array(result.mask)
    assert mask.sum() == 0
    assert result.aggregate_confidence == 0.0
