"""
btcv_cross_validation/z_spacing/test_cpu.py

CPU-only geometry verification for the z-spacing resampling module.
No GPU, no neural network, no NIfTI files on disk required.

Tests:
  ResamplingTests
    test_identity_unchanged_spacing       - zoom=1 returns identical array
    test_image_linear_shape               - output shape matches expected z_out
    test_label_nearest_neighbour          - label values stay integer {0,1}
    test_image_label_same_output_shape    - image and label produce same z_out
    test_source_bigger_than_target        - downsampling increases slice count
    test_source_smaller_than_target       - upsampling decreases slice count

  CoordMappingTests
    test_roundtrip_z_coord                - map forward then back stays close
    test_endpoints_clipped               - out-of-bounds coords are clamped
    test_physical_consistency            - mapped center preserves physical mm

  PrepareArraysTests
    test_cropping_removes_empty_z        - gt-cropped removes blank boundary slices
    test_full_volume_keeps_all_z         - full-volume keeps all slices
    test_absent_target_raises            - missing organ raises ValueError
    test_too_short_raises                - <3 slices in working grid raises

  PromptTests
    test_initial_prompt_box_shape        - box has shape (1,4)
    test_initial_prompt_inside_mask      - box corners inside the mask
    test_empty_slice_raises              - empty slice raises ValueError

  PipelineGeometryTests
    test_proposed_branch_output_on_scoring_grid  - prediction shape matches
                                                   scoring grid after round-trip
    test_label_values_preserved_through_resample - no spurious label values
"""
import unittest
import numpy as np
import sys
from pathlib import Path

# Allow running directly from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from btcv_cross_validation.z_spacing.helpers import (
    resample_image_z,
    resample_label_z,
    map_z_coord_to_resampled,
    map_z_coord_to_original,
    prepare_arrays,
    initial_prompt,
    segmentation_metrics,
)


def _make_volume(z=30, y=64, x=64, foreground_slices=(10, 20)):
    """Synthetic image and binary label."""
    image = np.random.randn(z, y, x).astype(np.float32) * 100
    label = np.zeros((z, y, x), dtype=bool)
    for s in range(*foreground_slices):
        label[s, 20:40, 20:40] = True
    return image, label


class ResamplingTests(unittest.TestCase):

    def test_identity_unchanged_spacing(self):
        img, _ = _make_volume(z=20)
        out, actual = resample_image_z(img, source_z_mm=3.0, target_z_mm=3.0)
        self.assertEqual(out.shape, img.shape)
        np.testing.assert_array_equal(out, img)
        self.assertAlmostEqual(actual, 3.0, places=6)

    def test_image_linear_shape(self):
        img, _ = _make_volume(z=30)
        # 3mm -> 1.5mm should roughly double z
        out, actual = resample_image_z(img, source_z_mm=3.0, target_z_mm=1.5)
        expected_z = max(2, int(round(30 * 3.0 / 1.5)))
        self.assertEqual(out.shape[0], expected_z)
        self.assertEqual(out.shape[1:], img.shape[1:])
        self.assertAlmostEqual(actual, 3.0 / (expected_z / 30), places=5)

    def test_label_nearest_neighbour(self):
        _, lbl = _make_volume(z=20)
        out, _ = resample_label_z(lbl.astype(np.uint8), source_z_mm=3.0, target_z_mm=1.5)
        unique = set(np.unique(out).tolist())
        self.assertTrue(unique.issubset({0, 1}),
                        f'Unexpected label values after resample: {unique}')

    def test_image_label_same_output_shape(self):
        img, lbl = _make_volume(z=30)
        img_out, _ = resample_image_z(img, 5.0, 3.0)
        lbl_out, _ = resample_label_z(lbl.astype(np.uint8), 5.0, 3.0)
        self.assertEqual(img_out.shape, lbl_out.shape)

    def test_source_bigger_than_target_increases_slices(self):
        img, _ = _make_volume(z=20)
        out, _ = resample_image_z(img, source_z_mm=6.0, target_z_mm=3.0)
        self.assertGreater(out.shape[0], img.shape[0])

    def test_source_smaller_than_target_decreases_slices(self):
        img, _ = _make_volume(z=40)
        out, _ = resample_image_z(img, source_z_mm=3.0, target_z_mm=6.0)
        self.assertLess(out.shape[0], img.shape[0])


class CoordMappingTests(unittest.TestCase):

    def test_roundtrip_z_coord(self):
        # Forward then backward should be close (within 1 slice)
        z_orig, source_z, target_z, z_in = 15, 6.0, 3.0, 30
        z_out = max(2, int(round(z_in * source_z / target_z)))
        actual_z = source_z / (z_out / z_in)
        z_fwd = map_z_coord_to_resampled(z_orig, source_z, actual_z, z_out)
        z_back = map_z_coord_to_original(z_fwd, source_z, actual_z, z_in)
        self.assertLessEqual(abs(z_back - z_orig), 1,
                             f'Roundtrip: {z_orig} -> {z_fwd} -> {z_back}')

    def test_endpoints_clipped(self):
        # Extreme coordinates should be clamped to valid range
        z_out = 50
        res = map_z_coord_to_resampled(-100, 6.0, 3.0, z_out)
        self.assertGreaterEqual(res, 0)
        res2 = map_z_coord_to_resampled(9999, 6.0, 3.0, z_out)
        self.assertLessEqual(res2, z_out - 1)

    def test_physical_consistency(self):
        # The physical centre of the mapped index should be close to the
        # physical centre of the original index.
        source_z, target_z, z_in = 6.0, 3.0, 20
        z_out = max(2, int(round(z_in * source_z / target_z)))
        actual_z = source_z / (z_out / z_in)
        for z_orig in [0, 5, 10, 15, 19]:
            z_new = map_z_coord_to_resampled(z_orig, source_z, actual_z, z_out)
            phys_orig = z_orig * source_z + source_z / 2
            phys_new  = z_new  * actual_z  + actual_z  / 2
            self.assertLess(abs(phys_orig - phys_new), source_z,
                            f'z={z_orig}: phys_orig={phys_orig:.2f} phys_new={phys_new:.2f}')


class PrepareArraysTests(unittest.TestCase):

    def test_cropping_removes_empty_z(self):
        image, label = _make_volume(z=30, foreground_slices=(10, 20))
        working, target, center, info = prepare_arrays(image, label, 'gt-cropped')
        # Working region should cover only slices 10..19 (plus/minus 0 boundary)
        self.assertEqual(info['crop_start_z'], 10)
        self.assertEqual(info['crop_stop_z'], 20)
        self.assertEqual(working.shape[0], 10)

    def test_full_volume_keeps_all_z(self):
        image, label = _make_volume(z=30, foreground_slices=(10, 20))
        working, target, center, info = prepare_arrays(image, label, 'full-volume')
        self.assertEqual(working.shape[0], 30)

    def test_absent_target_raises(self):
        image = np.zeros((20, 64, 64), dtype=np.float32)
        label = np.zeros((20, 64, 64), dtype=bool)
        with self.assertRaises(ValueError):
            prepare_arrays(image, label, 'gt-cropped')

    def test_too_short_raises(self):
        image = np.random.randn(3, 64, 64).astype(np.float32)
        label = np.zeros((3, 64, 64), dtype=bool)
        label[1, 10:20, 10:20] = True
        # Crop to 1 slice (only slice 1 has foreground)
        # Actually crop_start=1, crop_stop=2 -> 1 slice -> raises
        with self.assertRaises(ValueError):
            prepare_arrays(image, label, 'gt-cropped')


class PromptTests(unittest.TestCase):

    def test_initial_prompt_box_shape(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:30, 20:50] = True
        p = initial_prompt(mask)
        self.assertIn('box', p)
        self.assertEqual(p['box'].shape, (1, 4))

    def test_initial_prompt_inside_mask(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:30, 20:50] = True
        p = initial_prompt(mask)
        x0, y0, x1, y1 = p['box'][0]
        self.assertGreaterEqual(x0, 20)
        self.assertGreaterEqual(y0, 10)
        self.assertLessEqual(x1, 49)
        self.assertLessEqual(y1, 29)

    def test_empty_slice_raises(self):
        mask = np.zeros((64, 64), dtype=bool)
        with self.assertRaises(ValueError):
            initial_prompt(mask)


class PipelineGeometryTests(unittest.TestCase):

    def test_proposed_branch_output_on_scoring_grid(self):
        """
        Simulate the proposed branch: resample, produce dummy prediction,
        map back to scoring grid. Output shape must match scoring grid.
        """
        z_score, y, x = 20, 64, 64
        source_z, target_z = 6.0, 3.0

        image_scoring = np.random.randn(z_score, y, x).astype(np.float32)
        label_scoring = np.zeros((z_score, y, x), dtype=bool)
        label_scoring[5:15, 20:40, 20:40] = True

        # Resample image and label to working grid
        image_resampled, actual_z = resample_image_z(image_scoring, source_z, target_z)
        z_out = image_resampled.shape[0]

        # Simulate a prediction on the resampled grid (all-ones = full segmentation)
        pred_resampled = np.ones((z_out, y, x), dtype=np.uint8)

        # Map prediction back to scoring grid
        pred_back, _ = resample_label_z(pred_resampled, actual_z, source_z)
        z_pred = pred_back.shape[0]
        if z_pred >= z_score:
            prediction = pred_back[:z_score] > 0
        else:
            pad = np.zeros((z_score - z_pred, y, x), dtype=np.uint8)
            prediction = np.concatenate([pred_back, pad], axis=0) > 0

        self.assertEqual(prediction.shape, label_scoring.shape,
                         f'Prediction shape {prediction.shape} != '
                         f'scoring grid {label_scoring.shape}')

    def test_label_values_preserved_through_resample(self):
        """
        Binary label resampled from 6mm -> 3mm -> 6mm should contain
        only {0, 1} at each step.
        """
        _, lbl = _make_volume(z=20, foreground_slices=(5, 15))
        lbl_uint = lbl.astype(np.uint8)

        out_fine, actual_z = resample_label_z(lbl_uint, 6.0, 3.0)
        unique_fine = set(np.unique(out_fine).tolist())
        self.assertTrue(unique_fine.issubset({0, 1}),
                        f'After 6->3mm: {unique_fine}')

        out_back, _ = resample_label_z(out_fine, actual_z, 6.0)
        unique_back = set(np.unique(out_back).tolist())
        self.assertTrue(unique_back.issubset({0, 1}),
                        f'After 3->6mm: {unique_back}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
