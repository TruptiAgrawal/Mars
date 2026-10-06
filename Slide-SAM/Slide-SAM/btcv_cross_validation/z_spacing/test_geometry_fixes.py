"""
btcv_cross_validation/z_spacing/test_geometry_fixes.py

CPU tests for the two geometry fixes in evaluate_v3.py:

Fix 1 — _crop_origin: the working grid's physical z-origin must be
         origin_z_full + crop_start_z * z_spacing_mm.

Fix 2 — Exact scoring-grid restore: resample_label_z with the cropped
         origin and scoring spacing must return exactly the same z-size
         as the input scoring grid.

CropOriginTests
  test_axial_identity         - standard axial orientation, known shift
  test_zero_crop_unchanged    - crop_start_z=0 -> origin unchanged
  test_non_zero_crop_shifts_z - crop shifts z-component by expected amount
  test_non_axial_direction    - oblique direction: shift applied along z-column

ScoringGridRestoreTests
  test_restored_shape_exact            - roundtrip shape matches scoring z
  test_restored_shape_after_crop_origin - fix 1 + fix 2 together: shape exact
  test_no_prediction_shape_mismatch    - no mismatch key in extra_info

IntegrationTests
  test_control_branch_unaffected  - control branch ignores origin fix (no resample)
  test_proposed_uses_crop_origin  - proposed extra_info records cropped origin
"""
import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from btcv_cross_validation.z_spacing.evaluate_v3 import _crop_origin, run_volume
from btcv_cross_validation.z_spacing.helpers import (
    resample_image_z, resample_label_z,
    initial_prompt, segmentation_metrics,
    needs_resampling,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _axial_direction():
    return (1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0)


def _make_scoring_volume(z=20, y=32, x=32, foreground=(6, 16)):
    img = np.random.randn(z, y, x).astype(np.float32) * 100
    lbl = np.zeros((z, y, x), dtype=bool)
    for s in range(*foreground):
        lbl[s, 8:24, 8:24] = True
    return img, lbl


class DummyPredictor:
    """Returns a full-foreground prediction to avoid empty-mask issues."""
    prediction_status = 'ok'
    all_prompts = {}

    def predict_volume(self, image, template_slice_id, return_stability=False, **kwargs):
        pred = np.ones(image.shape, dtype=bool)
        self.all_prompts = {i: np.array([[0, 0, 100, 100]]) for i in range(image.shape[0])}
        return pred, None


# ── CropOriginTests ───────────────────────────────────────────────────────────

class CropOriginTests(unittest.TestCase):

    def test_axial_identity(self):
        """Standard axial: z shift = crop_start_z * z_spacing_mm."""
        origin = (10.0, 20.0, -50.0)
        z_mm   = 6.0
        crop   = 8
        result = _crop_origin(origin, crop, z_mm, _axial_direction())
        expected_z = -50.0 + 8 * 6.0  # = -2.0
        self.assertAlmostEqual(result[0], 10.0, places=6)
        self.assertAlmostEqual(result[1], 20.0, places=6)
        self.assertAlmostEqual(result[2], expected_z, places=6)

    def test_zero_crop_unchanged(self):
        """crop_start_z=0 must leave origin unchanged."""
        origin = (5.0, -3.0, 100.0)
        result = _crop_origin(origin, 0, 3.0, _axial_direction())
        self.assertAlmostEqual(result[0], origin[0], places=6)
        self.assertAlmostEqual(result[1], origin[1], places=6)
        self.assertAlmostEqual(result[2], origin[2], places=6)

    def test_non_zero_crop_shifts_z(self):
        """Shift magnitude = crop_start_z * z_spacing_mm."""
        origin = (0.0, 0.0, 0.0)
        crop, z_mm = 52, 6.0
        result = _crop_origin(origin, crop, z_mm, _axial_direction())
        self.assertAlmostEqual(result[2], crop * z_mm, places=5)
        self.assertAlmostEqual(result[0], 0.0, places=6)
        self.assertAlmostEqual(result[1], 0.0, places=6)

    def test_non_axial_direction(self):
        """Oblique orientation: shift applied along z-column of direction matrix.

        Direction matrix (row-major, 3x3):
            [[1, 0, 0],
             [0, 0, 1],
             [0,-1, 0]]
        z-column (col 2) = [0, 1, 0]  (y-component of the z direction vector).
        shift = crop * z_mm * z_col = 10*5*(0,1,0) = (0,50,0)
        """
        direction = (1.0, 0.0, 0.0,
                     0.0, 0.0, 1.0,
                     0.0, -1.0, 0.0)
        origin = (0.0, 0.0, 0.0)
        crop, z_mm = 10, 5.0
        result = _crop_origin(origin, crop, z_mm, direction)
        # z-column is (0, 1, 0) so shift = 10*5*(0,1,0) = (0,50,0)
        self.assertAlmostEqual(result[0], 0.0, places=5)
        self.assertAlmostEqual(result[1], 50.0, places=5)
        self.assertAlmostEqual(result[2], 0.0, places=5)


# ── ScoringGridRestoreTests ───────────────────────────────────────────────────

class ScoringGridRestoreTests(unittest.TestCase):

    def _resample_restore(self, z_score=20, source_z=6.0, target_z=3.0,
                          crop_start_z=10):
        """
        Simulate Fix 1 + Fix 2:
          - Compute cropped origin
          - Resample scoring array to fine spacing (forward)
          - Resample back to scoring spacing (restore)
          - Check shape
        """
        origin_full = (0.0, 0.0, 0.0)
        dirn = _axial_direction()
        sp_scoring = (source_z, 1.0, 1.0)

        origin_working = _crop_origin(origin_full, crop_start_z, source_z, dirn)

        img_scoring, lbl_scoring = _make_scoring_volume(z=z_score)
        lbl_uint = lbl_scoring.astype(np.uint8)

        # Forward: scoring -> fine
        lbl_fine, out_sp = resample_label_z(
            lbl_uint, sp_scoring, origin_working, dirn, target_z)

        # Restore: fine -> scoring (FIX 2: use origin_working)
        lbl_restored, _ = resample_label_z(
            lbl_fine, out_sp, origin_working, dirn, source_z)

        return lbl_restored.shape[0], z_score

    def test_restored_shape_exact(self):
        z_pred, z_score = self._resample_restore(z_score=20, crop_start_z=0)
        self.assertEqual(z_pred, z_score,
                         f'Restored z={z_pred} != scoring z={z_score}')

    def test_restored_shape_after_crop_origin(self):
        """Crop start > 0 must not affect the restored shape."""
        z_pred, z_score = self._resample_restore(z_score=18, crop_start_z=37)
        self.assertEqual(z_pred, z_score,
                         f'With crop_start_z=37: restored z={z_pred} != {z_score}')

    def test_different_crop_values(self):
        """Test several crop_start_z values to confirm invariance."""
        for crop in [0, 10, 37, 52, 100]:
            z_pred, z_score = self._resample_restore(z_score=20, crop_start_z=crop)
            self.assertEqual(z_pred, z_score,
                             f'crop_start_z={crop}: z_pred={z_pred} != {z_score}')

    def test_no_prediction_shape_mismatch(self):
        """
        run_volume proposed branch should not set prediction_shape_mismatch
        when both geometry fixes are applied.
        """
        z_score, y, x = 20, 32, 32
        source_z, target_z = 6.0, 3.0
        crop_start_z = 37
        origin_full = (0.0, 0.0, 0.0)
        dirn = _axial_direction()
        sp_scoring = (source_z, 1.0, 1.0)

        img_scoring, lbl_scoring = _make_scoring_volume(z=z_score, y=y, x=x)
        center = int(np.argmax(lbl_scoring.sum(axis=(1, 2))))

        predictor = DummyPredictor()

        metrics, extra, pred_arr = run_volume(
            predictor=predictor,
            image_working=img_scoring,
            target_scoring=lbl_scoring,
            spacing_scoring_zyx=sp_scoring,
            origin_xyz_full=origin_full,
            direction=dirn,
            crop_start_z=crop_start_z,
            center_scoring=center,
            branch='proposed',
            target_z_mm=target_z,
            trigger_mm=5.5,
            nsd_tolerance_mm=1.0,
        )

        self.assertNotIn(
            'prediction_shape_mismatch', extra,
            f'Unexpected shape mismatch: {extra.get("prediction_shape_mismatch")}')
        self.assertEqual(pred_arr.shape[0], z_score)


# ── IntegrationTests ──────────────────────────────────────────────────────────

class IntegrationTests(unittest.TestCase):

    def setUp(self):
        self.z, self.y, self.x = 18, 32, 32
        self.img, self.lbl = _make_scoring_volume(z=self.z, y=self.y, x=self.x)
        self.center = int(np.argmax(self.lbl.sum(axis=(1, 2))))
        self.sp = (6.0, 1.0, 1.0)
        self.origin = (0.0, 0.0, 0.0)
        self.dirn = _axial_direction()
        self.predictor = DummyPredictor()

    def _run(self, branch, crop_start_z=37):
        return run_volume(
            predictor=self.predictor,
            image_working=self.img,
            target_scoring=self.lbl,
            spacing_scoring_zyx=self.sp,
            origin_xyz_full=self.origin,
            direction=self.dirn,
            crop_start_z=crop_start_z,
            center_scoring=self.center,
            branch=branch,
            target_z_mm=3.0,
            trigger_mm=5.5,
            nsd_tolerance_mm=1.0,
        )

    def test_control_branch_prediction_shape(self):
        """Control branch: prediction shape must equal scoring grid."""
        _, _, pred_arr = self._run('control')
        self.assertEqual(pred_arr.shape[0], self.z)

    def test_proposed_branch_records_cropped_origin(self):
        """Proposed branch must record origin_xyz_working in extra_info."""
        _, extra, _ = self._run('proposed')
        self.assertIn('origin_xyz_working', extra)
        # With crop_start_z=37, z_mm=6.0, origin_z shifts by 37*6=222 mm
        oz_working = extra['origin_xyz_working'][2]
        self.assertAlmostEqual(oz_working, 37 * 6.0, places=4)

    def test_proposed_branch_no_shape_mismatch(self):
        """Proposed branch must not produce a shape mismatch."""
        _, extra, pred_arr = self._run('proposed')
        self.assertNotIn('prediction_shape_mismatch', extra)
        self.assertEqual(pred_arr.shape[0], self.z)

    def test_control_origin_fix_has_no_effect(self):
        """
        Control branch does not resample, so crop origin is computed but
        not used for resampling.  Scores must be identical regardless.
        """
        _, extra0, pred0 = self._run('control', crop_start_z=0)
        _, extra1, pred1 = self._run('control', crop_start_z=37)
        # Both should produce identical boolean predictions (same model call)
        np.testing.assert_array_equal(pred0, pred1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
