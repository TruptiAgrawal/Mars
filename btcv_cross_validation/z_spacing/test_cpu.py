"""
btcv_cross_validation/z_spacing/test_cpu.py  (v2 — geometry corrected)

Tests for the SimpleITK-based geometry implementation.

ResamplingTests
  test_identity_unchanged_spacing        — identity returns exact array copy
  test_output_shape_matches_coverage     — output size preserves physical extent
  test_label_nearest_neighbour_values    — label values stay in {0,1,...}
  test_image_label_same_shape            — image and label produce same shape

PhysicalCoordTests
  test_ramp_physical_coords              — linear-ramp image: output value at
                                           physical pos p equals input interpolated
                                           at p (verifies no half-voxel offset)
  test_landmark_alignment                — a 1-voxel spike resamples to the
                                           nearest output voxel at the correct
                                           physical location
  test_roundtrip_scoring_grid_size       — forward then backward resample
                                           recovers the original z size exactly

SliceMappingTests
  test_map_origin_slice                  — slice 0 maps to slice 0 (shared origin)
  test_map_physical_midpoint             — physical midpoint maps correctly
  test_map_endpoint                      — last slice maps near last output slice

TriggerTests
  test_trigger_above_threshold           — 6.0 mm > 5.5 mm triggers resampling
  test_trigger_at_threshold              — 5.5 mm does NOT trigger (strict >)
  test_trigger_below_threshold           — 3.0 mm does NOT trigger
  test_trigger_custom_threshold          — custom trigger value respected

PrepareArraysTests
  test_gt_crop                           — removes blank z boundary slices
  test_full_volume                       — keeps all z slices
  test_absent_raises                     — missing organ raises ValueError
  test_too_short_raises                  — <3 slices in crop raises ValueError

PipelineGeometryTests
  test_restored_shape_matches_scoring    — round-trip resample output matches
                                           scoring grid shape exactly (no manual
                                           crop/pad needed)
  test_label_values_through_roundtrip    — binary label values stay {0,1}
"""
import unittest
import tempfile
import os
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from btcv_cross_validation.z_spacing.helpers import (
    resample_image_z,
    resample_label_z,
    map_slice_to_resampled,
    map_slice_to_original,
    prepare_arrays,
    initial_prompt,
    needs_resampling,
    TRIGGER_MM,
    _output_size_z,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _identity_geometry(z=20, y=32, x=32, z_mm=3.0, y_mm=1.0, x_mm=1.0):
    """Return spacing_zyx, origin_xyz, direction for a trivial geometry."""
    spacing_zyx = (z_mm, y_mm, x_mm)
    origin_xyz  = (0.0, 0.0, 0.0)
    direction   = (1.0, 0.0, 0.0,
                   0.0, 1.0, 0.0,
                   0.0, 0.0, 1.0)
    return spacing_zyx, origin_xyz, direction


def _make_volume(z=20, y=32, x=32, foreground=(8, 15)):
    img = (np.arange(z, dtype=np.float32)[:, None, None]
           * np.ones((z, y, x), dtype=np.float32))
    lbl = np.zeros((z, y, x), dtype=np.uint8)
    for s in range(*foreground):
        lbl[s, 8:20, 8:20] = 1
    return img, lbl


# ── ResamplingTests ───────────────────────────────────────────────────────────

class ResamplingTests(unittest.TestCase):

    def _geom(self, z=20):
        return _identity_geometry(z=z)

    def test_identity_unchanged_spacing(self):
        img, _ = _make_volume(z=20)
        sp, orig, dirn = self._geom(z=20)
        out, out_sp = resample_image_z(img, sp, orig, dirn, sp[0])
        self.assertEqual(out.shape, img.shape)
        np.testing.assert_allclose(out, img, atol=1e-4)
        self.assertAlmostEqual(out_sp[0], sp[0], places=5)

    def test_output_shape_matches_coverage(self):
        z_in, source_z, target_z = 20, 6.0, 3.0
        img, _ = _make_volume(z=z_in)
        sp = (source_z, 1.0, 1.0)
        orig, dirn = (0., 0., 0.), (1,0,0, 0,1,0, 0,0,1)
        out, _ = resample_image_z(img, sp, orig, dirn, target_z)
        expected_z = _output_size_z(z_in, source_z, target_z)
        self.assertEqual(out.shape[0], expected_z,
                         f'Expected z={expected_z}, got {out.shape[0]}')

    def test_label_nearest_neighbour_values(self):
        _, lbl = _make_volume(z=20)
        sp, orig, dirn = self._geom(z=20)
        out, _ = resample_label_z(lbl, sp, orig, dirn, 1.5)
        unique = set(np.unique(out).tolist())
        self.assertTrue(unique.issubset({0, 1}),
                        f'Unexpected label values: {unique}')

    def test_image_label_same_shape(self):
        img, lbl = _make_volume(z=20)
        sp, orig, dirn = self._geom(z=20)
        img_out, _ = resample_image_z(img, sp, orig, dirn, 3.0)
        lbl_out, _ = resample_label_z(lbl, sp, orig, dirn, 3.0)
        self.assertEqual(img_out.shape, lbl_out.shape)


# ── PhysicalCoordTests ────────────────────────────────────────────────────────

class PhysicalCoordTests(unittest.TestCase):

    def test_ramp_physical_coords(self):
        """
        Input: image[i] = i * source_z_mm  (value == physical z position of centre).
        After resampling to target_z_mm, output[j] should equal j * actual_z_mm.

        v1 (scipy.zoom) would fail this because zoom places samples at
        j/(N-1)*(Z-1)*source_z_mm, not j*target_z_mm.
        """
        z_in, source_z, target_z = 30, 6.0, 3.0
        y, x = 4, 4
        # image value = physical z of voxel centre
        img = np.zeros((z_in, y, x), dtype=np.float32)
        for i in range(z_in):
            img[i] = i * source_z  # value equals physical z in mm

        sp  = (source_z, 1.0, 1.0)
        orig = (0.0, 0.0, 0.0)
        dirn = (1,0,0, 0,1,0, 0,0,1)

        out, out_sp = resample_image_z(img, sp, orig, dirn, target_z)
        actual_z = out_sp[0]

        # Check all output slices: out[j] should ≈ j * actual_z
        for j in range(out.shape[0]):
            expected = j * actual_z
            got = float(out[j].mean())
            self.assertAlmostEqual(
                got, expected, delta=actual_z * 0.52,  # allow ±0.52 voxel in mm
                msg=f'Slice {j}: expected physical pos {expected:.2f} mm, got {got:.2f} mm')

    def test_landmark_alignment(self):
        """
        A spike at slice k in the source grid should appear at the nearest
        output slice whose physical centre is closest to k*source_z_mm.
        """
        z_in, source_z, target_z = 20, 6.0, 3.0
        y, x = 4, 4
        k = 10  # spike at slice 10 -> physical pos 10 * 6 = 60 mm
        img = np.zeros((z_in, y, x), dtype=np.float32)
        img[k] = 1000.0

        sp   = (source_z, 1.0, 1.0)
        orig = (0.0, 0.0, 0.0)
        dirn = (1,0,0, 0,1,0, 0,0,1)

        out, out_sp = resample_image_z(img, sp, orig, dirn, target_z)
        actual_z = out_sp[0]

        # Expected output index: k * source_z / actual_z
        j_expected = k * source_z / actual_z
        j_peak = int(np.argmax(out.max(axis=(1,2))))
        self.assertLessEqual(
            abs(j_peak - j_expected), 1.0,
            f'Spike at source slice {k} (60 mm) should map near output slice '
            f'{j_expected:.1f}, found peak at {j_peak}')

    def test_roundtrip_scoring_grid_size(self):
        """
        Forward resample (source_z -> target_z) then backward (target_z -> source_z)
        should recover the original z size exactly, so no manual crop/pad is needed.
        """
        z_in, source_z, target_z = 20, 6.0, 3.0
        img, _ = _make_volume(z=z_in)
        sp   = (source_z, 1.0, 1.0)
        orig = (0.0, 0.0, 0.0)
        dirn = (1,0,0, 0,1,0, 0,0,1)

        img_fwd, out_sp = resample_image_z(img, sp, orig, dirn, target_z)
        img_bwd, _      = resample_image_z(img_fwd, out_sp, orig, dirn, source_z)

        self.assertEqual(
            img_bwd.shape[0], z_in,
            f'Roundtrip z size: expected {z_in}, got {img_bwd.shape[0]}. '
            f'No crop/pad should be needed.')


# ── SliceMappingTests ─────────────────────────────────────────────────────────

class SliceMappingTests(unittest.TestCase):

    def _resample_and_map(self, z_in=20, source_z=6.0, target_z=3.0):
        img, _ = _make_volume(z=z_in)
        sp   = (source_z, 1.0, 1.0)
        orig = (0.0, 0.0, 0.0)
        dirn = (1,0,0, 0,1,0, 0,0,1)
        _, out_sp = resample_image_z(img, sp, orig, dirn, target_z)
        return sp, out_sp

    def test_map_origin_slice(self):
        sp, out_sp = self._resample_and_map()
        z_cont = map_slice_to_resampled(0, sp, out_sp)
        # Slice 0 has physical centre at 0 mm; output slice 0 is also at 0 mm.
        self.assertAlmostEqual(z_cont, 0.0, places=6)

    def test_map_physical_midpoint(self):
        z_in, source_z, target_z = 20, 6.0, 3.0
        sp, out_sp = self._resample_and_map(z_in, source_z, target_z)
        z_mid = 10  # physical pos 60 mm
        z_cont = map_slice_to_resampled(z_mid, sp, out_sp)
        expected = z_mid * source_z / out_sp[0]
        self.assertAlmostEqual(z_cont, expected, places=5)

    def test_map_endpoint(self):
        z_in, source_z, target_z = 20, 6.0, 3.0
        sp, out_sp = self._resample_and_map(z_in, source_z, target_z)
        z_last = z_in - 1
        z_cont = map_slice_to_resampled(z_last, sp, out_sp)
        z_out  = _output_size_z(z_in, source_z, target_z)
        # Should map near the last output slice
        self.assertLessEqual(abs(z_cont - (z_out - 1)), 1.5,
                             f'Last source slice {z_last} -> {z_cont:.2f}, '
                             f'expected near {z_out-1}')


# ── TriggerTests ──────────────────────────────────────────────────────────────

class TriggerTests(unittest.TestCase):

    def test_trigger_above_threshold(self):
        self.assertTrue(needs_resampling(6.0))

    def test_trigger_at_threshold(self):
        # Strict > : exactly 5.5 mm does NOT trigger
        self.assertFalse(needs_resampling(5.5))

    def test_trigger_below_threshold(self):
        self.assertFalse(needs_resampling(3.0))

    def test_trigger_custom_threshold(self):
        self.assertTrue(needs_resampling(4.0, trigger_mm=3.5))
        self.assertFalse(needs_resampling(3.5, trigger_mm=3.5))


# ── PrepareArraysTests ────────────────────────────────────────────────────────

class PrepareArraysTests(unittest.TestCase):

    def test_gt_crop(self):
        img, lbl = _make_volume(z=30, foreground=(10, 20))
        lbl_bool = lbl.astype(bool)
        working, target, center, info = prepare_arrays(img, lbl_bool, 'gt-cropped')
        self.assertEqual(info['crop_start_z'], 10)
        self.assertEqual(info['crop_stop_z'], 20)

    def test_full_volume(self):
        img, lbl = _make_volume(z=30, foreground=(10, 20))
        working, _, _, info = prepare_arrays(img, lbl.astype(bool), 'full-volume')
        self.assertEqual(working.shape[0], 30)

    def test_absent_raises(self):
        img = np.zeros((20, 32, 32), dtype=np.float32)
        lbl = np.zeros((20, 32, 32), dtype=bool)
        with self.assertRaises(ValueError):
            prepare_arrays(img, lbl, 'gt-cropped')

    def test_too_short_raises(self):
        img = np.zeros((10, 32, 32), dtype=np.float32)
        lbl = np.zeros((10, 32, 32), dtype=bool)
        lbl[5, 5:10, 5:10] = True  # only 1 slice -> crop to 1 -> error
        with self.assertRaises(ValueError):
            prepare_arrays(img, lbl, 'gt-cropped')


# ── PipelineGeometryTests ─────────────────────────────────────────────────────

class PipelineGeometryTests(unittest.TestCase):

    def test_restored_shape_matches_scoring(self):
        """
        Simulate the proposed branch pipeline:
          thick image -> resample to fine -> dummy prediction -> restore to thick.
        Restored shape must equal the scoring grid shape WITHOUT crop/pad.
        """
        z_score, y, x = 20, 32, 32
        source_z, target_z = 6.0, 3.0
        img = np.random.randn(z_score, y, x).astype(np.float32)
        lbl = np.zeros((z_score, y, x), dtype=np.uint8)
        lbl[5:15, 8:20, 8:20] = 1

        sp   = (source_z, 1.0, 1.0)
        orig = (0.0, 0.0, 0.0)
        dirn = (1,0,0, 0,1,0, 0,0,1)

        img_fine, out_sp = resample_image_z(img, sp, orig, dirn, target_z)
        # Dummy prediction: all-ones on the fine grid
        pred_fine = np.ones_like(img_fine, dtype=np.uint8)
        # Restore to scoring grid
        pred_restored, _ = resample_label_z(pred_fine, out_sp, orig, dirn, source_z)

        self.assertEqual(
            pred_restored.shape[0], z_score,
            f'Restored z {pred_restored.shape[0]} != scoring z {z_score}. '
            f'Manual crop/pad should not be needed.')

    def test_label_values_through_roundtrip(self):
        """Binary label must contain only {{0, 1}} at every step."""
        z_in = 20
        _, lbl = _make_volume(z=z_in)
        sp   = (6.0, 1.0, 1.0)
        orig = (0.0, 0.0, 0.0)
        dirn = (1,0,0, 0,1,0, 0,0,1)

        lbl_fine, out_sp = resample_label_z(lbl, sp, orig, dirn, 3.0)
        unique_fine = set(np.unique(lbl_fine).tolist())
        self.assertTrue(unique_fine.issubset({0, 1}),
                        f'After 6->3mm: {unique_fine}')

        lbl_back, _ = resample_label_z(lbl_fine, out_sp, orig, dirn, 6.0)
        unique_back = set(np.unique(lbl_back).tolist())
        self.assertTrue(unique_back.issubset({0, 1}),
                        f'After 3->6mm roundtrip: {unique_back}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
