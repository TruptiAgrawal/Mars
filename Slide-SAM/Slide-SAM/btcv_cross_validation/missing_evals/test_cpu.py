"""CPU checks for the copied five-box and synthetic box-mode evaluators.
No neural network is built and no checkpoint is read.

    PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
      /home/jupyter-team2pop/.conda/envs/slidesam/bin/python \
      -m unittest btcv_cross_validation.missing_evals.test_cpu -v
"""
import ast
import hashlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np

CV = Path(__file__).resolve().parents[1]
ROOT = CV.parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_enlarge(path):
    """Extract BoxPromptGenerator.enlarge from a source file without importing it."""
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BoxPromptGenerator')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'enlarge')
    ns = {'__package__': 'btcv_cross_validation.five_box_padding'}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
    return ns['enlarge']


class SourceIdentityTests(unittest.TestCase):
    def test_unchanged_copies_are_byte_identical(self):
        for name in ('__init__', 'predictor', 'data_engine', 'helpers', 'loader', 'author_mbox'):
            self.assertEqual(digest(CV / f'five_box_padding/{name}.py'), digest(CV / f'five_box/{name}.py'), name)
        self.assertEqual(digest(CV / 'five_box_padding/originals/evaluate.py'), digest(CV / 'five_box/evaluate.py'))
        for name in ('__init__', 'helpers', 'loader'):
            self.assertEqual(digest(CV / f'z_spacing_boxmode/{name}.py'), digest(CV / f'z_spacing/{name}.py'), name)
        self.assertEqual(digest(CV / 'z_spacing_boxmode/originals/evaluate_v3.py'),
                         digest(CV / 'z_spacing/evaluate_v3.py'))

    def test_reused_sources_unchanged_since_copy(self):
        recorded = {}
        for line in (CV / 'missing_evals/source_sha256_at_copy.txt').read_text().splitlines():
            h, path = line.split(maxsplit=1)
            recorded[path] = h
        self.assertTrue(recorded)
        for path, h in recorded.items():
            self.assertEqual(digest(CV / path), h, path)


class HistoricalArithmeticTests(unittest.TestCase):
    """Historical mode must be the ORIGINAL enlarge() arithmetic, bugs included."""

    def setUp(self):
        self.original = load_enlarge(ROOT / 'datasets/data_engine.py')
        self.copy = load_enlarge(CV / 'five_box_padding/data_engine.py')
        rng = np.random.default_rng(0)
        a = rng.integers(0, 1024, size=(2000, 2, 2))
        self.boxes = [(int(min(p[0])), int(min(p[1])), int(max(p[0])), int(max(p[1]))) for p in a]

    def bounds(self, mode, fraction=0.0):
        return type('Bounds', (), {'size': (1024, 1024), 'padding_mode': mode, 'padding_fraction': fraction})()

    def test_historical_reproduces_original_exactly(self):
        for box in self.boxes:
            self.assertEqual(self.copy(self.bounds('historical'), box), self.original(self.bounds('historical'), box))
        # Known legacy behaviour: x1/y1 shrink, y0 uses margin_x.
        self.assertEqual(self.copy(self.bounds('historical'), (100, 200, 300, 300)), (90, 190, 295, 295))

    def test_corrected_zero_is_tight_box(self):
        for box in self.boxes:
            self.assertEqual(self.copy(self.bounds('none'), box), tuple(float(v) for v in box))

    def test_image_padding_is_five_percent_of_image_per_side(self):
        got = self.copy(self.bounds('image', 0.05), (100, 200, 300, 300))
        np.testing.assert_allclose(got, [100 - 51.2, 200 - 51.2, 300 + 51.2, 300 + 51.2])
        np.testing.assert_allclose(self.copy(self.bounds('image', 0.05), (10, 10, 1020, 1020)), [0, 0, 1023, 1023])


class PredictorWiringTests(unittest.TestCase):
    """Padding reaches propagated boxes only, through the copied five-box predictor."""

    def test_modes_route_to_propagated_boxes(self):
        from types import SimpleNamespace
        import torch
        from btcv_cross_validation.five_box_padding.predictor import VolumePredictor
        from btcv_cross_validation.five_box_padding.helpers import expand_box
        model = SimpleNamespace(image_encoder=SimpleNamespace(img_size=1024), device=torch.device('cpu'))
        mask = np.zeros((1, 3, 20, 40), dtype=np.float32)
        mask[:, :, 4:12, 5:15] = 1
        previous = {'masks': mask, 'tags': np.array(['organ'])}
        boxes = {}
        for mode, p in (('historical', 0), ('none', 0), ('image', .05)):
            predictor = VolumePredictor(model, padding_mode=mode, padding_fraction=p)
            boxes[mode], _ = predictor.generate_prompts_from_previous_masks(previous, 'down')
        original = load_enlarge(ROOT / 'datasets/data_engine.py')
        legacy = original(type('B', (), {'size': (1024, 1024)})(), tuple(int(v) for v in boxes['none'][0]))
        np.testing.assert_array_equal(boxes['historical'][0], legacy)
        np.testing.assert_allclose(boxes['image'][0], expand_box(boxes['none'][0], (1024, 1024), .05))
        with self.assertRaises(ValueError):
            VolumePredictor(model, padding_mode='historical', padding_fraction=0.05)

    def test_initial_and_corrective_boxes_bypass_enlarge(self):
        src = (CV / 'five_box_padding/predictor.py').read_text()
        self.assertEqual(src.count('.enlarge('), 1)
        self.assertIn('def generate_prompts_from_previous_masks', src)
        mbox = (CV / 'five_box_padding/author_mbox.py').read_text()
        self.assertNotIn('enlarge', mbox)
        self.assertIn('BoxPromptGenerator(size=None).mask_to_bbox(label[template_slice_id])', mbox)


class FiveBoxCliTests(unittest.TestCase):
    def run_cli(self, *args):
        env = dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
        return subprocess.run([sys.executable, '-m', 'btcv_cross_validation.five_box_padding.evaluate', *args],
                              cwd=ROOT, env=env, capture_output=True, text=True)

    def test_preflight_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'fold1'
            r = self.run_cli('--fold', '1', '--padding-mode', 'historical', '--output', str(out))
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn('PREFLIGHT 1', r.stdout)
            self.assertIn(str(ROOT / 'btcv_cross_validation/runs/fold1'), r.stdout)
            self.assertFalse(out.exists())

    def test_invalid_mode_fraction_pairs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for args in (('--padding-mode', 'image'), ('--padding-mode', 'historical', '--padding-fraction', '0.05'),
                         ('--padding-mode', 'none', '--padding-fraction', '0.05')):
                r = self.run_cli('--fold', '1', *args, '--output', str(Path(tmp) / 'x'))
                self.assertNotEqual(r.returncode, 0, args)

    def test_existing_output_rejected(self):
        before = digest(CV / 'five_box/results/fold1/results.json')
        r = self.run_cli('--fold', '1', '--padding-mode', 'none', '--output', str(CV / 'five_box/results/fold1'))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('Preserve existing output', r.stderr)
        self.assertEqual(digest(CV / 'five_box/results/fold1/results.json'), before)


class RecordingPredictor:
    prediction_status = 'ok'
    all_prompts = {}

    def __init__(self):
        self.calls = []

    def predict_volume(self, image, template_slice_id, return_stability=False, **prompt):
        self.calls.append((image.shape, template_slice_id, {k: np.asarray(v).tolist() for k, v in prompt.items()}))
        return np.ones(image.shape, dtype=bool), None


class SyntheticBoxModeTests(unittest.TestCase):
    def setUp(self):
        from btcv_cross_validation.z_spacing_boxmode import evaluate as ev
        self.ev = ev
        self.target = np.zeros((18, 32, 32), dtype=bool)
        self.target[3:15, 8:24, 10:20] = True
        self.image = np.random.default_rng(0).normal(size=self.target.shape).astype(np.float32)
        self.center = 9

    def run_case(self, branch, predictor):
        return self.ev.run_volume(predictor, self.image, self.target, (6.0, 1.0, 1.0), (0.0, 0.0, 0.0),
                                  (1, 0, 0, 0, 1, 0, 0, 0, 1), 37, self.center, branch, 3.0, 5.5, 1.0)

    def test_box_modes(self):
        self.assertEqual(self.ev.BOX_MODES, {'historical': 'historical', 'corrected_tight': 'none'})
        parser = self.ev.build_parser()
        base = ['--synthetic-root', 'x', '--manifest', 'x', '--train-manifest', 'x', '--checkpoint', 'x',
                '--checkpoint-kind', 'finetuned', '--label-idx', '6', '--branch', 'control',
                '--nsd-tolerance-mm', '1', '--output', 'x']
        with self.assertRaises(SystemExit):
            parser.parse_args(base)  # box mode is mandatory
        self.assertEqual(parser.parse_args(base + ['--box-mode', 'historical']).box_mode, 'historical')

    def test_initial_prompt_is_independent_of_box_mode(self):
        # run_volume never sees the box mode: it only reaches propagation via the predictor.
        self.assertFalse({'box_mode', 'padding_mode'} & set(inspect.signature(self.ev.run_volume).parameters))
        from btcv_cross_validation.z_spacing_boxmode.helpers import initial_prompt
        p = RecordingPredictor()
        self.run_case('control', p)
        self.assertEqual(p.calls[0][1], self.center)
        self.assertEqual(p.calls[0][2], {k: np.asarray(v).tolist() for k, v in initial_prompt(self.target[self.center]).items()})

    def test_recovery_on_restores_exact_scoring_grid(self):
        p = RecordingPredictor()
        _, extra, pred = self.run_case('proposed', p)
        self.assertTrue(extra['resampling_applied'])
        self.assertEqual(p.calls[0][0][0], extra['resampled_shape_zyx'][0])
        self.assertAlmostEqual(extra['actual_z_mm'], 3.0)
        self.assertEqual(pred.shape, self.target.shape)
        self.assertAlmostEqual(extra['origin_xyz_working'][2], 37 * 6.0)

    def test_restore_mismatch_raises_instead_of_crop_or_pad(self):
        real = self.ev.resample_label_z

        def off_by_one(array, spacing, origin, direction, target_z):
            out, sp = real(array, spacing, origin, direction, target_z)
            return (np.concatenate([out, out[-1:]]) if target_z == 6.0 else out), sp

        with mock.patch.object(self.ev, 'resample_label_z', off_by_one):
            with self.assertRaises(ValueError):
                self.run_case('proposed', RecordingPredictor())
        src = inspect.getsource(self.ev.run_volume)
        self.assertNotIn('np.concatenate', src)
        self.assertNotIn('prediction_shape_mismatch', src)


if __name__ == '__main__':
    unittest.main(verbosity=2)
