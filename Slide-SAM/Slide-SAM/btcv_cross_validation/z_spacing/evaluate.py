"""
btcv_cross_validation/z_spacing/evaluate.py  (v2 — geometry corrected)

Changes from v1:
  - All resampling uses SimpleITK with explicit origin/direction/spacing/size.
  - Prediction restoration uses the same reference grid as the forward
    resample — no manual crop/pad needed.
  - Prompt slice is mapped through physical voxel-centre coordinates
    (z_idx * spacing, shared origin cancels).
  - The >5.5 mm conditional trigger is implemented: control and proposed
    branches receive identical treatment when spacing <= 5.5 mm.
  - Per-case runtime recorded separately for resampling and inference.
  - 16 total predictions (2 patients x 4 organs x 2 branches) are expected.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
from .helpers import (
    checked_manifests, fingerprint,
    initial_prompt, segmentation_metrics,
    resample_image_z, resample_label_z,
    map_slice_to_resampled,
    prepare_arrays, needs_resampling, TRIGGER_MM,
)
from .loader import read_synthetic_case, case_paths


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--synthetic-root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--train-manifest', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint-kind', choices=['finetuned', 'official'], required=True)
    p.add_argument('--label-idx', type=int, required=True)
    p.add_argument('--protocol', choices=['gt-cropped', 'full-volume'], default='gt-cropped')
    p.add_argument('--branch', choices=['control', 'proposed'], required=True)
    p.add_argument('--target-z-mm', type=float, default=3.0,
                   help='Target z spacing for proposed branch (default 3.0 mm).')
    p.add_argument('--trigger-mm', type=float, default=TRIGGER_MM,
                   help=f'Resample only when z > this value (default {TRIGGER_MM} mm).')
    p.add_argument('--nsd-tolerance-mm', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--save-predictions', action='store_true')
    p.add_argument('--run-inference', action='store_true')
    return p


def preflight(args):
    if not np.isfinite(args.nsd_tolerance_mm) or args.nsd_tolerance_mm <= 0:
        raise ValueError('NSD tolerance must be positive.')
    if not 1 <= args.label_idx <= 13:
        raise ValueError('Label index must be in [1,13].')
    if args.branch == 'proposed':
        if not np.isfinite(args.target_z_mm) or args.target_z_mm <= 0:
            raise ValueError('target-z-mm must be positive.')
    training, evaluation = checked_manifests(args.train_manifest, args.manifest)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite: {args.output}')
    for vid in evaluation:
        for path in case_paths(args.synthetic_root, vid):
            if not path.is_file():
                raise FileNotFoundError(
                    f'Synthetic volume not found: {path}. Run make_synthetic.py first.')
    return training, evaluation


def save_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def run_volume(predictor, image_working, target_scoring,
               spacing_scoring_zyx, origin_xyz, direction,
               center_scoring, branch, target_z_mm, trigger_mm,
               nsd_tolerance_mm):
    """
    Run one volume through the chosen branch.

    Both branches score on the thick-spacing (scoring) grid.
    The proposed branch resamples only when spacing_scoring_zyx[0] > trigger_mm.

    Returns
    -------
    metrics    : dict
    extra_info : dict
    pred_arr   : np.ndarray uint8, shape == target_scoring.shape
    """
    source_z_mm = float(spacing_scoring_zyx[0])
    extra = {
        'branch': branch,
        'source_z_mm': source_z_mm,
        'trigger_mm': trigger_mm,
        'resampling_triggered': False,
    }

    if branch == 'control' or not needs_resampling(source_z_mm, trigger_mm):
        # Segment directly on the thick-spacing grid.
        prompt = initial_prompt(target_scoring[center_scoring])
        t_infer = time.perf_counter()
        prediction, _ = predictor.predict_volume(
            image_working, template_slice_id=center_scoring,
            return_stability=True, **prompt)
        extra['inference_seconds'] = time.perf_counter() - t_infer
        extra['resampling_applied'] = False
        if branch == 'proposed':
            extra['resampling_triggered'] = False
            extra['resampling_skipped_reason'] = (
                f'z spacing {source_z_mm:.3f} mm <= trigger {trigger_mm} mm')

    else:
        # Proposed branch with trigger activated.
        extra['resampling_triggered'] = True

        # 1. Resample image to target z spacing using SimpleITK.
        t_resample = time.perf_counter()
        image_resampled, out_spacing_zyx = resample_image_z(
            image_working, spacing_scoring_zyx, origin_xyz, direction, target_z_mm)
        extra['resample_seconds'] = time.perf_counter() - t_resample
        extra['target_z_mm'] = target_z_mm
        extra['actual_z_mm'] = out_spacing_zyx[0]
        extra['resampled_shape_zyx'] = list(image_resampled.shape)

        # 2. Map center slice to resampled grid via physical voxel centres.
        z_cont = map_slice_to_resampled(center_scoring, spacing_scoring_zyx, out_spacing_zyx)
        z_out = image_resampled.shape[0]

        # 3. Resample label to resampled grid to find valid prompt center.
        label_resampled, _ = resample_label_z(
            target_scoring.astype(np.uint8), spacing_scoring_zyx,
            origin_xyz, direction, target_z_mm)
        areas_r = label_resampled.sum(axis=(1, 2))
        valid_r = np.flatnonzero(areas_r)
        valid_r = valid_r[(valid_r >= 1) & (valid_r < z_out - 1)]
        if not len(valid_r):
            raise ValueError('No valid foreground center in resampled grid.')
        center_r = int(valid_r[np.argmin(abs(valid_r - z_cont))])
        extra['center_scoring_grid'] = center_scoring
        extra['center_continuous_resampled'] = float(z_cont)
        extra['center_resampled_actual'] = center_r
        extra['center_physical_displacement_mm'] = abs(
            center_scoring * source_z_mm - center_r * out_spacing_zyx[0])

        prompt = initial_prompt(label_resampled[center_r])

        # 4. Segment on resampled grid.
        t_infer = time.perf_counter()
        prediction_resampled, _ = predictor.predict_volume(
            image_resampled, template_slice_id=center_r,
            return_stability=True, **prompt)
        extra['inference_seconds'] = time.perf_counter() - t_infer

        # 5. Map prediction back to scoring grid using the same reference grid
        #    as the forward resample (SimpleITK, nearest-neighbour).
        #    Output size is exactly target_scoring.shape[0] — no crop/pad needed.
        pred_uint8 = prediction_resampled.astype(np.uint8)
        # Build a reference grid matching the scoring volume's z extent.
        pred_restored, _ = resample_label_z(
            pred_uint8, out_spacing_zyx, origin_xyz, direction, source_z_mm)

        # The restored shape should match the scoring grid exactly because
        # _output_size_z is symmetric. Assert and trim/pad only as a safety net.
        z_score = target_scoring.shape[0]
        z_pred = pred_restored.shape[0]
        if z_pred == z_score:
            prediction = pred_restored > 0
        elif z_pred > z_score:
            prediction = pred_restored[:z_score] > 0
        else:
            # Should not happen with correct SimpleITK grids; record if it does.
            pad = np.zeros((z_score - z_pred, *target_scoring.shape[1:]), dtype=np.uint8)
            prediction = np.concatenate([pred_restored, pad], axis=0) > 0
            extra['prediction_shape_warning'] = (
                f'restored {z_pred} != scoring {z_score}; zero-padded')

        extra['resampling_applied'] = True
        extra['restored_shape_zyx'] = list(prediction.shape)

    metrics = segmentation_metrics(
        prediction, target_scoring, spacing_scoring_zyx, nsd_tolerance_mm)
    return metrics, extra, np.asarray(prediction, dtype=np.uint8)


def main(argv=None):
    args = build_parser().parse_args(argv)
    training, evaluation = preflight(args)

    print(json.dumps({
        'preflight': 'passed',
        'training_count': len(training),
        'evaluation_ids': evaluation,
        'branch': args.branch,
        'protocol': args.protocol,
        'target_z_mm': args.target_z_mm if args.branch == 'proposed' else None,
        'trigger_mm': args.trigger_mm,
        'gpu_inference_requested': args.run_inference,
    }, indent=2))

    if not args.run_inference:
        print('Preflight only: no model imports, checkpoint reads, volume reads or GPU work.')
        return

    import torch
    from core.learner_sub1 import SamLearner
    from modeling.build_sam3d2 import sam_model_registry
    from btcv_cross_validation.box_padding.data_engine import DataManager
    from btcv_cross_validation.box_padding.predictor import VolumePredictor

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable.')

    args.output.mkdir(parents=True, exist_ok=False)
    stat = args.checkpoint.stat()

    report = {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'running',
        'geometry_version': 'v2_simpleitk',
        'arguments': {k: str(v.resolve()) if isinstance(v, Path) else v
                      for k, v in vars(args).items()},
        'training_ids': training,
        'evaluation_ids': evaluation,
        'manifest_sha256': fingerprint(args.manifest),
        'training_manifest_sha256': fingerprint(args.train_manifest),
        'checkpoint_bytes': stat.st_size,
        'checkpoint_mtime_ns': stat.st_mtime_ns,
        'checkpoint_sha256': fingerprint(args.checkpoint),
        'experiment_source_sha256': {
            p.name: fingerprint(p)
            for p in Path(__file__).parent.glob('*.py')
        },
        'warning': (
            'v2: SimpleITK resampling with explicit reference grid. '
            'Scoring on synthetic thick-spacing grid for both branches. '
            'Proposed branch resamples only when z spacing > trigger_mm.'
        ),
        'cases': [],
    }
    report_path = args.output / 'results.json'
    save_json(report_path, report)

    try:
        sam = sam_model_registry['vit_b'](checkpoint=None)
        learner = SamLearner(
            sam_model=sam, config={},
            data_engine=DataManager(img_size=(1024, 1024)))
        learner.use_lora()
        if args.checkpoint_kind == 'finetuned':
            learner.use_lora_sub()
        learner.load_well_trained_model(str(args.checkpoint))
        learner.cuda().eval()

        predictor = VolumePredictor(
            learner.model,
            use_postprocess=True,
            use_noise_remove=True,
            padding_mode='none',
            padding_fraction=0.0,
        )

        for vid in evaluation:
            (image_w, target_s, center_s,
             spacing_s, origin_xyz, direction, info) = read_synthetic_case(
                args.synthetic_root, vid, args.label_idx, args.protocol)

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()

            with torch.inference_mode():
                metrics, extra, pred_arr = run_volume(
                    predictor, image_w, target_s,
                    spacing_s, origin_xyz, direction,
                    center_s, args.branch, args.target_z_mm,
                    args.trigger_mm, args.nsd_tolerance_mm)

            torch.cuda.synchronize()
            info['total_seconds'] = time.perf_counter() - t0
            info['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated()
            info['metrics'] = metrics
            info['resampling_info'] = extra
            info['prediction_status'] = predictor.prediction_status
            info['propagation_prompts'] = {
                str(k): np.asarray(v).tolist()
                for k, v in predictor.all_prompts.items()
            }

            if args.save_predictions:
                np.savez_compressed(
                    args.output / f'{vid}_label{args.label_idx}.npz',
                    prediction=pred_arr,
                    spacing_zyx=np.asarray(spacing_s),
                    crop_start_z=info['crop_start_z'],
                )

            report['cases'].append(info)
            save_json(report_path, report)
            print(vid, metrics, flush=True)

        summary = {}
        for key in ('dice', 'nsd', 'hd95_mm',
                    'foreground_recall', 'foreground_precision', 'false_positive_ml'):
            values = [c['metrics'][key] for c in report['cases']
                      if c['metrics'][key] is not None]
            summary[key] = {
                'mean': float(np.mean(values)) if values else None,
                'valid_count': len(values),
                'undefined_count': len(evaluation) - len(values),
            }
        report.update(
            status='complete',
            summary=summary,
            finished_utc=datetime.now(timezone.utc).isoformat(),
        )
        save_json(report_path, report)

    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save_json(report_path, report)
        raise


if __name__ == '__main__':
    main()
