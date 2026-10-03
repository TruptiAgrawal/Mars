"""
btcv_cross_validation/z_spacing/evaluate.py

Isolated z-spacing evaluator. Default is preflight only — NOT inference.

Two branches, both receiving the SAME synthetic thick-spacing input:

  control  : segment directly with corrected zero padding (padding-mode none)
  proposed : resample z toward target_z_mm, segment, map predictions back to
             the scoring grid (the original thick-spacing grid)

Scoring is always on the declared thick-spacing grid for both branches.
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
    map_z_coord_to_resampled, map_z_coord_to_original,
    prepare_arrays,
)
from .loader import read_synthetic_case, case_paths


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--synthetic-root', type=Path, required=True,
                   help='Directory containing synthetic thick-spacing img/ and label/.')
    p.add_argument('--manifest', type=Path, required=True,
                   help='Manifest listing the synthetic volume IDs to evaluate.')
    p.add_argument('--train-manifest', type=Path, required=True,
                   help='Training manifest for the supplied checkpoint.')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint-kind', choices=['finetuned', 'official'], required=True)
    p.add_argument('--label-idx', type=int, required=True)
    p.add_argument('--protocol', choices=['gt-cropped', 'full-volume'], default='gt-cropped')
    p.add_argument('--branch', choices=['control', 'proposed'], required=True,
                   help='control = segment directly; proposed = resample then segment.')
    p.add_argument('--target-z-mm', type=float, default=3.0,
                   help='Target z spacing (mm) for the proposed branch. Ignored by control.')
    p.add_argument('--nsd-tolerance-mm', type=float, required=True)
    p.add_argument('--output', type=Path, required=True,
                   help='New directory; existing outputs are never overwritten.')
    p.add_argument('--save-predictions', action='store_true')
    p.add_argument('--run-inference', action='store_true',
                   help='Load model and run GPU inference. Omit for preflight only.')
    return p


def preflight(args):
    if not np.isfinite(args.nsd_tolerance_mm) or args.nsd_tolerance_mm <= 0:
        raise ValueError('NSD tolerance must be positive millimetres.')
    if not 1 <= args.label_idx <= 13:
        raise ValueError('BTCV label must be in [1,13].')
    if args.branch == 'proposed':
        if not np.isfinite(args.target_z_mm) or args.target_z_mm <= 0:
            raise ValueError('target-z-mm must be a positive float.')
    training, evaluation = checked_manifests(args.train_manifest, args.manifest)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {args.output}')
    for vid in evaluation:
        for path in case_paths(args.synthetic_root, vid):
            if not path.is_file():
                raise FileNotFoundError(
                    f'Synthetic volume not found: {path}. '
                    f'Run make_synthetic.py first.'
                )
    return training, evaluation


def save_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def run_volume(predictor, image_working, target_scoring, spacing_scoring_zyx,
               center_scoring, branch, target_z_mm, nsd_tolerance_mm):
    """
    Run one volume through the chosen branch.

    image_working    : (Z, Y, X) float32 — already clipped/cropped
    target_scoring   : (Z, Y, X) bool    — scoring grid (thick-spacing)
    spacing_scoring_zyx : (z_mm, y_mm, x_mm) of the scoring grid
    center_scoring   : int — initial-prompt slice on the scoring grid
    branch           : 'control' or 'proposed'
    target_z_mm      : desired z spacing for the proposed branch

    Returns
    -------
    metrics    : dict
    extra_info : dict  (geometry/resampling provenance)
    """
    source_z_mm = float(spacing_scoring_zyx[0])
    extra = {'branch': branch, 'source_z_mm': source_z_mm}

    if branch == 'control':
        # Segment directly on the thick-spacing working grid.
        prompt = initial_prompt(target_scoring[center_scoring])
        prediction, _ = predictor.predict_volume(
            image_working, template_slice_id=center_scoring,
            return_stability=True, **prompt)
        extra['resampling_applied'] = False

    elif branch == 'proposed':
        # 1. Resample image to target z spacing.
        image_resampled, actual_z_mm = resample_image_z(
            image_working, source_z_mm, target_z_mm)
        extra['target_z_mm'] = target_z_mm
        extra['actual_z_mm_after_resample'] = actual_z_mm
        extra['resampled_shape_zyx'] = list(image_resampled.shape)

        # 2. Map center slice from scoring grid to resampled grid.
        z_out = image_resampled.shape[0]
        center_resampled = map_z_coord_to_resampled(
            center_scoring, source_z_mm, actual_z_mm, z_out)
        extra['center_scoring_grid'] = center_scoring
        extra['center_resampled_grid'] = center_resampled

        # 3. Build prompt on the resampled grid using the SAME physical
        #    location — map the scoring-grid label to the resampled grid
        #    to find the foreground on the nearest resampled slice.
        label_resampled, _ = resample_label_z(
            target_scoring, source_z_mm, actual_z_mm)
        # Use nearest valid foreground center on the resampled grid
        areas_r = label_resampled.sum(axis=(1, 2))
        valid_r = np.flatnonzero(areas_r)
        valid_r = valid_r[(valid_r >= 1) & (valid_r < z_out - 1)]
        if not len(valid_r):
            raise ValueError('No valid foreground center in resampled grid.')
        center_r = int(valid_r[np.argmin(abs(valid_r - center_resampled))])
        extra['center_resampled_actual'] = center_r
        prompt = initial_prompt(label_resampled[center_r])

        # 4. Segment on the resampled grid.
        prediction_resampled, _ = predictor.predict_volume(
            image_resampled, template_slice_id=center_r,
            return_stability=True, **prompt)

        # 5. Map prediction back to scoring grid using nearest-neighbour.
        pred_label, _ = resample_label_z(
            prediction_resampled.astype(np.uint8), actual_z_mm, source_z_mm)
        # Crop/pad to match scoring grid size exactly.
        z_score = target_scoring.shape[0]
        z_pred = pred_label.shape[0]
        if z_pred >= z_score:
            prediction = pred_label[:z_score] > 0
        else:
            pad = np.zeros((z_score - z_pred, *target_scoring.shape[1:]), dtype=np.uint8)
            prediction = np.concatenate([pred_label, pad], axis=0) > 0
        extra['resampling_applied'] = True

    else:
        raise ValueError(f'Unknown branch: {branch!r}')

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
        'gpu_inference_requested': args.run_inference,
    }, indent=2))

    if not args.run_inference:
        print('Preflight only: no model imports, checkpoint reads, volume reads or GPU work.')
        return

    # Import GPU dependencies only after an explicit inference request.
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
            'Scoring on declared synthetic thick-spacing grid for both branches. '
            'Proposed branch uses linear image resampling and nearest-neighbour '
            'label/prediction resampling. No fine-spacing original information used.'
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

        # Always use corrected zero padding (none) for both branches.
        predictor = VolumePredictor(
            learner.model,
            use_postprocess=True,
            use_noise_remove=True,
            padding_mode='none',
            padding_fraction=0.0,
        )

        for vid in evaluation:
            image_w, target_s, center_s, spacing_s, info = read_synthetic_case(
                args.synthetic_root, vid, args.label_idx, args.protocol)

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()

            with torch.inference_mode():
                metrics, extra, pred_arr = run_volume(
                    predictor, image_w, target_s, spacing_s,
                    center_s, args.branch, args.target_z_mm,
                    args.nsd_tolerance_mm)

            torch.cuda.synchronize()
            info['inference_seconds'] = time.perf_counter() - t0
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

        # Summary
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
