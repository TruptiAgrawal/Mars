"""Isolated box-padding evaluator. Default is manifest preflight, NOT inference."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
from .helpers import checked_manifests, fingerprint, initial_prompt, segmentation_metrics
from .loader import case_paths, read_case


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('datasets'))
    p.add_argument('--manifest', type=Path, required=True, help='Exact evaluation manifest (old val or fold test).')
    p.add_argument('--train-manifest', type=Path, required=True, help='Training manifest for the supplied checkpoint.')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint-kind', choices=['finetuned','official'], required=True)
    p.add_argument('--label-idx', type=int, required=True)
    p.add_argument('--prompt', choices=['box','point'], default='box')
    p.add_argument('--protocol', choices=['gt-cropped','full-volume'], required=True)
    p.add_argument('--padding-mode', choices=['historical','none','image'], required=True)
    p.add_argument('--padding-fraction', type=float, default=0.0,
                   help='Fraction of image dimensions per side. Use 0.05 with image mode.')
    p.add_argument('--nsd-tolerance-mm', type=float, required=True)
    p.add_argument('--output', type=Path, required=True, help='A new directory; existing outputs are never overwritten.')
    p.add_argument('--save-predictions', action='store_true')
    p.add_argument('--run-inference', action='store_true', help='Explicitly load model and use the GPU; omit for preflight.')
    return p


def preflight(args):
    if not np.isfinite(args.padding_fraction) or not 0 <= args.padding_fraction <= 1:
        raise ValueError('Padding fraction must be finite and in [0,1].')
    if args.padding_mode != 'image' and args.padding_fraction != 0:
        raise ValueError('Only image mode accepts a nonzero fraction.')
    if not np.isfinite(args.nsd_tolerance_mm) or args.nsd_tolerance_mm <= 0:
        raise ValueError('NSD tolerance must be positive millimeters.')
    if not 1 <= args.label_idx <= 13:
        raise ValueError('BTCV label must be in [1,13]; 2 is right kidney, 6 is liver.')
    training, evaluation = checked_manifests(args.train_manifest,args.manifest)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    for vid in evaluation:
        for path in case_paths(args.data_root,vid):
            if not path.is_file():
                raise FileNotFoundError(path)
    return training, evaluation


def save_json(path, data):
    # Each output directory is exclusive; replace only our own progress file.
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def main(argv=None):
    args = build_parser().parse_args(argv)
    training, evaluation = preflight(args)
    print(json.dumps({'preflight':'passed','training_count':len(training),
                      'evaluation_ids':evaluation,'protocol':args.protocol,
                      'padding_mode':args.padding_mode,'padding_fraction':args.padding_fraction,
                      'gpu_inference_requested':args.run_inference},indent=2))
    if not args.run_inference:
        print('Preflight only: no model imports, checkpoint deserialization, volume reads or GPU work.')
        return

    # Import model dependencies only after an explicit inference request.
    import torch
    from core.learner_sub1 import SamLearner
    from modeling.build_sam3d2 import sam_model_registry
    from .data_engine import DataManager
    from .predictor import VolumePredictor
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Preflight/CPU tests can still run without GPU.')
    args.output.mkdir(parents=True,exist_ok=False)
    stat = args.checkpoint.stat()
    report = {'started_utc':datetime.now(timezone.utc).isoformat(), 'status':'running',
              'arguments':{k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'training_ids':training, 'evaluation_ids':evaluation,
              'manifest_sha256':fingerprint(args.manifest),
              'training_manifest_sha256':fingerprint(args.train_manifest),
              'checkpoint_bytes':stat.st_size,'checkpoint_mtime_ns':stat.st_mtime_ns,
              'checkpoint_sha256':fingerprint(args.checkpoint),
              'experiment_source_sha256':{p.name:fingerprint(p) for p in Path(__file__).parent.glob('*.py')},
              'warning':'Ground-truth-assisted initial prompt; gt-cropped additionally uses target extent. Historical padding is within the corrected evaluator, not an exact legacy-score reproduction.',
              'cases':[]}
    report_path = args.output/'results.json'
    save_json(report_path,report)
    try:
        sam = sam_model_registry['vit_b'](checkpoint=None)
        learner = SamLearner(sam_model=sam,config={},data_engine=DataManager(img_size=(1024,1024)))
        learner.use_lora()
        if args.checkpoint_kind == 'finetuned':
            learner.use_lora_sub()
        learner.load_well_trained_model(str(args.checkpoint))
        learner.cuda().eval()
        predictor = VolumePredictor(learner.model,use_postprocess=True,use_noise_remove=True,
                                    padding_mode=args.padding_mode,padding_fraction=args.padding_fraction)
        for vid in evaluation:
            image, target, center, spacing, info = read_case(args.data_root,vid,args.label_idx,args.protocol)
            prompt = initial_prompt(target[center],args.prompt)
            info['initial_prompt']={k:v.tolist() for k,v in prompt.items()}
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode():
                prediction, stability = predictor.predict_volume(image,template_slice_id=center,
                                                                 return_stability=True,**prompt)
            torch.cuda.synchronize()
            info['inference_seconds']=time.perf_counter()-started
            info['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated()
            info['metrics']=segmentation_metrics(prediction,target,spacing,args.nsd_tolerance_mm)
            info['prediction_status']=predictor.prediction_status
            info['propagation_prompts']={str(k):np.asarray(v).tolist() for k,v in predictor.all_prompts.items()}
            if args.save_predictions:
                np.savez_compressed(args.output/f'{vid}_label{args.label_idx}.npz',
                                    prediction=np.asarray(prediction,dtype=np.uint8),
                                    spacing_zyx=np.asarray(spacing),crop_start_z=info['crop_start_z'])
            report['cases'].append(info)
            save_json(report_path,report)
            print(vid,info['metrics'],flush=True)
        summary={}
        for key in ('dice','nsd','hd95_mm','foreground_recall','foreground_precision','false_positive_ml'):
            values=[c['metrics'][key] for c in report['cases'] if c['metrics'][key] is not None]
            summary[key]={'mean':float(np.mean(values)) if values else None,
                          'valid_count':len(values),'undefined_count':len(evaluation)-len(values)}
        report.update(status='complete',summary=summary,finished_utc=datetime.now(timezone.utc).isoformat())
        save_json(report_path,report)
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        save_json(report_path,report)
        raise


if __name__ == '__main__':
    main()
