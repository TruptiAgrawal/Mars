"""Sequential launcher for the missing BTCV padding x prompt-budget evaluations
(native, section A) and the historical-box synthetic-spacing arms (section B).

Default is PREFLIGHT ONLY: provenance/cohort/geometry-input checks, evaluator
preflights on CPU, and an execution inventory. GPU inference starts only with
--execute. Runs are skipped only when their output validates as complete;
any other existing output stops the launcher and is left untouched.

    PY=/home/jupyter-team2pop/.conda/envs/slidesam/bin/python
    $PY btcv_cross_validation/missing_evals/run_missing_evals.py            # preflight
    $PY btcv_cross_validation/missing_evals/run_missing_evals.py --execute  # GPU
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CV = ROOT / 'btcv_cross_validation'
OLD_ROOT = Path('/home/jupyter-team2pop/Slide-SAM/Slide-SAM')
PYTHON = Path('/home/jupyter-team2pop/.conda/envs/slidesam/bin/python')
FOLDS = [1, 2, 3, 4, 5]
ORGANS = {6: 'Liver', 7: 'Stomach', 3: 'Left kidney', 1: 'Spleen'}
FRACTION = 0.05
NSD_TOL = 1.0
TARGET_Z, TRIGGER = 3.0, 5.5
# Fixed synthetic-spacing cohort (native 3 mm held-out patients), from the request.
SYNTH_COHORT = {1: ['img0001', 'img0002', 'img0003', 'img0004', 'img0005', 'img0006'],
                2: ['img0007', 'img0008', 'img0010', 'img0021'],
                3: ['img0024'],
                4: ['img0029', 'img0030', 'img0031', 'img0032'],
                5: ['img0036', 'img0040']}
SEEDED = ("import random,runpy,numpy as np,torch; random.seed(0); np.random.seed(0); "
          "torch.manual_seed(0); torch.backends.cudnn.enabled=False; "
          "torch.backends.cudnn.benchmark=False; "
          "runpy.run_module('{module}',run_name='__main__',alter_sys=True)")


class Mismatch(Exception):
    pass


def check(cond, message):
    if not cond:
        raise Mismatch(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def manifest_ids(path):
    return [Path(line.split()[0]).name.replace('.nii.gz', '')
            for line in Path(path).read_text().splitlines() if line.strip()]


def fold_paths(fold):
    lists = CV / 'dataset_list' / f'fold{fold}'
    return {'train': lists / '01_BTCV_train_4to1.txt', 'test': lists / '01_BTCV_test_4to1.txt',
            'checkpoint': CV / f'runs/fold{fold}/ddp_sub_cv/cv_fold{fold}/ckpt/model_epoch_20.pth',
            'run_config': CV / f'runs/fold{fold}/ddp_sub_cv/cv_fold{fold}/config.yaml',
            'legacy_eval': CV / f'logs/fold{fold}_evaluation.json'}


def valid_unit(x):
    return isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1


# ---------------------------------------------------------------- provenance

def verify_folds():
    """Per-fold manifests, disjointness, checkpoint path/hash agreement with reused runs."""
    folds = {}
    for fold in FOLDS:
        p = fold_paths(fold)
        train, test = manifest_ids(p['train']), manifest_ids(p['test'])
        check(len(test) == 6 and len(set(test)) == 6, f'fold{fold}: expected 6 held-out IDs, got {test}')
        check(len(train) == 24, f'fold{fold}: expected 24 training IDs, got {len(train)}')
        check(not set(train) & set(test), f'fold{fold}: train/test overlap')
        cfg = p['run_config'].read_text()
        check(f'dataset_list/fold{fold}/' in cfg, f'fold{fold}: run config does not use fold{fold} manifests')
        recorded = Path(json.loads(p['legacy_eval'].read_text())['checkpoint'])
        check(ROOT / recorded.relative_to(OLD_ROOT) == p['checkpoint'],
              f'fold{fold}: legacy checkpoint path {recorded} does not map to {p["checkpoint"]}')
        check(p['checkpoint'].is_file(), f'missing {p["checkpoint"]}')
        digest = sha(p['checkpoint'])
        ref = json.loads((CV / f'five_box/results/fold{fold}/results.json').read_text())
        check(ref['checkpoint_sha256'] == digest, f'fold{fold}: checkpoint hash differs from five_box run')
        for vid in test:
            for kind, name in (('img', f'{vid}.nii.gz'), ('label', f'label{vid[3:]}.nii.gz')):
                check((ROOT / f'datasets/01_BCV-Abdomen/Training/{kind}/{name}').is_file(), f'missing data {name}')
        folds[fold] = {'train': train, 'test': test, 'checkpoint': p['checkpoint'], 'sha256': digest}
    all_test = [v for f in folds.values() for v in f['test']]
    check(len(set(all_test)) == 30, 'held-out sets do not cover 30 distinct patients')
    return folds


def verify_synthetic_cohort(folds):
    """17-patient cohort: eligibility list, synthetic manifests, v3 results, hashes."""
    eligible = json.loads((CV / 'z_spacing/eligible_patients.json').read_text())
    three_mm = {}
    for e in eligible:
        if e['z_spacing_mm'] == 3.0:
            three_mm.setdefault(e['fold'], []).append(e['volume_id'])
    check(three_mm == SYNTH_COHORT, f'eligibility list (3 mm) {three_mm} != requested cohort')
    inputs = {}
    for fold, ids in SYNTH_COHORT.items():
        synth = CV / f'z_spacing/synthetic_6mm_v3/fold{fold}'
        check(manifest_ids(synth / 'synthetic_manifest.txt') == ids, f'fold{fold}: synthetic manifest IDs differ')
        check(set(ids) <= set(folds[fold]['test']) and not set(ids) & set(folds[fold]['train']),
              f'fold{fold}: synthetic cohort not within held-out set')
        meta = json.loads((synth / 'meta.json').read_text())
        for organ in ORGANS:
            v3 = {}
            for arm in ('b00_control', 'b01_proposed'):
                r = json.loads((CV / f'z_spacing/results_v3/fold{fold}/{arm}_label{organ}/results.json').read_text())
                check(r['status'] == 'complete' and [c['volume_id'] for c in r['cases']] == ids,
                      f'fold{fold} {arm} label{organ}: v3 result incomplete or wrong IDs')
                check(r['checkpoint_sha256'] == folds[fold]['sha256'], f'fold{fold}: v3 checkpoint hash differs')
                v3[arm] = {c['volume_id']: c for c in r['cases']}
            for vid in ids:
                reproduce_v3_geometry(synth, vid, organ, v3['b00_control'][vid], v3['b01_proposed'][vid])
        for vid in ids:
            check(meta[vid]['source_z_mm'] == 3.0 and abs(meta[vid]['actual_z_mm'] - 6.0) < 0.1,
                  f'{vid}: synthetic spacing metadata unexpected')
            for kind in ('img', 'label'):
                inputs[f'fold{fold}/{kind}/{vid}'] = sha(synth / kind / f'{vid}.nii.gz')
    return inputs


def reproduce_v3_geometry(synth, vid, organ, b00, b01):
    """File mtimes were reset by the repo move, so instead recompute, from the saved
    synthetic inputs, every geometry value the completed v3 runs recorded: grid,
    crop, initial slice, cropped origin, resampled prompt slice and exact restore."""
    import numpy as np
    from btcv_cross_validation.z_spacing_boxmode.loader import read_synthetic_case
    from btcv_cross_validation.z_spacing_boxmode.evaluate import _crop_origin
    from btcv_cross_validation.z_spacing_boxmode.helpers import (
        resample_label_z, map_slice_to_resampled)
    image, target, center, spacing, origin, direction, info = read_synthetic_case(synth, vid, organ, 'gt-cropped')
    where = f'{vid} label{organ}'
    for key in ('crop_start_z', 'crop_stop_z', 'original_shape_zyx', 'working_shape_zyx',
                'initial_slice_original_z', 'spacing_zyx_mm', 'origin_xyz', 'direction'):
        for arm, rec in (('B00', b00), ('B01', b01)):
            check(np.allclose(np.asarray(info[key], float), np.asarray(rec[key], float), atol=1e-6),
                  f'{where}: {key} differs from v3 {arm}')
    check(image.shape == target.shape, f'{where}: image/label working shapes differ')
    origin_w = _crop_origin(origin, info['crop_start_z'], spacing[0], direction)
    rec = b01['resampling_info']
    check(np.allclose(origin_w, rec['origin_xyz_working'], atol=1e-6), f'{where}: cropped origin differs from v3')
    label_r, out_spacing = resample_label_z(target.astype(np.uint8), spacing, origin_w, direction, TARGET_Z)
    check(list(label_r.shape) == rec['resampled_shape_zyx'], f'{where}: resampled shape differs from v3')
    z_cont = map_slice_to_resampled(center, spacing, out_spacing)
    valid = np.flatnonzero(label_r.sum(axis=(1, 2)))
    valid = valid[(valid >= 1) & (valid < label_r.shape[0] - 1)]
    check(int(valid[np.argmin(abs(valid - z_cont))]) == rec['center_resampled_actual'],
          f'{where}: resampled initial slice differs from v3')
    restored, _ = resample_label_z(label_r, out_spacing, origin_w, direction, spacing[0])
    check(restored.shape == target.shape, f'{where}: restoration does not land on the scoring grid')


# ------------------------------------------------------------- validators

def load_report(out, name='results.json'):
    path = out / name
    if not path.is_file():
        raise Mismatch(f'{out}: no {name}')
    return json.loads(path.read_text())


def validate_one_box(out, fold, organ, mode, fraction, ids, digest):
    r = load_report(out)
    a = r['arguments']
    check(r['status'] == 'complete', f'{out}: status {r["status"]}')
    check(a['padding_mode'] == mode and a['padding_fraction'] == fraction, f'{out}: padding config differs')
    check(a['label_idx'] == organ and a['protocol'] == 'gt-cropped' and a['prompt'] == 'box'
          and a['nsd_tolerance_mm'] == NSD_TOL and a['checkpoint_kind'] == 'finetuned', f'{out}: settings differ')
    check(Path(a['manifest']).parts[-2:] == (f'fold{fold}', '01_BTCV_test_4to1.txt'), f'{out}: wrong manifest')
    check(r['checkpoint_sha256'] == digest, f'{out}: checkpoint hash differs')
    check(r['evaluation_ids'] == ids and [c['volume_id'] for c in r['cases']] == ids, f'{out}: patient IDs differ')
    for c in r['cases']:
        check(c['prediction_status'] == 'ok' and valid_unit(c['metrics']['dice']), f'{out}: invalid case {c["volume_id"]}')
    return r


def validate_five_box(out, fold, mode, fraction, ids, digest):
    r = load_report(out)
    check(r['status'] == 'complete', f'{out}: status {r["status"]}')
    check(r.get('padding_mode', 'none') == mode and r.get('padding_fraction', 0.0) == fraction,
          f'{out}: padding config differs')
    if 'padding_mode' not in r:  # completed corrected-zero run predates the field
        check(r['protocol'].startswith('gt-cropped; corrected-zero'), f'{out}: unexpected protocol')
    check(r['fold'] == fold and r['prompt_budget'] == 5 and r['cudnn_enabled'] is False, f'{out}: settings differ')
    check(r['checkpoint_sha256'] == digest and r['evaluation_ids'] == ids, f'{out}: provenance differs')
    got = sorted((c['organ'], c['volume_id']) for c in r['cases'])
    check(got == sorted((o, v) for o in ORGANS.values() for v in ids), f'{out}: organ/patient cases differ')
    for c in r['cases']:
        check(len(c['dice_by_prompt_count']) == 5 and all(map(valid_unit, c['dice_by_prompt_count']))
              and len(set(c['prompt_slices_cropped'])) == 5, f'{out}: invalid case {c["volume_id"]}')
    return r


def validate_synthetic(out, fold, organ, box_mode, branch, digest, inputs):
    r = load_report(out)
    a = r['arguments']
    check(r['status'] == 'complete', f'{out}: status {r["status"]}')
    check(r['box_mode'] == box_mode and r['padding_mode'] == {'historical': 'historical',
          'corrected_tight': 'none'}[box_mode], f'{out}: box mode differs')
    check(a['branch'] == branch and a['label_idx'] == organ and a['target_z_mm'] == TARGET_Z
          and a['trigger_mm'] == TRIGGER and a['nsd_tolerance_mm'] == NSD_TOL, f'{out}: settings differ')
    ids = SYNTH_COHORT[fold]
    check(r['checkpoint_sha256'] == digest and [c['volume_id'] for c in r['cases']] == ids, f'{out}: provenance differs')
    for c in r['cases']:
        vid = c['volume_id']
        check(c['prediction_status'] == 'ok' and valid_unit(c['metrics']['dice']), f'{out}: invalid case {vid}')
        check(c['resampling_info']['resampling_applied'] == (branch == 'proposed'), f'{out}: recovery flag wrong for {vid}')
        check(c['synthetic_input_sha256'] == {k: inputs[f'fold{fold}/{k}/{vid}'] for k in ('img', 'label')},
              f'{out}: synthetic inputs differ for {vid}')
    return r


# ------------------------------------------------------------------- arms

def build_arms(folds, inputs):
    arms = []
    for fold in FOLDS:
        f = folds[fold]
        p = fold_paths(fold)
        # A1: 5% image-relative padding, one box, folds 1-4 (fold 5 completed).
        if fold != 5:
            for organ in ORGANS:
                out = CV / f'box_padding/results/fold{fold}_e20_image_label{organ}'
                args = ['--data-root', ROOT / 'datasets', '--manifest', p['test'], '--train-manifest', p['train'],
                        '--checkpoint', f['checkpoint'], '--checkpoint-kind', 'finetuned', '--label-idx', organ,
                        '--prompt', 'box', '--protocol', 'gt-cropped', '--padding-mode', 'image',
                        '--padding-fraction', FRACTION, '--nsd-tolerance-mm', NSD_TOL, '--output', out]
                arms.append(dict(id=f'A1_fold{fold}_image_1box_label{organ}', section='A', fold=fold,
                                 cohort=f['test'], padding='image 5%', budget=1, spacing='native',
                                 organs=[organ], checkpoint=f['checkpoint'], output=out,
                                 cmd=['-c', SEEDED.format(module='btcv_cross_validation.box_padding.evaluate'), *args],
                                 validate=lambda out=out, fold=fold, organ=organ, f=f: validate_one_box(
                                     out, fold, organ, 'image', FRACTION, f['test'], f['sha256'])))
    # A2/A3: historical and 5% padding, five boxes, folds 1-5.
    for tag, mode, fraction in (('A2', 'historical', 0.0), ('A3', 'image', FRACTION)):
        for fold in FOLDS:
            f = folds[fold]
            out = CV / f'five_box_padding/results/{mode}_p{fraction:g}/fold{fold}'
            args = ['--fold', fold, '--padding-mode', mode, '--padding-fraction', fraction, '--output', out]
            arms.append(dict(id=f'{tag}_fold{fold}_{mode}_5box', section='A', fold=fold, cohort=f['test'],
                             padding='historical' if mode == 'historical' else 'image 5%', budget=5,
                             spacing='native', organs=list(ORGANS), checkpoint=f['checkpoint'], output=out,
                             cmd=['-m', 'btcv_cross_validation.five_box_padding.evaluate', *args],
                             validate=lambda out=out, fold=fold, mode=mode, fraction=fraction, f=f:
                                 validate_five_box(out, fold, mode, fraction, f['test'], f['sha256'])))
    # B: historical boxes x recovery OFF/ON on the saved v3 synthetic inputs.
    for fold in FOLDS:
        f = folds[fold]
        synth = CV / f'z_spacing/synthetic_6mm_v3/fold{fold}'
        for branch, recovery in (('control', 'off'), ('proposed', 'on')):
            for organ in ORGANS:
                out = CV / f'z_spacing_boxmode/results/fold{fold}/historical_box_recovery_{recovery}_label{organ}'
                args = ['--synthetic-root', synth, '--manifest', synth / 'synthetic_manifest.txt',
                        '--train-manifest', fold_paths(fold)['train'], '--checkpoint', f['checkpoint'],
                        '--checkpoint-kind', 'finetuned', '--label-idx', organ, '--protocol', 'gt-cropped',
                        '--branch', branch, '--box-mode', 'historical', '--target-z-mm', TARGET_Z,
                        '--trigger-mm', TRIGGER, '--nsd-tolerance-mm', NSD_TOL, '--output', out]
                arms.append(dict(id=f'B_fold{fold}_historical_recovery_{recovery}_label{organ}', section='B',
                                 fold=fold, cohort=SYNTH_COHORT[fold], padding='historical', budget=1,
                                 spacing=f'synthetic ~6 mm, recovery {recovery.upper()}', organs=[organ],
                                 checkpoint=f['checkpoint'], output=out,
                                 cmd=['-c', SEEDED.format(module='btcv_cross_validation.z_spacing_boxmode.evaluate'), *args],
                                 validate=lambda out=out, fold=fold, organ=organ, branch=branch, f=f: validate_synthetic(
                                     out, fold, organ, 'historical', branch, f['sha256'], inputs)))
    return arms


def reused_results(folds):
    """Completed results this plan reuses; each must validate before anything runs."""
    reused = []
    for fold in FOLDS:
        f = folds[fold]
        legacy = json.loads(fold_paths(fold)['legacy_eval'].read_text())
        check(legacy['epoch'] == 20 and legacy['prompt'] == 'box', f'fold{fold}: legacy eval settings differ')
        for organ, name in ORGANS.items():
            row = legacy['organs'][{'Left kidney': 'Left Kidney'}.get(name, name)]
            check(row['label_idx'] == organ and sorted(row['per_volume_dice']) == f['test']
                  and all(map(valid_unit, row['per_volume_dice'].values())), f'fold{fold}: legacy {name} invalid')
        reused.append((f'fold{fold} historical 1-box (legacy evaluator, Dice only)', fold_paths(fold)['legacy_eval']))
        validate_five_box(CV / f'five_box/results/fold{fold}', fold, 'none', 0.0, f['test'], f['sha256'])
        reused.append((f'fold{fold} corrected-zero 1-box and 5-box', CV / f'five_box/results/fold{fold}/results.json'))
        for organ in ORGANS:
            for branch, arm in (('control', 'b00_control'), ('proposed', 'b01_proposed')):
                out = CV / f'z_spacing/results_v3/fold{fold}/{arm}_label{organ}'
                r = load_report(out)
                check(r['status'] == 'complete' and r['arguments']['branch'] == branch
                      and r['arguments']['label_idx'] == organ, f'{out}: invalid')
        reused.append((f'fold{fold} synthetic corrected-tight recovery OFF/ON (v3 B00/B01)',
                       CV / f'z_spacing/results_v3/fold{fold}'))
    five = folds[5]
    for mode, fraction in (('historical', 0.0), ('none', 0.0), ('image', FRACTION)):
        for organ in ORGANS:
            validate_one_box(CV / f'box_padding/results/fold5_e20_{mode}_label{organ}', 5, organ, mode, fraction,
                             five['test'], five['sha256'])
    reused.append(('fold5 historical/corrected-zero/5% 1-box (new evaluator)', CV / 'box_padding/results/fold5_e20_*'))
    # The new evaluator's historical mode reproduces legacy fold-5 Dice exactly.
    legacy = json.loads(fold_paths(5)['legacy_eval'].read_text())['organs']
    for organ, name in ORGANS.items():
        new = load_report(CV / f'box_padding/results/fold5_e20_historical_label{organ}')
        old = legacy[{'Left kidney': 'Left Kidney'}.get(name, name)]['per_volume_dice']
        check(all(c['metrics']['dice'] == old[c['volume_id']] for c in new['cases']),
              f'fold5 {name}: new historical Dice differs from legacy evaluator')
    return reused


# ------------------------------------------------------------------ main

def env(cpu_only):
    e = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
             OPENBLAS_NUM_THREADS='1', MPLCONFIGDIR='/tmp/slidesam-missing-evals-mpl')
    if cpu_only:
        e['CUDA_VISIBLE_DEVICES'] = ''
    return e


def status_of(arm):
    if not arm['output'].exists():
        return 'pending'
    try:
        arm['validate']()
        return 'complete'
    except (Mismatch, KeyError, ValueError) as exc:
        return f'INVALID: {exc}'


def write_inventory(arms, reused, folds, inputs):
    rel = lambda p: str(Path(p).relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p)
    lines = ['# Execution inventory', '', f'Generated {datetime.now(timezone.utc).isoformat()} by '
             '`btcv_cross_validation/missing_evals/run_missing_evals.py`.', '',
             '## Checkpoints (epoch 20, strict load, verified against completed five-box runs)', '',
             '| Fold | Checkpoint | SHA-256 |', '| --- | --- | --- |']
    lines += [f'| {k} | `{rel(v["checkpoint"])}` | `{v["sha256"]}` |' for k, v in folds.items()]
    lines += ['', '## Reused completed results (validated, not rerun)', '']
    lines += [f'- {name}: `{rel(path)}`' for name, path in reused]
    lines += ['', '## Planned runs', '',
              '| # | Arm | Fold | Cohort (n) | Padding | Boxes | Spacing branch | Checkpoint | Output | Status |',
              '| ---: | --- | ---: | --- | --- | ---: | --- | --- | --- | --- |']
    for i, a in enumerate(arms, 1):
        lines.append(f'| {i} | {a["id"]} | {a["fold"]} | {", ".join(a["cohort"])} ({len(a["cohort"])}) | '
                     f'{a["padding"]} | {a["budget"]} | {a["spacing"]} | `{rel(a["checkpoint"])}` | '
                     f'`{rel(a["output"])}` | {a["status"]} |')
    (HERE / 'execution_inventory.md').write_text('\n'.join(lines) + '\n')
    (HERE / 'synthetic_input_sha256.json').write_text(json.dumps(inputs, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--execute', action='store_true', help='Run GPU inference for pending arms.')
    parser.add_argument('--only', choices=['A1', 'A2', 'A3', 'B'], nargs='+',
                        help='Restrict to these groups (default: all).')
    args = parser.parse_args()
    check(Path(sys.executable).resolve() == PYTHON.resolve(), f'Use {PYTHON}, not {sys.executable}')
    check(str(ROOT) in os.environ.get('PYTHONPATH', '').split(os.pathsep), f'Set PYTHONPATH={ROOT}')

    print('== CPU test suites', flush=True)
    for suite in ('missing_evals.test_cpu', 'box_padding.test_cpu', 'z_spacing.test_geometry_fixes'):
        r = subprocess.run([str(PYTHON), '-m', 'unittest', f'btcv_cross_validation.{suite}'], cwd=ROOT,
                           env=dict(env(True), OMP_NUM_THREADS='1'), capture_output=True, text=True)
        check(r.returncode == 0, f'{suite} failed:\n{r.stderr[-2000:]}')
        print(f'  {suite}: {r.stderr.strip().splitlines()[-3]} OK', flush=True)

    print('== Provenance and cohort checks', flush=True)
    folds = verify_folds()
    inputs = verify_synthetic_cohort(folds)
    reused = reused_results(folds)
    arms = [a for a in build_arms(folds, inputs) if not args.only or a['id'].split('_')[0] in args.only]
    outputs = [a['output'] for a in arms]
    check(len(set(outputs)) == len(outputs), 'output directories are not unique')
    for a in arms:
        a['status'] = status_of(a)
        print(f'  {a["id"]:<48} {a["status"]}', flush=True)
    write_inventory(arms, reused, folds, inputs)
    bad = [a for a in arms if a['status'].startswith('INVALID')]
    if bad:
        sys.exit('Existing outputs are not verifiably complete; they were left untouched. Inspect and move '
                 'them aside manually:\n' + '\n'.join(str(a['output']) for a in bad))
    pending = [a for a in arms if a['status'] == 'pending']

    print(f'== Evaluator preflights on CPU for {len(pending)} pending arms', flush=True)
    for a in pending:
        r = subprocess.run([str(PYTHON), *map(str, a['cmd'])], cwd=ROOT, env=env(True),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        check(r.returncode == 0, f'preflight failed for {a["id"]}:\n{r.stdout[-1500:]}')
        check(not a['output'].exists(), f'preflight created {a["output"]}')
    print('All preflights passed. Inventory: btcv_cross_validation/missing_evals/execution_inventory.md')
    if not args.execute:
        print('Preflight only; no GPU work. Add --execute to run the pending arms sequentially.')
        return

    logs = HERE / 'logs'
    logs.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for i, a in enumerate(pending, 1):
        log = logs / f'{stamp}_{a["id"]}.log'
        print(f'[{i}/{len(pending)}] {a["id"]} -> {log.name}', flush=True)
        with open(log, 'w') as handle:
            r = subprocess.run([str(PYTHON), '-u', *map(str, [*a['cmd'], '--run-inference'])], cwd=ROOT,
                               env=env(False), stdout=handle, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            sys.exit(f'{a["id"]} failed (exit {r.returncode}); partial output preserved at {a["output"]}. See {log}')
        a['validate']()
        a['status'] = 'complete'
        write_inventory(arms, reused, folds, inputs)
    print('All pending arms complete. Next: summarize.py')


if __name__ == '__main__':
    try:
        main()
    except Mismatch as exc:
        sys.exit(f'STOPPED (no GPU work started for unchecked arms): {exc}')
