"""Per-fold report for the padding x prompt-budget (native) and box-mode x
recovery (synthetic) evaluations. Missing arms are shown as 'pending'; no
across-fold means are computed and native/synthetic scores are never pooled.

    /home/jupyter-team2pop/.conda/envs/slidesam/bin/python \
      btcv_cross_validation/missing_evals/summarize.py
"""
import csv
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
CV = HERE.parent
FOLDS = [1, 2, 3, 4, 5]
ORGANS = [(6, 'Liver'), (7, 'Stomach'), (3, 'Left kidney'), (1, 'Spleen')]
NATIVE = ['H1', 'Z1', 'P1', 'H5', 'Z5', 'P5']
NATIVE_NAMES = {'H1': 'Historical 1-box', 'Z1': 'Corrected-zero 1-box', 'P1': '5% padding 1-box',
                'H5': 'Historical 5-box', 'Z5': 'Corrected-zero 5-box', 'P5': '5% padding 5-box'}
SYNTH = ['H-OFF', 'H-ON', 'C-OFF', 'C-ON']
SYNTH_NAMES = {'H-OFF': 'Historical, recovery OFF', 'H-ON': 'Historical, recovery ON',
               'C-OFF': 'Corrected tight, recovery OFF (v3 B00)', 'C-ON': 'Corrected tight, recovery ON (v3 B01)'}
AUTHORS_5BOX = {'Liver': 95.62, 'Stomach': 90.88, 'Left kidney': 93.49, 'Spleen': 91.88}


def load(path):
    path = Path(path)
    if not path.is_file():
        return None
    r = json.loads(path.read_text())
    return r if r.get('status') == 'complete' else None


def native_scores(fold):
    """{arm: {(organ, patient): (dice, seconds)}}; arms absent if incomplete."""
    out = {}
    legacy = json.loads((CV / f'logs/fold{fold}_evaluation.json').read_text())['organs']
    if fold == 5:
        sources = {'H1': 'historical', 'P1': 'image'}
    else:
        sources = {'P1': 'image'}
        out['H1'] = {(name, v): (d, None) for label, name in ORGANS
                     for v, d in legacy['Left Kidney' if name == 'Left kidney' else name]['per_volume_dice'].items()}
    for arm, mode in sources.items():
        cells = {}
        for label, name in ORGANS:
            r = load(CV / f'box_padding/results/fold{fold}_e20_{mode}_label{label}/results.json')
            if r:
                cells.update({(name, c['volume_id']): (c['metrics']['dice'], c['inference_seconds']) for c in r['cases']})
        if len(cells) == 24:
            out[arm] = cells
    for prefix, path in (('Z', f'five_box/results/fold{fold}'), ('H', f'five_box_padding/results/historical_p0/fold{fold}'),
                         ('P', f'five_box_padding/results/image_p0.05/fold{fold}')):
        r = load(CV / path / 'results.json')
        if r:
            if prefix == 'Z':
                out['Z1'] = {(c['organ'], c['volume_id']): (c['one_box_dice'], None) for c in r['cases']}
            out[f'{prefix}5'] = {(c['organ'], c['volume_id']): (c['five_box_dice'], c['total_seconds']) for c in r['cases']}
            out[f'{prefix}5_stage1'] = {(c['organ'], c['volume_id']): c['one_box_dice'] for c in r['cases']}
    return out


def synthetic_scores(fold):
    out = {}
    paths = {'H-OFF': 'z_spacing_boxmode/results/fold{f}/historical_box_recovery_off_label{l}',
             'H-ON': 'z_spacing_boxmode/results/fold{f}/historical_box_recovery_on_label{l}',
             'C-OFF': 'z_spacing/results_v3/fold{f}/b00_control_label{l}',
             'C-ON': 'z_spacing/results_v3/fold{f}/b01_proposed_label{l}'}
    for arm, pattern in paths.items():
        cells, modes = {}, set()
        for label, name in ORGANS:
            r = load(CV / pattern.format(f=fold, l=label) / 'results.json')
            if r:
                modes.add(r.get('padding_mode', 'none'))  # v3 B00/B01 hardcoded corrected-zero ('none')
                cells.update({(name, c['volume_id']): (c['metrics']['dice'], c['total_seconds']) for c in r['cases']})
        if cells and len(cells) == 4 * len({p for _, p in cells}):
            assert modes == ({'historical'} if arm.startswith('H') else {'none'}), (fold, arm, modes)
            out[arm] = cells
    return out


def pct(x):
    return 'pending' if x is None else f'{100 * x:.2f}'


def diff(a, b):
    return 'pending' if a is None or b is None else f'{100 * (a - b):+.2f}'


def organ_mean(cells, organ):
    vals = [d for (o, _), (d, _) in cells.items() if o == organ]
    return mean(vals) if vals else None


def fold_table(scores, arms, names, patients):
    lines = ['| Organ | ' + ' | '.join(names[a] for a in arms) + ' |', '| --- |' + ' ---: |' * len(arms)]
    means = {a: [] for a in arms}
    for _, organ in ORGANS:
        row = []
        for a in arms:
            m = organ_mean(scores[a], organ) if a in scores else None
            means[a].append(m)
            row.append(pct(m))
        lines.append(f'| {organ} | ' + ' | '.join(row) + ' |')
    lines.append('| **Four-organ mean** | ' + ' | '.join(
        f'**{pct(mean(v))}**' if None not in v else 'pending' for v in means.values()) + ' |')
    return lines


def paired_lines(scores, contrasts, patients):
    lines = ['| Organ | Patient | ' + ' | '.join(f'{a} − {b} (pp)' for a, b in contrasts) + ' |',
             '| --- | --- |' + ' ---: |' * len(contrasts)]
    summary = []
    for _, organ in ORGANS:
        for p in patients:
            get = lambda a: scores[a][(organ, p)][0] if a in scores else None
            lines.append(f'| {organ} | {p} | ' + ' | '.join(diff(get(a), get(b)) for a, b in contrasts) + ' |')
        for a, b in contrasts:
            if a in scores and b in scores:
                d = [scores[a][(organ, p)][0] - scores[b][(organ, p)][0] for p in patients]
                summary.append(f'{organ} {a}−{b}: {sum(x > 0 for x in d)} better / {sum(x < 0 for x in d)} worse '
                               f'of {len(d)}; worst {100 * min(d):+.2f} pp')
    return lines, summary


def runtime(scores, arms):
    parts = []
    for a in arms:
        secs = [s for _, s in scores.get(a, {}).values() if s is not None]
        if secs:
            parts.append(f'{a}: {sum(secs) / 60:.1f} min total, {mean(secs):.1f} s/case')
    return '; '.join(parts) or 'n/a'


def main():
    out = ['# Padding × prompt budget (native) and box mode × recovery (synthetic): per-fold results', '',
           'Generated by `btcv_cross_validation/missing_evals/summarize.py`. Dice in %. Folds are reported '
           'separately; no across-fold mean is computed. Native and synthetic scores are never pooled. '
           'Cells marked *pending* have no verified complete result yet.', '',
           '**Native experiment.** All six held-out patients per fold, native scans, gt-cropped, '
           'oracle GT max-area initial box. Padding applies to automatically propagated boxes only. '
           'H = historical `enlarge()` arithmetic, Z = corrected zero (tight box), P = 5% of image '
           'width/height per side. Five-box = initial box + four oracle lowest-Dice corrective boxes '
           'with the authors’ merge rule.', '',
           '**Synthetic experiment (exploratory robustness study).** Native 3 mm held-out patients only '
           '(not all 30), the saved v3 ~6 mm synthetic inputs, one box. Recovery OFF segments 6 mm directly; '
           'ON resamples >5.5 mm to ~3 mm and restores to the exact 6 mm scoring grid. "Corrected tight" is NOT '
           '5% padding. C-OFF/C-ON are the existing v3 B00/B01 runs. Interpolation does not recover missing '
           'anatomy; these scores are not comparable with native BTCV or the authors’ scores.', '']
    csv_rows = []
    checks = []
    for fold in FOLDS:
        native = native_scores(fold)
        patients = sorted({p for _, p in next(iter(native.values()))})
        out += [f'## Fold {fold}', '', f'### Native, n = {len(patients)} patients ({", ".join(patients)})', '']
        out += fold_table(native, NATIVE, NATIVE_NAMES, patients)
        lines, summary = paired_lines(native, [('Z1', 'H1'), ('P1', 'Z1'), ('Z5', 'H5'), ('P5', 'Z5'),
                                               ('H5', 'H1'), ('P5', 'P1')], patients)
        out += ['', '<details><summary>Paired per-patient differences</summary>', '', *lines, '', '</details>', '',
                'Organ-level paired counts: ' + ('; '.join(summary) or 'pending'), '',
                f'Runtime: {runtime(native, ["P1", "H5", "P5"])}. H1 folds 1–4 come from the legacy evaluator '
                '(Dice only, no timing).', '']
        for prefix, one in (('H', 'H1'), ('P', 'P1'), ('Z', 'Z1')):
            stage = native.get(f'{prefix}5_stage1')
            if stage and one in native:
                worst = max(abs(stage[k] - native[one][k][0]) for k in stage)
                checks.append(f'fold {fold} {prefix}: five-box stage-1 vs {one} max |ΔDice| = {worst:.3g}')
        synth = synthetic_scores(fold)
        sp = sorted({p for arm in synth.values() for _, p in arm})
        out += [f'### Synthetic ~6 mm, n = {len(sp)} patients ({", ".join(sp)})', '']
        out += fold_table(synth, SYNTH, SYNTH_NAMES, sp)
        lines, summary = paired_lines(synth, [('H-ON', 'H-OFF'), ('C-ON', 'C-OFF'), ('C-OFF', 'H-OFF'),
                                              ('C-ON', 'H-ON')], sp)
        out += ['', '<details><summary>Paired per-patient differences</summary>', '', *lines, '', '</details>', '',
                'Organ-level paired counts: ' + ('; '.join(summary) or 'pending'), '',
                f'Runtime: {runtime(synth, SYNTH)}.', '']
        for arm, cells in {**{a: native[a] for a in NATIVE if a in native}, **synth}.items():
            for (organ, p), (d, s) in cells.items():
                csv_rows.append([fold, 'native' if arm in NATIVE else 'synthetic', arm, organ, p, d, s])
    out += ['## Authors (five boxes only, descriptive)', '',
            'Paper Table 1, Ours (5 boxes): ' + ', '.join(f'{k} {v:.2f}' for k, v in AUTHORS_5BOX.items()) +
            '. Compare only with our five-box columns. The authors’ evaluation split and training provenance '
            'are not established as identical to these folds; differences are not attributable to padding.', '',
            '## Consistency checks', '', *[f'- {c}' for c in checks],
            '- Expected 0 for all: stage 1 of a five-box run is the same single-box prediction.', '']
    (HERE / 'report.md').write_text('\n'.join(out))
    with open(HERE / 'per_patient.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['fold', 'experiment', 'arm', 'organ', 'patient', 'dice', 'seconds'])
        w.writerows(csv_rows)
    print('\n'.join(out[:40]))


if __name__ == '__main__':
    main()
