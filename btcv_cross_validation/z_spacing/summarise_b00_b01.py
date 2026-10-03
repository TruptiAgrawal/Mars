"""
btcv_cross_validation/z_spacing/summarise_b00_b01.py

Extract and summarise results from the 17-patient B00 vs B01 comparison.

Run from repo root:
    python btcv_cross_validation/z_spacing/summarise_b00_b01.py
"""
import json
import numpy as np
from pathlib import Path

label_map = {6: 'Liver', 7: 'Stomach', 3: 'Left Kidney', 1: 'Spleen'}
branches  = ['control', 'proposed']
folds     = [1, 2, 3, 4, 5]

# Load all case records
all_cases = []
for fold in folds:
    for branch in branches:
        prefix = 'b01' if branch == 'proposed' else 'b00'
        for label_idx, organ in label_map.items():
            path = Path(
                f'btcv_cross_validation/z_spacing/results_v3/'
                f'fold{fold}/{prefix}_{branch}_label{label_idx}/results.json')
            if not path.exists():
                print(f'MISSING: {path}')
                continue
            with open(path) as f:
                d = json.load(f)
            for c in d['cases']:
                m = c['metrics']
                all_cases.append({
                    'fold': fold, 'branch': branch,
                    'organ': organ, 'label_idx': label_idx,
                    'patient': c['volume_id'],
                    'dice':    m['dice'],
                    'nsd':     m['nsd'],
                    'hd95':    m['hd95_mm'],
                    'recall':  m['foreground_recall'],
                    'prec':    m['foreground_precision'],
                    'fp_ml':   m['false_positive_ml'],
                    'seconds': c.get('total_seconds', 0),
                })

print(f'Total case records loaded: {len(all_cases)}')
print(f'Expected: 17 patients x 4 organs x 2 branches = 136')
print()

# Build paired lookup: (patient, organ) -> {branch: row}
paired = {}
for r in all_cases:
    key = (r['patient'], r['organ'])
    paired.setdefault(key, {})[r['branch']] = r

complete_pairs = {k: v for k, v in paired.items()
                  if 'control' in v and 'proposed' in v}
print(f'Complete paired cases: {len(complete_pairs)} / {len(paired)}')
print()

# ── Organ-level mean Dice ────────────────────────────────────────────────────
print('=== ORGAN-LEVEL MEAN DICE (%) — equal patient weight ===')
print(f'{"Organ":<14} {"n":>4} {"B00 (ctrl)":>12} {"B01 (prop)":>12} {"Diff":>8}')
print('-' * 54)
all_ctrl, all_prop = [], []
for label_idx, organ in label_map.items():
    ctrl_d = [v['control']['dice'] * 100 for (p, o), v in complete_pairs.items() if o == organ]
    prop_d = [v['proposed']['dice'] * 100 for (p, o), v in complete_pairs.items() if o == organ]
    n = len(ctrl_d)
    cm, pm = np.mean(ctrl_d), np.mean(prop_d)
    all_ctrl.append(cm); all_prop.append(pm)
    print(f'{organ:<14} {n:>4} {cm:>12.4f} {pm:>12.4f} {pm - cm:>+8.4f}')
print('-' * 54)
print(f'{"4-organ mean":<14} {"":>4} {np.mean(all_ctrl):>12.4f} {np.mean(all_prop):>12.4f} '
      f'{np.mean(all_prop) - np.mean(all_ctrl):>+8.4f}')
print()

# ── NSD ──────────────────────────────────────────────────────────────────────
print('=== MEAN NSD @ 1mm (%) ===')
for label_idx, organ in label_map.items():
    c_nsd = [v['control']['nsd'] * 100  for (p, o), v in complete_pairs.items()
             if o == organ and v['control']['nsd'] is not None]
    p_nsd = [v['proposed']['nsd'] * 100 for (p, o), v in complete_pairs.items()
             if o == organ and v['proposed']['nsd'] is not None]
    print(f'  {organ:<14}: B00={np.mean(c_nsd):.4f}  B01={np.mean(p_nsd):.4f}  '
          f'diff={np.mean(p_nsd) - np.mean(c_nsd):+.4f}')
print()

# ── HD95 ─────────────────────────────────────────────────────────────────────
print('=== MEAN HD95 (mm) ===')
for label_idx, organ in label_map.items():
    c_hd = [v['control']['hd95']  for (p, o), v in complete_pairs.items()
            if o == organ and v['control']['hd95'] is not None]
    p_hd = [v['proposed']['hd95'] for (p, o), v in complete_pairs.items()
            if o == organ and v['proposed']['hd95'] is not None]
    print(f'  {organ:<14}: B00={np.mean(c_hd):.2f}  B01={np.mean(p_hd):.2f}  '
          f'diff={np.mean(p_hd) - np.mean(c_hd):+.2f}')
print()

# ── Recall / Precision ───────────────────────────────────────────────────────
print('=== MEAN RECALL / PRECISION ===')
for label_idx, organ in label_map.items():
    cr = [v['control']['recall'] * 100  for (p, o), v in complete_pairs.items()
          if o == organ and v['control']['recall'] is not None]
    pr = [v['proposed']['recall'] * 100 for (p, o), v in complete_pairs.items()
          if o == organ and v['proposed']['recall'] is not None]
    cp = [v['control']['prec'] * 100    for (p, o), v in complete_pairs.items()
          if o == organ and v['control']['prec'] is not None]
    pp = [v['proposed']['prec'] * 100   for (p, o), v in complete_pairs.items()
          if o == organ and v['proposed']['prec'] is not None]
    print(f'  {organ:<14}: recall B00={np.mean(cr):.2f}% B01={np.mean(pr):.2f}% '
          f'({np.mean(pr)-np.mean(cr):+.2f})  '
          f'prec B00={np.mean(cp):.2f}% B01={np.mean(pp):.2f}% '
          f'({np.mean(pp)-np.mean(cp):+.2f})')
print()

# ── Per-patient paired Dice ──────────────────────────────────────────────────
print('=== PER-PATIENT PAIRED DICE (%) ===')
print(f'{"Patient":<10} {"Fold":>5} {"Organ":<14} {"B00":>8} {"B01":>8} {"Diff":>8}')
print('-' * 60)
for (patient, organ) in sorted(complete_pairs.keys()):
    arms = complete_pairs[(patient, organ)]
    fold = arms['control']['fold']
    c = arms['control']['dice'] * 100
    p = arms['proposed']['dice'] * 100
    print(f'{patient:<10} {fold:>5} {organ:<14} {c:>8.4f} {p:>8.4f} {p - c:>+8.4f}')
print()

# ── Per-organ win/loss count ─────────────────────────────────────────────────
print('=== WIN/LOSS COUNT (B01 > B00) ===')
for label_idx, organ in label_map.items():
    diffs = [(v['proposed']['dice'] - v['control']['dice']) * 100
             for (p, o), v in complete_pairs.items() if o == organ]
    wins  = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    ties   = sum(1 for d in diffs if d == 0)
    print(f'  {organ:<14}: wins={wins}  losses={losses}  ties={ties}  '
          f'mean_diff={np.mean(diffs):+.4f}pp')
print()

# ── Runtime ──────────────────────────────────────────────────────────────────
print('=== RUNTIME (mean seconds per patient per organ) ===')
for branch in branches:
    times = [r['seconds'] for r in all_cases if r['branch'] == branch]
    print(f'  {branch:<10}: mean={np.mean(times):.2f}s  '
          f'total_est={np.sum(times)/60:.1f} min')
