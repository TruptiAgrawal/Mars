"""Aggregate all held-out patients and compare with the paper's five-box row."""
import csv
import json
from pathlib import Path
from statistics import mean

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
reports=[json.loads((HERE/f'results/fold{i}/results.json').read_text()) for i in range(1,6)]
assert all(r['status']=='complete' and len(r['cases'])==24 for r in reports)
assert len({p for r in reports for p in r['evaluation_ids']})==30
for r in reports:
    assert not set(r['training_ids']) & set(r['evaluation_ids'])
    for c in r['cases']:
        assert len(c['prompt_slices_cropped'])==len(set(c['prompt_slices_cropped']))==5
organs=['Liver','Stomach','Left kidney','Spleen']
paper=[95.62,90.88,93.49,91.88]
cases=[dict(c,fold=r['fold']) for r in reports for c in r['cases']]
lines=['# Five-box corrected-zero evaluation versus the authors', '',
       'All five held-out BTCV folds, epoch-20 BS=4 checkpoints, native volumes, four organs. All 120 patient-organ cases completed with five distinct box prompts each (600 prompt stages). No retraining or z-resampling.', '',
       '| Organ | Authors five-box Dice (%) | Our one-box corrected-zero (%) | Our five-box corrected-zero (%) | Five − one (pp) | Our five − authors (pp) |',
       '| --- | ---: | ---: | ---: | ---: | ---: |']
one=[];five=[]
for o,a in zip(organs,paper):
    subset=[c for c in cases if c['organ']==o];assert len(subset)==30
    x=mean(c['one_box_dice'] for c in subset)*100;y=mean(c['five_box_dice'] for c in subset)*100
    one.append(x);five.append(y)
    lines.append(f'| {o} | {a:.2f} | {x:.2f} | {y:.2f} | {y-x:+.2f} | {y-a:+.2f} |')
lines += [f'| **Mean** | **92.96** | **{mean(one):.2f}** | **{mean(five):.2f}** | **{mean(five)-mean(one):+.2f}** | **{mean(five)-92.96:+.2f}** |', '',
          'Authors’ numbers are from Table 1, Ours (5 boxes), [published paper](https://raw.githubusercontent.com/mlresearch/v250/main/assets/quan24a/quan24a.pdf). The paper prints mean 92.96%; averaging its rounded organ values gives 92.9675%. We retain the published mean as printed.', '',
          '## Protocol and limits', '',
          'The first box is taken from the ground-truth maximum-area valid slice. Four further boxes use the unchanged outer authors’ `Evaluater.finetune_with_more_prompt`: choose the lowest per-slice Dice among unused interior slices, derive a ground-truth box, propagate, then merge using the authors’ confidence rule (including doubled confidence on the correction slice). This is oracle-assisted interactive correction, not five uniformly spaced boxes or automatic prompting. The inference method named “finetune” does not train weights.', '',
          'The original correction method and predictor state/merge behavior are preserved. Corrected-zero propagated boxes replace the malformed historical adjustment. The loader uses explicit z-y-x geometry and ground-truth extent cropping. All five prompts are distinct. Additional prompts are not guaranteed to improve Dice because we do not select the best result across prompt counts.', '',
          'One-box and five-box scores above are paired within the same run. The one-box row is newly evaluated across all 30 patients; it is not the earlier six-patient Box Z score. Each patient uses its own held-out fold checkpoint. Comparison with the paper is descriptive: author split identities and complete model/training provenance are not established, and corrected-zero propagation intentionally differs from the historical code.', '',
          '## Per-fold means', '',
          '| Fold | Patients | One box (%) | Five boxes (%) | Change (pp) |',
          '| --- | ---: | ---: | ---: | ---: |']
for r in reports:
    x=mean(c['one_box_dice'] for c in r['cases'])*100;y=mean(c['five_box_dice'] for c in r['cases'])*100
    lines.append(f"| {r['fold']} | 6 | {x:.2f} | {y:.2f} | {y-x:+.2f} |")
lines += ['', '## Checks and provenance', '',
          '- Same checkpoint, patient, image and initial prompt within every one/five-box pair.',
          '- Each fold has 24 completed patient-organ cases; all 30 patients appear in one held-out fold.',
          '- Native CUDA convolution, cuDNN disabled, float32 evaluation, fixed seeds, no gradients.',
          '- Original source copies and author multi-box evaluator are preserved under `btcv_cross_validation/five_box/originals/`.',
          '- Results record checkpoint/source hashes, prompt slices and boxes, and Dice after every prompt.',
          '- The failed first launch used incompatible system Python and produced no scores. The successful run uses the explicit `slidesam` environment.',
          '- This evaluation reports Dice, not NSD/HD95 or clinical annotation time.', '',
          'Raw files: `btcv_cross_validation/five_box/results/fold{1-5}/results.json`, `per_patient.csv`, `run.log` and `summary.json`.', '']
with (HERE/'per_patient.csv').open('w') as f:
    writer=csv.writer(f);writer.writerow(['fold','patient','organ','one_box_dice','five_box_dice','difference_pp','seconds'])
    for c in cases:writer.writerow([c['fold'],c['volume_id'],c['organ'],c['one_box_dice'],c['five_box_dice'],100*(c['five_box_dice']-c['one_box_dice']),c['total_seconds']])
summary={'organs':{o:{'one_box_percent':x,'five_box_percent':y,'authors_five_box_percent':a} for o,x,y,a in zip(organs,one,five,paper)},
         'one_box_mean_percent':mean(one),'five_box_mean_percent':mean(five),'authors_reported_mean_percent':92.96,'cases':120}
(HERE/'summary.json').write_text(json.dumps(summary,indent=2))
(ROOT/'KT/Five_Box_Corrected_Zero_Results.md').write_text('\n'.join(lines))
print('\n'.join(lines[:12]))
