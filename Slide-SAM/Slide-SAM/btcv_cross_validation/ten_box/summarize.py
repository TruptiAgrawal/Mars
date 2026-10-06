"""Aggregate all held-out patients: one, five and ten boxes, with checks against the five-box run."""
import csv
import json
from pathlib import Path
from statistics import mean

HERE=Path(__file__).resolve().parent
FIVE=HERE.parent/'five_box'
PROMPTS=10
reports=[json.loads((HERE/f'results/fold{i}/results.json').read_text()) for i in range(1,6)]
previous=[json.loads((FIVE/f'results/fold{i}/results.json').read_text()) for i in range(1,6)]
assert all(r['status']=='complete' and len(r['cases'])==24 for r in reports)
assert len({p for r in reports for p in r['evaluation_ids']})==30
for r in reports:
    assert not set(r['training_ids']) & set(r['evaluation_ids'])
    for c in r['cases']:
        assert len(c['prompt_slices_cropped'])==len(set(c['prompt_slices_cropped']))==PROMPTS
        assert len(c['dice_by_prompt_count'])==PROMPTS

# Consistency with the five-box run: same checkpoints, same first five prompts and Dice.
checkpoint_match=all(r['checkpoint_sha256']==p['checkpoint_sha256'] for r,p in zip(reports,previous))
max_diff=0.0;slices_match=True
for r,p in zip(reports,previous):
    old={(c['volume_id'],c['organ']):c for c in p['cases']}
    for c in r['cases']:
        o=old[(c['volume_id'],c['organ'])]
        slices_match&=c['prompt_slices_cropped'][:5]==o['prompt_slices_cropped']
        max_diff=max(max_diff,*(abs(a-b) for a,b in zip(c['dice_by_prompt_count'][:5],o['dice_by_prompt_count'])))

organs=['Liver','Stomach','Left kidney','Spleen']
paper=[95.62,90.88,93.49,91.88]
cases=[dict(c,fold=r['fold']) for r in reports for c in r['cases']]
lines=['# Ten-box corrected-zero evaluation', '',
       'All five held-out BTCV folds, epoch-20 BS=4 checkpoints, native volumes, four organs. All 120 patient-organ cases completed with ten distinct box prompts each (1200 prompt stages). No retraining or z-resampling. Same protocol as the five-box experiment, extended from four to nine corrective boxes.', '',
       '| Organ | Authors five-box (%) | Our one box (%) | Our five boxes (%) | Our ten boxes (%) | Ten − five (pp) | Ten − one (pp) | Our ten − authors five (pp) |',
       '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
one=[];five=[];ten=[]
for o,a in zip(organs,paper):
    subset=[c for c in cases if c['organ']==o];assert len(subset)==30
    x=mean(c['one_box_dice'] for c in subset)*100;y=mean(c['five_box_dice'] for c in subset)*100;z=mean(c['ten_box_dice'] for c in subset)*100
    one.append(x);five.append(y);ten.append(z)
    lines.append(f'| {o} | {a:.2f} | {x:.2f} | {y:.2f} | {z:.2f} | {z-y:+.2f} | {z-x:+.2f} | {z-a:+.2f} |')
lines += [f'| **Mean** | **92.96** | **{mean(one):.2f}** | **{mean(five):.2f}** | **{mean(ten):.2f}** | **{mean(ten)-mean(five):+.2f}** | **{mean(ten)-mean(one):+.2f}** | **{mean(ten)-92.96:+.2f}** |', '',
          'The paper reports five boxes only (Table 1, Ours (5 boxes), [published paper](https://raw.githubusercontent.com/mlresearch/v250/main/assets/quan24a/quan24a.pdf)); the last column compares our ten boxes against their five and is descriptive only.', '',
          '## Dice by number of boxes (%)', '',
          '| Organ | '+' | '.join(str(k) for k in range(1,PROMPTS+1))+' |',
          '| --- |'+' ---: |'*PROMPTS]
curve={}
for o in organs:
    subset=[c for c in cases if c['organ']==o]
    curve[o]=[mean(c['dice_by_prompt_count'][k] for c in subset)*100 for k in range(PROMPTS)]
    lines.append(f'| {o} | '+' | '.join(f'{v:.2f}' for v in curve[o])+' |')
curve['Mean']=[mean(curve[o][k] for o in organs) for k in range(PROMPTS)]
lines.append('| **Mean** | '+' | '.join(f'**{v:.2f}**' for v in curve['Mean'])+' |')
lines += ['', '## Protocol and limits', '',
          'The first box is taken from the ground-truth maximum-area valid slice. Nine further boxes use the unchanged outer authors’ `Evaluater.finetune_with_more_prompt`: choose the lowest per-slice Dice among unused interior slices, derive a ground-truth box, propagate, then merge using the authors’ confidence rule. This is oracle-assisted interactive correction, not automatic prompting. We report the Dice after the tenth box, not the best over prompt counts, so additional boxes can lower Dice.', '',
          'Checkpoints were loaded from their current location under the repository; the fold evaluation logs record the former repository path. Checkpoint hashes are compared with the five-box run below.', '',
          '## Per-fold means', '',
          '| Fold | Patients | One box (%) | Five boxes (%) | Ten boxes (%) | Ten − five (pp) |',
          '| --- | ---: | ---: | ---: | ---: | ---: |']
for r in reports:
    x=mean(c['one_box_dice'] for c in r['cases'])*100;y=mean(c['five_box_dice'] for c in r['cases'])*100;z=mean(c['ten_box_dice'] for c in r['cases'])*100
    lines.append(f"| {r['fold']} | 6 | {x:.2f} | {y:.2f} | {z:.2f} | {z-y:+.2f} |")
lines += ['', '## Checks and provenance', '',
          f'- Checkpoint SHA-256 identical to the five-box run for all folds: {checkpoint_match}.',
          f'- First five prompt slices identical to the five-box run in every case: {slices_match}.',
          f'- Maximum absolute difference from the five-box run in Dice at prompt counts 1–5: {max_diff:.2e}.',
          '- Each fold has 24 completed patient-organ cases; all 30 patients appear in one held-out fold; ten distinct prompts per case.',
          '- Native CUDA convolution, cuDNN disabled, float32 evaluation, fixed seeds, no gradients.',
          '- Code copied from `btcv_cross_validation/five_box/`; only `evaluate.py`, `summarize.py` and `README.md` differ.', '',
          'Raw files: `btcv_cross_validation/ten_box/results/fold{1-5}/results.json`, `per_patient.csv`, `run.log`, `summary.json` and `verification.json`.', '']
with (HERE/'per_patient.csv').open('w') as f:
    writer=csv.writer(f);writer.writerow(['fold','patient','organ','one_box_dice','five_box_dice','ten_box_dice','ten_minus_five_pp','seconds'])
    for c in cases:writer.writerow([c['fold'],c['volume_id'],c['organ'],c['one_box_dice'],c['five_box_dice'],c['ten_box_dice'],100*(c['ten_box_dice']-c['five_box_dice']),c['total_seconds']])
summary={'organs':{o:{'one_box_percent':x,'five_box_percent':y,'ten_box_percent':z,'authors_five_box_percent':a,'by_prompt_count_percent':curve[o]} for o,x,y,z,a in zip(organs,one,five,ten,paper)},
         'one_box_mean_percent':mean(one),'five_box_mean_percent':mean(five),'ten_box_mean_percent':mean(ten),
         'mean_by_prompt_count_percent':curve['Mean'],'authors_reported_five_box_mean_percent':92.96,'cases':120}
(HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
verification={'folds_complete':5,'patients':30,'patient_organ_cases':120,'prompt_stages':120*PROMPTS,
              'distinct_ten_prompts_each':True,'checkpoint_sha256_matches_five_box':checkpoint_match,
              'first_five_prompt_slices_match_five_box':slices_match,'max_abs_dice_diff_vs_five_box_counts_1_to_5':max_diff,
              'no_training_test_manifest_overlap':True}
(HERE/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
(HERE/'Ten_Box_Corrected_Zero_Results.md').write_text('\n'.join(lines))
print('\n'.join(lines[:22]))
