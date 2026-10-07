# Missing padding × prompt-budget and synthetic box-mode evaluations

Prepared 7 October 2026 from `btcv_cross_validation/Missing_Padding_Evaluations_Prompt.txt`.
**No GPU inference has been run.** Completed results were not modified, no
synthetic inputs were regenerated, and no training or cache rebuilding was done.

## What runs (66 arms, sequential)

| Group | What | Folds | Cohort | Code | Output |
| --- | --- | --- | --- | --- | --- |
| A1 | 5% image padding, 1 box | 1–4 | all 6 held-out | `box_padding/evaluate.py` (unchanged) | `box_padding/results/fold{N}_e20_image_label{L}` |
| A2 | historical padding, 5 boxes | 1–5 | all 6 held-out | `five_box_padding/evaluate.py` (copy) | `five_box_padding/results/historical_p0/fold{N}` |
| A3 | 5% image padding, 5 boxes | 1–5 | all 6 held-out | `five_box_padding/evaluate.py` (copy) | `five_box_padding/results/image_p0.05/fold{N}` |
| B | historical boxes, recovery OFF and ON, 1 box | 1–5 | 17 native-3 mm patients (6/4/1/4/2) | `z_spacing_boxmode/evaluate.py` (copy of v3) | `z_spacing_boxmode/results/fold{N}/historical_box_recovery_{off,on}_label{L}` |

`execution_inventory.md` lists every arm with fold, patient IDs, padding, prompt budget, spacing branch, checkpoint, output path and status. It is regenerated on every launcher invocation.

## Reused, not rerun

| Setting | Source | Note |
| --- | --- | --- |
| Historical 1-box, folds 1–4 | `logs/fold{N}_evaluation.json` (legacy evaluator) | Dice only. On fold 5 the new evaluator's historical mode reproduces the legacy per-patient Dice **bit-exactly (24/24, max diff 0)**; the launcher re-asserts this. |
| Historical / corrected-zero / 5% 1-box, fold 5 | `box_padding/results/fold5_e20_*` | 12 arms, validated |
| Corrected-zero 1-box and 5-box, folds 1–5 | `five_box/results/fold{N}` | stage-1 Dice equals `box_padding` corrected-zero fold-5 Dice exactly |
| Corrected tight, recovery OFF/ON (synthetic) | `z_spacing/results_v3/fold{N}/b00_control_*`, `b01_proposed_*` | 40 arms, validated. B00 and B01 **both** use corrected tight boxes. |

## Code changes (copies only; originals byte-identical, enforced by tests)

- `five_box_padding/evaluate.py`: copy of `five_box/evaluate.py` plus `--padding-mode`, `--padding-fraction`, `--output`; padding mode/fraction recorded in the report and every case; stale checkpoint path repaired exactly as `ten_box` does. All other modules are byte-identical copies of `five_box/`. The authors' corrective-slice selection and merge (`author_mbox.py`) are unchanged.
- `z_spacing_boxmode/evaluate.py`: copy of `z_spacing/evaluate_v3.py` plus a mandatory `--box-mode {historical,corrected_tight}` (→ predictor `padding_mode` `historical`/`none`). The v3 crop/pad fallback is replaced by a hard error; it never fired in the 40 completed v3 arms. Every case records `box_mode`, `padding_mode` and SHA-256 of its synthetic image/label. The cropped-origin and exact-reference restoration are unchanged. Images use linear interpolation and labels use nearest neighbour, both from the unchanged `helpers.py`.
- Padding reaches only automatically propagated boxes (`predictor.generate_prompts_from_previous_masks`). Initial GT boxes and the authors' corrective GT boxes (`mask_to_bbox`, no `enlarge`) are never padded. 5% means 0.05 × the 1024-pixel model-input width/height on each side, which is 51.2 px.

## Before launching: preflight (CPU only, ~2 min)

From the repository root:

```bash
cd /home/jupyter-team2pop/Mars/Slide-SAM/Slide-SAM
PYTHONPATH="$PWD" /home/jupyter-team2pop/.conda/envs/slidesam/bin/python \
  btcv_cross_validation/missing_evals/run_missing_evals.py
```

Preflight runs three CPU suites (`missing_evals.test_cpu` 14 tests, `box_padding.test_cpu` 19 tests, `z_spacing.test_geometry_fixes` 12 tests). It also checks:

- train/test disjointness, 6 held-out IDs per fold, and 30 distinct IDs overall;
- each fold's run `config.yaml` points to that fold's manifests;
- each checkpoint path maps from the legacy record, with its SHA-256 equal to the completed five-box run;
- every completed reused result validates;
- the 17-patient cohort equals the eligibility list filtered to 3 mm (it excludes img0009 at 2.5 mm), the synthetic manifests, and the v3 result IDs.

For every patient × organ, it reloads the saved synthetic input and recomputes crop, initial slice, spacing, origin, direction, cropped origin, resampled shape, resampled initial slice and exact-shape restoration. Each value must equal what the v3 runs recorded. This replaces a file-time check, because the repo move reset all mtimes to 2026-10-03 21:28. Finally, preflight runs each evaluator's own preflight. It never imports model weights or touches CUDA.

## Launch (GPU; not started)

```bash
cd /home/jupyter-team2pop/Mars/Slide-SAM/Slide-SAM
PYTHONPATH="$PWD" nohup /home/jupyter-team2pop/.conda/envs/slidesam/bin/python \
  btcv_cross_validation/missing_evals/run_missing_evals.py --execute \
  > btcv_cross_validation/missing_evals/launcher.log 2>&1 &
```

`--only A1 A2 A3 B` restricts groups. Estimated time is about 1–1.5 h on the A6000: A1 ≈ 15 min, A2+A3 ≈ 25 min, B ≈ 20 min, plus model loads. The estimate comes from recorded per-case times.

Runtime settings: the `slidesam` interpreter, cuDNN disabled, seeds 0, float32, `OMP_NUM_THREADS=4`. Each arm writes its own log under `missing_evals/logs/`. Each evaluator writes `results.json` incrementally.

**Skip and stop rules:**

- An arm is skipped only if its output validates as complete: status, all patient × organ IDs, padding/box mode, settings, checkpoint hash, synthetic input hashes, recovery flag, and valid Dice.
- Any other existing output, such as a partial or failed run, stops the launcher before GPU work and is left untouched. Move it aside manually to rerun.
- A failed arm stops the sequence, with its partial output preserved.

## After the run

```bash
/home/jupyter-team2pop/.conda/envs/slidesam/bin/python btcv_cross_validation/missing_evals/summarize.py
```

This writes `report.md` and `per_patient.csv`. Per fold, it reports:

- **Native:** liver, stomach, left kidney, spleen and the four-organ mean for H/Z/P × 1/5 boxes, plus paired per-patient differences, organ-level better/worse counts, worst regression and runtime.
- **Synthetic:** a separate table for the four box-mode × recovery settings, labelled with the patient count (6, 4, 1, 4, 2).

It does not compute an across-fold mean or pool native with synthetic. The authors' five-box numbers appear only as a descriptive five-box comparison, with a split/provenance disclaimer. A consistency check confirms that stage 1 of every five-box run equals its 1-box counterpart; it should be exactly 0.

## Out of scope (not prepared)

- Five-box synthetic-spacing runs.
- 5% padding combined with synthetic spacing.
- Native 2.5 mm/5 mm patients and all-30-patient synthetic runs.
- Re-generating synthetic inputs; tuning spacing, trigger, padding fraction or organ rules.
- Rerunning completed arms, including a separate fold-5-only synthetic run.
