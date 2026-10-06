# Five-box corrected-zero evaluation versus the authors

All five held-out BTCV folds, epoch-20 BS=4 checkpoints, native volumes, four organs. All 120 patient-organ cases completed with five distinct box prompts each (600 prompt stages). No retraining or z-resampling.

| Organ | Authors five-box Dice (%) | Our one-box corrected-zero (%) | Our five-box corrected-zero (%) | Five − one (pp) | Our five − authors (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Liver | 95.62 | 90.15 | 91.03 | +0.88 | -4.59 |
| Stomach | 90.88 | 66.89 | 80.98 | +14.09 | -9.90 |
| Left kidney | 93.49 | 68.85 | 82.77 | +13.92 | -10.72 |
| Spleen | 91.88 | 86.35 | 87.38 | +1.03 | -4.50 |
| **Mean** | **92.96** | **78.06** | **85.54** | **+7.48** | **-7.42** |

Authors’ numbers are from Table 1, Ours (5 boxes), [published paper](https://raw.githubusercontent.com/mlresearch/v250/main/assets/quan24a/quan24a.pdf). The paper prints mean 92.96%; averaging its rounded organ values gives 92.9675%. We retain the published mean as printed.

## Protocol and limits

The first box is taken from the ground-truth maximum-area valid slice. Four further boxes use the unchanged outer authors’ `Evaluater.finetune_with_more_prompt`: choose the lowest per-slice Dice among unused interior slices, derive a ground-truth box, propagate, then merge using the authors’ confidence rule (including doubled confidence on the correction slice). This is oracle-assisted interactive correction, not five uniformly spaced boxes or automatic prompting. The inference method named “finetune” does not train weights.

The original correction method and predictor state/merge behavior are preserved. Corrected-zero propagated boxes replace the malformed historical adjustment. The loader uses explicit z-y-x geometry and ground-truth extent cropping. All five prompts are distinct. Additional prompts are not guaranteed to improve Dice because we do not select the best result across prompt counts.

One-box and five-box scores above are paired within the same run. The one-box row is newly evaluated across all 30 patients; it is not the earlier six-patient Box Z score. Each patient uses its own held-out fold checkpoint. Comparison with the paper is descriptive: author split identities and complete model/training provenance are not established, and corrected-zero propagation intentionally differs from the historical code.

## Per-fold means

| Fold | Patients | One box (%) | Five boxes (%) | Change (pp) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 6 | 78.56 | 86.93 | +8.38 |
| 2 | 6 | 74.67 | 84.16 | +9.49 |
| 3 | 6 | 78.10 | 84.76 | +6.66 |
| 4 | 6 | 79.21 | 85.49 | +6.28 |
| 5 | 6 | 79.77 | 86.36 | +6.59 |

## Checks and provenance

- Same checkpoint, patient, image and initial prompt within every one/five-box pair.
- Each fold has 24 completed patient-organ cases; all 30 patients appear in one held-out fold.
- Native CUDA convolution, cuDNN disabled, float32 evaluation, fixed seeds, no gradients.
- Original source copies and author multi-box evaluator are preserved under `btcv_cross_validation/five_box/originals/`.
- Results record checkpoint/source hashes, prompt slices and boxes, and Dice after every prompt.
- The failed first launch used incompatible system Python and produced no scores. The successful run uses the explicit `slidesam` environment.
- This evaluation reports Dice, not NSD/HD95 or clinical annotation time.

Raw files: `btcv_cross_validation/five_box/results/fold{1-5}/results.json`, `per_patient.csv`, `run.log` and `summary.json`.
