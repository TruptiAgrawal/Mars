# Ten-box corrected-zero evaluation

All five held-out BTCV folds, epoch-20 BS=4 checkpoints, native volumes, four organs. All 120 patient-organ cases completed with ten distinct box prompts each (1200 prompt stages). No retraining or z-resampling. Same protocol as the five-box experiment, extended from four to nine corrective boxes.

| Organ | Authors five-box (%) | Our one box (%) | Our five boxes (%) | Our ten boxes (%) | Ten − five (pp) | Ten − one (pp) | Our ten − authors five (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Liver | 95.62 | 90.15 | 91.03 | 91.05 | +0.02 | +0.90 | -4.57 |
| Stomach | 90.88 | 66.89 | 80.98 | 80.88 | -0.10 | +13.98 | -10.00 |
| Left kidney | 93.49 | 68.85 | 82.77 | 83.38 | +0.62 | +14.54 | -10.11 |
| Spleen | 91.88 | 86.35 | 87.38 | 86.63 | -0.74 | +0.29 | -5.25 |
| **Mean** | **92.96** | **78.06** | **85.54** | **85.49** | **-0.05** | **+7.43** | **-7.47** |

The paper reports five boxes only (Table 1, Ours (5 boxes), [published paper](https://raw.githubusercontent.com/mlresearch/v250/main/assets/quan24a/quan24a.pdf)); the last column compares our ten boxes against their five and is descriptive only.

## Dice by number of boxes (%)

| Organ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Liver | 90.15 | 90.21 | 90.68 | 91.00 | 91.03 | 91.02 | 91.01 | 91.04 | 91.05 | 91.05 |
| Stomach | 66.89 | 70.90 | 76.45 | 79.34 | 80.98 | 81.27 | 81.57 | 81.36 | 81.17 | 80.88 |
| Left kidney | 68.85 | 70.80 | 77.76 | 81.12 | 82.77 | 83.59 | 83.82 | 83.70 | 83.56 | 83.38 |
| Spleen | 86.35 | 86.58 | 87.26 | 87.19 | 87.38 | 87.18 | 86.92 | 86.77 | 86.69 | 86.63 |
| **Mean** | **78.06** | **79.62** | **83.04** | **84.66** | **85.54** | **85.76** | **85.83** | **85.72** | **85.62** | **85.49** |

## Protocol and limits

The first box is taken from the ground-truth maximum-area valid slice. Nine further boxes use the unchanged outer authors’ `Evaluater.finetune_with_more_prompt`: choose the lowest per-slice Dice among unused interior slices, derive a ground-truth box, propagate, then merge using the authors’ confidence rule. This is oracle-assisted interactive correction, not automatic prompting. We report the Dice after the tenth box, not the best over prompt counts, so additional boxes can lower Dice.

Checkpoints were loaded from their current location under the repository; the fold evaluation logs record the former repository path. Checkpoint hashes are compared with the five-box run below.

## Per-fold means

| Fold | Patients | One box (%) | Five boxes (%) | Ten boxes (%) | Ten − five (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 6 | 78.56 | 86.93 | 87.61 | +0.68 |
| 2 | 6 | 74.67 | 84.16 | 84.92 | +0.76 |
| 3 | 6 | 78.10 | 84.76 | 83.54 | -1.22 |
| 4 | 6 | 79.21 | 85.49 | 85.43 | -0.06 |
| 5 | 6 | 79.77 | 86.36 | 85.93 | -0.42 |

## Checks and provenance

- Checkpoint SHA-256 identical to the five-box run for all folds: True.
- First five prompt slices identical to the five-box run in every case: True.
- Maximum absolute difference from the five-box run in Dice at prompt counts 1–5: 0.00e+00.
- Each fold has 24 completed patient-organ cases; all 30 patients appear in one held-out fold; ten distinct prompts per case.
- Native CUDA convolution, cuDNN disabled, float32 evaluation, fixed seeds, no gradients.
- Code copied from `btcv_cross_validation/five_box/`; only `evaluate.py`, `summarize.py` and `README.md` differ.

Raw files: `btcv_cross_validation/ten_box/results/fold{1-5}/results.json`, `per_patient.csv`, `run.log`, `summary.json` and `verification.json`.
