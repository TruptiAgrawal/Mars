# BTCV five-fold cross-validation

Epoch 20; prompt: box. All 30 held-out volumes validated against manifests.

Each fold score is the unweighted mean over its six held-out volumes. Final mean and sample SD (ddof=1) are calculated across the five fold scores. The macro row averages four organs within each fold before calculating CV mean/SD.

CSV/JSON retain original metric units. Dice in this table is a percentage; other metrics retain their input units.

| Organ | Metric | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± SD |
|---|---|---:|---:|---:|---:|---:|---:|
| Liver | dice | 74.87 | 68.39 | 77.64 | 77.14 | 79.02 | 75.41 ± 4.20 |
| Stomach | dice | 57.83 | 57.55 | 61.27 | 63.32 | 61.56 | 60.31 ± 2.51 |
| Left Kidney | dice | 70.19 | 61.71 | 69.02 | 69.62 | 67.74 | 67.66 ± 3.45 |
| Spleen | dice | 76.25 | 67.15 | 79.43 | 75.05 | 76.48 | 74.87 ± 4.61 |
| Four-organ macro average | dice | 69.78 | 63.70 | 71.84 | 71.28 | 71.20 | 69.56 ± 3.36 |

Only metrics present as per-volume measurements in all folds are aggregated. The current eval_fold.py records Dice only; box-prompt NSD=0 is a placeholder, so it must not be reported as measured NSD. This script validates saved results and manifests; it does not rerun inference or independently verify checkpoint provenance.
