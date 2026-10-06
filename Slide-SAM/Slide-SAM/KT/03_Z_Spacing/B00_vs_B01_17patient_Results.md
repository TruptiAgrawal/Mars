# B00 vs B01: 17-Patient Spacing Experiment Results

**Date:** 2026-10-01  
**Geometry version:** v3 — SimpleITK resampling, cropped z-origin (Fix 1), exact scoring-grid restore (Fix 2)  
**Checkpoint:** Per-fold epoch-20 checkpoints (finetuned, BS=4)  
**Patients:** 17 eligible patients with z ≤ 3 mm from held-out test sets of all 5 folds  
**Input:** Synthetic 6 mm z-spacing volumes (downsampled from 3 mm originals via SimpleITK)  
**Organs:** Liver=6, Stomach=7, Left Kidney=3, Spleen=1  
**Protocol:** gt-cropped, box prompt, corrected-zero padding (none), NSD tolerance 1 mm  
**Trigger:** B01 resamples only when z spacing > 5.5 mm (6 mm synthetic inputs trigger this)  
**Total predictions:** 17 × 4 × 2 = **136. All complete, zero failures.**

---

## 1. Cohort

| Fold | Patients | Checkpoint |
|---|---|---|
| 1 | img0001–img0006 (6 patients, 3 mm) | fold1/cv_fold1/ckpt/model_epoch_20.pth |
| 2 | img0007, img0008, img0010, img0021 (4 patients, 3 mm) | fold2/cv_fold2/ckpt/model_epoch_20.pth |
| 3 | img0024 (1 patient, 3 mm) | fold3/cv_fold3/ckpt/model_epoch_20.pth |
| 4 | img0029–img0032 (4 patients, 3 mm) | fold4/cv_fold4/ckpt/model_epoch_20.pth |
| 5 | img0036, img0040 (2 patients, 3 mm) | fold5/cv_fold5/ckpt/model_epoch_20.pth |

Note: img0009 (fold 2, 2.5 mm) was found eligible but skipped by make_synthetic.py which only processes exactly 3 mm volumes. Total is 17 patients. Each patient is evaluated with the checkpoint trained on the other 24 volumes of that fold — no patient-checkpoint leakage.

---

## 2. Summary results

| Organ | B00 (%) | B01 (%) | Diff (pp) | Wins | Losses |
|---|---:|---:|---:|---:|---:|
| Liver | 85.78 | 89.83 | **+4.05** | 16/17 | 1/17 |
| Stomach | 64.48 | 64.22 | −0.26 | 9/17 | 8/17 |
| Left Kidney | 72.40 | 68.57 | **−3.84** | 2/17 | 15/17 |
| Spleen | 81.88 | 85.23 | **+3.36** | 14/17 | 3/17 |
| **4-organ mean** | **76.13** | **76.96** | **+0.83** | — | — |

B00 = control (segment 6 mm directly, corrected-zero padding)  
B01 = proposed (resample 6 mm → ~3 mm, segment, restore to 6 mm scoring grid)

---

## 3. Secondary metrics

### NSD @ 1 mm (%)
| Organ | B00 | B01 | Diff |
|---|---:|---:|---:|
| Liver | 34.42 | 49.78 | **+15.36** |
| Stomach | 23.57 | 30.58 | **+7.01** |
| Left Kidney | 41.68 | 41.54 | −0.14 |
| Spleen | 41.19 | 53.57 | **+12.38** |

NSD improvements for liver, stomach, and spleen are substantially larger than Dice improvements. Left kidney NSD is nearly identical.

### HD95 (mm)
| Organ | B00 | B01 | Diff |
|---|---:|---:|---:|
| Liver | 18.54 | 13.43 | −5.10 |
| Stomach | 62.29 | 67.87 | +5.58 |
| Left Kidney | 16.68 | 21.64 | +4.96 |
| Spleen | 12.25 | 10.52 | −1.73 |

### Recall and Precision
| Organ | B00 recall | B01 recall | Diff | B00 prec | B01 prec | Diff |
|---|---:|---:|---:|---:|---:|---:|
| Liver | 93.68% | 94.30% | +0.62 | 79.69% | 86.32% | **+6.63** |
| Stomach | 63.74% | 58.79% | −4.94 | 67.98% | 74.04% | **+6.06** |
| Left Kidney | 71.58% | 63.25% | **−8.33** | 74.19% | 76.69% | +2.50 |
| Spleen | 90.93% | 92.68% | +1.76 | 75.18% | 79.52% | **+4.34** |

A consistent pattern across all organs: precision increases with B01 (tighter, more accurate predictions on the denser resampled grid). Recall drops for stomach and kidney; it is stable or slightly higher for liver and spleen.

---

## 4. Per-patient paired Dice (%)

| Patient | Fold | Liver B00→B01 | Stomach B00→B01 | Kidney B00→B01 | Spleen B00→B01 |
|---|---|---|---|---|---|
| img0001 | 1 | 91.09→92.50 (+1.41) | 75.84→72.33 (−3.51) | 79.93→71.67 (−8.26) | 87.30→91.43 (+4.13) |
| img0002 | 1 | 90.47→90.84 (+0.37) | 58.09→54.48 (−3.61) | 79.06→70.11 (−8.94) | 79.61→81.99 (+2.38) |
| img0003 | 1 | 88.30→91.69 (+3.39) | 83.08→84.60 (+1.52) | 74.24→75.85 (+1.61) | 81.93→83.26 (+1.33) |
| img0004 | 1 | 84.92→90.97 (+6.05) | 61.97→57.24 (−4.73) | 86.11→88.17 (+2.06) | 85.19→89.48 (+4.29) |
| img0005 | 1 | 75.41→69.31 (−6.10) | 79.09→78.48 (−0.60) | 70.59→69.80 (−0.79) | 89.51→91.72 (+2.21) |
| img0006 | 1 | 88.12→90.52 (+2.40) | 44.18→46.41 (+2.23) | 76.23→71.05 (−5.18) | 68.86→65.57 (−3.29) |
| img0007 | 2 | 86.27→90.94 (+4.68) | 48.04→43.05 (−4.98) | 58.93→52.29 (−6.63) | 68.25→85.66 (+17.42) |
| img0008 | 2 | 67.32→89.06 (+21.74) | 65.95→63.30 (−2.64) | 24.93→22.50 (−2.42) | 82.88→85.98 (+3.10) |
| img0010 | 2 | 87.10→93.10 (+6.00) | 65.07→59.02 (−6.05) | 82.77→81.85 (−0.92) | 87.45→85.88 (−1.57) |
| img0021 | 2 | 88.34→91.82 (+3.48) | 65.37→68.19 (+2.82) | 75.13→67.20 (−7.93) | 75.95→74.18 (−1.76) |
| img0024 | 3 | 85.58→87.76 (+2.18) | 68.30→70.25 (+1.96) | 72.96→68.12 (−4.84) | 84.68→88.47 (+3.79) |
| img0029 | 4 | 87.79→91.90 (+4.11) | 58.83→54.70 (−4.13) | 70.77→56.15 (−14.62) | 81.52→84.44 (+2.92) |
| img0030 | 4 | 87.98→92.31 (+4.32) | 63.17→66.73 (+3.57) | 70.93→70.39 (−0.53) | 87.01→91.87 (+4.85) |
| img0031 | 4 | 84.38→87.65 (+3.27) | 58.18→62.30 (+4.12) | 65.45→65.21 (−0.24) | 77.67→84.41 (+6.73) |
| img0032 | 4 | 86.40→91.39 (+4.98) | 69.64→76.08 (+6.43) | 69.88→67.20 (−2.68) | 84.78→86.89 (+2.11) |
| img0036 | 5 | 90.86→94.08 (+3.23) | 64.94→67.81 (+2.87) | 87.92→85.51 (−2.42) | 82.89→88.97 (+6.08) |
| img0040 | 5 | 87.92→91.33 (+3.41) | 66.40→66.80 (+0.40) | 85.05→82.53 (−2.52) | 86.41→88.78 (+2.37) |

---

## 5. Interpretation

### Liver and spleen: consistent and large gain
Liver improves in 16/17 patients (+4.05 pp mean Dice, +15.36 pp NSD). Spleen improves in 14/17 (+3.36 pp Dice, +12.38 pp NSD). These are large and consistent. Resampling to finer spacing substantially helps the model produce accurate, tight predictions for these large organs. The single liver outlier (img0005, −6.10 pp) is worth investigating in a future audit.

### Kidney: consistent and large regression
Left kidney worsens in 15/17 patients (−3.84 pp mean Dice). The two wins are small (+1.61 pp for img0003, +2.06 pp for img0004). The worst loss is img0029 (−14.62 pp). This is not a marginal effect — it is a clear and consistent pattern.

From the kidney audit (2 patients): the regression is caused by recall drops (not precision). The proposed branch sees ~2× as many slices in the resampled grid, meaning more propagation steps. Some of those steps produce empty masks ("No masks / Ends for no mask"), reducing recall. The kidney is smaller than liver and spleen, making it more vulnerable to propagation drift across the additional steps.

### Stomach: no reliable effect
Stomach shows 9/17 wins with −0.26 pp mean Dice — effectively neutral. The NSD improvement (+7.01 pp) suggests boundary quality improves on average even when Dice is mixed. The large Dice variance for stomach (range −6.05 to +6.43 pp) reflects that stomach segmentation is inherently variable.

### Precision increases for all organs
All four organs show higher precision with B01. The model makes tighter, higher-confidence predictions at the finer resampled spacing. This comes at the cost of recall for smaller organs (stomach, kidney).

### The net result at 4-organ mean is modest (+0.83 pp)
The liver/spleen gains (+4.05 and +3.36 pp) are partially cancelled by the kidney loss (−3.84 pp). Equal-organ weighting hides that liver and spleen see the largest benefit. If the method were restricted to liver and spleen only, the gain would be +3.70 pp mean Dice with 30/34 wins.

---

## 6. What this does and does not establish

- ✅ Resampling improves liver (16/17) and spleen (14/17) consistently across 5 folds.
- ✅ Resampling reliably worsens left kidney (15/17) across 5 folds.
- ✅ Precision increases for all organs — resampling produces tighter predictions.
- ✅ NSD improvements are larger than Dice improvements for all benefiting organs.
- ❌ The 4-organ mean gain (+0.83 pp) is modest and depends heavily on the kidney regression.
- ❌ These are all synthetic thick-spacing inputs, not native thick-slice acquisitions.
- ❌ Each fold uses its own fine-tuned checkpoint. Cross-fold variation in checkpoint quality may contribute to some per-patient variance.
- ❌ Stomach result is inconclusive.
- ❌ The kidney regression mechanism (recall drop from additional propagation steps) is identified but not yet fixed.

---

## 7. Runtime

| Branch | Mean seconds per patient per organ | Total GPU time |
|---|---:|---:|
| B00 (control) | 5.01 s | ~5.7 min |
| B01 (proposed) | 10.40 s | ~11.8 min |

B01 is approximately 2× slower due to doubled slice count in the resampled grid.

---

## 8. Next decision

The kidney regression is the blocking issue. The likely cause is that doubling the slice count during propagation accumulates termination events. Candidate mitigations (to be investigated on separate development patients, not these test cases):

1. Use a less aggressive target spacing (e.g. 4–5 mm instead of 3 mm) to reduce slice-count inflation
2. Investigate whether the kidney crop extent is unusually short (fewer starting slices, so doubling is more harmful)
3. Test whether restricting resampling to a smaller crop window reduces accumulated error

The full spacing × padding factorial (B00/B10/B01/B11) remains deferred until the kidney issue is understood.

---

## 9. Provenance

| Item | Value |
|---|---|
| Geometry version | v3 (evaluate_v3.py) |
| CPU geometry tests | 12/12 pass (test_geometry_fixes.py) |
| Synthetic volumes | `btcv_cross_validation/z_spacing/synthetic_6mm_v3/fold{1-5}/` |
| Results | `btcv_cross_validation/z_spacing/results_v3/fold{1-5}/` |
| Logs | `btcv_cross_validation/z_spacing/logs/inference_v3_fold{1-5}.log` |
| Eligible patient list | `btcv_cross_validation/z_spacing/eligible_patients.json` |
| Summary script | `btcv_cross_validation/z_spacing/summarise_b00_b01.py` |
