# Box Padding First Results

**Date:** 2026-10-01  
**Checkpoint:** `btcv_cross_validation/runs/fold5/ddp_sub_cv/cv_fold5/ckpt/model_epoch_20.pth` (finetuned, BS=4)  
**Patients:** img0035–img0040 (fold-5 held-out, 6 patients)  
**Organs:** Liver=6, Stomach=7, Left Kidney=3, Spleen=1  
**Protocol:** gt-cropped, box prompt, NSD tolerance 1 mm  
**Evaluator:** `btcv_cross_validation/box_padding/` (isolated experimental copy)  
**All 72 predictions completed. Zero empty predictions.**

---

## 1. Key result

**Corrected zero padding (Z) improves the four-organ mean Dice by +8.57 pp over historical (H).**  
**5% image-relative padding (P5) degrades performance by −21.54 pp versus Z, and −12.98 pp versus H.**

P5 is not a useful setting for this checkpoint at this fraction. Z is the effective contribution: fixing the historical `enlarge()` bug alone (wrong signs, mixed x/y margins) produces the gain.

---

## 2. Mean Dice summary

| Organ | H (%) | Z (%) | P5 (%) | P5−Z (pp) | Z−H (pp) | P5−H (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Liver | 79.0197 | 91.2589 | 80.7201 | −10.54 | **+12.24** | +1.70 |
| Stomach | 61.5634 | 70.3396 | 56.2438 | −14.10 | **+8.78** | −5.32 |
| Left Kidney | 67.7364 | 72.4751 | 30.9726 | −41.50 | **+4.74** | −36.76 |
| Spleen | 76.4808 | 84.9941 | 64.9563 | −20.04 | **+8.51** | −11.52 |
| **4-organ mean** | **71.2001** | **79.7669** | **58.2232** | **−21.54** | **+8.57** | **−12.98** |

H = historical `enlarge()` arithmetic · Z = corrected tight box · P5 = image-relative 5% padding per side

---

## 3. Per-patient Dice

### Liver (label 6)
| Patient | H (%) | Z (%) | P5 (%) | P5−Z | Z−H |
|---|---:|---:|---:|---:|---:|
| img0035 | 80.57 | 89.92 | 83.33 | −6.59 | +9.34 |
| img0036 | 75.80 | 94.35 | 83.12 | −11.23 | +18.55 |
| img0037 | 82.73 | 88.74 | 74.76 | −13.98 | +6.01 |
| img0038 | 79.93 | 89.61 | 73.59 | −16.02 | +9.67 |
| img0039 | 83.77 | 92.45 | 82.82 | −9.62 | +8.68 |
| img0040 | 71.31 | 92.49 | 86.70 | −5.79 | +21.18 |

### Stomach (label 7)
| Patient | H (%) | Z (%) | P5 (%) | P5−Z | Z−H |
|---|---:|---:|---:|---:|---:|
| img0035 | 59.31 | 67.29 | 57.62 | −9.67 | +7.98 |
| img0036 | 60.99 | 66.78 | 48.35 | −18.43 | +5.79 |
| img0037 | 69.15 | 77.58 | 41.42 | −36.16 | +8.43 |
| img0038 | 67.82 | 78.35 | 76.86 | −1.48 | +10.53 |
| img0039 | 60.78 | 62.73 | 49.32 | −13.42 | +1.95 |
| img0040 | 51.33 | 69.32 | 63.90 | −5.42 | +17.99 |

### Left Kidney (label 3)
| Patient | H (%) | Z (%) | P5 (%) | P5−Z | Z−H |
|---|---:|---:|---:|---:|---:|
| img0035 | 64.57 | 72.77 | 26.76 | −46.01 | +8.20 |
| img0036 | 77.78 | 91.62 | 24.75 | −66.87 | +13.84 |
| img0037 | 66.44 | 69.83 | 46.53 | −23.30 | +3.39 |
| img0038 | 65.53 | 63.09 | 38.14 | −24.95 | −2.44 |
| img0039 | 60.67 | 55.97 | 27.84 | −28.13 | −4.70 |
| img0040 | 71.43 | 81.58 | 21.82 | −59.76 | +10.15 |

### Spleen (label 1)
| Patient | H (%) | Z (%) | P5 (%) | P5−Z | Z−H |
|---|---:|---:|---:|---:|---:|
| img0035 | 67.58 | 83.36 | 72.91 | −10.45 | +15.78 |
| img0036 | 79.68 | 90.22 | 65.27 | −24.95 | +10.53 |
| img0037 | 64.59 | 70.25 | 64.06 | −6.19 | +5.66 |
| img0038 | 76.33 | 86.13 | 67.22 | −18.91 | +9.80 |
| img0039 | 85.29 | 90.68 | 70.66 | −20.02 | +5.39 |
| img0040 | 85.41 | 89.32 | 49.62 | −39.71 | +3.91 |

---

## 4. Secondary metrics

### NSD @ 1 mm (%)
| Organ | H | Z | P5 |
|---|---:|---:|---:|
| Liver | 28.65 | 42.28 | 16.40 |
| Stomach | 26.26 | 28.62 | 14.66 |
| Left Kidney | 38.05 | 39.27 | 16.58 |
| Spleen | 30.19 | 42.90 | 14.61 |

Z improves NSD consistently over H. P5 degrades NSD for all organs.

### HD95 (mm)
| Organ | H | Z | P5 |
|---|---:|---:|---:|
| Liver | 40.64 | 12.13 | 39.83 |
| Stomach | 67.24 | 61.55 | 46.04 |
| Left Kidney | 27.70 | 22.16 | 55.10 |
| Spleen | 24.22 | 12.02 | 40.75 |

Z substantially reduces HD95 for liver and spleen. P5 increases HD95 for left kidney and spleen.

### Recall / Precision / False Positive Volume
| Organ | Mode | Recall (%) | Precision (%) | FP (mL) |
|---|---|---:|---:|---:|
| Liver | H | 71.44 | 89.32 | 156.4 |
| Liver | Z | 98.68 | 84.92 | 318.9 |
| Liver | P5 | 81.35 | 80.48 | 351.8 |
| Stomach | H | 49.01 | 84.18 | 46.5 |
| Stomach | Z | 64.29 | 79.91 | 91.1 |
| Stomach | P5 | 52.68 | 66.55 | 162.6 |
| Left Kidney | H | 60.82 | 76.46 | 32.9 |
| Left Kidney | Z | 68.78 | 78.16 | 32.7 |
| Left Kidney | P5 | 22.91 | 61.73 | 49.2 |
| Spleen | H | 76.32 | 77.14 | 68.4 |
| Spleen | Z | 89.19 | 81.48 | 61.6 |
| Spleen | P5 | 65.59 | 68.46 | 104.6 |

---

## 5. Interpretation

### What worked: Z (corrected tight box)
The historical `enlarge()` implementation in `BoxPromptGenerator` has three bugs: it subtracts the margin from `x1`/`y1` instead of adding (shrinks the upper-right), uses `margin_x` for `y0`, and ignores its `margin` argument. Fixing this alone — i.e., using a clean tight box with no expansion — produces +8.57 pp on the four-organ mean Dice. This is the Z−H effect. It is a **baseline correction**, not a novel algorithm contribution. Every four-organ mean with Z exceeds H.

### What did not work: P5 (5% image-relative padding)
Expanding propagated boxes by 5% of the image dimensions on each side severely degrades performance, especially for left kidney (−41.50 pp vs Z). The P5 recall numbers partially recover vs H for liver and stomach, but false-positive volume increases substantially for all organs, and precision drops. The Dice and NSD degradation is consistent and large.

The most likely cause: 5% of a 1024×1024 model-input image is ~51 pixels per side. For a small organ like the kidney, this roughly doubles the box area, pulling in competing anatomy and generating false-positive segmentations. The propagated mask quality deteriorates across slices as false positives compound.

### Left kidney is most sensitive
P5 drops left kidney Dice to 30.97% (from 72.47% at Z). Per-patient scores range 21.82–46.53%, compared to 55.97–91.62% at Z. The kidney is small and surrounded by adjacent anatomy; large box expansion is particularly harmful here.

### Z vs the historical comparison checkpoint
The historical trajectory result (71.2001%) used the authors' frozen evaluator, not this experimental evaluator. The H arm here (71.2001%) coincidentally matches that figure for the four-organ mean Dice, but metrics like NSD and HD95 are computed differently. Do not subtract old evaluator scores from new evaluator scores; compare only within this experiment.

---

## 6. What this does and does not establish

- ✅ Fixing the `enlarge()` bug improves Dice, NSD and HD95 consistently for this checkpoint on these 6 patients.
- ✅ 5% image-relative expansion at propagation time hurts performance for this checkpoint on these 6 patients.
- ❌ Does not establish what padding fraction (if any) would help — P5 may be too large; a smaller fraction was not tested here.
- ❌ Does not generalise to other checkpoints, folds, or patient populations.
- ❌ Does not explain or fix the fine-tuning degradation identified in `Early_Finetuning_Audit.md`.
- ❌ These 6 patients have been inspected repeatedly. This is exploratory, not a confirmatory result.

---

## 7. Next steps per Novelty_First_Next_Steps_2026-10-01.md

1. **Separate the bug-fix from the padding contribution.** Z−H (+8.57 pp) is a code correction. Any future padding comparison must use Z as the control, not H.
2. **Test smaller padding fractions** (e.g. p=0.01, 0.02) on development/validation patients only — not on these 6 test patients again.
3. **Consider box-relative padding** as an alternative definition (5% of box width/height rather than image dimensions) — this would add far fewer pixels for small organs.
4. **Prepare z-spacing handling** (Section 8 of Novelty_First_Next_Steps) as the next novelty experiment.
5. **Do not re-tune on these 6 patients.** Use inner-validation patients for any hyperparameter search.

---

## 8. Provenance

| Item | Value |
|---|---|
| Checkpoint SHA256 | `235b75d82b5e7e71c9e00c65b2a91d3f02b4cc60f79ccdb6518f138105730785` |
| Manifest SHA256 | `224f08d7cd9f1660fb5158326dc95dbbb2ca1193a56ecbc9202e54ed87680457` |
| Training manifest SHA256 | `8315f3cb35b20e023f2f923ac9f32e017955f69e3645385295a69f0c1cee4445` |
| Results directories | `btcv_cross_validation/box_padding/results/fold5_e20_{historical,none,image}_label{6,7,3,1}/` |
| Logs | `btcv_cross_validation/box_padding/logs/20261001_110432_{historical,none,image}.log` |
| Run script | `btcv_cross_validation/box_padding/run_padding_eval.sh` |
| Peak GPU memory | ~11.6 GB (per run) |
| Inference time | ~9 sec per patient per organ |
