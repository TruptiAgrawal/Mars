# Z-Spacing Resampling First Results

**Date:** 2026-10-01  
**Checkpoint:** `btcv_cross_validation/runs/fold5/ddp_sub_cv/cv_fold5/ckpt/model_epoch_20.pth` (finetuned, BS=4)  
**Input:** Synthetic 6 mm z-spacing volumes downsampled from 3 mm BTCV originals  
**Patients:** img0036, img0040 (only 2 of 6 fold-5 patients are at 3 mm; the other 4 are natively 5 mm)  
**Organs:** Liver=6, Stomach=7, Left Kidney=3, Spleen=1  
**Protocol:** gt-cropped, box prompt, NSD tolerance 1 mm, corrected zero padding (none) in both branches  
**Scoring grid:** synthetic 6 mm grid for both branches  
**All 8 predictions completed successfully.**

---

## 1. Key result

**Resampling 6 mm → 3 mm before segmentation improves the 4-organ mean Dice by +1.96 pp and NSD by +11.37 pp** on these 2 patients. The benefit is concentrated in liver and spleen; left kidney shows a small regression (−1.21 pp Dice).

Importantly, precision improves and false-positive volume drops for all organs, meaning the improvement comes from cleaner segmentation, not just larger predictions.

---

## 2. Mean Dice summary

| Organ | Control (%) | Proposed (%) | Diff (pp) |
|---|---:|---:|---:|
| Liver | 89.50 | 92.88 | **+3.38** |
| Stomach | 65.52 | 65.84 | +0.31 |
| Left Kidney | 85.51 | 84.30 | −1.21 |
| Spleen | 84.74 | 90.08 | **+5.34** |
| **4-organ mean** | **81.32** | **83.27** | **+1.96** |

Control = segment directly on 6 mm grid with corrected zero padding  
Proposed = resample 6 mm → 3 mm, segment, map predictions back to 6 mm grid

---

## 3. Per-patient Dice (%)

| Organ | Patient | Control | Proposed | Diff |
|---|---|---:|---:|---:|
| Liver | img0036 | 91.07 | 94.43 | +3.35 |
| Liver | img0040 | 87.92 | 91.34 | +3.41 |
| Stomach | img0036 | 64.65 | 66.30 | +1.65 |
| Stomach | img0040 | 66.40 | 65.38 | −1.03 |
| Left Kidney | img0036 | 85.96 | 85.68 | −0.28 |
| Left Kidney | img0040 | 85.05 | 82.92 | −2.13 |
| Spleen | img0036 | 83.08 | 90.03 | +6.95 |
| Spleen | img0040 | 86.41 | 90.13 | +3.73 |

Both patients show consistent gains for liver and spleen. Left kidney is consistently slightly worse in both.

---

## 4. Secondary metrics

### NSD @ 1 mm (%)
| Organ | Control | Proposed | Diff |
|---|---:|---:|---:|
| Liver | 40.23 | 55.58 | **+15.35** |
| Stomach | 23.77 | 32.24 | +8.47 |
| Left Kidney | 48.67 | 49.63 | +0.96 |
| Spleen | 40.02 | 60.72 | **+20.70** |

NSD improvements are consistently larger than Dice improvements — resampling particularly helps boundary accuracy.

### HD95 (mm)
| Organ | Control | Proposed | Diff |
|---|---:|---:|---:|
| Liver | 14.42 | 7.47 | −6.95 |
| Stomach | 75.31 | 82.14 | +6.83 |
| Left Kidney | 11.48 | 11.40 | −0.08 |
| Spleen | 8.63 | 5.61 | −3.02 |

Liver and spleen HD95 improve substantially. Stomach HD95 worsens slightly.

### Recall / Precision / False Positive Volume
| Organ | Control recall | Proposed recall | Control prec | Proposed prec | Control FP (mL) | Proposed FP (mL) |
|---|---:|---:|---:|---:|---:|---:|
| Liver | 99.3% | 99.0% | 81.5% | 87.5% | 410.2 | 252.1 |
| Stomach | 57.6% | 53.3% | 76.0% | 86.2% | 99.9 | 45.2 |
| Left Kidney | 91.9% | 86.8% | 80.2% | 82.0% | 46.3 | 39.1 |
| Spleen | 96.8% | 95.0% | 75.5% | 85.7% | 83.2 | 39.2 |

A consistent pattern: resampling slightly reduces recall but substantially increases precision and reduces false positive volume across all organs. The model produces tighter, more accurate predictions on the denser resampled grid.

---

## 5. Interpretation

### What worked
Resampling thick-spacing (6 mm) volumes to finer spacing (3 mm) before inference improves Dice and dramatically improves NSD and false positive volume for liver and spleen. The model was trained on 3 mm and 5 mm data — presenting thick-slice inputs at a finer apparent spacing is beneficial for these organs.

### Left kidney regression
Left kidney shows a consistent small Dice regression (−1.21 pp) in both patients. Possible reasons:
- The kidney is smaller; resampling doubles the slice count, giving the propagator more steps to accumulate errors.
- The model's propagation terminates ("No masks / Ends for no mask") for stomach in the proposed branch — the same termination risk exists for kidney.
- The nearest-neighbour mapping of predictions back to the 6 mm grid may lose thin structures.

This is worth watching but is small (−1.21 pp) relative to the liver/spleen gains.

### "No masks" events
The predictor printed `No masks / Ends for no mask` for stomach in both branches and for spleen in the proposed branch on some windows. These are mid-propagation terminations, not total failures — the final Dice scores are reasonable. They indicate the propagator drops the mask on some windows and then resumes.

### What this does not establish
- Only 2 patients. Both happen to be the 3 mm cases in the fold-5 test set. Results cannot be generalised.
- The 5.5 mm trigger was not tested — these 6 mm synthetic volumes are above it, so the rule would activate. But these are synthetically degraded, not naturally thick.
- Resampling 3 mm → 6 mm → back is not equivalent to native thick-slice acquisition; interpolation artefacts differ.
- The proposed branch sees denser slices (~2× the count), so the propagator does more work. Runtime and memory are not reported here but are higher for the proposed branch.

---

## 6. Next steps

1. **Extend to more patients.** Use 3 mm training patients (img0001–img0010, img0021–img0034 excluding 5 mm ones) for a larger synthetic cohort — still exploratory but more informative.
2. **Investigate left kidney regression.** Check whether the propagation termination events and prediction mapping are causing the drop.
3. **Test on natively thick data.** The 5 mm BTCV patients (img0035, img0037–img0039) are not thick enough to trigger the >5.5 mm rule. For a fair test of the proposed rule, need genuinely thick-slice external data or larger synthetic degradation (e.g. 3 mm → 9 mm).
4. **Compare against the box-padding Z result.** On img0036 and img0040 under the box_padding experiment (Z arm), liver scored 94.35% and 92.49% respectively. The proposed resampling branch scores 94.43% and 91.34% — comparable on liver, different dataset (native 3 mm vs synthetic 6 mm).
5. **Do not tune on these 2 patients.** They are test cases. Any parameter search (target spacing, trigger threshold) must use development/training patients.

---

## 7. Provenance

| Item | Value |
|---|---|
| Checkpoint SHA256 | `235b75d82b5e7e71c9e00c65b2a91d3f02b4cc60f79ccdb6518f138105730785` |
| Synthetic volumes | `btcv_cross_validation/z_spacing/synthetic_6mm/` |
| Synthetic meta | `btcv_cross_validation/z_spacing/synthetic_6mm/meta.json` |
| Results | `btcv_cross_validation/z_spacing/results/fold5_e20_{control,proposed}_label{6,7,3,1}/` |
| Setup log | `btcv_cross_validation/z_spacing/logs/setup.log` |
| Inference logs | `btcv_cross_validation/z_spacing/logs/inference_{control,proposed}.log` |
| Scoring grid | Synthetic 6 mm z-spacing grid (both branches) |
| Source volumes | img0036 (3 mm → 6.000 mm, 184→92 slices), img0040 (3 mm → 5.969 mm, 195→98 slices) |
