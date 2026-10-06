# Z-Spacing Resampling — Corrected Results (v2)

**Date:** 2026-10-01  
**Geometry version:** v2 — SimpleITK resampling with explicit reference grid  
**Checkpoint:** `btcv_cross_validation/runs/fold5/ddp_sub_cv/cv_fold5/ckpt/model_epoch_20.pth` (finetuned, BS=4)  
**Input:** Synthetic 6 mm z-spacing volumes in `synthetic_6mm_v2/` (SimpleITK, from 3 mm originals)  
**Patients:** img0036, img0040 (2 patients × 4 organs × 2 branches = **16 predictions**)  
**Protocol:** gt-cropped, box prompt, corrected zero padding (none), NSD tolerance 1 mm  
**Trigger:** resampling applied only when z spacing > 5.5 mm (6 mm synthetic volumes trigger this)  
**Scoring grid:** synthetic 6 mm grid for both branches  
**All 16 predictions completed. Zero failures.**

---

## 1. Geometry corrections made (v1 → v2)

| Issue | v1 | v2 |
|---|---|---|
| Resampler | `scipy.ndimage.zoom` — samples at fractions of `[0, N-1]`, not at voxel centres | `SimpleITK.Resample` with explicit origin/direction/spacing/size |
| Physical origin | Shifted by ~0.5 voxel in output grid | Preserved exactly — shared origin between source and reference grids |
| Slice mapping | `pos = z * source_z + source_z/2` then divided by target_z — extra half-voxel terms | `z_cont = z * source_z / actual_z` — shared origin cancels, no offset |
| Prediction restoration | Manual crop/pad after second zoom call | Same SimpleITK reference grid used forward and backward; output size is symmetric by construction |
| Conditional trigger | Always resampled | Resamples only when z spacing > 5.5 mm (`needs_resampling()`) |
| Synthetic volume creation | `scipy.ndimage.zoom` | `SimpleITK.Resample` with same convention |
| CPU tests | Shape/roundtrip only | Added: physical ramp, landmark alignment, roundtrip scoring-grid size, trigger check (20 tests, all pass) |

---

## 2. Mean Dice — v2 results

| Organ | Control (%) | Proposed (%) | Diff (pp) |
|---|---:|---:|---:|
| Liver | 89.39 | 92.71 | **+3.32** |
| Stomach | 65.67 | 67.30 | **+1.63** |
| Left Kidney | 86.49 | 84.02 | −2.47 |
| Spleen | 84.65 | 88.88 | **+4.23** |
| **4-organ mean** | **81.55** | **83.23** | **+1.68** |

The +1.96 pp gain reported in v1 persists after geometry correction: v2 shows +1.68 pp. The direction and rough magnitude are consistent. Left kidney continues to regress (−2.47 pp in v2 vs −1.21 pp in v1).

---

## 3. v1 vs v2 comparison — geometry correction effect

### Control branch (same input, no resampling — pure synthetic-volume geometry effect)

| Organ | v1 (%) | v2 (%) | Diff |
|---|---:|---:|---:|
| Liver | 89.50 | 89.39 | −0.11 |
| Stomach | 65.52 | 65.67 | +0.15 |
| Left Kidney | 85.51 | 86.49 | +0.98 |
| Spleen | 84.74 | 84.65 | −0.09 |

Control scores are nearly identical (max change 0.98 pp). The synthetic volume geometry fix (scipy → SimpleITK) had minimal effect on scores because both branches see the same input.

### Proposed branch (resampling geometry corrected)

| Organ | v1 (%) | v2 (%) | Diff |
|---|---:|---:|---:|
| Liver | 92.88 | 92.71 | −0.17 |
| Stomach | 65.84 | 67.30 | +1.47 |
| Left Kidney | 84.30 | 84.02 | −0.28 |
| Spleen | 90.08 | 88.88 | −1.21 |

The geometry fix changes proposed-branch scores by up to ±1.5 pp. The net direction of gain (proposed > control) is unchanged for liver, stomach, and spleen. The kidney regression direction is also unchanged.

---

## 4. Per-patient Dice — v2

| Organ | Patient | Control (%) | Proposed (%) | Diff |
|---|---|---:|---:|---:|
| Liver | img0036 | 90.86 | 94.08 | +3.23 |
| Liver | img0040 | 87.92 | 91.33 | +3.41 |
| Stomach | img0036 | 64.94 | 67.81 | +2.87 |
| Stomach | img0040 | 66.40 | 66.80 | +0.40 |
| Left Kidney | img0036 | 87.92 | 85.51 | −2.42 |
| Left Kidney | img0040 | 85.05 | 82.53 | −2.52 |
| Spleen | img0036 | 82.89 | 88.97 | +6.08 |
| Spleen | img0040 | 86.41 | 88.78 | +2.37 |

Gains are consistent across both patients for liver, stomach, and spleen. Left kidney regression is consistent across both patients (−2.42 and −2.52 pp).

---

## 5. Secondary metrics — v2

### NSD @ 1 mm (%)
| Organ | Control | Proposed | Diff |
|---|---:|---:|---:|
| Liver | 39.16 | 52.73 | **+13.57** |
| Stomach | 22.44 | 34.02 | **+11.58** |
| Left Kidney | 51.30 | 49.82 | −1.48 |
| Spleen | 39.55 | 55.01 | **+15.46** |

NSD improvements are consistently larger than Dice improvements for liver, stomach, and spleen. Resampling particularly helps boundary accuracy on these organs.

### HD95 (mm)
| Organ | Control | Proposed | Diff |
|---|---:|---:|---:|
| Liver | 14.38 | 6.90 | −7.48 |
| Stomach | 73.34 | 80.46 | +7.12 |
| Left Kidney | 11.24 | 11.61 | +0.37 |
| Spleen | 8.38 | 5.83 | −2.55 |

Liver and spleen HD95 improve substantially. Stomach and kidney slightly worsen.

---

## 6. Runtime

| Branch | Organ | Mean seconds/patient |
|---|---|---:|
| Control | Liver | 9.59 |
| Control | Stomach | 4.73 |
| Control | Left Kidney | 4.60 |
| Control | Spleen | 4.22 |
| Proposed | Liver | 19.93 |
| Proposed | Stomach | 10.20 |
| Proposed | Left Kidney | 9.14 |
| Proposed | Spleen | 7.32 |

The proposed branch is approximately 2× slower, primarily because resampling doubles the slice count, requiring more encoder passes. This is expected and recorded; it is not a disqualifying factor for an inference-time preprocessing step.

---

## 7. Interpretation

### The gain is real and survives geometry correction
The v1 +1.96 pp gain reduces to +1.68 pp in v2 after fixing the physical-coordinate convention. The sign and approximate magnitude are consistent. This cannot be dismissed as a geometry artefact.

### Left kidney consistently regresses
Both v1 (−1.21 pp) and v2 (−2.47 pp) show left kidney regression. Both patients show the same direction. The kidney is the smallest organ in this set. Doubling the slice count during propagation gives the predictor more steps to accumulate errors, and the nearest-neighbour restoration back to the scoring grid may lose fine structure. This is a genuine weakness of the approach for small organs.

### Stomach improves more in v2 (+1.63 pp) than v1 (+0.31 pp)
The geometry fix matters most for stomach. In v1 the half-voxel offset in prompt mapping may have placed the initial box on a slightly wrong slice for this organ, suppressing the gain. In v2 the physical coordinate mapping is correct and the gain is clearly positive.

### NSD improvements are larger than Dice improvements
This is consistent across all benefiting organs (liver, stomach, spleen). Resampling to finer spacing helps the model produce spatially tighter predictions, improving surface-level accuracy more than volumetric overlap.

### Conditional trigger behaves correctly
The trigger fires for the 6 mm synthetic inputs (> 5.5 mm threshold). For native BTCV scans (max 5.0 mm), the trigger would not fire and both branches would produce identical results. The experiment isolates the resampling effect correctly.

---

## 8. What this does and does not establish

- ✅ With corrected geometry, resampling 6 mm → 3 mm before segmentation improves 4-organ mean Dice (+1.68 pp) and NSD (+13.5 pp mean across 3 organs) on these 2 patients.
- ✅ The geometry correction (v1 → v2) changes scores by at most ±1.5 pp — the earlier v1 result was directionally correct.
- ✅ The `>5.5 mm` conditional trigger is implemented and tested.
- ❌ Only 2 patients. No statistical confidence. Cannot generalise.
- ❌ Left kidney regresses consistently — the approach may need modification for small organs.
- ❌ Synthetic degradation is not equivalent to native thick-slice acquisition.
- ❌ The proposed branch always resamples to 3 mm; the target spacing is not tuned.
- ❌ These patients have been inspected before (fold-5 test set). This is exploratory.

---

## 9. Next steps

1. The left kidney regression warrants investigation before reporting this as a general improvement. Possible approaches: check propagation termination events, try a more conservative target spacing, or restrict resampling to organs where it is consistently beneficial.
2. To extend the cohort without touching test patients, convert 3 mm training patients to 6 mm synthetic volumes and evaluate on those.
3. Do not tune the target spacing or trigger threshold on these 2 patients.
4. The factorial B00/B10/B01/B11 experiment (spacing × padding) is deferred until the left-kidney issue is understood.

---

## 10. Provenance

| Item | Value |
|---|---|
| Checkpoint SHA256 | `235b75d82b5e7e71c9e00c65b2a91d3f02b4cc60f79ccdb6518f138105730785` |
| Synthetic volumes | `btcv_cross_validation/z_spacing/synthetic_6mm_v2/` |
| Results | `btcv_cross_validation/z_spacing/results_v2/fold5_e20_{control,proposed}_label{6,7,3,1}/` |
| Setup log | `btcv_cross_validation/z_spacing/logs/setup_v2.log` |
| Inference logs | `btcv_cross_validation/z_spacing/logs/inference_{control,proposed}_v2.log` |
| v1 backup | `btcv_cross_validation/z_spacing/v1_backup/` |
| v1 results (preserved) | `btcv_cross_validation/z_spacing/results/` |
| CPU tests | 20 tests, all pass (physical ramp, landmark, roundtrip size, trigger) |
