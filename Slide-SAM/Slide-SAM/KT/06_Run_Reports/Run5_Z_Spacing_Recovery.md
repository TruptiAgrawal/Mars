# Run 5 — Z-spacing recovery on synthetic thick scans (v3, 17 patients)

Recovery helped liver and spleen but hurt left kidney in 15 of 17 patients. The overall gain was small: +0.83 pp across the 17 patients.

**Goal.** Test whether thick-slice scans segment better if they are first resampled to thinner slices.

**What was done.** We took the 17 held-out patients with native 3 mm slices and made synthetic 6 mm versions by removing slices. Each was segmented twice with one corrected-zero box:

- **B00, recovery OFF:** segment the 6 mm scan directly.
- **B01, recovery ON:** resample to about 3 mm, segment, then map the result back to the 6 mm grid.

Both arms are scored against the same 6 mm labels. The patients per fold are 6, 4, 1, 4 and 2, not all 30.

**What changed, and why.** The resampling step, triggered only when slices are thicker than 5.5 mm. It took three versions: v1 and v2 had geometry errors (half-slice shifts and a wrong crop origin) that were fixed before these results.

**Before and after (four-organ mean Dice %).**

| Fold (patients) | B00, OFF (before) | B01, ON (after) | Change (pp) |
| --- | ---: | ---: | ---: |
| 1 (6) | 78.30 | 77.89 | −0.41 |
| 2 (4) | 70.61 | 72.13 | +1.52 |
| 3 (1) | 77.88 | 78.65 | +0.77 |
| 4 (4) | 75.28 | 76.85 | +1.57 |
| 5 (2) | 81.55 | 83.23 | +1.68 |

**What improved.** Liver (+4.05 pp, better in 16/17 patients) and spleen (+3.36 pp, 14/17). Boundary accuracy (NSD) rose for liver, stomach and spleen.

**Still not working.**

- Left kidney got worse: −3.84 pp, lower in 15/17 patients, because more kidney was missed.
- Stomach is a coin flip (9 better, 8 worse).
- Folds 3 and 5 have only 1 and 2 patients.

**Key findings.** Resampling makes predictions tighter (higher precision) but can miss parts of small organs. It cannot restore anatomy removed from the thick scan. These synthetic scores can't be compared with native-scan or paper scores.

Sources: `KT/03_Z_Spacing/B00_vs_B01_17patient_Results.md`, `btcv_cross_validation/z_spacing/results_v3/`.
