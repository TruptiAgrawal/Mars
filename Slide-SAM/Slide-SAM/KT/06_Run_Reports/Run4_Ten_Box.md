# Run 4 — Ten boxes with corrected-zero boxes (all folds)

Going from five to ten boxes made no real difference: the fold means changed by −1.22 to +0.76 pp.

**Goal.** Check whether more corrective boxes keep improving the result.

**What was done.** Same as Run 3, with nine corrective boxes instead of four. The first five boxes matched Run 3 exactly, which confirms the setup was the same.

**What changed, and why.** Only the number of boxes, to find where the benefit levels off.

**Before and after (four-organ mean Dice %, 6 patients per fold).**

| Fold | 5 boxes (before) | 10 boxes (after) | Change (pp) |
| --- | ---: | ---: | ---: |
| 1 | 86.93 | 87.61 | +0.68 |
| 2 | 84.16 | 84.92 | +0.76 |
| 3 | 84.76 | 83.54 | −1.22 |
| 4 | 85.49 | 85.43 | −0.06 |
| 5 | 86.36 | 85.93 | −0.42 |

**What improved.** Little. Left kidney gained 0.62 pp; spleen lost 0.74 pp.

**Still not working.** Extra boxes can lower Dice: the score after the last box is reported, not the best along the way.

**Key findings.** Across all 30 patients, the mean peaks at about seven boxes (85.83%) and flattens after five. Five boxes is a sensible budget.

Sources: `KT/02_BOX/Ten_Box_Corrected_Zero_Results.md`, `btcv_cross_validation/ten_box/results/`.
