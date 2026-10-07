# Run 3 — Five boxes with corrected-zero boxes (all folds)

Four extra corrective boxes raised the mean Dice by 6.28–9.49 pp in every fold, reaching 84.16–86.93%.

**Goal.** Match the paper's main setting (five boxes) and see how much extra user boxes help once the box bug is fixed.

**What was done.** All 30 patients, each scored by its own fold's checkpoint. After the first box, the authors' own correction loop adds four boxes. Each new box goes on the worst-scoring slice, is drawn from the ground truth, and is merged using the authors' rule. Corrected-zero boxes are used throughout.

**What changed, and why.** Only the box bug fix from Run 2. The authors' correction and merge code is unchanged, so the result is comparable to their five-box setting.

**Before and after (four-organ mean Dice %, 6 patients per fold).**

| Fold | 1 box (before) | 5 boxes (after) | Change (pp) |
| --- | ---: | ---: | ---: |
| 1 | 78.56 | 86.93 | +8.38 |
| 2 | 74.67 | 84.16 | +9.49 |
| 3 | 78.10 | 84.76 | +6.66 |
| 4 | 79.21 | 85.49 | +6.28 |
| 5 | 79.77 | 86.36 | +6.59 |

**What improved.** Stomach (+14.09 pp) and left kidney (+13.92 pp) gained the most. Liver and spleen, already high, gained about 1 pp.

**Still not working.** We are still below the paper's five-box result, by 4.5 pp for liver and spleen and about 10 pp for stomach and left kidney (paper mean 92.96%). The paper's splits and training are not the same as ours.

**Key findings.** Extra boxes mostly fix the organs that one box handles badly. The boxes come from the ground truth, so this measures guided correction, not automatic segmentation.

Sources: `KT/02_BOX/Five_Box_Corrected_Zero_Results.md`, `btcv_cross_validation/five_box/results/`.
