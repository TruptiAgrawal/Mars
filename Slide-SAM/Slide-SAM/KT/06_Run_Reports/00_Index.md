# Slide-SAM BTCV Run Reports — Index

Updated: 7 October 2026. One simple report per experiment/run.

The biggest gain so far came from fixing a bug in how propagated boxes were drawn (corrected-zero boxes). Adding more user boxes also helped up to about five. The two new ideas tested, 5% box padding and z-spacing recovery, did not give a reliable improvement.

## Setup shared by every run

- Data: BTCV abdominal CT, 30 patients, 5-fold cross-validation (24 train / 6 held-out per fold).
- Model: Slide-SAM fine-tuned per fold (rank-1 LoRA, batch size 4), epoch-20 checkpoint.
- Organs scored: liver (6), stomach (7), left kidney (3), spleen (1). Metric: Dice (%).
- Prompting: the first box comes from the ground-truth slice with the largest organ area, so scores are oracle-assisted, not fully automatic.

## Runs

| # | Report | Question | Answer |
| --- | --- | --- | --- |
| 1 | [5-fold baseline](Run1_5Fold_Baseline.md) | How does our fine-tuned model do with 1 box? | 63.7–71.8 per fold, far below the paper's 88.33 |
| 2 | [Box padding pilot (fold 5)](Run2_Box_Padding_Pilot.md) | Does fixing or padding the propagated box help? | Fix: +8.57 pp. 5% padding: −21.54 pp |
| 3 | [Five boxes](Run3_Five_Box.md) | How much do 4 extra corrective boxes add? | +6.3 to +9.5 pp per fold |
| 4 | [Ten boxes](Run4_Ten_Box.md) | Do 9 extra boxes add more? | No; flat versus five boxes |
| 5 | [Z-spacing recovery](Run5_Z_Spacing_Recovery.md) | Does resampling thick (6 mm) scans to ~3 mm help? | Small, mixed; hurts left kidney |
| 6 | [Padding × box count, all folds](Run6_Padding_x_Box_Count.md) | Do the padding results hold on every fold and with 5 boxes? | Yes; corrected-zero best everywhere |
| 7 | [Box type × recovery](Run7_Box_Type_x_Recovery.md) | Is recovery still mixed with the old (historical) boxes? | With historical boxes, recovery hurts in every fold |

See also [Open problems and caveats](Open_Problems_and_Caveats.md) and [Runs 1, 3, 6 and 7 explained](Runs_1_3_6_7_Explained.md).
