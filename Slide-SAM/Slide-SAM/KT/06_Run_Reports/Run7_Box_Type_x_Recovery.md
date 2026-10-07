# Run 7 — Historical vs corrected-tight boxes × recovery (synthetic scans)

With historical boxes, recovery lowered the mean Dice in every fold, by 1.6–6.2 pp. Corrected-tight boxes beat historical boxes in every fold, with recovery off or on.

**Goal.** Complete the 2 × 2 comparison from Run 5: does recovery behave the same with the old historical boxes as with corrected-tight boxes?

**What was done.** 40 new runs, one box, on the same 17 patients and the same saved 6 mm inputs as Run 5 (6, 4, 1, 4 and 2 patients per fold). Scoring labels and the first box were the same as in Run 5. The corrected-tight results are Run 5's B00/B01, reused unchanged.

**What changed, and why.** A copy of the v3 evaluator gained a setting to choose historical or corrected-tight boxes. It now stops with an error, instead of silently patching the geometry, if a resampled scan doesn't line up.

**Before and after (four-organ mean Dice %).** Columns marked ★ are new.

| Fold (patients) | Historical, OFF ★ | Historical, ON ★ | Corrected-tight, OFF | Corrected-tight, ON |
| --- | ---: | ---: | ---: | ---: |
| 1 (6) | 74.22 | 68.06 | 78.30 | 77.89 |
| 2 (4) | 64.77 | 63.14 | 70.61 | 72.13 |
| 3 (1) | 75.27 | 70.34 | 77.88 | 78.65 |
| 4 (4) | 74.03 | 68.32 | 75.28 | 76.85 |
| 5 (2) | 74.83 | 69.88 | 81.55 | 83.23 |

**What improved.** Nothing new improved. The run shows that the corrected-tight box fix also helps on thick synthetic scans.

**Still not working.**

- With historical boxes, recovery cut liver Dice by 7.9–10.6 pp in every fold.
- With either box type, recovery does not reliably help, and it lowers left kidney Dice in every fold with corrected-tight boxes.

**Key findings.** Recovery only looks useful when paired with corrected-tight boxes, and even then it trades kidney accuracy for liver and spleen. This is an exploratory robustness test on synthetic scans, not a benchmark result.

Sources: `btcv_cross_validation/missing_evals/report.md`, `btcv_cross_validation/z_spacing_boxmode/results/`.
