# Run 6 — Padding × number of boxes, all folds (native scans)

The fold-5 pilot holds on every fold. Corrected-zero is best and 5% padding is worst, with 1 box and with 5 boxes.

**Goal.** Fill in the missing padding × box-count combinations, so all three box types are compared with 1 and 5 boxes on all six held-out patients of every fold.

**What was done.** 26 new runs on native scans, finished 7 October 2026 in about one hour with no failures:

- 5% padding, 1 box: folds 1–4 (fold 5 was done in Run 2).
- Historical boxes, 5 boxes: folds 1–5.
- 5% padding, 5 boxes: folds 1–5.

**What changed, and why.** The five-box evaluator could only use corrected-zero boxes, so a copy was extended to accept the padding type. The authors' correction and merge code is unchanged. A check confirmed that the first box of every 5-box run gives the same Dice as the matching 1-box run.

**Before and after (four-organ mean Dice %, 6 patients per fold).** Columns marked ★ are new in this run; the rest come from Runs 1–3.

| Fold | Historical, 1 box | Corrected-zero, 1 box | 5% padding, 1 box | Historical, 5 boxes ★ | Corrected-zero, 5 boxes | 5% padding, 5 boxes ★ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 69.78 | 78.56 | 52.53 ★ | 84.18 | 86.93 | 65.73 |
| 2 | 63.70 | 74.67 | 55.03 ★ | 80.37 | 84.16 | 68.05 |
| 3 | 71.84 | 78.10 | 61.64 ★ | 82.58 | 84.76 | 70.49 |
| 4 | 71.28 | 79.21 | 57.82 ★ | 84.40 | 85.49 | 67.46 |
| 5 | 71.20 | 79.77 | 58.22 | 83.22 | 86.36 | 71.38 |

**What improved.** Five boxes beat one box for every box type in every fold. They also shrink the gap between historical and corrected-zero boxes, from 6.3–11.0 pp with 1 box to 1.1–3.8 pp with 5 boxes.

**Still not working.**

- 5% padding hurts in every fold. Left kidney falls to 28.77–46.12% with 1 box, against 57.55–73.67% for corrected-zero.
- With 5 boxes, historical boxes beat corrected-zero for left kidney in all 5 folds (by 0.6–3.1 pp), and for stomach in fold 4.

**Key findings.** Corrected-zero should stay the default. Do not adopt organ-specific choices, such as historical boxes for kidney, from these numbers: they come from the test patients and would need separate validation data.

Sources: `btcv_cross_validation/missing_evals/report.md` (per-organ and per-patient tables), `per_patient.csv`.
