# Runs 1, 3, 6 and 7 — plain explanation

Updated: 7 October 2026. Short, plain-language guide to what each of these four runs did. Full details are in each run's own report.

## Two ideas used throughout

- **Box prompt.** You draw one box around the organ on a single slice. The model segments that slice, then draws its own box from that result for the next slice, and repeats slice by slice through the scan.
- **Historical vs corrected box.** The original code had a bug: the box the model draws for itself shrank on two sides and mixed up x and y. "Historical" means that buggy box. "Corrected-zero" or "corrected tight" means the bug is fixed: a tight box around the previous slice's result.

## Run 1 — Baseline (the starting point)

- **What:** fine-tune the model separately for each of 5 folds. Each fold trains on 24 patients and tests on the other 6, so all 30 patients get tested once.
- **Setup:** 1 box per organ, the original **historical** (buggy) box, normal scans.
- **Result:** 63.7–71.8% Dice per fold, well below the paper's 88.3%.
- **Why it matters:** every later run is compared against this.

Report: [Run1_5Fold_Baseline.md](Run1_5Fold_Baseline.md)

## Run 3 — Five boxes with the bug fixed

- **What:** switch to **corrected** boxes, then let the user add 4 more boxes. Each extra box goes on the slice the model got most wrong, following the authors' own method.
- **Setup:** all 5 folds, normal scans, compared with 1 box vs 5 boxes.
- **Result:** 1 box scores 74.7–79.8%; 5 boxes score 84.2–86.9%, about +6 to +9.5 points per fold.
- **Why it matters:** the bug fix plus extra boxes is our best result so far.

Report: [Run3_Five_Box.md](Run3_Five_Box.md)

## Run 6 — Box type × number of boxes (run 7 October 2026, normal scans)

- **What:** fill in the combinations that hadn't been tested on every fold.
- **Setup:** three box types (historical, corrected, and corrected plus 5% padding) × 1 or 5 boxes × all 5 folds. This run added 5% padding with 1 box (folds 1–4), historical with 5 boxes, and 5% padding with 5 boxes.
- **Result:** in every fold, corrected is best and 5% padding is worst; 5 boxes always beat 1 box.
- **Why it matters:** it shows the fold-5 pilot finding holds on all folds, not just one.

Report: [Run6_Padding_x_Box_Count.md](Run6_Padding_x_Box_Count.md)

## Run 7 — Box type × z-spacing recovery (run 7 October 2026, thick synthetic scans)

- **What:** test both box types on thick scans, with and without "recovery".
  - We took the 17 patients with 3 mm slices and made 6 mm versions by removing slices.
  - **Recovery OFF:** segment the 6 mm scan directly.
  - **Recovery ON:** resample it to about 3 mm first, segment, then map the result back to 6 mm.
- **Setup:** historical vs corrected box × recovery OFF vs ON, 1 box, all 5 folds (6/4/1/4/2 patients). This run added the two historical-box settings; the two corrected-box settings come from the earlier run (Run 5, v3 B00/B01).
- **Result:**
  - Corrected boxes beat historical boxes in every fold.
  - With historical boxes, recovery hurts in every fold.
  - With corrected boxes, recovery is mixed: it helps liver and spleen but hurts left kidney.
- **Why it matters:** it shows recovery isn't a reliable improvement whichever box type is used.

Report: [Run7_Box_Type_x_Recovery.md](Run7_Box_Type_x_Recovery.md)

## How they connect

```
Run 1  baseline: historical box, 1 box, normal scans
  └─ Run 3  fix the box bug and add 4 more boxes  → big gain
       └─ Run 6  all box types × 1 or 5 boxes      → fix stays best, padding hurts
Run 7  thick scans: box type × recovery           → fix helps, recovery doesn't reliably help
```
