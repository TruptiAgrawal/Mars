# Run 2 — Box padding pilot (fold 5)

Fixing the box bug raised fold 5's mean Dice from 71.20% to 79.77% (+8.57 pp). Adding 5% padding dropped it to 58.22%.

**Goal.** Test whether enlarging the box passed from one slice to the next helps the model keep the whole organ in view.

**What was done.** Re-scored fold 5's 6 held-out patients with one box, in three ways, on the same checkpoint (72 predictions):

- **H, historical:** the original code's box enlargement.
- **Z, corrected zero:** a tight box around the previous slice's mask, no enlargement.
- **P5, 5% padding:** the tight box grown by 5% of the image size on each side (about 51 px).

Padding only changes the boxes the model passes to itself. The user's first box is never changed.

**What changed, and why.** While adding padding we found the original `enlarge()` code was wrong: it shrank the box on two sides and mixed up x and y. Z removes that bug; P5 then tests real padding on top of the fixed box.

**Before and after (Dice %, fold 5, 6 patients).**

| Organ | H (before) | Z | P5 |
| --- | ---: | ---: | ---: |
| Liver | 79.02 | 91.26 | 80.72 |
| Stomach | 61.56 | 70.34 | 56.24 |
| Left kidney | 67.74 | 72.48 | 30.97 |
| Spleen | 76.48 | 84.99 | 64.96 |
| **Mean** | **71.20** | **79.77** | **58.22** |

**What improved.** Z beat H for every organ, and it also improved boundary accuracy (NSD and HD95).

**Still not working.**

- 5% padding hurt every organ, and left kidney fell by 41.5 pp. The extra 51 px pulls neighbouring anatomy into a small organ's box.
- Z is worse than H for left kidney in 2 of 6 patients (img0038, img0039).

**Key findings.** The gain is a bug fix, not a new method. From here on, Z is the baseline that any padding idea must beat.

Sources: `KT/02_BOX/Box_Padding_First_Results.md`, `btcv_cross_validation/box_padding/results/fold5_e20_*`.
