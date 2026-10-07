# Open problems and caveats (all runs)

The main open problem is that fine-tuning lowers accuracy. The pretrained model scored 79.12% on fold 5; our epoch-20 model scored 71.20%.

## Still not working

- **Fine-tuning hurts.** Training loss falls while test Dice drops (Run 1). The cause is not yet fixed.
- **Gap to the paper.** We are 17–25 pp below the paper with 1 box and about 4.5–10 pp below with 5 boxes, depending on the organ.
- **Left kidney** is the most fragile organ. It is hurt most by padding (Runs 2 and 6) and by z-spacing recovery (Runs 5 and 7).
- **No positive novelty result yet.** 5% padding and z-spacing recovery both failed to give a reliable gain.

## How to read these numbers

- All prompts come from the ground truth, so these are guided scores, not automatic ones.
- The paper's patient splits and training differ from ours; comparisons with it are descriptive only.
- Native-scan (Runs 1–4, 6) and synthetic-scan (Runs 5, 7) scores use different patients and grids. Don't compare them directly.
- The same test patients have been looked at many times. Any further tuning (padding size, organ-specific rules) needs separate validation patients.
