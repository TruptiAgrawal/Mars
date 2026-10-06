# MARS Progress: Slide-SAM as the Segmentation Agent

Snapshot taken 24 Sep 2026, 13:10 IST.

**End goal:** a 4-agent system: **segmentation → clinical cross-check → confidence check →
report generation**. Slide-SAM (LoRA fine-tuned on BTCV) is the candidate for the segmentation
agent.

## 1. Where things stand today

- **Fold 1 is finished and evaluated.** It used the final epoch 20 checkpoint on img0001–0006
  (`btcv_cross_validation/logs/fold1_evaluation.json`):

  | Liver | Spleen | L-Kidney | Stomach | Mean |
  |---|---|---|---|---|
  | 0.749 | 0.762 | 0.702 | 0.578 | **0.698** |

- **Fold 2 is training** (`core.ddp_sub_cv`, started 09:54 on 24 Sep). It has finished epoch 4
  of 20 at about 43 min per epoch, so it should end around 00:30 on 25 Sep.
- Folds 3–5 are still to run, about 15 h each, one at a time. That puts the full CV result
  around **27–28 Sep**.
- **Batch size is consistent across folds.** The configs for folds 2–5 have an uncommitted
  change from 2 to 4, and the KT notes still say "bs=2". But Fold 1's committed config is also 4,
  and both fold logs show about 1905 batches per epoch. So the notes are out of date, not the
  runs.

## 2. How far along we are

| Layer | Progress | Comment |
|---|---|---|
| Slide-SAM adaptation (research result) | **~50–60%** | Pipeline, LoRA, leak-proof CV and eval are all done. What's left is mostly GPU time, plus fixing NSD and a like-for-like baseline comparison. |
| Segmentation *agent* (usable in a system) | **~25%** | We have a model that scores well **when given a ground-truth box**. An agent can't get that at inference time (see §3). |
| Full 4-agent system | **~10–15%** | Clinical cross-check, confidence and report agents haven't been started, and neither has the interface between them. |

The hardest engineering, trustworthy training and evaluation, is done. But the research result
and the product are drifting apart, and that is the main thing to address.

## 3. The critical gap: where the prompts come from

`test/volume_eval_cv.py:85` builds the box from the **ground-truth label** on the template
slice, and the template slice itself is also chosen from the label. So every Dice number so far
(0.698, 0.724, 0.791) is an **oracle-prompt** score. Inside an autonomous pipeline, nobody
supplies that box.

Before building the other agents, the team needs to answer one question: **what role does
Slide-SAM play in the segmentation agent?**

- **(a) The segmenter itself.** We then need an automatic prompt generator: a detector, or a
  cheap coarse segmenter (such as a small nnU-Net or TotalSegmentator) turned into boxes. Expect
  Dice to drop compared with oracle boxes.
- **(b) A refiner or interactive corrector** on top of a fully automatic model. This is a strong
  fit for the *confidence* agent: low-confidence organs get re-prompted and refined by
  Slide-SAM, possibly with a clinician's click.
- **(c) Structures that off-the-shelf models don't cover**, such as lesions.

If a pretrained automatic model already segments these 4 organs at about 0.9 Dice, reviewers
will ask why Slide-SAM is there. Option (b) gives the clearest answer.

## 4. Next steps, in order

### While the GPU is busy (CPU-only work, no conflict with the running folds)
1. **Prompt-robustness study.** Using the Fold 1 checkpoint, re-run the eval with perturbed boxes
   (shifted or scaled by ±5–20%) and with non-central template slices. This tells us how much
   Dice we lose with realistic prompts, and it decides between (a), (b) and (c). Wait until the
   GPU is free, or run small subsets.
2. **Fix NSD** (currently always logged as 0.0). Clinical reviewers care about boundary metrics.
3. **Define the contract between agents now**, before building them. For example, the
   segmentation agent outputs per organ: mask, volume in mL, the prompt used, SAM's
   predicted-IoU per slice, and how consistent the mask is from slice to slice. That last pair
   is exactly what the **confidence agent** would consume, and Slide-SAM already produces the
   IoU scores almost for free.

### After CV completes (~28 Sep)
4. Report **mean ± SD Dice per organ** across the 5 folds. Also re-evaluate the
   `btcv_4to1_20epoch` and `word_mini` checkpoints through `btcv_cross_validation/tests/eval_fold.py`
   for a like-for-like comparison. Fold 1's 0.698 is lower than word_mini's 0.791, though on
   different volumes, so it's still unclear whether adding WORD helps.
5. If it does, run a CV with WORD added. Stomach, the weakest organ at 0.58, is where it would
   matter most.

### Then build an end-to-end skeleton with stub agents
6. Wire segmentation → clinical cross-check (simple rules: organ volumes within normal ranges,
   left kidney on the left, expected Hounsfield-unit (HU) ranges) → confidence (IoU scores plus
   slice consistency plus rule violations) → report (a template filled in by an LLM from
   structured findings). A working pipeline, even a crude one, will show which agent actually
   limits quality far sooner than more Dice tuning will.
7. **Scope check for the report agent:** BTCV mostly contains normal organs, so the reports can
   only say things like "liver volume X mL". If the reports should describe *findings*
   (tumours, cysts), we'll need lesion data (for example LiTS, KiTS or MSD) at some point. Plan
   that dataset decision early.

## 5. Summary

Let the CV finish untouched. Use the waiting time to measure how much we lose with realistic
prompts and to define the interfaces between agents. Then build a crude end-to-end pipeline
before optimising any single agent further.
