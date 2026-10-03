# Slide-SAM — Weekly Progress Report (16 – 23 Sep 2026)

Prepared 23 Sep 2026, ~17:40 IST. All numbers below are taken directly from the logs in
`logs/` and `btcv_cross_validation/logs/`.

## 1. What we are working on

Adapting the official **Slide-SAM-B** checkpoint (`weights/slidesam_b.pth`) to abdominal CT
using **LoRA fine-tuning** on BTCV (optionally mixed with WORD). We evaluate on the four organs
the paper uses: **Liver (label 6), Stomach (7), Left Kidney (3), Spleen (1)**, with a box prompt
and volume-level Dice.

This week had two goals:
1. Improve on our earlier BTCV fine-tuned model, and
2. Replace the single fixed train/test split with **5-fold cross-validation**, so the results
   we report hold up under review.

## 2. What we did this week

| Date | Work | Outcome |
|---|---|---|
| 15 Sep | Wrapped up the 20-epoch BTCV run (`btcv_4to1_20epoch`, trained 5 Sep); ended the BTCV + WORD-50 pilot and "final5" attempts, which did not get past epoch 1 | Final loss 0.39 (epoch 20); this is the **"before" model** |
| 22 Sep | Visualisations of the 20-epoch model, including the right kidney. Configs cleaned up and committed | Committed (`88bbd822`, `7db0e9b0`) |
| 22 Sep | BTCV (24 vols) + WORD-50 run at batch size 4 (`btcv_4to1_20epoch_bs4`) | Loss 0.479 → 0.441 over 2 epochs, then stopped (~33.7 GB GPU at bs=4 is too much on a shared GPU) |
| 22 Sep | `btcv_word_mini` run: 3 BTCV + 3 WORD volumes, 5 epochs (first attempt crashed, retry succeeded) | Loss 0.580 → 0.543 |
| 23 Sep (AM) | Evaluated the word_mini model on the 6 held-out volumes, for all 4 organs | **Better than the old model on every organ** (see §3) |
| 22–23 Sep | Designed, built and verified the complete **5-fold CV pipeline** | Steps 0–7 of the roadmap done (see §4) |
| 23 Sep 09:50 | **Fold 1 training started** | Running now, epoch 10 of 20 done (see §5) |

## 3. Results — before vs. now

Test set: the same 6 held-out BTCV volumes (img0035–img0040), box prompt, Dice per organ.

| Organ | Before: `btcv_4to1_20epoch` (6 Sep eval) | Now: `btcv_word_mini` (23 Sep eval) | Change |
|---|---|---|---|
| Liver | 0.800 | **0.829** | +0.029 |
| Spleen | 0.779 | **0.878** | **+0.099** |
| Left Kidney | 0.690 | **0.744** | +0.054 |
| Stomach | 0.628 | **0.711** | +0.083 |
| **Mean (4 organs)** | **0.724** | **0.791** | **+0.066** |

Sources: `logs/BTCV evals/eval_btcv_e1_*.log` (before) and `logs/eval_word_mini_*.log` (now).

**Caveats: read these before quoting the numbers.**
- The "now" model trained on only **3 BTCV + 3 WORD volumes for 5 epochs**, but it beat a model
  trained on 24 BTCV volumes for 20 epochs. That is encouraging (the old model may have
  overfit, or the WORD data helps), but it is surprising enough to need checking.
- The two evaluations used **different eval scripts** (`volume_eval_btcv_e1.py` vs. the new
  `FoldTestLoader` path). Both ran on the same 6 volumes with identical template slices, but
  before we claim an improvement, the old checkpoint should be re-run through the new eval
  script.
- Only 6 test volumes, so a single volume can move the mean a lot. For example, word_mini liver
  Dice on img0040 is 0.54, while the other five volumes are 0.84–0.92.
- **NSD is logged as 0.0 in every eval.** It is not actually being computed and still needs to
  be fixed.

These caveats are the reason for the 5-fold CV work below: it tests on all 30 volumes and
reports mean ± SD.

## 4. Achievement: a verified 5-fold cross-validation pipeline

BTCV-only, patient/volume-level folds: 24 train / 6 test per fold, every volume held out
exactly once. Everything lives in `btcv_cross_validation/` or in new files. **No existing file
was modified.**

| Step | What | Status |
|---|---|---|
| 1 | Generated 5 deterministic folds from the actual files on disk (`scripts/make_folds.py`) | Done |
| 2 | Checked the folds mathematically (`scripts/verify_folds.py`): 30 unique vols, 24/6 per fold, train ∩ test = ∅, test sets partition all 30, byte-identical on regeneration, all files exist | **All checks passed** |
| 3 | One shared cache of all 30 volumes, so every fold gets identical preprocessing | Done: 2,074 slices, 667 MB |
| 4 | New CV loader `datasets/cache_dataset3d_cv.py` that **raises an error if any held-out sample leaks into training** | **0 leaked samples on all 5 folds** |
| 5 | Fold configs (bs=2, 21 epochs, lr 2e-5; each fold starts fresh from `slidesam_b.pth`) and the `core/ddp_sub_cv.py` entrypoint | Done |
| 6 | Fold-aware evaluation: `loader_cv.py` and `test/volume_eval_cv.py` | Done, checked on CPU for all 5 folds |
| 7 | Evaluation targets fixed as Liver 6, Stomach 7, L-Kidney 3, Spleen 1 | Done |
| 8 | Train folds one at a time | **In progress: Fold 1** |
| 9 | Report mean ± SD Dice per organ across the folds | Not started |

Bugs found and avoided while building this:
- The old loader (`cache_dataset3d_e2.py`) **silently mixes in WORD-50** and ignores
  `dataset_list` from the config.
- The default eval loader would have **silently evaluated on the old 3-volume test file**
  instead of the fold's 6 held-out volumes.
- An example eval config used `label_idx=2` with the comment "liver". Label 2 is actually the
  **right kidney**.
- The `specific_label` config key is not used anywhere in the pipeline.

## 5. Currently running: CV Fold 1 (do not interrupt)

- Command: `python -m core.ddp_sub_cv --config btcv_cross_validation/configs/fold1.yaml --tag cv_fold1` (PID 320117)
- Log: `btcv_cross_validation/logs/train_fold1.log`
- Test volumes: img0001–img0006

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Loss | 0.537 | 0.455 | 0.448 | 0.442 | 0.438 | 0.429 | 0.431 | 0.425 | 0.423 | 0.421 |

Each epoch takes about 43 minutes. Epoch 10 finished at 17:11 on 23 Sep, so the run should
finish at about **00:30 on 24 Sep**. The loss curve is tracking the old 20-epoch run
(0.410 at epoch 10), which is what we would expect.

## 6. Issues / risks

1. **Shared GPU** (one RTX A6000, 49 GB): at 99% utilisation with ~34 GB in use. Folds must run
   one after another, so the remaining 4 folds need about 15 h of training each, or about 2.5
   days in total.
2. **NSD metric not implemented** (always 0.0).
3. The "before vs. now" comparison needs to be re-run through the same eval script (see §3).
4. Uncommitted: `KT/`, `datasets/dataset_list/btcv_word_mini/01_BTCV_test_4to1.txt`.

## 7. Next steps

1. When Fold 1 finishes, evaluate it on img0001–0006 for all 4 organs
   (`test/volume_eval_cv.py --label_idx {6,7,3,1}`).
2. Train and evaluate Folds 2 → 5, one after another.
3. Aggregate the per-organ Dice as **mean ± SD across the 5 folds**. This becomes the headline
   result.
4. Re-evaluate the old `btcv_4to1_20epoch` checkpoint with `volume_eval_cv.py` so the
   before/after comparison is like-for-like.
5. Fix the NSD computation.
