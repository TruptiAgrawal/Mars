# Slide-SAM — Overall Progress Report (Jul – Sep 2026)

Prepared 23 Sep 2026. Built from the git history of `Slide-SAM/Slide-SAM`, the training and eval
logs in `logs/`, the run folders in `../runs/sam/`, and `btcv_cross_validation/`.
For this week's detail, see `KT/Weekly_Progress_16-23Sep2026.md`.

## 1. Project goal

Adapt **Slide-SAM**, a SAM-based medical segmentation model that propagates a prompt through a
3-D CT volume slice by slice, to abdominal CT. We do this by **LoRA fine-tuning** the official
Slide-SAM-B checkpoint on **BTCV** (30 labelled volumes), optionally adding **WORD**. We report
volume-level Dice on the four organs the paper uses:

| Organ | Liver | Stomach | Left Kidney | Spleen |
|---|---|---|---|---|
| BTCV label | 6 | 7 | 3 | 1 |

The end goal is a result that holds up under review: **5-fold cross-validation, mean ± SD Dice
per organ**.

## 2. How we got here: the sequence

```
Phase 1  Jul 25–30   Set up + reproduce the original code         → pipeline runs end-to-end
Phase 2  Jul 26–30   Make training robust (resume-safe checkpoints)
Phase 3  Aug 18–19   First fine-tuning on WORD                     → learned the data pipeline
Phase 4  Sep 3–4     Move to BTCV + LoRA fine-tuning (ddp_sub)     → first BTCV models
Phase 5  Sep 5–9     20-epoch BTCV run + fixed 24/6 split + eval   → baseline Dice ("before")
Phase 6  Sep 9–15    BTCV + WORD-50 combined experiments (E2)
Phase 7  Sep 22–23   bs=4 run, word_mini run + eval               → better Dice ("now")
Phase 8  Sep 22–23   5-fold cross-validation pipeline              → built + verified
Phase 9  Sep 23 →    CV training, fold by fold                     → Fold 1 running
```

### Phase 1: Setup and reproducing the original code (25–30 Jul)
- Got the environment (`slidesam` conda env), data and code running on the shared GPU node.
- Ran the original `core/ddp.py` training (`test_train`, `reproduce2`) and the original volume
  evaluation.
- **First evaluation number:** Dice **0.577** with a box prompt (`label_idx 13`,
  `logs/volume_eval.log`). This proved the evaluation pipeline worked end to end.

### Phase 2: Making training robust (26–30 Jul)
- Training runs were being cut short, so we added **resume-safe checkpointing** and tested it by
  resuming mid-run (`resume3`, `resume_from20000_test_1`, `resume_safe_ckpt`,
  `resume_safe_ckpt_resume3`).
- Result: runs can now be stopped and resumed from a saved iteration without losing progress.
  Every later experiment depends on this.

### Phase 3: First fine-tuning on WORD (18–19 Aug)
- Trained on WORD, restricted to label 2 (`word_label2_fresh`, then `word_label2_epoch2_3`).
  The epoch-1 loss was 0.653 on the fresh run and 0.525 on the continued run.
- Lesson: the epoch 2–3 run **failed while writing a checkpoint** (`PytorchStreamWriter failed
  writing`), probably because disk or I/O ran out. We learned to watch disk space.

### Phase 4: Moving to BTCV with LoRA (3–4 Sep)
- Switched to the **LoRA fine-tuning path** (`core/ddp_sub.py` + `learner_sub1.py`), which starts
  from the official `slidesam_b.pth` weights. This is the only path that supports LoRA.
- Built the **data partitioning and visualiser tooling** (`Visulaizer/`: dataset stats,
  flowchart, TensorBoard loss loader) and started committing to our own git history
  (3 Sep).
- `btcv_liver` / `btcv_liver_run2`, 4 epochs: loss 0.516 → 0.439.
- ⚠ **Discovered later:** these runs set `specific_label: [2]` to mean "liver". But in BTCV,
  label 2 is the **right kidney**, and the `specific_label` key isn't read anywhere in the
  pipeline. So these runs actually trained on all organs, not just the liver.

### Phase 5: 20-epoch BTCV baseline (5–9 Sep)
- Created a fixed **"4-to-1" split**: 24 training volumes (img0001–10, img0021–34) and 6 test
  volumes (img0035–40).
- **`btcv_4to1_20epoch`**: 20 epochs, about 45 min per epoch. Loss fell 0.510 → 0.394.
- Evaluated on the 6 test volumes (`test/volume_eval_btcv_e1.py`). **This is our baseline:**

  | Liver | Spleen | L-Kidney | Stomach | Mean |
  |---|---|---|---|---|
  | 0.800 | 0.779 | 0.690 | 0.628 | **0.724** |

- Produced per-organ visualisations for every case, including the right kidney (commits 9 Sep
  and 22 Sep).

### Phase 6: BTCV + WORD-50 combined experiments, "E2" (9–15 Sep)
- Wrote `datasets/cache_dataset3d_e2.py`, which trains on 24 BTCV + 50 WORD volumes
  (8,159 + 25,913 = 34,072 cached slices) and prints a dataset audit.
- Runs: `e2_run1`, `pilot5`, `recovery_test`, `clean`, `final5`. Epoch-1 loss was 0.469.
- These runs were mostly held back by infrastructure: repeated restarts, a `state_dict` loading
  error in the recovery test, and checkpoint-logic fixes to `ddp_sub.py` (backups kept as
  `ddp_sub.py.*_backup`). None of them reached a full evaluation.

### Phase 7: Batch-size and word_mini experiments (22–23 Sep)
- **`btcv_4to1_20epoch_bs4`** (BTCV + WORD-50, batch size 4): loss 0.479 → 0.441 over 2 epochs,
  then stopped. It used about 34 GB of the shared GPU, which is not sustainable.
- **`btcv_word_mini_5epoch`** (3 BTCV + 3 WORD volumes, 5 epochs): loss 0.580 → 0.543.
- Evaluated it on the same 6 test volumes:

  | Liver | Spleen | L-Kidney | Stomach | Mean |
  |---|---|---|---|---|
  | 0.829 | 0.878 | 0.744 | 0.711 | **0.791** |

### Phase 8: 5-fold cross-validation pipeline (22–23 Sep)
A single 6-volume test set is too small to trust, so we built a proper CV setup. It is isolated
in `btcv_cross_validation/` plus new files only; **no existing file was changed**.
1. Generated 5 deterministic volume-level folds, 24 train / 6 test each.
2. **Checked them mathematically:** no overlap, every volume held out exactly once,
   reproducible, all files exist. All checks passed.
3. Built one shared cache of all 30 volumes (2,074 slices, 667 MB) so preprocessing is identical
   across folds.
4. Wrote a new training loader that **raises an error if a test volume leaks into training**.
   It found 0 leaked samples on all 5 folds.
5. Created fold configs (bs=2, 21 epochs, lr 2e-5; every fold starts fresh from
   `slidesam_b.pth`) and the `core/ddp_sub_cv.py` entrypoint.
6. Built fold-aware evaluation (`loader_cv.py`, `test/volume_eval_cv.py`), which requires the
   organ label to be passed explicitly.

### Phase 9: CV training (23 Sep → now)
- **Fold 1** has been running since 09:50 on 23 Sep. Loss has fallen 0.537 → 0.421 over
  10 of 20 epochs, close to the Phase 5 run's curve. It should finish around 00:30 on 24 Sep.

## 3. Results so far

### Dice on the fixed 6-volume test set (img0035–0040, box prompt)

| Organ | July sanity check | Baseline: 20-ep BTCV (Phase 5) | Latest: word_mini (Phase 7) |
|---|---|---|---|
| Liver | – | 0.800 | **0.829** |
| Spleen | – | 0.779 | **0.878** |
| Left Kidney | – | 0.690 | **0.744** |
| Stomach | – | 0.628 | **0.711** |
| **Mean** | 0.577 (single class, original code) | **0.724** | **0.791 (+0.066)** |

### Training loss (final epoch reached)

| Run | Data | Epochs | Final loss |
|---|---|---|---|
| btcv_liver_run2 | BTCV | 3 | 0.439 |
| btcv_4to1_20epoch | BTCV 24 vols | 20 | **0.394** |
| btcv_4to1_20epoch_bs4 | BTCV 24 + WORD 50 | 2 | 0.441 |
| btcv_word_mini | BTCV 3 + WORD 3 | 4 | 0.543 |
| cv_fold1 (running) | BTCV 24 vols (fold 1) | 10 / 20 | 0.421 |

**How far to trust these numbers:**
- The "latest" model trained on far less data than the baseline, yet scored higher. That may
  mean the baseline overfit, or that WORD helps. But the two models were evaluated with
  different eval scripts, so the comparison must be re-run through `volume_eval_cv.py` before we
  claim it.
- Six test volumes is a small sample. One hard case (img0040 liver, Dice 0.54) moves the mean
  noticeably. The 5-fold CV will settle this.
- **NSD is always logged as 0.0.** It is not being computed yet.

## 4. Key achievements

1. **Working, resumable LoRA fine-tuning pipeline** for Slide-SAM on BTCV and WORD.
2. **Baseline established:** mean Dice 0.724 on 4 organs. The latest model reaches **0.791**
   (still needs to be re-checked through the same eval script).
3. **A leak-proof 5-fold CV framework** with mathematical checks and hard failures on leakage.
   This is the basis for results we can publish.
4. **Found and neutralised hidden bugs** that would have invalidated results:
   - `cache_dataset3d_e2.py` silently mixes in WORD-50 and ignores `dataset_list`.
   - The default eval loader silently tests on an old 3-volume list.
   - `label_idx=2` / `specific_label: [2]` was labelled "liver" but is the right kidney.
   - `specific_label` is not read anywhere in the pipeline.
5. **Visualisation and analysis tooling:** per-organ overlays for all cases, dataset stats,
   loss curves.

## 5. Where we stand

| Milestone | Status |
|---|---|
| Reproduce Slide-SAM | ✅ Done |
| Robust / resumable training | ✅ Done |
| LoRA fine-tuning on BTCV | ✅ Done |
| Baseline on fixed split | ✅ Done (0.724 mean Dice) |
| Improved model | ✅ 0.791 mean Dice, needs a like-for-like re-eval |
| BTCV + WORD-50 full run | ⏸ Partial, no final evaluation |
| 5-fold CV pipeline | ✅ Built and verified |
| CV training | 🔄 Fold 1 of 5 at 50% |
| CV results (mean ± SD) | ⏳ Not started |
| NSD metric | ❌ Not working |

All the infrastructure for the CV headline result is done. What's left is mostly GPU time (about 15 h per fold, so about 2.5 days for Folds 2–5 run one after another
on the shared GPU).

## 6. Next steps

1. Fold 1 finishes → evaluate on img0001–0006 for all 4 organs.
2. Train and evaluate Folds 2 → 5, one after another.
3. Report **mean ± SD Dice per organ** across the 5 folds.
4. Re-evaluate the baseline checkpoint with `volume_eval_cv.py` for a like-for-like comparison.
5. Fix NSD.
6. Optional: run the same 5-fold CV with WORD added, to test whether WORD really helps.
