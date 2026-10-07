# Mars disk cleanup review

Survey taken 6 Oct 2026. **Nothing has been deleted.** This is a list of candidates for review.

Context: the disk is at 100% (about 3 GB free of 834 GB). `Mars/` uses about 126 GB, almost
all of it under `Slide-SAM/`. None of the folders listed below are tracked by git (they are
covered by `.gitignore`), so removing them would not change the repository, except where noted.

Paths are relative to `Mars/`. `SS/` is short for `Slide-SAM/Slide-SAM/`.

## Summary

| Category                                    | Approx. size | Effect on reported results                              |
|---------------------------------------------|-------------:|---------------------------------------------------------|
| 1. Intermediate CV fold checkpoints         | 33 GB        | None. Evaluations load only epoch 20                    |
| 2. July upstream training runs              | 39 GB        | None. Superseded by cross-validation                    |
| 3. September pre-CV training runs           | up to 29 GB  | Keep each run's final checkpoint if you may re-evaluate |
| 4. Regenerable visualizations and old cache | 11.5 GB      | None. Can be regenerated                                |
| 5. Small clutter                            | under 10 MB  | None                                                    |
| **Total potential**                         | **about 75–110 GB**                                                    |

## 1. Intermediate CV fold checkpoints (about 33 GB)

Location: `SS/btcv_cross_validation/runs/fold{1-5}/ddp_sub_cv/cv_fold{N}/ckpt/`

Training saved a full model (`model_epoch_N.pth`, about 360 MB) every epoch. Every evaluation
(fold evals, box padding, five-box, ten-box) loads only `model_epoch_20.pth`, as recorded in
`SS/btcv_cross_validation/logs/fold*_evaluation.json`.

| Fold | Intermediate `.pth` files (not epoch 20) | Size |
|------|-----------------------------------------:|-----:|
| 1    | 37                                       | 9.8 GB |
| 2    | 37                                       | 9.8 GB |
| 3    | 23                                       | 4.6 GB |
| 4    | 21                                       | 4.2 GB |
| 5    | 23                                       | 4.6 GB |
| **Total** | **141** | **33.0 GB** |

**Keep:** `model_epoch_20.pth`, all `*_lora.safetensors` files (about 150 KB each), and
`ckpt_v/model_latest.pth` with its LoRA file. Checksums show `model_latest.pth` is **not** the
same file as epoch 20, so it is a distinct checkpoint. Also keep `tb/`, `log/` and `scripts/`.

Consider keeping a few intermediate epochs (for example 5, 10 and 15) if a learning-curve
analysis per epoch may be needed later.

## 2. July upstream training runs (39 GB)

Location: `Slide-SAM/runs/sam/ddp/`

These are early runs from 20 Jul to 19 Aug, mainly to debug the training script, test resuming
and test the WORD labels. The later cross-validation runs supersede them.

| Run | Size |
|---|---:|
| word_label2_epoch2_3    | 5.4 GB |
| word_label2_fresh_4     | 4.3 GB |
| resume_from20000_test_1 | 3.2 GB |
| resume20000             | 2.9 GB |
| resume3                 | 2.9 GB |
| resume_safe_ckpt        | 2.9 GB |
| reproduce2              | 2.5 GB |
| resume_safe_ckpt_resume | 2.5 GB |
| debug_resume            | 2.2 GB |
| fresh_resume            | 2.2 GB |
| resume2                 | 1.8 GB |
| test_train              | 1.8 GB |
| fresh_resume_1          | 1.1 GB |
| reproduce               | 1.1 GB |
| resume20000_v2          | 1.1 GB |
| resume_safe_ckpt_resume3 | 1.1 GB |
| resume_from20000_test   | 0.4 GB |
| Empty or aborted (under 1 MB each): `2026-07-20-17:09:25.610633`, `2026-07-20-17:49:28.813868`, `fresh_resume_test`, `resume_safe_ckpt_resume2`, `word_label2_fresh`, `word_label2_fresh_2`, `word_label2_fresh_3`, `word_label2_fresh_4_resume`, `word_label2_fresh_5_resume` | about 0 |

Some run names appear in old logs and docs under `SS/logs/` and `SS/KT/`. Those are historical
references only; no current code loads these checkpoints.

## 3. September pre-CV training runs (up to 29 GB)

Location: `Slide-SAM/runs/sam/ddp_sub/`

| Run | Size | Note |
|---|---:|---|
| btcv_4to1_20epoch           | 14.0 GB | The earlier "e1" BTCV evaluation in the 23 Sep progress notes |
| btcv_4to1_20epoch_bs4       | 3.2 GB | BS=4 BTCV run |
| btcv_word_mini_5epoch_retry | 2.9 GB | Likely source of the word_mini evaluation (23 Sep) |
| btcv_word50_e2_pilot5       | 2.9 GB | Pilot run |
| btcv_liver_run2             | 2.9 GB | Liver-only run |
| btcv_word50_e2_run1         | 1.4 GB | |
| btcv_liver                  | 1.1 GB | Liver-only run |
| btcv_word_mini_5epoch       | 1.1 GB | First attempt, followed by `_retry` |
| btcv_word50_e2_final5       | 51 MB | |
| Empty or aborted (under 1 MB each): `2026-09-14-07:54:16.746173`, `btcv_word50_e2`, `btcv_word50_e2_clean`, `btcv_word50_e2_recovery_test` | about 0 | |

Recommendation: if any of these results may be quoted or re-evaluated, keep only the **final
checkpoint** of each run (about 360 MB each) and remove the intermediate epochs. Otherwise the
whole folder can go.

## 4. Regenerable visualizations and old cache (11.5 GB)

| Path | Size | Why it is not needed |
|---|---:|---|
| `SS/visualizations_copilot/`          | 6.9 GB | Generated output; recreated by `SS/copilot-visualisation.py` |
| `SS/epoch_20_visualizations_copilot/` | 1.1 GB | Generated output; no code references it |
| `SS/cached_dataset2/`                 | 3.5 GB | Used only by the older configs (`SS/configs/vit_sub*.yaml`). Cross-validation uses `SS/btcv_cross_validation/cache/`. Can be rebuilt; delete only if those configs won't be rerun |

## 5. Small clutter (under 10 MB)

| Item | Note |
|---|---|
| 89 `.ipynb_checkpoints/` folders | Jupyter autosaves; ignored by git |
| 14 `__pycache__/` folders        | Python bytecode; ignored by git |
| `Segmentation_Agent/` | Contains only an `.ipynb_checkpoints/` folder |
| `Slide-SAM/Untitled.ipynb` | Empty notebook (72 bytes) |
| `SS/visual/` (1.4 MB) and `SS/visualizations/` (1.2 MB) | Overlap with `SS/Visulaizer/` (misspelled; used by `SS/epoch_progression.py`). Consider merging into one folder |

## Tracked file to review before pushing

`SS/my-session.json` (2.1 MB) **is tracked by git**. It is an export of a Kiro AI-assistant
session from 1 Oct 2026, and may contain the full conversation and local paths. If it should
not be on GitHub, untrack it and remove it from history before the next push.

## Keep (required)

| Path | Size | Reason |
|---|---:|---|
| `SS/weights/slidesam_b.pth`     | 360 MB | Base Slide-SAM-B checkpoint that all fine-tuning starts from |
| `SS/sam_vit_b_01ec64.pth`       | 358 MB | Used by `SS/core/ddp.py`, `SS/core/ddp_sub.py`, `SS/core/lora_sam.py` and `SS/configs/vit_b.yaml` |
| `SS/datasets/01_BCV-Abdomen/`   | 1.6 GB | Raw BTCV data |
| `SS/datasets/07_WORD/`          | 6.3 GB | Raw WORD data, used by the word_mini config and the predictors |
| `SS/btcv_cross_validation/cache/`     | 667 MB | Cross-validation training cache |
| `SS/btcv_cross_validation/z_spacing/` | 421 MB | Synthetic spacing experiment data |
| Epoch 20, `ckpt_v` and LoRA checkpoints for each fold | about 2 GB | Used by every reported CV result |
| All code, `logs/`, `KT/`, `results/`, `docs/`, `backend/`, `frontend/` | | |

## Other space outside Mars (for reference)

`~/git-backups-2026-10-06/` (about 1.5 GB) holds the git backups made during the repository
merge on 6 Oct 2026. It can be removed once the Mars push has succeeded and been checked.

## Suggested order

1. Section 1 (intermediate fold checkpoints) and section 2 (July runs): about 72 GB, with no
   effect on any reported result.
2. Section 4 visualizations: about 8 GB.
3. Section 3: decide per run whether to keep its final checkpoint.
4. Before deleting, write the exact file list and check it.
