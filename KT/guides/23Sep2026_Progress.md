# Slide-SAM Project Progress — 23 Sep 2026

Snapshot taken 23 Sep 2026, ~16:15.

## Project overview
LoRA fine-tuning of the official Slide-SAM-B checkpoint (`weights/slidesam_b.pth`) on BTCV
abdominal CT (optionally combined with WORD), evaluated on the four paper-aligned organs:

| Organ | BTCV label |
|---|---|
| Spleen | 1 |
| Left Kidney | 3 |
| Liver | 6 |
| Stomach | 7 |

(Right Kidney = 2 is not part of the paper's 4-organ evaluation.)

Working copy: `/home/jupyter-team2pop/Slide-SAM/Slide-SAM` (has its own git history; the outer
`Slide-SAM/` repo is the original upstream code).

## Timeline
- **Jul 25–30** — First training runs, resume-safe checkpointing, volume-level evaluation.
- **Aug 18–19** — WORD label experiments, epoch 2–3 training.
- **Sep 3–15** — Data partitioning + visualiser, BTCV liver runs, BTCV + WORD-50 pilot,
  20-epoch run, visualisations (incl. right kidney).
- **Sep 22** — BTCV bs=4 training; "btcv_word_mini" 5-epoch run (epoch loss 0.58 → 0.54 over epochs 2–4).
- **Sep 23 (early)** — Evaluation of word_mini model on the fixed test volumes.

## Latest evaluation results (Dice, fixed test split)

| Organ | Earlier "e1" eval | word_mini eval (23 Sep) |
|---|---|---|
| Liver | 0.800 | **0.829** |
| Spleen | 0.779 | **0.878** |
| Left Kidney | 0.690 | **0.744** |
| Stomach | 0.628 | **0.711** |

Note: NSD is reported as 0.0 in all eval logs — appears not to be computed.

Logs: `logs/eval_word_mini_*.log`, `logs/BTCV evals/eval_btcv_e1_*.log`

## Current focus: BTCV-only 5-fold cross-validation
Plan: `btcv_cross_validation/5foldClaudeRoadmap.txt`
Detailed session notes: `btcv_cross_validation/ClaudeMemory.txt`

| Step | Description | Status |
|---|---|---|
| 0 | CV folder skeleton | Done |
| 1 | Generate 5 volume-level folds (24 train / 6 test each) | Done |
| 2 | Mathematically verify folds (disjoint, full coverage, deterministic, files exist) | Done — all checks passed |
| 3 | Shared cache of all 30 volumes (2074 slices, 667 MB) | Done |
| 4 | CV dataset loader `datasets/cache_dataset3d_cv.py` with leakage hard-fail | Done — 0 leaked samples on all 5 folds |
| 5 | Fold configs + entrypoint `core/ddp_sub_cv.py` | Done |
| 6 | Fold-aware eval: `datasets/eval_dataloader/loader_cv.py`, `test/volume_eval_cv.py` | Done |
| 7 | Evaluation targets (Liver 6, Stomach 7, Left Kidney 3, Spleen 1) | Done (by design) |
| 8 | Train folds sequentially on GPU | **In progress — Fold 1 running** |
| 9 | Aggregate mean ± SD Dice per organ across folds | Not started |

Fold held-out test volumes:
- Fold 1: img0001–0006
- Fold 2: img0007–0010, img0021–0022
- Fold 3: img0023–0028
- Fold 4: img0029–0034
- Fold 5: img0035–0040

### Fold 1 training status (live)
- Command: `python -m core.ddp_sub_cv --config btcv_cross_validation/configs/fold1.yaml --tag cv_fold1`
- Log: `btcv_cross_validation/logs/train_fold1.log`
- Started ~09:50, 23 Sep 2026.
- Epoch 8 of 21 (epochs 0–20) completed at 15:46; epoch loss plateauing around 0.42–0.43.
- ~43 min per epoch → expected finish around **00:30, 24 Sep 2026**.
- Checkpoints saved every epoch under `btcv_cross_validation/runs/fold1/`.

## Issues / things to watch
1. **GPU is shared and nearly full:** single RTX A6000, 100% utilisation, ~39/49 GB used
   (fold 1 ≈ 34 GB, plus another user's job ≈ 5.6 GB). Folds must run one at a time.
2. **Uncommitted file:** `datasets/dataset_list/btcv_word_mini/01_BTCV_test_4to1.txt`.

## Next steps
1. When fold 1 finishes, evaluate it on its 6 held-out volumes for all 4 organs
   (`test/volume_eval_cv.py --label_idx {6,7,3,1}`).
2. Decide the batch size for folds 2–5 (match fold 1 or rerun fold 1).
3. Train folds 2 → 5 sequentially, evaluating each.
4. Aggregate mean ± SD Dice per organ across the 5 folds.
