# Ten-box evaluation with corrected-zero propagation

Extension of `../five_box/` from five to ten box prompts. All 30 native BTCV patients are evaluated in their respective held-out folds using the saved BS=4 epoch-20 checkpoints. Each of four organs receives one initial ground-truth box plus nine corrective boxes. No training, cache rebuilding, z-resampling or original-source edits are performed.

All code was copied from `../five_box/`. `predictor.py`, `loader.py`, `data_engine.py`, `helpers.py`, `author_mbox.py` and `originals/` are byte-identical copies. Only these files differ:

- `evaluate.py`: nine corrective rounds instead of four (`PROMPTS = 10`), at least ten valid interior slices required, `ten_box_dice` recorded alongside `five_box_dice` (Dice after box 5). Checkpoints are loaded from their current location under this repository: `logs/fold{N}_evaluation.json` records the former root `/home/jupyter-team2pop/Slide-SAM/Slide-SAM`, which is remapped to the current root. Both paths are stored in the results.
- `summarize.py`: reports one/five/ten boxes and the per-prompt-count Dice curve. It checks against `../five_box/results` that checkpoint hashes, the first five prompt slices and the Dice at prompt counts 1–5 match.

## Exact protocol

Identical to the five-box experiment except for the prompt budget: ground-truth organ-extent crop; maximum-area valid initial slice; corrective slice = lowest current Dice among unused interior slices (authors' ranking); tight ground-truth box; corrected-zero propagation; unchanged authors' confidence merge. Dice is recorded at prompt counts 1–10, and the tenth result is reported, not the maximum over prompt counts. This is oracle-assisted interactive evaluation.

## Execution

From the inner repository root:

```bash
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  /home/jupyter-team2pop/.conda/envs/slidesam/bin/python \
  -m btcv_cross_validation.ten_box.evaluate --fold 1 --run-inference
```

Repeat for folds 2–5 sequentially (existing result directories are rejected). Omit `--run-inference` for preflight only. Then run:

```bash
/home/jupyter-team2pop/.conda/envs/slidesam/bin/python -m btcv_cross_validation.ten_box.summarize
```

Outputs, all in this folder: `results/fold{1-5}/results.json`, `run.log`, `per_patient.csv`, `summary.json`, `verification.json` and the report `Ten_Box_Corrected_Zero_Results.md`.
