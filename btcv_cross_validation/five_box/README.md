# Five-box evaluation with corrected-zero propagation

This experiment evaluates all 30 native BTCV patients in their respective held-out folds using the saved BS=4 epoch-20 checkpoints. Each of four organs receives one initial ground-truth box plus four corrective boxes. No training, cache rebuilding, z-resampling or original-source edits are performed.

The predictor, loader, data engine and helpers were copied from the isolated box-padding experiment before use. The outer authors’ `test/volume_eval_mbox.py` is copied unchanged to `author_mbox.py`; its `finetune_with_more_prompt` method is invoked directly. Original copies are in `originals/`.

## Exact protocol

- Ground-truth organ-extent crop and maximum-area valid initial slice, matching the corrected-zero one-box experiment.
- Corrective slice: lowest current Dice among unused interior slices, using the authors’ ranking.
- Corrective prompt: tight ground-truth box on the selected slice.
- Propagation: corrected-zero boxes, with the existing predictor thresholds and postprocessing.
- Merge: unchanged authors’ confidence selection; corrective-slice confidence is doubled. The predictor’s original state retention across prompt calls is preserved.
- Record Dice at prompt counts 1, 2, 3, 4 and 5. Report the actual fifth result, not the maximum over stages.
- Strict checkpoint loading with nested rank-8/rank-1 LoRA; native CUDA backend with cuDNN disabled for the local runtime.
- Evaluation requires five distinct valid interior foreground slices. Unsupported cases fail explicitly instead of being silently omitted or assigned a different prompting policy.

Ground truth is used to choose corrective slices. This is an oracle-assisted interactive evaluation, not automatic five-box prediction. The authors’ method name “finetune” refers to correction during inference and does not update weights.

## Execution

Run from the inner repository root using the explicit environment. An existing result directory is rejected; preserve or archive outputs before planning any rerun.

```bash
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  /home/jupyter-team2pop/.conda/envs/slidesam/bin/python \
  -m btcv_cross_validation.five_box.evaluate --fold 1 --run-inference
```

Repeat with fold numbers 2–5 sequentially. Omit `--run-inference` for path/manifest preflight only. Generate the aggregate report after all five folds complete:

```bash
/home/jupyter-team2pop/.conda/envs/slidesam/bin/python \
  -m btcv_cross_validation.five_box.summarize
```

Results contain 24 patient-organ cases per fold, all five prompt boxes/slice IDs, all five Dice scores and checkpoint/source hashes. `shared_source_sha256.json` records imported model/support source files; the experiment copies have their own hashes. `failed_wrong_python.log` preserves a pre-inference import failure from the initial system-Python launch.

The final report is `KT/Five_Box_Corrected_Zero_Results.md`. Compare one versus five boxes within this experiment. Comparison with the paper’s five-box BTCV row is descriptive because the authors’ patient split and training provenance are not established as identical, and our corrected-zero propagation intentionally differs from the historical implementation.
