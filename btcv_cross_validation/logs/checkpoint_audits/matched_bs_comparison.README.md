> Archive update (1 October 2026): copied source trees and runner scripts were removed with user approval after completion. Commands and source paths below describe the historical run; they are not runnable archive instructions. Raw JSON/CSV/manifest records are preserved verbatim inside the matching `.evidence.log`.

# Matched BS=2 versus BS=4 checkpoint evaluation

Both epoch-20 checkpoints are evaluated on the same six held-out BTCV patients (0035–0040), using a frozen copy of the outer repository's authors' evaluator and its dependencies. Original project files and checkpoints are unchanged.

## Protocol

| Setting | Shared value |
| --- | --- |
| Evaluator | `author_source/test/volume_eval_sublora.py`, unchanged `Evaluater.eval_step` |
| Data loader | Authors' `AbstractLoader` |
| Model | ViT-B, nested rank-8 and rank-1 LoRA; strict checkpoint loading |
| Organs | Liver (6), stomach (7), left kidney (3), spleen (1) |
| Prompt | One ground-truth bounding box on maximum-area slice |
| Evaluation extent | Ground-truth organ extent along z, as in authors' loader |
| Propagation | Authors' original box behavior, postprocessing and noise removal enabled |
| New padding / resampling | Neither applied |
| Inference slice batch | 4 for both models; training batch sizes are 2 and 4 |
| Precision | Float32; model in evaluation mode; no gradients |
| Backend | GPU CUDA, cuDNN disabled equally for both models |
| Aggregation | Mean of six patient Dice scores per organ; equal-weight mean of four organs |

The initial cuDNN attempt failed before producing any scores. A standalone convolution confirmed native CUDA works, so the wrapper disables cuDNN for both models. This changes the runtime backend, not the evaluation algorithm. Small numerical differences from historical runs remain possible. The failed log and pre-fix wrapper are preserved.

`PYTHONPATH` explicitly selects the frozen authors' source. The wrapper also verifies that all project module imports resolve there. No source fixes are applied to the copied authors' evaluator, loader, model, or propagation code.

## Audit records and results

- `source_sha256.json`: hashes of the 43 copied Python source files.
- `split_audit.json`: matching 24 training patients, matching six test patients, and no train/test overlap.
- `bs2_checkpoint_equivalence.json`: all 422 tensors in BS=2 epoch 20 equal those in the historical `ckpt_v/model_latest.pth`.
- `results.json`: final checkpoint hashes, imported modules, organ means and per-patient measurements.
- `per_patient.csv`: all 48 patient-organ-checkpoint scores.
- `run.log`: complete inference output. The authors' printed NSD value is a placeholder; NSD was not computed.
- `report.md`: final comparison and interpretation, generated after completion.

This is a comparison between two trained checkpoints. It does not establish a causal batch-size effect, because training update counts and random training trajectories were not controlled. It uses the authors' ground-truth-assisted protocol and should not be described as full-volume deployment evaluation. Keep these test cases out of padding hyperparameter selection.

## Reproduce

Run from the inner repository root in the `slidesam` environment with GPU access. Archive existing outputs first if preserving this run.

```sh
PYTHONPATH="$PWD/btcv_cross_validation/matched_bs_comparison/author_source" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python -u btcv_cross_validation/matched_bs_comparison/run_comparison.py
```
