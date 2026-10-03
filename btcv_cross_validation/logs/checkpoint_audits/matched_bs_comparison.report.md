> Archive update (1 October 2026): copied source trees and runner scripts were removed with user approval after completion. Commands and source paths below describe the historical run; they are not runnable archive instructions. Raw JSON/CSV/manifest records are preserved verbatim inside the matching `.evidence.log`.

# BS=2 versus BS=4: matched authors’ evaluation

Completed evaluation of both epoch-20 checkpoints on the same six held-out BTCV patients (0035–0040), using the same frozen outer authors’ evaluator.

| Organ | BS=2 Dice (%) | BS=4 fold 5 Dice (%) | BS=2 − BS=4 (percentage points) |
| --- | ---: | ---: | ---: |
| Liver | 80.0005 | 79.0197 | +0.9808 |
| Stomach | 62.8340 | 61.5634 | +1.2706 |
| Left kidney | 68.9815 | 67.7364 | +1.2451 |
| Spleen | 77.9389 | 76.4808 | +1.4581 |
| **Four-organ mean** | **72.4387** | **71.2001** | **+1.2387** |

The BS=2 checkpoint scores 1.2387 percentage points higher under this matched evaluation. A difference between evaluator implementations is therefore not required to explain the observed gap. This comparison does not prove batch size caused the difference: training update counts and random training trajectories also differ.

## Comparison with previous results

| Organ | BS=2 change from historical score (pp) | BS=4 change from historical score (pp) |
| --- | ---: | ---: |
| Liver | +0.00000000 | +0.00000000 |
| Stomach | +0.00000000 | +0.00000000 |
| Left kidney | +0.00000000 | +0.00000000 |
| Spleen | +0.00000000 | +0.00000000 |

## Paired patient results

Each patient score below is the mean of its four organ Dice scores.

| Patient | BS=2 Dice (%) | BS=4 Dice (%) | Difference (pp) |
| --- | ---: | ---: | ---: |
| img0035 | 69.9363 | 68.0094 | +1.9268 |
| img0036 | 73.6754 | 73.5655 | +0.1100 |
| img0037 | 73.5682 | 70.7276 | +2.8406 |
| img0038 | 73.3839 | 72.4010 | +0.9829 |
| img0039 | 73.0716 | 72.6263 | +0.4453 |
| img0040 | 70.9969 | 69.8705 | +1.1264 |

## Protocol and verification

- All 48 evaluations completed: two checkpoints × four organs × six patients.
- Same training patient IDs and held-out patient IDs verified; no train/test overlap.
- Same copied authors’ model, loader, evaluator, box propagation, thresholds, postprocessing and Dice calculation.
- Identical input shapes, spacing and prompt-slice indices verified for each paired case.
- Ground-truth organ extent determines the z crop; one ground-truth box is provided on the maximum-area slice. This is the authors’ evaluation protocol, not a full-volume deployment benchmark.
- No new box-padding method, z-resampling, retraining, checkpoint updates or training-code changes.
- Explicit PYTHONPATH selects the frozen outer authors’ source. All project module paths were verified.
- cuDNN failed before any initial scores were produced. Both complete runs use the same working native CUDA fallback, with float32 inference and model.eval().
- All 422 BS=2 epoch-20 checkpoint tensors equal the historical ckpt_v/model_latest.pth tensors.
- NSD was not computed; ignore the original evaluator’s printed zero placeholder.
- Reported training BS=2/BS=4 is distinct from inference slice_per_batch=4, which was identical for both runs.

## Checkpoints

- **bs2_epoch20**: `/home/jupyter-team2pop/Slide-SAM/runs/sam/ddp_sub/btcv_4to1_20epoch/ckpt/model_epoch_20.pth`; SHA-256 `4b94b46a0272727bb6d2ed2fbb0c865efe56a23afc66408992bba8351150a94b`.
- **bs4_fold5_epoch20**: `/home/jupyter-team2pop/Slide-SAM/Slide-SAM/btcv_cross_validation/runs/fold5/ddp_sub_cv/cv_fold5/ckpt/model_epoch_20.pth`; SHA-256 `235b75d82b5e7e71c9e00c65b2a91d3f02b4cc60f79ccdb6518f138105730785`.

See [README.md](matched_bs_comparison.README.md) for reproduction and [per_patient.csv](matched_bs_comparison.evidence.log) / [results.json](matched_bs_comparison.evidence.log) for the raw measurements. These test results should not be used to tune padding settings.
