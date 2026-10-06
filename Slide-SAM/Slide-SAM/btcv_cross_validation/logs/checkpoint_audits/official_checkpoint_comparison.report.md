> Archive update (1 October 2026): copied source trees and runner scripts were removed with user approval after completion. Commands and source paths below describe the historical run; they are not runnable archive instructions. Raw JSON/CSV/manifest records are preserved verbatim inside the matching `.evidence.log`.

# Official checkpoint versus BTCV fine-tuning

The local official `weights/slidesam_b.pth` was evaluated without additional fine-tuning on patients 0035–0040, with the same frozen authors’ evaluator used for the BS=2 and BS=4 comparison.

| Organ | Official pretrained Dice (%) | BS=2 Dice (%) | BS=4 fold 5 Dice (%) |
| --- | ---: | ---: | ---: |
| Liver | 83.4941 | 80.0005 | 79.0197 |
| Stomach | 69.7184 | 62.8340 | 61.5634 |
| Left kidney | 74.9429 | 68.9815 | 67.7364 |
| Spleen | 88.3187 | 77.9389 | 76.4808 |
| **Mean** | **79.1185** | **72.4387** | **71.2001** |

## Effect of fine-tuning on this held-out set

- BS=2 epoch 20: -6.6798 percentage points relative to the official checkpoint.
- BS=4 fold 5 epoch 20: -7.9185 percentage points relative to the official checkpoint.

These are measurements on six patients, not evidence that a particular batch size is universally better. Fine-tuning changes can improve some organs while reducing others; inspect the organ and patient scores before attributing a mechanism.

## Verification and protocol

- All 24 official-checkpoint predictions completed (four organs × six patients).
- All 43 copied source files match the previously frozen authors’ source hashes.
- Patient order, cropped image shapes, spacing and prompt-slice indices match the previous evaluation.
- The official checkpoint uses rank-8 LoRA only and loads strictly. The fine-tuned models use their trained nested rank-8/rank-1 adapters.
- Same ViT-B architecture, float32, model.eval(), native CUDA backend with cuDNN disabled, and inference slice batch of four.
- Same ground-truth-based z crop and one box on the maximum-area slice; this is not a full-volume deployment benchmark.
- Original predictor defaults retained, including IoU threshold 0.5 and stability threshold 0.6. No padding changes or z-resampling.
- The printed NSD zero is an authors’ placeholder; NSD was not computed.
- Original checkpoints, previous evaluation outputs and training code were not edited.
- The local file is the checkpoint used to initialize the training runs. Its hash records identity; these evaluations do not independently certify the release’s training-data provenance.

## Artifacts

- Official checkpoint: `/home/jupyter-team2pop/Slide-SAM/Slide-SAM/weights/slidesam_b.pth`.
- SHA-256: `d47a6bbf6d5045f8fd1f69fe24101417b0c1f1b23f479dcd4f060b980ebe213a`.
- [results.json](official_checkpoint_comparison.evidence.log): scores, timings, geometry, source imports and checkpoint hash.
- [per_patient.csv](official_checkpoint_comparison.evidence.log): 24 individual scores.
- [run.log](official_checkpoint_comparison.run.log): inference log.
- `run_comparison.py` (removed after completion): copied runner; original saved as `run_comparison.original.py`.
- [Previous matched results](matched_bs_comparison.report.md).

For subsequent hyperparameter selection, use validation patients excluded from the corresponding training run. These six test patients must not become a repeated tuning target.
