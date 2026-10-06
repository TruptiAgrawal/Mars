> Archive update (1 October 2026): copied source trees and runner scripts were removed with user approval after completion. Commands and source paths below describe the historical run; they are not runnable archive instructions. Raw JSON/CSV/manifest records are preserved verbatim inside the matching `.evidence.log`.

# Fold-5 checkpoint trajectory

Post-hoc diagnostic on the six previously evaluated held-out patients. This is not validation and does not select a best checkpoint.

| Checkpoint | Liver (%) | Stomach (%) | Left kidney (%) | Spleen (%) | Mean (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official pretrained | 83.4941 | 69.7184 | 74.9429 | 88.3187 | 79.1185 |
| BS=4 epoch 0 | 83.4941 | 69.7184 | 74.9429 | 88.3187 | 79.1185 |
| BS=4 epoch 5 | 75.9218 | 62.8912 | 66.9910 | 75.4067 | 70.3027 |
| BS=4 epoch 10 | 77.1713 | 61.9877 | 66.5432 | 75.3285 | 70.2577 |
| BS=4 epoch 15 | 78.7640 | 61.8940 | 66.7244 | 75.3469 | 70.6823 |
| BS=4 epoch 20 | 79.0197 | 61.5634 | 67.7364 | 76.4808 | 71.2001 |

Epoch 0 is the saved debug-training checkpoint, not untouched initialization. The official checkpoint is the pre-fine-tuning reference.

All new scores use the same frozen authors’ evaluator, original thresholds, ground-truth z crop and one box prompt, with no new padding or resampling. Official and epoch-20 scores are reused from the completed matched evaluations.

A decline over epochs would support performance degradation during fine-tuning on these cases, but would not by itself prove overfitting or identify its cause. Choosing training duration or other hyperparameters requires validation cases excluded from training, within each outer training fold. These test results must not be used to choose the best fold or stopping epoch.
