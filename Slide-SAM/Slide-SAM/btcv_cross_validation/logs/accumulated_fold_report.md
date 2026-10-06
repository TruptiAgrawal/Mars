# BTCV cross-validation: accumulated fold1, fold2, fold3 and fold4 report

All four folds use epoch-20 checkpoints and box prompts. Each fold contains six held-out volumes. Combined results pool 24 distinct held-out volumes; each volume is predicted by its corresponding fold model. Dice is expressed as a percentage. SD is the sample standard deviation across volumes, not across folds.

| Organ | Fold1 mean ± SD | Fold2 mean ± SD | Fold3 mean ± SD | Fold4 mean ± SD | Combined mean ± SD |
|---|---:|---:|---:|---:|---:|
| Liver | 74.87 ± 8.90 | 68.39 ± 10.61 | 77.64 ± 6.15 | 77.14 ± 7.02 | 74.51 ± 8.65 |
| Stomach | 57.83 ± 8.80 | 57.55 ± 12.14 | 61.27 ± 7.74 | 63.32 ± 9.12 | 59.99 ± 9.28 |
| Left Kidney | 70.19 ± 4.53 | 61.71 ± 11.17 | 69.02 ± 10.39 | 69.62 ± 11.03 | 67.64 ± 9.69 |
| Spleen | 76.25 ± 6.44 | 67.15 ± 12.68 | 79.43 ± 4.81 | 75.05 ± 5.33 | 74.47 ± 8.74 |
| **Four-organ macro average** | **69.78** | **63.70** | **71.84** | **71.28** | **69.15** |

The four-organ macro average weights each organ equally; no SD is attached to this aggregate.

Fold3 achieves the highest four-organ mean Dice at 71.84%, followed by fold4 (71.28%), fold1 (69.78%), and fold2 (63.70%). Fold4 trails fold3 by 0.56 percentage points and exceeds fold1 and fold2 by 1.50 and 7.58 points, respectively. Fold3 has the highest liver and spleen Dice, fold4 has the highest stomach Dice, and fold1 has the highest left kidney Dice. Combined, liver has the highest Dice (74.51%), followed by spleen (74.47%), left kidney (67.64%), and stomach (59.99%).

Training summary: all four runs completed 20 epochs. Fold1 training loss declined from 0.5365 to 0.4013 (25.2%); fold2 declined from 0.5491 to 0.4018 (26.8%); fold3 declined from 0.5286 to 0.3931 (25.6%); fold4 declined from 0.5447 to 0.4034 (25.9%). Similar final training losses did not translate into equal held-out Dice. Recorded epoch times total 14.20 hours for fold1, 16.31 hours for fold2, 16.80 hours for fold3, and 16.01 hours for fold4. Training dataset audits reported zero held-out samples selected in each fold.

Fold3 analysis: all 20 epoch summaries and all 420 expected per-100-iteration log entries were present. Point-prompt loss declined from 0.3200 to 0.2126 and box-prompt loss from 0.2086 to 0.1805. The lowest epoch training loss was 0.3930 at epoch 18; the reported validation results use epoch 20. No per-epoch validation curve is available.

Fold3 LoRA analysis: 48 rank-1 tensors contain 36,864 trainable values. The L2 distance of the LoRA A/B tensors from initialization increased from 4.00 at epoch 5 to 7.06 at epoch 20. Updates between consecutive saved checkpoints had a mean cosine similarity of 0.81. Block 11 accounted for the largest share of final change (17%), while query and value projections accounted for 39% and 61%, respectively. Saved weight epochs were 0, 5, 10, 15, and 20; update measurements span these checkpoint intervals.

Fold4 analysis: all 20 epoch summaries and all 390 expected per-100-iteration log entries were present (1,948 batches per epoch). Point-prompt loss declined from 0.3287 to 0.2179 and box-prompt loss from 0.2161 to 0.1855. The lowest epoch training loss was 0.4034 at epoch 20, matching the checkpoint used for the reported validation results. No per-epoch validation curve is available.

Fold4 LoRA analysis: 48 rank-1 tensors contain 36,864 trainable values. The L2 distance of the LoRA A/B tensors from initialization increased from 3.95 at epoch 5 to 6.86 at epoch 20. Updates between consecutive saved checkpoints had a mean cosine similarity of 0.78. Block 11 accounted for the largest share of final effective-weight change (17%), while query and value projections accounted for 39% and 61%, respectively. Saved weight epochs were 0, 5, 10, 15, and 20; update measurements span these checkpoint intervals. All 13 packed epoch and iteration checkpoints reconstructed bit-for-bit from the epoch-0 checkpoint and the LoRA pack.

These results cover four of the planned five folds, not the complete five-fold cross-validation experiment. Combined results are pooled held-out predictions, not an ensemble. No significance claim is made from the observed fold differences.

Sources: [fold1 evaluation](fold1_evaluation.json), [fold2 evaluation](fold2_evaluation.json), [fold3 evaluation](fold3_evaluation.json), [fold4 evaluation](fold4_evaluation.json), [fold1 training log](train_fold1.log), [fold2 training log](train_fold2.log), [fold3 training log](train_fold3.log), [fold4 training log](train_fold4.log), [fold3 analysis summary](fold3_analysis/fold3_analysis_summary.txt), [fold3 epoch losses](fold3_analysis/fold3_epoch_losses.csv), [fold4 analysis summary](fold4_analysis/fold4_analysis_summary.txt), and [fold4 epoch losses](fold4_analysis/fold4_epoch_losses.csv).
