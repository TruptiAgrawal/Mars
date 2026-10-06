# Authors’ BTCV results versus our five-fold and novelty experiments

Updated: 3 October 2026. All local values below are recomputed from completed result JSON files. Dice is expressed as a percentage; differences are percentage points (pp). No new inference or training was run to prepare this comparison.

## 1. How to read this comparison

The authors’ Table 1 reports **one-box BTCV results**. Our local five-fold baseline, fold-5 padding experiment, and synthetic-spacing experiment use different patient groups and/or grids. They must not be presented as successive improvements on a single identical benchmark.

- **Within a padding table:** compare H/Z/P5 on the same six patients, checkpoint and evaluator.
- **Within a spacing version:** compare B00/B01 on the same synthetic inputs and scoring grid, with matched checkpoints.
- **Across cohorts, versions or the paper:** numerical gaps are descriptive, not measured intervention effects.
- The paper specifies a 4:1 BTCV split; identical author/local patient IDs and full training provenance have not been established.
- The local protocols use ground-truth-assisted prompts and target-extent cropping. These are not fully automatic deployment scores.


## 2. Main comparison at a glance

| Method / experiment | Liver (%) | Stomach (%) | Left kidney (%) | Spleen (%) | Mean (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Authors: one box, paper Table 1 | 95.44 | 74.31 | 92.42 | 91.16 | 88.33 |
| Our five-fold baseline: all 30 patients | 75.41 | 60.31 | 67.66 | 74.87 | 69.56 |
| Our fold-5 baseline: six native patients | 79.02 | 61.56 | 67.74 | 76.48 | 71.20 |
| Box H: historical, six native patients | 79.02 | 61.56 | 67.74 | 76.48 | 71.20 |
| Box Z: corrected zero, six native patients | 91.26 | 70.34 | 72.48 | 84.99 | 79.77 |
| Box P5: image-relative 5%, six native patients | 80.72 | 56.24 | 30.97 | 64.96 | 58.22 |
| Spacing v2 B00: two synthetic patients | 89.39 | 65.67 | 86.49 | 84.65 | 81.55 |
| Spacing v2 B01: two synthetic patients | 92.71 | 67.30 | 84.02 | 88.88 | 83.23 |
| Spacing v3 B00: 17 synthetic patients | 85.78 | 64.48 | 72.40 | 81.88 | 76.13 |
| Spacing v3 B01: 17 synthetic patients | 89.83 | 64.22 | 68.57 | 85.23 | 76.96 |

**Do not rank these rows as a single leaderboard.** For example, 79.77% from Box Z and 76.96% from v3 B01 are measured on different cohorts and inputs. Their difference is not the effect of adding z-resampling.

## 3. Authors versus each cross-validation fold


| Method / experiment | Liver (%) | Stomach (%) | Left kidney (%) | Spleen (%) | Mean (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Authors: one box | 95.44 | 74.31 | 92.42 | 91.16 | 88.33 |
| Our fold 1 | 74.87 | 57.83 | 70.19 | 76.25 | 69.78 |
| Our fold 2 | 68.39 | 57.55 | 61.71 | 67.15 | 63.70 |
| Our fold 3 | 77.64 | 61.27 | 69.02 | 79.43 | 71.84 |
| Our fold 4 | 77.14 | 63.32 | 69.62 | 75.05 | 71.28 |
| Our fold 5 | 79.02 | 61.56 | 67.74 | 76.48 | 71.20 |
| Our five-fold mean | 75.41 | 60.31 | 67.66 | 74.87 | 69.56 |

Each fold evaluates six different held-out patients with its own epoch-20 checkpoint. Consequently, the equal-fold mean here also equals the equal-patient mean. Do not choose the best fold to represent the five-fold result.

Numerical shortfall from the authors’ reported value (authors minus our five-fold mean):

| Liver (pp) | Stomach (pp) | Left kidney (pp) | Spleen (pp) | Mean (pp) |
| ---: | ---: | ---: | ---: | ---: |
| 20.03 | 14.00 | 24.76 | 16.29 | 18.77 |

This shortfall includes differences in training, patient selection and potentially evaluation protocol. It is not attributable to batch size alone. The paper’s broad medical pretraining and our additional BTCV-only fine-tuning are different training stages.

## 4. Box-padding comparison: matched six-patient experiment

All arms use the same fold-5 epoch-20 checkpoint, patients 0035–0040, initial box, experimental evaluator and original NIfTI data. There are 72 predictions: 6 patients × 4 organs × 3 arms.


| Method / experiment | Liver (%) | Stomach (%) | Left kidney (%) | Spleen (%) | Mean (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| H: historical box arithmetic | 79.02 | 61.56 | 67.74 | 76.48 | 71.20 |
| Z: tight box, explicit zero padding | 91.26 | 70.34 | 72.48 | 84.99 | 79.77 |
| P5: 5% image-relative padding per side | 80.72 | 56.24 | 30.97 | 64.96 | 58.22 |

| Comparison | Mean Dice change (pp) | Interpretation |
| --- | ---: | --- |
| Z − H | +8.57 | Effect of replacing the faulty historical box adjustment with a tight box |
| P5 − Z | -21.54 | Effect of the proposed 5% expansion |
| P5 − H | -12.98 | Total difference from historical behavior |

**Conclusion:** corrected zero improves the baseline. The tested 5% expansion worsens performance. The baseline correction is not a novel algorithm, and its gain must not be credited to positive padding. H matches the previous fold-5 Dice results; the within-experiment H/Z/P5 comparison is still the appropriate controlled analysis.

## 5. Z-spacing and B00/B01 are the same intervention comparison

B00 and B01 are factorial cell names, not two additional methods separate from z-spacing:

| Cell | Positive box padding | Conditional z-resampling |
| --- | --- | --- |
| B00 | OFF: corrected zero | OFF: segment thick input directly |
| B01 | OFF: corrected zero | ON: thick input → approximately 3 mm → segment → restore |
| B10 | ON | OFF |
| B11 | ON | ON |

The complete factorial remains deferred. B10/P5 has been evaluated on six native patients, but that is not a matched B10 arm for the 17-patient synthetic-spacing cohort. B11 has not been established by these results.

Both spacing branches receive the same synthetic thick-spacing image. Both are scored on the same thick-spacing reference grid. The proposed branch interpolates the input; it does not receive missing information from the fine-spacing original.

### Spacing study progression


| Method / experiment | Liver (%) | Stomach (%) | Left kidney (%) | Spleen (%) | Mean (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 B00: two patients, old geometry | 89.50 | 65.52 | 85.51 | 84.74 | 81.32 |
| v1 B01: two patients, old geometry | 92.88 | 65.84 | 84.30 | 90.08 | 83.27 |
| v2 B00: two patients, corrected resampling | 89.39 | 65.67 | 86.49 | 84.65 | 81.55 |
| v2 B01: two patients, corrected resampling | 92.71 | 67.30 | 84.02 | 88.88 | 83.23 |
| v3 B00: 17 patients, final geometry fixes | 85.78 | 64.48 | 72.40 | 81.88 | 76.13 |
| v3 B01: 17 patients, final geometry fixes | 89.83 | 64.22 | 68.57 | 85.23 | 76.96 |

v1 is retained only as historical evidence because its physical-coordinate convention was inconsistent. v2 corrected resampling and prompt mapping. v3 additionally updates the cropped origin and restores predictions onto the exact scoring reference. Version/cohort changes mean the net gains should not be compared as if only the method changed.

### Latest result: v3, 17 patients across all five folds

Eligible patient counts are 6/4/1/4/2 across folds 1–5. These are the 17 native-3-mm cases; the 2.5-mm case was not processed by the exact-3-mm synthetic generator. Each patient uses its held-out fold checkpoint. All 136 predictions completed. Aggregate by patient, not by an unweighted average of the five fold means.

| Organ | B01 − B00 Dice (pp) | Patients improved | Patients worsened |
| --- | ---: | ---: | ---: |
| Liver | +4.05 | 16/17 | 1/17 |
| Stomach | -0.26 | 9/17 | 8/17 |
| Left kidney | -3.84 | 2/17 | 15/17 |
| Spleen | +3.36 | 14/17 | 3/17 |
| Four-organ mean | +0.83 | — | — |

**Conclusion:** liver and spleen benefit in most cases, stomach has mixed Dice results, and left kidney worsens in most cases. The overall gain is modest because kidney losses offset other gains. The lower kidney recall is measured; extra propagation steps causing it remains a hypothesis, not a proven mechanism. These results do not justify a post-hoc organ-selection policy as a validated improvement.

The reported runtime is approximately 5.01 seconds for B00 versus 10.40 seconds for B01 per patient-organ case. These are local measurements, not directly comparable with paper timing on different hardware/protocols. All spacing evidence is synthetic, not a native thick-slice clinical benchmark.

## 6. Scope and status

| Work | Status | Interpretation |
| --- | --- | --- |
| Five-fold fine-tuning/evaluation | Complete | Baseline reference |
| Matched historical/zero/5% padding | Complete on six patients | Baseline correction helps; 5% padding hurts |
| Z-spacing B00/B01 | Complete on 17 patients | Modest overall gain, substantial organ differences |
| Combined B11 | Deferred | No combined improvement established |
| Novelty-specific retraining | Not required for completed interventions | Experiments use existing checkpoints |
| Generalisation to native thick-spacing data | Not established | Requires additional evidence |

Do not add the padding correction gain (+8.57 pp) and spacing gain (+0.83 pp): their cohorts, inputs and controls differ. Similarly, 83.23% from the two-patient spacing pilot is not evidence that the five-fold baseline rose from 69.56% to 83.23%.

## 7. Sources and raw evidence

- [Authors’ published paper, Table 1](https://raw.githubusercontent.com/mlresearch/v250/main/assets/quan24a/quan24a.pdf): one-box BTCV values supplied and verified earlier in this project.
- Local baseline: `btcv_cross_validation/logs/fold1_evaluation.json` through `fold5_evaluation.json`.
- [Box-padding report](Box_Padding_First_Results.md); raw results under `btcv_cross_validation/box_padding/results/`.
- [Spacing v1 report](Z_Spacing_First_Results.md); raw results under `btcv_cross_validation/z_spacing/results/`.
- [Spacing v2 report](Z_Spacing_Corrected_Results.md); raw results under `btcv_cross_validation/z_spacing/results_v2/`.
- [Spacing v3 / 17-patient report](B00_vs_B01_17patient_Results.md); raw results under `btcv_cross_validation/z_spacing/results_v3/`.
- [Archived matched-checkpoint diagnostics](../btcv_cross_validation/logs/checkpoint_audits/README.md).

Values are rounded only for display. Organ scores are equally weighted in the four-organ mean. The displayed paper mean is 88.33%; averaging its four rounded organ scores yields 88.3325%, which rounds to the same value.

