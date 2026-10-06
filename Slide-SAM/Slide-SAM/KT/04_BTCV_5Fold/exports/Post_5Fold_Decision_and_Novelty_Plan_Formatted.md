# After five-fold training: experiment choice and novelty plan

Date: 26 September 2026.

Companion document: [Detailed research proposals](SlideSAM_Research_Novelty_Proposal.md).

## 1. Direct recommendation: start with box padding

**Box padding is easier than z-resampling and is the recommended first experiment.** Start with the completed fold checkpoints; neither inference-only padding nor inference-only resampling requires retraining the segmentation model.

**Table 1. Box padding versus conditional z-resampling**

| Consideration                   | Automatic box padding                                           | Conditional z-resampling                                                   |
| ------------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Scope                           | Box coordinates during propagation                              | Image geometry, input grid, prompts, and restored predictions              |
| Main code                       | `datasets/data_engine.py`, `core/volume_predictor.py`           | Evaluation loader plus preprocessing/restoration wrapper and predictor     |
| Retraining for first experiment | No                                                              | No                                                                         |
| Extra inference cost            | Usually small, but expanded predictions can extend propagation  | Potentially substantial because more slices are processed                  |
| Main implementation risks       | Wrong signs, dimensions, coordinate scale, or excessive padding | Axis/orientation mistakes, physical-grid mismatch, interpolation artifacts |
| Data eligibility                | Applicable wherever nonempty masks generate boxes               | Proposed rule activates only when through-plane spacing exceeds 5.5 mm     |
| First priority                  | Yes                                                             | After metadata audit confirms suitable scans                               |

This is an engineering-effort judgment, not a prediction that padding will improve Dice more. Padding can increase false positives; resampling can smooth structures without recovering missing anatomy.

### Important baseline correction

The current path is not actually a zero-padding implementation. `generate_prompts_from_previous_masks()` calls `BoxPromptGenerator.enlarge()`, whose inspected code hardcodes 5% box-relative margins but subtracts from the upper edges and mixes horizontal/vertical margins. Its `margin` argument is unused.

Preserve the unchanged implementation as **H**, a historical reference. Implement a consistent, explicit zero-padding version as **B00**. Then compare positive padding with B00. Keep gains from correcting the existing function separate from gains caused by intentional expansion. No such code correction has been made as part of writing this plan.

The team's primary padding rule is **5% of the image width/height per side**. This is different from 5% of the box dimensions. On a 512-pixel-wide image it adds 25.6 pixels to each horizontal side before clipping. Keep that definition fixed; a box-relative policy is a separate comparison.

## 2. How to use the “within 6–7% of the authors” target

Interpret the proposed tolerance as **6–7 absolute Dice percentage points**, unless the team explicitly intends a relative percentage. For example, 88.33% to 81.33% is a 7-point decrease; a 7% relative decrease from 88.33% gives approximately 82.15%.

The manuscript reports the following BTCV one-box results. These are contextual targets, not a controlled reproduction benchmark unless the data split, model, preprocessing, and prompt protocol match. [Slide-SAM, Table 1](https://arxiv.org/html/2311.10121v3).

**Table 2. Reference Dice values and proposed tolerance bounds**

| Metric                | Authors' one-box Dice (%) | Lower bound within 7 points | Lower bound within 6 points |
| --------------------- | ------------------------: | --------------------------: | --------------------------: |
| Liver                 |                     95.44 |                       88.44 |                       89.44 |
| Stomach               |                     74.31 |                       67.31 |                       68.31 |
| Left kidney           |                     92.42 |                       85.42 |                       86.42 |
| Spleen                |                     91.16 |                       84.16 |                       85.16 |
| Four-organ macro mean |                     88.33 |                       81.33 |                       82.33 |

These lower bounds mean “no more than this much below,” not that a result above the authors' value should be rejected. Specify whether the target applies to the macro mean, every organ, or both before inspecting the remaining results. Recommended: use the macro mean as a descriptive target and always show every organ, so strong liver results cannot conceal poor stomach segmentation.

The saved local report currently gives fold 1 a macro Dice of 69.78% and fold 2 63.70%; their gaps from 88.33% are 18.55 and 24.63 points. Neither meets the suggested macro target. These are report values, not newly executed evaluations, and this document makes no assumption about the outcomes of folds 3–5. Source: `btcv_cross_validation/logs/fold1_fold2_report.md`.

**If no fold meets the target:** report that honestly, audit the pipeline, and compare the official pretrained checkpoint with the fine-tuned models under one identical evaluation protocol. Do not search among folds, organs, or prompts until a favorable number appears. A trustworthy baseline can still support research even if it does not match the authors' absolute score; what matters is a credible explanation of the setup and a controlled improvement.

## 3. Selecting a fold: development is different from final evaluation

The team's proposed sequence is understandable: finish training → identify a promising fold → develop a novelty on it. The adjustment is to select development data without turning the selected test set into a tuning set.

**Do not choose the fold closest to the paper's score and present subsequent gains on that same test fold as unbiased evidence.** Each fold has different patients; a high score may reflect easier anatomy rather than a superior transferable checkpoint. Selecting by held-out outcomes and repeatedly adjusting the method to those outcomes introduces selection bias.

### Recommended workflow

```text
Complete all five baseline training runs
    → evaluate each checkpoint only on its corresponding held-out patients
    → report all five folds and pooled out-of-fold scores
    → freeze baseline configurations, code, and predictions
    → choose development setup by feasibility/data coverage, not highest test Dice
    → tune the intervention on development/inner-validation patients
    → lock the method and its settings
    → compare baseline and intervention across the same five folds
    → confirm on a separate untouched external cohort where available
```

For fast engineering, a fixed fold checkpoint and a few training-side cases can test whether code runs and coordinates are correct. Those cases are not validation evidence if the checkpoint trained on them.

For parameter selection, create an inner training/validation split within the outer training set, and train an inner model excluding its validation patients. Tune padding, spacing rules, or learned modules there. A checkpoint already trained on all 24 outer-training volumes cannot provide an independent inner-validation result on any of those same 24 volumes. Alternatively, use a separately reserved development cohort with documented non-overlap, or prespecify the inference settings without tuning.

If the team nevertheless selects a fold using its observed test Dice, label all subsequent work on that fold **exploratory**. It remains useful for demonstrations and failure analysis, but confirmation needs untouched data. Re-running all five folds does not erase earlier tuning on their test outcomes.

Keep each patient's out-of-fold prediction from the model that excluded that patient. Do not ensemble all five checkpoints on the 30 BTCV patients and call that held-out CV: most of those checkpoints trained on each patient.

## 4. Concrete next steps after training

### Step A — Audit and freeze the reference

- Verify label mapping, checkpoint loading, intensity preprocessing, image orientation, and prompt transforms.
- Repair/verify surface metrics: current box-mode NSD is a placeholder zero. Calculate NSD and HD95 with correctly ordered physical spacing and explicit empty-mask handling.
- Save per-patient, per-organ Dice, predictions, initial prompts, and inference settings. Record checkpoint identities and source revision.
- Audit source scan spacing. Count patients above 5.5 mm, preserving their fold membership.

### Step B — Run padding first

Implement a configurable padding function with deterministic coordinate handling. Check zero padding, border clipping, non-square images, and non-square boxes. Keep the initial prompt unchanged; expand automatic propagation boxes only.

Compare historical H, corrected zero-padding B00, and 5% padding B10. Save false-positive volume, foreground recall, boundary metrics, and premature-stop events alongside Dice. Padding does not help when the mask was rejected before a bounding box was generated.

### Step C — Add resampling if the data support it

Apply the proposed >5.5 mm trigger and ≈3.0 mm target in physical space. Resample image intensities linearly and discrete labels with nearest-neighbor interpolation if labels are needed for simulation. Preserve the initial prompt's physical location and restore predictions to the original grid before scoring against untouched labels.

Three-millimeter z-spacing is not generally isotropic. If no native patient meets the trigger, report that the intervention is inactive on BTCV and use a declared synthetic spacing stress test or an appropriate external cohort. Do not claim a thick-slice clinical result from an inactive factor.

### Step D — Complete the original 2×2 study

**Table 3. The 2 × 2 evaluation matrix**

| Cell | Automatic padding          | Conditional z-resampling |
| ---- | -------------------------- | ------------------------ |
| B00  | 0%                         | OFF                      |
| B10  | 5% image-relative per side | OFF                      |
| B01  | 0%                         | ON                       |
| B11  | 5% image-relative per side | ON                       |

Keep the checkpoint, patients, prompts, fusion, and other thresholds identical within a fold. Report the eligible spacing subgroup separately from the full cohort. Five folds × four cells means **20 evaluation configurations**, not 20 training runs; historical H adds five reference configurations. Each configuration still covers the selected organs and patients.

Report paired patient-level changes and uncertainty intervals, plus per-fold summaries. The factorial interaction on a higher-is-better metric is `B11 − B10 − B01 + B00`; positive values suggest more-than-additive benefit on that metric scale but require uncertainty assessment.

## 5. Research directions beyond the authors' stated limitations

These candidates are motivated by local code behavior and broader segmentation research. They are not certified new discoveries. Each needs a focused nearest-work review and evidence that its specific mechanism improves on simpler baselines.

### Candidate 1 — Distinguish true organ endings from propagation failure

**Question:** when a predicted mask disappears, has the organ ended, or has the model lost it?

**Proposed mechanism:** replace immediate termination with a bounded state controller: continue, inspect/retry, or stop. Use appearance features, recent mask evolution, spacing, and overlap disagreement to distinguish genuine endings from temporary loss. Probe only a limited physical distance beyond an uncertain failure and require evidence before accepting a recovered mask.

```text
Small/empty prediction → assess end-versus-failure evidence
    → confident ending: stop
    → uncertain: bounded retry using last reliable target context
    → successful recovery: continue; otherwise stop or request correction
```

**Code:** `core/volume_predictor.py`, especially small-mask rejection and propagation-loop termination. A learned controller also needs sequences including true organ endpoints and background slices; the current positive-mask cache may underrepresent these cases.

**Experiments:** compare immediate stop, fixed extra-step retry, and the proposed controller at matched extra decoder-call budgets. Measure terminal extent error in millimeters, premature stops, recovered foreground, and spurious continuation outside the organ.

**Effort:** medium. **Risk:** recovery may hallucinate organ continuation. **Candidate novelty:** explicit end-versus-failure discrimination with physically bounded recovery, not generic memory or propagation. Modern memory-based methods such as [MedSAM2](https://arxiv.org/abs/2504.03600) are relevant comparators. This is my preferred additional research hypothesis if trajectory logs show frequent early termination.

### Candidate 2 — Refine boundaries using disagreement between slice-output roles

**Question:** can a small refinement model correct boundary regions where central-slice and neighboring-window outputs disagree?

**Proposed mechanism:** collect probabilities for the same slice from different window positions, identify an uncertainty band around their boundaries in physical units, and predict a residual correction from local image evidence plus the competing predictions. Preserve confident interior/exterior regions unless evidence supports changing them.

```text
Overlapping predictions → disagreement band → local image-conditioned refinement
    → corrected slice probabilities → volume assembly
```

**Code:** expose unthresholded outputs in `core/volume_predictor.py`; add a small refiner under `modeling/` and train it only on training patients. Prefer cross-fitted training predictions or realistic perturbations over unrealistically clean in-sample masks.

**Experiments:** compare union, probability averaging, standard morphology, a generic refiner, and the role-aware refiner. Measure NSD, HD95, thin peripheral-region recall, Dice, and added runtime. Include an ablation removing slice-role information.

**Effort:** medium to high. **Risk:** erasing small structures or only learning the training model's error pattern. Generic mask refinement already exists in [SAMRefiner](https://proceedings.iclr.cc/paper_files/paper/2025/hash/90800f46f84381b7891e1378ee850013-Abstract-Conference.html); the candidate distinction is how overlapping three-slice roles locate and resolve errors, not merely adding a refinement network.

### Candidate 3 — Separate segmentation from calibrated quality screening

**Question:** can the system flag unreliable organ volumes without access to their ground truth, instead of returning every output with equal confidence?

**Proposed mechanism:** predict a volume-level failure score from propagation records and calibrate a review/abstention threshold on independent calibration patients. Return both the mask and a review flag. If data are sufficient, investigate conformal risk control for a precisely defined volume-level loss.

```text
Segmentation + propagation diagnostics → volume-quality score
    → independently calibrated threshold → accept or flag for review
```

**Code:** export diagnostics from the predictor and add a separate quality module/evaluation script. This does not require changing the encoder initially.

**Experiments:** compare predicted IoU, mean stability, and the proposed trajectory score. Report failure-detection sensitivity, precision, calibration, and risk-versus-coverage curves; retain the all-case segmentation results. Removing flagged patients must not be presented as a Dice improvement on the original cohort.

**Effort:** medium for empirical screening; high for rigorous risk guarantees. **Risk:** BTCV's 30 patients provide little independent calibration data, and dataset shift can invalidate assumptions. Use patients, not correlated slices, as the calibration unit. Do not promise patient-specific safety guarantees.

Conformal calibration is already used in [ConformalSAM](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_ConformalSAM_Unlocking_the_Potential_of_Foundational_Segmentation_Models_in_Semi-Supervised_ICCV_2025_paper.pdf) and [Conformal Lesion Segmentation](https://arxiv.org/abs/2510.17897). A candidate contribution must specifically demonstrate useful propagation-derived quality evidence and correctly scoped calibration, not simply use the word conformal.

### Candidate 4 — Preserve target identity when nearby organs compete

**Question:** can independently prompted organ predictions be reconciled when they overlap or propagation drifts into a neighboring organ?

**Proposed mechanism:** use the prompted target's representation plus competing organ logits to resolve inconsistent assignments while allowing background/abstention. Enforce mutual exclusion only for labels that are genuinely mutually exclusive in the dataset; do not apply it to nested labels such as an organ and its lesion.

```text
Per-organ prompts → per-organ probability volumes
    → detect label conflicts → target-conditioned arbitration → consistent label volume
```

**Code:** orchestrate multiple organ predictions in evaluation and add a conflict-resolution module. Keep inference labels unavailable; target identity must come from the declared user/task prompt.

**Experiments:** compare independent predictions, calibrated argmax arbitration, and the proposed resolver. Measure conflicting voxel volume, organ swaps, per-organ Dice/NSD, and correction of both false positives and false negatives. Use identical organ prompt budgets for all methods.

**Effort:** medium to high. **Risk:** a dominant large-organ prediction can suppress a correct small organ. Anatomical priors are already explored in [KG-SAM](https://arxiv.org/abs/2509.21750), so the research claim needs a concrete target-identity and propagation-specific distinction.

### Bridge from the first experiment — Adaptive padding

If fixed padding helps some cases but harms others, a natural next hypothesis is to choose padding based on physical scale, mask uncertainty, and observed contour change. Compare against tuned fixed image-relative, box-relative, and millimeter-based margins. Count all candidate-box decoder calls and keep their budget equal across policies.

This is an extension of the team's original idea rather than an independent topic. Adaptive/elastic box generation already appears in [SAMRefiner](https://proceedings.iclr.cc/paper_files/paper/2025/hash/90800f46f84381b7891e1378ee850013-Abstract-Conference.html); calling padding “adaptive” does not itself establish novelty.

## 6. Choose the novelty from the observed failure, not the highest fold score

**Table 4. Choosing a research direction from observed errors**

| Dominant error after the simple baselines              | Next research candidate                                                     |
| ------------------------------------------------------ | --------------------------------------------------------------------------- |
| Masks vanish before the true organ endpoint            | End-versus-failure controller                                               |
| Organ extent is correct but contours remain inaccurate | Role-aware boundary refinement                                              |
| Most cases work but failures are confidently missed    | Calibrated volume-quality screening                                         |
| Predictions drift into or overlap neighboring organs   | Target-identity conflict resolution                                         |
| Fixed padding has strongly variable effects            | Adaptive padding, with strong fixed-margin baselines                        |
| Failures remain concentrated in large spacing gaps     | Spacing-conditioned context or rollout training from the companion proposal |

**Recommended sequence:** finish and document all five folds → audit evaluation → padding first → conditional resampling where eligible → complete the 2×2 study → choose one supported research mechanism → develop on separate development data → lock settings → report paired fold-wise results and external confirmation where possible.

Your 6–7-point target is useful for diagnosing the baseline. It should not determine which test patients are retained, which folds are reported, or whether an otherwise sound research experiment is allowed to proceed. A convincing contribution is a reproducible improvement on a clearly defined problem, with its costs and failure cases reported.
