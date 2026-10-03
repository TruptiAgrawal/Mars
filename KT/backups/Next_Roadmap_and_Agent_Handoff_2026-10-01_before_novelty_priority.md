# Slide-SAM: next roadmap and CLI/agent handoff

Updated: 1 October 2026, Asia/Kolkata. This document records verified local results and the next research decisions. Read it before launching training or evaluation.

## 1. Current decision

**Do not start BS=8 training or repeat completed evaluations. First audit why BTCV fine-tuning reduces performance relative to the supplied pretrained model.** Then establish validation-based development and test box padding. Z-spacing handling follows only with suitable data.

The latest trajectory evaluation has already completed. The previous chat instruction to run `checkpoint_trajectory/run.sh` is now superseded by its saved results. The completed runner and copied source tree have now been removed with user approval; do not run the old command.

This roadmap does not authorize sending messages, publishing results, deleting experiments, or launching unspecified long training jobs. For the current handoff, the user requested runnable commands for subsequent experiments. Prepare and explain commands; let the user launch them unless they explicitly ask the agent to run them.

The three completed diagnostic folders were consolidated into [the audit archive](../btcv_cross_validation/logs/checkpoint_audits/README.md). Only Markdown and logs remain there. Training checkpoints and the box-padding experiment are unchanged.

## 2. Workspace, environment and operating rules

| Item | Location / rule |
| --- | --- |
| Working project | `/home/jupyter-team2pop/Slide-SAM/Slide-SAM` |
| Outer authors’ checkout | `/home/jupyter-team2pop/Slide-SAM` |
| Python environment | Conda environment `slidesam` |
| Observed GPU | NVIDIA RTX A6000, approximately 48 GiB; check current availability |
| Knowledge transfer | `KT/` |
| Experiment outputs | `btcv_cross_validation/` |
| Before code changes | Copy the source first; use a separate experiment directory and record the diff |
| Existing files | Preserve checkpoints, manifests, caches and prior results; do not reset unrelated changes |
| Imports | Set explicit `PYTHONPATH` to the intended source tree |
| CUDA runtime | Completed matched runs used native CUDA with `torch.backends.cudnn.enabled=False` after cuDNN initialization failed |

Initial commands, from a terminal with GPU access:

```bash
cd /home/jupyter-team2pop/Slide-SAM/Slide-SAM
conda activate slidesam
git status --short
nvidia-smi
cat KT/Next_Roadmap_and_Agent_Handoff_2026-10-01.md
```

The Git tree contains pre-existing changes. Do not treat them as disposable. In a sandbox, GPU access may require an approved execution outside the sandbox; do not interpret sandbox GPU invisibility as a missing driver without checking.

## 3. Results completed so far

### 3.1 Same-evaluator checkpoint comparison

All three models below were evaluated on the same six patients, `img0035` through `img0040`, with the frozen outer authors’ evaluator and dependencies.

| Organ | Label index | Official pretrained (%) | BS=2 epoch 20 (%) | BS=4 fold 5 epoch 20 (%) |
| --- | ---: | ---: | ---: | ---: |
| Liver | 6 | 83.4941 | 80.0005 | 79.0197 |
| Stomach | 7 | 69.7184 | 62.8340 | 61.5634 |
| Left kidney | 3 | 74.9429 | 68.9815 | 67.7364 |
| Spleen | 1 | 88.3187 | 77.9389 | 76.4808 |
| **Four-organ mean** | — | **79.1185** | **72.4387** | **71.2001** |

Fine-tuning reduced the mean by 6.6798 percentage points for BS=2 and 7.9185 for BS=4 on this cohort. BS=2 exceeds BS=4 by 1.2387 points. These observations do not establish a causal batch-size effect or explain the mechanism of degradation.

The re-evaluated BS=2 and BS=4 organ means reproduce their historical scores. All 422 tensors in the old BS=2 epoch-20 checkpoint equal those in its historical `ckpt_v/model_latest.pth`; their serialized file hashes differ, so file-hash inequality alone is not evidence of different weights.

Evidence:

- [Matched BS comparison](../btcv_cross_validation/logs/checkpoint_audits/matched_bs_comparison.report.md), with raw results and verification preserved in `matched_bs_comparison.evidence.log`, and console output in `matched_bs_comparison.run.log`.
- [Official comparison](../btcv_cross_validation/logs/checkpoint_audits/official_checkpoint_comparison.report.md), with the corresponding raw results and verification.

### 3.2 Fold-5 checkpoint trajectory — now completed

| Checkpoint | Liver (%) | Stomach (%) | Left kidney (%) | Spleen (%) | Mean (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official pretrained | 83.4941 | 69.7184 | 74.9429 | 88.3187 | 79.1185 |
| BS=4 epoch 0 | 83.4941 | 69.7184 | 74.9429 | 88.3187 | 79.1185 |
| BS=4 epoch 5 | 75.9218 | 62.8912 | 66.9910 | 75.4067 | 70.3027 |
| BS=4 epoch 10 | 77.1713 | 61.9877 | 66.5432 | 75.3285 | 70.2577 |
| BS=4 epoch 15 | 78.7640 | 61.8940 | 66.7244 | 75.3469 | 70.6823 |
| BS=4 epoch 20 | 79.0197 | 61.5634 | 67.7364 | 76.4808 | 71.2001 |

Interpretation: degradation is already substantial by epoch 5; this is not merely a final-epoch decline. Later epochs partly recover the mean. Epoch 0 is a debug-training checkpoint saved after a short training path, not untouched initialization. It matches the official scores at displayed precision, not bit-exactly: maximum absolute per-case Dice difference is approximately `1.24e-6` on the 0–1 scale.

The handoff audit verified 96 new case scores, matched geometry/prompt-slice metadata, and unchanged frozen source hashes. The official and epoch-20 values in the trajectory report reuse the earlier completed runs.

Evidence: [Trajectory report](../btcv_cross_validation/logs/checkpoint_audits/checkpoint_trajectory.report.md) with raw records in `checkpoint_trajectory.evidence.log` and console output in `checkpoint_trajectory.run.log`. This is a **post-hoc test-set diagnostic**, not a validation curve for checkpoint selection.

### 3.3 Existing five-fold results

These are the existing fold-specific evaluation results, not new re-evaluations of every fold through the frozen outer source.

| Fold | Four-organ mean Dice (%) |
| --- | ---: |
| 1 | 69.7833 |
| 2 | 63.7012 |
| 3 | 71.8380 |
| 4 | 71.2812 |
| 5 | 71.2001 |
| Equal-weight five-fold mean | 69.5607 |

Sources: `btcv_cross_validation/logs/fold1_evaluation.json` through `fold5_evaluation.json`. Some older aggregate Markdown reports cover only four folds; prefer the complete JSON records.

Do not select whichever fold happens to resemble the paper. Fold test sets contain different patients. A valid cross-validation report includes all folds; selecting a fold on test Dice introduces selection bias.

## 4. Checkpoints and evaluation protocol

Paths below are relative to the working project root.

| Checkpoint | Path | Required adapter loading |
| --- | --- | --- |
| Official local release | `weights/slidesam_b.pth` | Rank-8 only |
| BS=2 epoch 20 | `../runs/sam/ddp_sub/btcv_4to1_20epoch/ckpt/model_epoch_20.pth` | Rank-8, then rank-1, then strict load |
| BS=4 fold 5 epoch 20 | `btcv_cross_validation/runs/fold5/ddp_sub_cv/cv_fold5/ckpt/model_epoch_20.pth` | Rank-8, then rank-1, then strict load |
| Fold-5 earlier checkpoints | Same directory, epochs `0`, `5`, `10`, `15` | Same nested adapters |
| Old short pilot | `../runs/sam/ddp_sub/btcv_liver_run2/ckpt/model_epoch_3.pth` | Same nested adapters |

Use lowercase `cv_fold5`. Older notes mention a progressing `cv_Fold5` directory; that is historical and not the current confirmed checkpoint path. “Liver” in the old pilot filename/log does not identify the evaluated organ: label index 2 is **right kidney**, not liver.

The matched protocol uses:

- ViT-B, model evaluation mode, float32, no gradients, inference slice batch 4.
- The authors’ `AbstractLoader` and `Evaluater.eval_step` in copied `author_source/` trees.
- Ground-truth organ extent cropping along z; initial box on the maximum-area foreground slice, subject to the historical code’s handling.
- Original propagation, postprocessing and noise removal; IoU threshold 0.5, stability threshold 0.6.
- No new padding and no z-resampling.
- Mean patient Dice per organ, then equal-weight mean across four organs.

This is a ground-truth-assisted evaluation, not fully automatic whole-volume deployment. The authors’ printed `nsd=0` is a placeholder; do not report it as measured NSD. Checkpoint hashes and historical imported source paths are saved in the original `results.json` text embedded in each archive evidence log. Those copied source paths no longer exist.

The local official file is the model used to initialize the training runs. Its saved hash identifies the file; it does not certify the release’s training-data provenance or establish that our test patients were absent from its pretraining.

## 5. What the paper does and does not establish

Sources checked during this discussion:

- [Published MIDL paper](https://raw.githubusercontent.com/mlresearch/v250/main/assets/quan24a/quan24a.pdf).
- [Accessible manuscript](https://arxiv.org/html/2311.10121v3).
- [OpenReview paper and rebuttals](https://openreview.net/forum?id=LNdU9RTv3L); the user supplied the rebuttal text when browser access was challenged.
- [Authors’ repository](https://github.com/Curli-quan/Slide-SAM).

The paper’s one-box BTCV scores are liver 95.44, stomach 74.31, left kidney 92.42, spleen 91.16, mean 88.33%. It describes a 4:1 split, but we have not established identical patient IDs. The official local checkpoint is still 9.2115 percentage points below that reported mean on our cohort; do not promise that one setting will close this gap.

The paper describes over 4,000 CT/MRI volumes, multiple public/private datasets, pseudo-labels, AdamW with learning rate 0.0002 and weight decay 0.1, and 20 epochs on four GPUs. Our additional fine-tuning uses 24 BTCV training patients, learning rate 0.00002, weight decay 0.0001, and additional rank-1 adapters with the existing model frozen. The pretrained model already carries prior training; this is not a comparison of training from scratch on 24 versus 4,000 volumes.

The outer fine-tuning YAML says `batch_size: 2 # 8 for A100`. That comment is not evidence that BS=8 gives the paper’s score. The paper’s explicit batch-size-4 statement concerns inference. Appendix C mentions IoU filtering at 0.4, whereas the copied predictor defaults to 0.5. Audit this discrepancy, but do not silently alter the baseline or optimize thresholds on test cases.

At 8,159 samples per full epoch, one GPU and no gradient accumulation:

| Training batch size | Updates per full epoch | Updates in 20 full epochs |
| --- | ---: | ---: |
| 2 | 4,079 | 81,580 |
| 4 | 2,039 | 40,780 |
| 8 | 1,019 | 20,380 |

These exclude the short epoch-0 debug path. Equal epochs do not mean equal update budgets. BS=4 for roughly 40 epochs would match BS=2’s 20-epoch update budget but see more samples; it is a possible diagnostic, not a recommended immediate run or guaranteed improvement.

## 6. Next roadmap, in execution order

### Stage A — Read-only early-training audit: next task

Do this before another GPU experiment. Produce a new `KT/Early_Finetuning_Audit.md` with evidence and a proposed pilot; this file does not exist merely because it is named here.

1. Review `core/ddp_sub_cv.py`, `core/learner_sub1.py`, `core/lora_sam.py`, `core/learner2.py`, `core/learner3.py`, `core/loss.py`, `trans_utils/trainer_ddp.py`, `datasets/cache_dataset3d_cv.py`, its base loader and `datasets/data_engine.py`.
2. Verify trainable parameter names/counts and strict loading. Existing fold-5 analysis reports 48 rank-1 tensors / 36,864 trainable values. Check that updates are confined to intended parameters; frozen weights should not change.
3. Compare checkpoint adapter changes at epochs 0/5/10/15/20. Use nested LoRA key mappings; do not compare unmatched checkpoint key strings as if weights differ. Reuse `fold5_analysis/` artifacts before recomputing.
4. Check actual optimizer/scheduler behavior, prompt generation, loss terms, data normalization, label integrity and sampled organ distribution. Configuration fields alone do not prove a scheduler runs; inspected `configure_optimizers` returns no scheduler.
5. Inspect training logs and cached-label semantics. Existing cache byte equality confirms BS=2/BS=4 saw matching stored images/masks, not that JPEG labels are ideal. If a cache repair is needed, version it and treat it as a baseline correction.
6. Separate findings from hypotheses. Lower training loss plus worse held-out Dice is compatible with several causes; do not diagnose overfitting, forgetting or a learning-rate bug without evidence.

Useful read-only commands:

```bash
cat btcv_cross_validation/logs/checkpoint_audits/checkpoint_trajectory.report.md
cat btcv_cross_validation/logs/fold5_analysis/fold5_analysis_summary.txt
rg -n 'use_lora|load_well|validset|breakpoint' core/ddp_sub_cv.py
rg -n 'freeze_all|requires_grad|linear_a|linear_b' core/lora_sam.py
rg -n 'configure_optimizers|scheduler|AdamW' core/learner2.py
rg -n 'epoch == 0|optimizer.step|backward|validation' trans_utils/trainer_ddp.py
```

Some historical reports call final test evaluation “validation.” Do not inherit that terminology; the completed runs did not have a clean per-epoch inner-validation selection process.

### Stage B — Establish honest development splits

For any new training/tuning pilot, reserve validation patients from the 24 outer-training patients and retrain the pilot without those patients. Merely designating patients as validation after a checkpoint trained on them is invalid. Record explicit train/validation/outer-test manifests and assert patient disjointness; all slices and synthetic variants of one patient stay together.

The old 24:3:3 pilot has a distinct historical validation role, but its three validation cases overlap fold 5’s test patients. It can check mechanics; it cannot provide independent tuning evidence for fold 5. Likewise, trying padding on its short epoch-3 checkpoint cannot establish the best padding setting for trained CV models.

The current six test patients have already been inspected repeatedly. Disclose these diagnostics. Future confirmatory claims need a prespecified protocol and, ideally, additional untouched data. Do not choose a fold or checkpoint because it is within 6–7 points of the paper.

### Stage C — Small controlled training pilot, only if the audit justifies it

Keep a fixed pretrained initialization, manifests, seeds, prompt protocol, and evaluator. Include the untouched official checkpoint as a reference. Change one factor at a time: adaptation learning rate, training duration, or trainable-module capacity. Rank increases or decoder unfreezing are hypotheses, not established solutions.

Use inner-validation Dice and boundary metrics to determine stopping and settings. Record optimizer updates as well as epochs. Do not immediately launch five new folds, BS=8, a large learning-rate jump to the paper setting, or a 40-epoch extension. First provide a concrete copied configuration, budget, command and acceptance criteria to the user.

### Stage D — Padding pilot and baseline correction

Padding is the first practical novelty experiment because it is inference-only and requires no retraining of an existing checkpoint. Use the isolated code under `btcv_cross_validation/box_padding/`; read its README fully. No GPU padding results were found at handoff time. Previously prepared CPU tests passed; rerun them before modifying this code.

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/slidesam-padding-mpl \
  python -m unittest btcv_cross_validation.box_padding.test_cpu -v
```

The experimental evaluator changes some boundary/orientation/metric handling. Therefore evaluate **historical behavior, corrected zero padding, and positive padding through that same experimental evaluator**. Do not subtract an old author-evaluator score from a new experimental-evaluator score and call the difference a padding gain.

A mechanics-only preflight for the old short checkpoint is below. It does not start inference. The organ is right kidney, deliberately matching the earlier pilot context rather than the four-organ paper table.

```bash
python -m btcv_cross_validation.box_padding.evaluate \
  --data-root datasets \
  --manifest datasets/dataset_list/01_BTCV_val.txt \
  --train-manifest datasets/dataset_list/01_BTCV_train.txt \
  --checkpoint ../runs/sam/ddp_sub/btcv_liver_run2/ckpt/model_epoch_3.pth \
  --checkpoint-kind finetuned --label-idx 2 --prompt box \
  --protocol gt-cropped --padding-mode none --nsd-tolerance-mm 1.0 \
  --output btcv_cross_validation/box_padding/results/pilot_right_kidney_none
```

For an actual approved run, add `--run-inference`; first verify GPU access and runtime compatibility. The padding runner has not been verified on the GPU by the matched-run CUDA workaround. Use `--checkpoint-kind official` for official weights and `finetuned` for nested adapters. For a research experiment, substitute the properly established development manifests, checkpoint and label index; the checkpoint must actually correspond to the supplied training manifest. Manifest overlap checks cannot prove training provenance.

Use separate output directories for every checkpoint, organ, protocol and padding mode. Compare `historical`, `none`, and `image --padding-fraction 0.05`; consider 0.10 only under a prespecified development comparison. NSD 1 mm above is an example tolerance, not a clinically established choice.

### Stage E — Spacing robustness and stronger mechanisms

Proceed only after the baseline and padding result are interpretable. See Sections 7–8 below for the candidate mechanisms and data constraints. Freeze settings on development data before confirmation; report patient-level paired changes and all folds rather than the best fold.

## 7. Team’s two initial improvements

### 7.1 Prompt-noise robustness through propagated-box padding

The hypothesis is that propagated boxes can omit boundaries and contribute to early termination. On an image grid of width W and height H, expand each side using:

```text
x0' = max(0, x0 - p*W)       x1' = min(W-1, x1 + p*W)
y0' = max(0, y0 - p*H)       y1' = min(H-1, y1 + p*H)
```

Start with p=0.05; 0.10 is another candidate, not a selected optimum. This is **image-relative padding per side**, not box-relative padding. Apply only to automatically propagated boxes; keep the initial prompt and ground truth unchanged. Declare the coordinate grid and clipping convention. The prepared implementation uses the model-input grid and float coordinates.

Important: the original `BoxPromptGenerator.enlarge()` already applies malformed 5% box-relative arithmetic: upper coordinates are subtracted, x/y margins are mixed, and its margin argument is unused. Thus “historical baseline” is not identical to explicit p=0. Separate this bug/behavior correction from the padding contribution.

Padding cannot revive a mask rejected before box generation or an already empty mask. Evaluate precision, recall, terminal-slice coverage, premature stopping, Dice, NSD and HD95; larger boxes can also introduce false positives. Fixed padding alone is a useful baseline and possible empirical contribution, not a strong automatic novelty claim.

### 7.2 Conditional through-plane resampling

Proposed rule: read physical geometry; if through-plane spacing exceeds 5.5 **mm**, resample that axis toward 3.0 mm. Use linear interpolation for images and nearest-neighbor for discrete labels, preserve origin/direction and the prompt’s physical location, and map predictions back to the declared original evaluation grid.

Corrections to remember:

- Spacing is sample-center separation, not necessarily acquisition slice thickness.
- Changing only z to 3 mm is not isotropic unless x and y are also 3 mm.
- Interpolation cannot recover missing anatomy and can introduce artifacts.
- The authors increased spacing to test degradation. They did not establish 5.5 mm as a universal cutoff or show that interpolating back fixes it.
- The 5.5 mm trigger is our proposed setting, not an existing author-code rule.

The local header audit found zero native scans above 5.5 mm: BTCV has 30 scans at 2.5/3/5 mm; WORD train/validation/test have 100/20/30 scans at 2.5/3 mm. Thus the proposed rule is a no-op on these native datasets. Use genuinely thick-spacing data or explicitly labeled synthetic degradation experiments. Synthetic recovery must use only the degraded image, not information from the original fine-spacing image. Keep all variants grouped by patient.

### 7.3 Planned factorial design

| Cell | Propagated padding | Conditional resampling |
| --- | --- | --- |
| B00 | Corrected p=0 | OFF |
| B10 | Image-relative p=0.05 per side | OFF |
| B01 | Corrected p=0 | ON when spacing >5.5 mm; target ~3 mm |
| B11 | Image-relative p=0.05 per side | Same ON rule |

Keep an additional unchanged historical reference. The factorial comparison must otherwise share checkpoint, initial physical prompt, evaluator and scoring grid. On the current native cohorts, spacing ON/OFF cells should coincide because the trigger never activates; do not present that as a meaningful spacing-gain study.

## 8. Stronger candidate research contributions

| Candidate | Concrete mechanism | Training needed? | Required evidence |
| --- | --- | --- | --- |
| Physical-spacing-conditioned context | Learn how much adjacent slices contribute using physical distance and image/context features | Yes | Improvement over fixed windows and resampling across spacing regimes |
| Recovery from propagated prompt errors | Train with model-generated imperfect prompts and a recovery curriculum | Yes | Reduced error accumulation and premature stopping versus generic prompt jitter and fixed padding |
| Budgeted corrective prompting | Request corrections where they are predicted to improve downstream slices | Depends on selector | Dice/annotation-time or prompt-budget curves against random, uniform and uncertainty baselines |
| Same-slice agreement across overlapping windows | Compare predictions for a slice in different window positions; combine or flag disagreement | Possibly inference-only | Benefit beyond simple fusion, with runtime and failure analysis |

These connect the initial proposals to larger research questions: padding is a baseline for prompt-error recovery; resampling is a baseline for spacing-conditioned context. Their novelty is not established. Generic LoRA fine-tuning, five-fold CV, longer training, backbone replacement, bidirectional propagation and confidence fusion are not sufficient novelty claims alone. Recheck nearest prior work before writing publication claims; the prior-work notes in the older proposal are pointers, not a completed novelty review.

See [full novelty proposal](SlideSAM_Research_Novelty_Proposal.md) for mechanisms and ablations. Do not promise specific Dice gains, paper-level replication, or publication acceptance.

## 9. Completion criteria and handoff checklist

Before each new experiment, record:

1. The specific hypothesis and whether this is mechanics checking, diagnosis, development or confirmation.
2. Copied source/configuration paths, diff, source hashes and checkpoint hash.
3. Training/validation/test patient IDs and provenance; no patient-level overlap for the intended role.
4. Label mapping: liver 6, stomach 7, left kidney 3, spleen 1; right kidney 2.
5. Prompt budget, crop protocol, geometry, padding definition, resampling grid, thresholds and metric definitions.
6. Exact command, environment, runtime estimate, GPU check and unique output directory.
7. Completed case counts, failures/empty masks, raw patient scores and aggregate results.

For final research comparisons, use paired patient-level uncertainty estimates; do not count slices or synthetic variants as independent patients. Report runtime, peak GPU memory, false positives and undefined surface metrics transparently. Paper-vs-local differences require matching protocol and split, not just similar average Dice.

Suggested task for the next CLI/agent:

> Read this handoff and the three completed comparison reports. Do not rerun them. Perform Stage A as a read-only audit of the early fine-tuning degradation, reusing the existing fold-5 analysis. Identify verified implementation differences separately from hypotheses, document findings in a new KT file, and propose a small validation-based pilot with copied configuration and exact command. Do not launch new training or change padding/thresholds on the test cases without the user’s next instruction.

## 10. Historical documents and precedence

- [Basic audit and padding pilot](Basic_Audit_and_Padding_Pilot_Plan.md).
- [Original post-five-fold decision plan](Post_5Fold_Decision_and_Novelty_Plan.md).
- [Formatted plan for copying tables](Post_5Fold_Decision_and_Novelty_Plan_Formatted.md).
- [Z-spacing explanation](Z_Spacing_5_5mm_Explanation.md).
- [Padding implementation README](../btcv_cross_validation/box_padding/README.md).

Retain these for history. Statements that fold 5 is still running, that the matched/official/trajectory runs are pending, or that the current fold-5 directory is `cv_Fold5` are superseded by this dated handoff and the verified result files. The README’s statement that its padding code has not yet produced a GPU result remains applicable at this handoff.
