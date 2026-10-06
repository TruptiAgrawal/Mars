# Next steps: box padding first, z-spacing second

Updated: 1 October 2026. **This is the current execution plan.** It supersedes the ordering in `Next_Roadmap_and_Agent_Handoff_2026-10-01.md` and the training/cache pilot in `Early_Finetuning_Audit.md`. Those documents remain useful evidence, but their training investigations must not block this novelty experiment.

## 1. Agreed scope

1. Test **propagated-box padding** on the existing fold-5 epoch-20 checkpoint.
2. Report its measured effect using a controlled comparison.
3. Then prepare **conditional z-spacing handling**, with a suitable spacing-robustness dataset and the planned factorial comparison.

**No retraining or label-cache rebuild is required for step 1.** Evaluation reads original NIfTI images and labels, not cached JPEG training masks. Cache repair and fine-tuning diagnosis are parked. Do not change batch size, learning rate, adapters, training duration or checkpoint during the padding comparison.

User preferences: copy code before editing; preserve existing results; ask before deleting anything. This document supplies commands for the user to launch. Creating this plan does not start any GPU run.

## 2. What is already known

| Existing evaluation | Four-organ mean Dice (%) |
| --- | ---: |
| Untouched official checkpoint | 79.1185 |
| BS=2 epoch 20 | 72.4387 |
| BS=4 fold 5 epoch 20 | 71.2001 |
| BS=4 fold 5 epoch 5 | 70.3027 |

The official model outperformed fine-tuned models on these six patients. This finding does not prevent testing whether padding improves the selected fold-5 model. We are not testing why fine-tuning degraded here.

Completed diagnostics are archived in [checkpoint_audits](../btcv_cross_validation/logs/checkpoint_audits/README.md). Do not rerun the removed comparison/trajectory scripts.

## 3. Lock the padding experiment before execution

| Setting | Fixed choice |
| --- | --- |
| Checkpoint | `btcv_cross_validation/runs/fold5/ddp_sub_cv/cv_fold5/ckpt/model_epoch_20.pth` |
| Checkpoint type | Finetuned: nested rank-8 + rank-1 adapters |
| Patients | Fold-5 held-out patients `img0035`–`img0040` |
| Training manifest | `btcv_cross_validation/dataset_list/fold5/01_BTCV_train_4to1.txt` |
| Evaluation manifest | `btcv_cross_validation/dataset_list/fold5/01_BTCV_test_4to1.txt` |
| Organs | Liver=6, stomach=7, left kidney=3, spleen=1 |
| Initial prompt | Same deterministic ground-truth-derived box in every arm |
| Evaluation extent | `gt-cropped` in every arm |
| Evaluator | Existing isolated `btcv_cross_validation/box_padding/` implementation |
| Z-resampling | OFF |
| Primary metric | Mean patient Dice per organ, then equal-weight mean across four organs |
| Secondary metrics | NSD at a declared 1 mm tolerance, HD95, recall, precision and false-positive volume |
| Experimental role | Prespecified exploratory comparison; not validation-based hyperparameter selection |

The 1 mm NSD tolerance is an explicit experimental choice, not a clinical claim. These six cases have already been inspected. Do not use this experiment to choose a best fold, checkpoint or padding fraction. A later confirmatory study needs a locked protocol with validation-separated development and appropriate held-out data.

## 4. Run exactly three arms

| Arm | CLI mode | Meaning |
| --- | --- | --- |
| H: historical | `historical`, fraction 0 | Existing propagation-box arithmetic |
| Z: corrected zero | `none`, fraction 0 | Explicit tight box, valid bounds, no expansion |
| P5: proposed padding | `image`, fraction 0.05 | Expand each side by 5% of the image width/height |

The historical `enlarge()` implementation is not true zero padding: its signs and x/y margins are inconsistent. Therefore keep H separate from Z. The primary padding comparison is **P5 minus Z**. Report **Z minus H** as a behavior correction, and **P5 minus H** as the total change.

Only automatically propagated boxes change. Do not pad initial prompts or modify ground truth. On the model-input grid:

```text
x0' = max(0, x0 - 0.05*W)     x1' = min(W-1, x1 + 0.05*W)
y0' = max(0, y0 - 0.05*H)     y1' = min(H-1, y1 + 0.05*H)
```

This is image-relative padding per side, not box-relative padding. The initial proposal included 5–10%; this first comparison fixes 5%. Do not add a 10% sweep after looking at test scores. Padding may help boundary coverage or add false positives; neither outcome is assumed. It cannot recover masks already rejected before box generation.

**Re-evaluate H within the experimental evaluator.** This evaluator has some different boundary handling and metrics from the historical authors’ evaluator. The old 71.2001% score is context, not the numerical control to subtract from P5.

## 5. Commands: setup and CPU checks

Run these in a Bash terminal:

```bash
cd /home/jupyter-team2pop/Slide-SAM/Slide-SAM
conda activate slidesam
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/slidesam-padding-mpl

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  python -m unittest btcv_cross_validation.box_padding.test_cpu -v
```

Read [the padding README](../btcv_cross_validation/box_padding/README.md). If tests fail, resolve the specific failure in copied experimental code before inference. Do not modify original training files or weaken checks just to make tests pass.

## 6. Commands: preflight and GPU evaluation

Paste this function into the same Bash terminal. It uses the previously working native CUDA backend consistently across all arms without editing the evaluator. It fixes random seeds and disables cuDNN benchmarking. This wrapper does not guarantee cross-platform bit-exact determinism.

```bash
set -euo pipefail

run_padding_arm() {
  local mode="$1"
  local organ="$2"
  shift 2
  local fraction=0
  if [ "$mode" = image ]; then fraction=0.05; fi

  python -u -c '
import random, runpy
import numpy as np
import torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
runpy.run_module("btcv_cross_validation.box_padding.evaluate", run_name="__main__", alter_sys=True)
' \
    --data-root datasets \
    --manifest btcv_cross_validation/dataset_list/fold5/01_BTCV_test_4to1.txt \
    --train-manifest btcv_cross_validation/dataset_list/fold5/01_BTCV_train_4to1.txt \
    --checkpoint btcv_cross_validation/runs/fold5/ddp_sub_cv/cv_fold5/ckpt/model_epoch_20.pth \
    --checkpoint-kind finetuned \
    --label-idx "$organ" --prompt box --protocol gt-cropped \
    --padding-mode "$mode" --padding-fraction "$fraction" \
    --nsd-tolerance-mm 1.0 \
    --output "btcv_cross_validation/box_padding/results/fold5_e20_${mode}_label${organ}" \
    "$@"
}

# Check all 12 argument/path combinations without requesting inference.
for mode in historical none image; do
  for organ in 6 7 3 1; do
    run_padding_arm "$mode" "$organ"
  done
done
```

The preflight checks paths and manifest disjointness. It does not prove numerical correctness or GPU compatibility. The wrapper imports PyTorch, but no neural inference is requested in this first loop. Existing output directories are intentionally rejected.

After the CPU tests and preflights pass, check the GPU and launch the actual comparison:

```bash
nvidia-smi

mkdir -p btcv_cross_validation/box_padding/logs
padding_stamp=$(date +%Y%m%d_%H%M%S)

for mode in historical none image; do
  for organ in 6 7 3 1; do
    run_padding_arm "$mode" "$organ" --run-inference \
      2>&1 | tee "btcv_cross_validation/box_padding/logs/${padding_stamp}_${mode}_label${organ}.log"
  done
done
```

This is 12 runs, with six patients each: **72 patient-organ predictions**. Run sequentially. Runtime may exceed the previous Dice-only evaluations because this evaluator also computes surface metrics. If the GPU is occupied by another job, do not interfere with it.

If a run fails, retain its log and partial output. Diagnose the failure; do not delete or silently overwrite its directory. Agree on a new output name or request permission to archive/remove the failed output before restarting. Stop on failure rather than proceeding to comparisons with missing cases.

## 7. Required result report

After completion, create `KT/Box_Padding_First_Results.md`. This is a requested future output, not an existing report. Use raw `results.json` files in the 12 directories above and verify completed status and six cases in each.

Include:

| Organ | H Dice (%) | Z Dice (%) | P5 Dice (%) | P5−Z (pp) | Z−H (pp) | P5−H (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Liver | Pending | Pending | Pending | Pending | Pending | Pending |
| Stomach | Pending | Pending | Pending | Pending | Pending | Pending |
| Left kidney | Pending | Pending | Pending | Pending | Pending | Pending |
| Spleen | Pending | Pending | Pending | Pending | Pending | Pending |
| Four-organ mean | Pending | Pending | Pending | Pending | Pending | Pending |

Also report per-patient changes, secondary metrics, failures/empty predictions, runtime, GPU memory and checkpoint/source hashes. Keep undefined HD95 cases visible. Verify that initial prompts, patient order, checkpoint, geometry and all non-padding settings match across arms. Do not report terminal-slice or stopping improvements unless explicitly measured; those may need a subsequent instrumented analysis.

Interpretation should answer: does the fixed 5% intervention help this checkpoint under the declared protocol, which organs benefit or worsen, and how much comes from corrected box arithmetic? No gain is also a valid result. Do not claim the fine-tuning degradation is explained or that fixed padding is established publication novelty.

## 8. Second planned novelty: z-spacing handling

Only after reporting the padding experiment, prepare the spacing implementation and evaluation design. It is not currently wired into the padding commands above.

The proposed rule is: when through-plane spacing is **greater than 5.5 mm**, interpolate the image along that axis toward **3 mm**, segment, then restore predictions to the original physical grid. Images use linear interpolation; discrete labels use nearest-neighbor. Preserve origin, direction, spacing and initial prompt location in physical coordinates. Score against unchanged source reference labels on the declared evaluation grid.

Important constraints:

- 5.5 mm is our proposed experimental trigger, inspired by the authors’ spacing-degradation experiments. It is not an author-established corrective threshold.
- The local audit found no native BTCV or WORD scans above 5.5 mm. Turning on this rule would change none of them.
- Use a separate naturally thick-spacing cohort or clearly labeled synthetic degradation. Both comparison branches must receive the same degraded input; interpolated recovery cannot use the original fine-spacing image.
- Interpolation cannot restore missing anatomy. Changing z alone is through-plane normalization, not necessarily isotropic resampling.
- Keep variants of each patient together; they are not independent samples.

The intended factorial comparison remains:

| Cell | Corrected propagated-box padding | Conditional z-resampling |
| --- | --- | --- |
| B00 | p=0 | OFF |
| B10 | Image-relative p=0.05 | OFF |
| B01 | p=0 | ON |
| B11 | Image-relative p=0.05 | ON |

Retain H as a separate historical reference. On native scans where the threshold never triggers, ON/OFF equality is a no-op check, not evidence for a spacing contribution. See [z-spacing explanation](Z_Spacing_5_5mm_Explanation.md).

## 9. Instructions for the next CLI/agent

> Follow this novelty-first plan. Keep fold-5 epoch 20 fixed and compare historical, corrected-zero and 5% image-relative propagated-box padding using the same isolated evaluator on four organs. Do not require cache rebuilding, retraining, batch-size changes or further fine-tuning diagnosis before this evaluation. Preserve existing files and ask before deletion. Provide commands unless asked to launch them. Report the completed padding comparison before designing the later z-spacing experiment. Do not tune on the six test cases or promise gains.

The broader candidates—spacing-conditioned context, propagation-error recovery training, corrective prompting and overlapping-window agreement—remain in [the research proposal](SlideSAM_Research_Novelty_Proposal.md). They are deferred; do not expand this immediate experiment into those projects.
