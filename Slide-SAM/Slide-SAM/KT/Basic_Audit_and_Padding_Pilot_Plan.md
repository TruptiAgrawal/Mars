# Basic audit → padding pilot → five-fold evaluation

Audit date: 29 September 2026. Evidence snapshot: [Basic_Audit_2026-09-29.json](Basic_Audit_2026-09-29.json).

## 1. Does this affect fold 5 training?

The audit performed here does not change training behavior. It reads manifests, saved configurations, small logs/source files, checkpoint file metadata, and the headers of 30 image/label pairs. It also runs two isolated CPU checks on tiny synthetic inputs. It does not import the model, allocate GPU memory, deserialize checkpoints, read full image arrays, start evaluation, or change training files.

Only this report and the JSON evidence file were added under `KT/`. Hashes of the inspected source/configuration/manifest files were unchanged during the audit. These small reads have negligible resource demands compared with GPU evaluation; zero shared-system overhead cannot be guaranteed.

Leave training code, configurations, manifests, caches, and checkpoints unchanged while fold 5 runs. Defer GPU inference and large checkpoint reads until training has finished. A read-only audit is safe to do now; a full-volume pilot on the shared GPU should wait.

## 2. Structured plan

| Stage | Action | Output | Timing |
|---|---|---|---|
| A. Basic audit | Check splits, run identity, evaluation code, geometry headers, and basic helpers | Findings and evidence snapshot | Completed at the scope described below |
| B. Evaluation preparation | Preserve historical behavior; correct and verify the controlled evaluation path and box-coordinate helper | Versioned evaluator and explicit padding modes | After training; not implemented yet |
| C. Technical pilot | Use the old epoch-3 checkpoint; check shapes, coordinates, clipping, reproducibility, and output files | Implementation check, not a performance claim | After GPU is free |
| D. Optional development comparison | Compare historical behavior, corrected 0%, and prespecified 5% padding on old validation cases | Exploratory scores and masks | Disclose any parameter/method selection based on these cases |
| E. Freeze protocol | Lock initial prompts, checkpoint choice, padding definition, cropping, metrics, and thresholds | Written experiment configuration | Before final comparison |
| F. Five-fold comparison | Use each fold checkpoint only on that fold's held-out patients | Paired per-patient results and fold summaries | After fold 5 checkpoint completion is verified |

None of the inference-only padding stages requires retraining or modifying learned weights.

### Audit versus pilot

- **Audit:** establish whether the experiment is correctly defined and implemented.
- **Technical pilot:** establish whether the proposed implementation operates correctly.
- **Development comparison:** use scores or visual quality to choose a method or setting.
- **Final comparison:** evaluate the already frozen intervention, with development overlap disclosed.

Calling a comparison “just trying” does not preserve test independence if its outcomes influence the method. The old validation cases are also fold 5 test cases. A later audit cannot undo that exposure.

## 3. Completed checks and results

### 3.1 Fold manifests — PASS

| Fold | Training volumes | Held-out volumes | Within-fold overlap |
|---|---:|---:|---:|
| 1 | 24 | 6 | 0 |
| 2 | 24 | 6 | 0 |
| 3 | 24 | 6 | 0 |
| 4 | 24 | 6 | 0 |
| 5 | 24 | 6 | 0 |

All 30 on-disk image IDs are covered. Each is held out exactly once. Manifests have no duplicate IDs, each training set is the complement of its held-out set, and the corresponding image and label files exist. Historical validation is `img0035–0037`; historical test is `img0038–0040`. Together these are fold 5's held-out set. These checks establish manifest integrity, not absence of foundation-model pretraining overlap.

The existing `btcv_cross_validation/scripts/verify_folds.py` was inspected but deliberately not executed: despite its temporary-backup logic, it invokes the fold generator in place. The audit instead independently checked the existing manifests without regenerating them.

### 3.2 Image/label geometry headers — PASS within scope

Dimensions, spacing, spatial-unit codes, and qform/sform transform fields match between the 30 image/label pairs. Header matching does not verify segmentation content, image-label anatomical alignment, or cache correctness.

| Native header z-spacing | Number of volumes |
|---|---:|
| 2.5 mm | 1 |
| 3.0 mm | 17 |
| 5.0 mm | 12 |
| Greater than 5.5 mm | 0 |

**Decision:** the currently proposed `z > 5.5 mm` resampling rule changes none of these native BTCV scans. It cannot produce a distinct resampling treatment on this cohort. Keep padding first. Later use an explicitly declared synthetic spacing experiment or suitable external scans; do not silently lower the threshold to manufacture an effect. Mapping the NIfTI physical axes through the full loader still needs verification before a resampling implementation.

### 3.3 Run configuration and checkpoint identity — PARTIAL

The saved run configurations inspected for folds 1–5 all record batch size **4**, `num_epochs=21`, learning rate `2e-5`, weight decay `1e-4`, and initialization from `weights/slidesam_b.pth`. Earlier prose describing CV batch size 2 should not override these saved configurations. Save intervals differ: folds 1/2 use 1, folds 3/4/5 use 5.

Fold 1–4 epoch-20 checkpoint files are present, and the main training logs record epoch-20 saves. The audit checked file existence and sizes, not tensor contents or successful model loading.

At the snapshot, the fold 5 progress log last recorded **epoch 16, batch 1515/2039, iteration 32100**. The progressing run folder is `btcv_cross_validation/runs/fold5/ddp_sub_cv/cv_Fold5/`, whose run log records epoch 15 completed and saved. A separate lowercase `cv_fold5/` folder contains an earlier run that only logs completion through epoch 1. Select the exact final checkpoint path deliberately; do not select by a loosely matching folder name.

This is log evidence, not a live-process guarantee or completion-time estimate. The sandbox's process list only exposes its own short-lived processes, so absence of a training process there is not evidence that training stopped. No process was signaled.

### 3.4 Basic Dice arithmetic — PASS for nonempty targets

The isolated `trans_utils/metrics.py::compute_dice_np` function returns 1 for identical masks, 0 for disjoint masks, and 0 for an empty prediction with nonempty ground truth. It raises `AssertionError` when ground truth is empty, including both-empty masks. Define and test the absent-organ policy before evaluating datasets where this occurs.

## 4. Evaluation findings requiring action or explicit reporting

### A. Current box expansion is incorrect — confirmed with a synthetic example

`datasets/data_engine.py::BoxPromptGenerator.enlarge()` maps `(100, 200, 300, 300)` to `(90, 190, 295, 295)` with 1024×1024 bounds. Both upper edges move inward, and the lower y edge uses the horizontal margin. Passing `margin=0` or `margin=50` gives the same output because that argument is unused.

The propagation code calls this helper. Therefore the unchanged implementation is **historical H**, not a genuine 0%-padding reference. Build a separate consistent helper for controlled 0%/5% comparisons. The proposed primary 5% is image-relative per side; do not confuse it with the existing helper's box-relative arithmetic.

### B. NSD and point evaluation need correction

In `test/volume_eval_cv.py`, box-mode NSD is explicitly set to zero. That value is not a measured surface score. Point mode calls `compute_surface_dice` without defining or importing it in that module. Validate metric routines and physical spacing order on synthetic shapes before reporting NSD/HD95 or running the point track.

### C. Evaluation uses ground-truth-assisted cropping and initial-slice selection

`datasets/eval_dataloader/loader_abstract.py::_get_data()` finds the target's first and last nonempty slices from ground truth, crops both image and label to that range, and chooses the maximum-area labeled slice for the initial prompt. The evaluator derives a tight box from the target label on that slice.

This is an oracle-assisted, organ-extent-cropped evaluation protocol. Record it explicitly when describing existing Dice. It excludes opportunities for false-positive continuation outside the target's true extent; results from it cannot establish full-volume termination performance.

Preserve the historical protocol for comparison with existing results. If adding full-volume evaluation with an explicit initial-prompt policy, establish a new baseline under that same protocol before attributing any change to padding. Keep protocol changes separate from intervention effects.

### D. Slice-boundary safeguards contain errors

- The loader uses `template_slice_id == label.shape[0] // 2` where an assignment appears intended; this expression does not update the index.
- The evaluator tests the upper slice boundary using `data['img'].shape[0] - 1`, which refers to the batch dimension rather than slice depth for its batched input.
- The predictor selects the slice axis using the smallest array dimension, not physical orientation. This needs explicit validation before extending to differently shaped/resampled volumes.

These are source findings. The audit has not established which historical cases triggered them or quantified their effect on Dice.

### E. Normalization and masks need follow-up checks

The loader selects the CT/MRI clipping branch using the image minimum and the predictor min-max normalizes each three-slice window. The latter has no explicit constant-window denominator guard. Preserve normalization in the reference and test its edge cases separately.

Training cache labels use JPEG and are binarized at greater than zero. A comparison with source labels is still needed to assess whether compression changed label boundaries; that check was deferred to avoid scanning the cache during training.

### F. Model-loading path — inspected, not executed

The fold evaluator builds rank-8 then rank-1 LoRA and loads the checkpoint using `load_state_dict`. This matches the intended nested-adapter path. Full checkpoint integrity, tensor compatibility, and GPU inference still require verification after training. The official pretrained checkpoint requires its own correct load sequence; do not treat it as an interchangeable nested-adapter checkpoint.

## 5. Exact pilot scope and data-use decision

Use `../runs/sam/ddp_sub/btcv_liver_run2/ckpt/model_epoch_3.pth` with the historical validation manifest containing `img0035–0037`. The previous inspection established the checkpoint is readable and has nested adapters; repeat the model compatibility check before inference. The run has a debug epoch 0 and three complete training epochs, 1–3. The short run is adequate for engineering checks but does not establish performance transfer to longer runs.

The fold-specific loader always reads the six-case `01_BTCV_test_4to1.txt`; it must not silently substitute for the historical three-case validation loader. Explicitly verify the three patient IDs printed by the pilot.

First test coordinates and output behavior without using quality scores to select parameters. If the team then compares masks/Dice to decide whether to retain or change padding, label those cases development data. Their later fold 5 results are exploratory/development-overlap results, even if the whole CV table is retained for transparency. Fresh external confirmation is preferable for stronger claims, and outcomes already examined in other folds also need disclosure if they guide development.

Keep the historical three-case test set separate from tuning. Do not reuse the five fold models as an ensemble on the same 30 patients: most component models trained on each patient.

## 6. Ready now versus pending

**Completed now:** the structured plan, read-only manifest checks, configuration/checkpoint-file inventory, geometry-header audit, isolated box-helper demonstration, isolated Dice checks, and static evaluation review.

**Pending:** evaluation fixes in a separately versioned path, complete model-loading verification, cache-content/label audit, physical-axis verification through the loader, GPU pilot, final fold 5 completion check, and paired five-fold padding evaluation.

**Decision:** proceed with evaluation preparation after fold 5 finishes, then the padding pilot. Do not call the entire evaluation pipeline validated yet. The basic audit found concrete issues to resolve and a native-spacing result that makes the proposed >5.5 mm resampling intervention inactive on BTCV.
