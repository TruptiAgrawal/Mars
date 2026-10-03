> **Current priority:** Follow [Novelty-First Next Steps](Novelty_First_Next_Steps_2026-10-01.md). Box-padding evaluation comes first; cache rebuilding and further training are deferred, not prerequisites. This document is retained as background evidence.

# Early Fine-Tuning Degradation Audit

**Date:** 2026-10-01  
**Scope:** Read-only code + log audit for Stage A of the post-5-fold roadmap.  
**Question:** Why does BTCV fine-tuning reduce performance relative to the official pretrained model?

---

## 1. Observed degradation

| Checkpoint | Mean Dice (4 organs, 6 test patients) |
|---|---:|
| Official pretrained | 79.12% |
| BS=4 fold 5 epoch 0 (debug path) | 79.12% |
| BS=4 fold 5 epoch 5 | 70.30% |
| BS=4 fold 5 epoch 10 | 70.26% |
| BS=4 fold 5 epoch 15 | 70.68% |
| BS=4 fold 5 epoch 20 | 71.20% |

The drop is **8.82 pp from pretrained to epoch 5**, achieved in 10,196 gradient updates (1 debug-path update at epoch 0 + 5 × 2,039 full-epoch updates). Later epochs partially recover to −7.92 pp at epoch 20. The loss falls monotonically (epoch 1: 0.5412, epoch 20: 0.4019), so lower training loss coexists with worse held-out Dice throughout.

Source: `btcv_cross_validation/logs/checkpoint_audits/checkpoint_trajectory.report.md`, `fold5_analysis/fold5_epoch_losses.csv`.

---

## 2. Verified implementation facts

The following are confirmed by reading source code and logs. Each is a **fact**, not a hypothesis.

### 2.1 Trainable parameter scope: rank-1 LoRA only, 48 tensors

`core/learner_sub1.py → use_lora_sub()`:
```python
def use_lora_sub(self):
    lora_r = 1
    lora_sam = LoRA_Sam(self.model, lora_r, freeze_all=True)
    self.lora_module = lora_sam
```

`core/lora_sam.py → LoRA_Sam.__init__()` with `freeze_all=True`:
```python
if freeze_all:
    for param in sam_model.parameters():
        param.requires_grad = False
```
Then only the newly injected rank-1 `linear_a` / `linear_b` layers are `nn.Linear` modules with `requires_grad=True` (they are fresh parameters, not touched by the freeze loop).

**Result:** 48 tensors totaling 36,864 values are trainable. All other parameters — including the existing rank-8 LoRA adapters from pretraining — are frozen. Confirmed by `fold5_analysis/delta_pack_verification.log`: 422 tensors per checkpoint, 48 LoRA tensors, 376.9 MB frozen.

The rank-1 adapters are added on top of the already-LoRA-modified `_LoRA_qkv` modules, creating a nested structure. The rank-8 weights contribute to the forward pass throughout but receive no gradients.

### 2.2 No learning rate scheduler

`core/learner2.py → configure_optimizers()`:
```python
# scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule)
scheduler = None
return {'optimizer': optimizer, "scheduler": scheduler}
```

`trans_utils/trainer_ddp.py → train()`:
```python
if scheduler is not None:
    scheduler.step()
```

The scheduler call is guarded, so `scheduler=None` silently means no scheduling. **LR is fixed at 2e-5 for all 20 epochs.** Confirmed in training log: `lr:2.000000e-05` on every epoch line.

The paper uses AdamW with LR 2e-4 and weight decay 0.1. This fine-tuning uses LR 2e-5 (10× lower) and weight decay 1e-4 (1000× lower). The lower LR means updates are small but persistent; the lower weight decay means there is almost no regularisation pulling parameters back toward zero.

### 2.3 Both prompt types computed unconditionally, gradients summed

`core/learner3.py → training_step()` (used via `learner5 → learner_sub1`):
```python
# Point forward pass — always executed
pred_masks, iou_predictions, logits = self.predict_torch(
    point_coords=prompt_point, point_labels=point_label, ...)
loss_1, ... = ranked_combined_loss_with_indicators(...)

# Box forward pass — always executed
pred_masks, iou_predictions, logits = self.predict_torch(
    point_coords=None, boxes=prompt_box, ...)
loss_2, ... = ranked_combined_loss_with_indicators(...)

loss = loss_1 + loss_2
return {"loss": loss_1 + loss_2, ...}
```

Contrast with `core/learner2.py → training_step()`, which uses:
```python
if np.random.random() > 0.5:
    # point path
else:
    # box path
```

**The learner3/5/sub1 path runs both forward passes and sums both losses every step.** Compared to learner2's 50/50 random choice, each optimizer step here accumulates gradients from two loss terms rather than one. This increases the total gradient per step, but does not necessarily double it: `loss_1` (point) and `loss_2` (box) are computed from different prompt modes and may have different scales. The exact per-parameter gradient magnitude depends on those scales. What is established is that the update uses more information per step, and that the rank-1 adapters receive gradient signal from both prompt modes simultaneously on every batch — unlike learner2, where each batch trains on only one mode.

### 2.4 Loss function: ranked_combined_loss_with_indicators — indicators are all-ones

`core/loss.py`:
```python
def ranked_combined_loss_with_indicators(pred_mask, gt_mask, iou_pred, indicators):
    indicators = repeat(indicators, "b d -> b c d h w", ...)
    pred_mask = pred_mask * indicators
    gt_mask   = gt_mask   * indicators
    return ranked_combined_loss(pred_mask, gt_mask, iou_pred)
```

The indicators mechanism exists to zero out slices where the organ is absent. However, `cache_dataset3d3.py → __getitem__()` always sets:
```python
vector = np.ones(3)
ret_dict['indicators'] = vector
```

With all-ones indicators, multiplying pred_mask and gt_mask by indicators is a no-op — **no slices are masked out**. Absent-organ slices (where gt_mask is zero) contribute zero Dice loss naturally because the numerator is zero, but the focal loss still receives signal from those slices' predictions against an all-zero target. The indicators mechanism as coded here does not provide the absent-organ exclusion described in older notes. The `ranked_combined_loss` then selects the best-ranked prediction (among 3 outputs) and adds an IoU-regression term. The 20× focal weighting is applied before ranking.

### 2.5 Prompt generated from middle slice only

`datasets/data_engine.py → DataEngine.get_prompt()`:
```python
gt_mask = mask[1, :, :]   # always the middle of the 3-slice stack
```

The prompt point and prompt box are derived solely from slice index 1 (the middle slice of the cached 3-slice window). The model is then asked to predict all 3 slices. This is intentional for the sliding-window design, but it means the prompt never directly represents slices 0 or 2.

### 2.6 JPEG-compressed label cache — lossy label storage

`cache_dataset3d_cv.py` inherits `cache_dataset3d3.Dataset3D`, which reads from:
- `image/*.jpg` — image slices (JPEG, lossy)
- `label_jpg/*.jpg` — **label/segmentation masks** (JPEG, lossy)

JPEG is a lossy format. Storing binary or multi-class integer masks as JPEG introduces compression artefacts: boundary pixels may be altered, thin structures may disappear, and label values near the boundary are quantised. **This affects the ground-truth masks used for loss computation, not just the images.** The extent of the artefact depends on JPEG quality settings used when the cache was built; this was not checked for the current audit.

### 2.7 BoxPromptGenerator.enlarge() bug

`datasets/data_engine.py`:
```python
def enlarge(self, bbox, margin=0):
    x0, y0, x1, y1 = bbox
    margin_x = int((x1 - x0) * 0.05)
    margin_y = int((y1 - y0) * 0.05)
    x0 = max(x0 - margin_x, 0)
    y0 = max(y0 - margin_x, 0)   # BUG: uses margin_x, not margin_y
    x1 = min(x1 - margin_y, self.size[0]-1)   # BUG: subtracts (shrinks), should add
    y1 = min(y1 - margin_y, self.size[1]-1)   # BUG: subtracts (shrinks), should add
```

Three bugs:
1. `y0` uses `margin_x` instead of `margin_y`.
2. `x1` and `y1` are **decreased** by the margin instead of increased — the upper-right corner is shrunk.
3. The `margin` parameter passed in is ignored entirely.

**Effect:** `enlarge()` shrinks the upper-right of the box by ~5% of box dimensions while expanding only the lower-left corner. Called with `margin=0` in training, so this shrinkage is present during training. However, `enlarge()` is not called in the main `get_prompt_box()` training path — that calls `mask_to_bbox` + `add_random_noise_to_bbox`. The bug affects only callers that explicitly call `enlarge()`. Verify whether the evaluator or propagation code calls `enlarge()`.

### 2.8 Epoch 0 is a debug-path checkpoint, not clean initialization

`trans_utils/trainer_ddp.py → train()`:
```python
if epoch == 0:
    if self.logging_available:
        self.logger.info("[*] Debug Checking Pipeline !!!")
    del out
    return   # exits training loop after first batch
```

After epoch 0's single-batch run, `save()` is called immediately:
```
[2026-09-29 08:09:21] [*] Debug Checking Pipeline !!!
[2026-09-29 08:09:30] Epoch 0: Save checkpoint to /ckpt_v/model_latest.pth
[2026-09-29 08:09:34] Epoch 0: Save model to /ckpt/model_epoch_0.pth!
[2026-09-29 08:11:38] Iteration 0: Saved model_iter_0.pth and optim_iter_0.pth
```

So `model_epoch_0.pth` contains the state after **one mini-batch of gradient updates**, not untouched initialisation. The trajectory report shows it matches the official checkpoint scores at displayed precision (max abs Dice difference ~1.24e-6), meaning one batch of 4 samples caused no measurable change — consistent with the very small gradient magnitude for rank-1 LoRA with LR 2e-5.

### 2.9 No validation set

`core/ddp_sub_cv.py → ddp_train()`:
```python
ddp_trainer.fit(learner, trainset=data_engine, validset=None)
```

There is no per-epoch inner validation. The only feedback during training is training loss. Stopping and checkpoint selection were done at epoch 20 by design, not by any validation criterion. All trajectory evaluations were post-hoc on test patients.

### 2.10 Training dataset: 24 volumes, 8,159 cached samples, fold-5 split

From the training log:
```
Training volumes        : 24
Held-out volumes        : 6
Cached training samples : 8159
```

At BS=4: 2,039 batches per epoch. With no gradient accumulation and one GPU:
- Epoch 5: ~10,195 gradient updates from initialisation (5 × 2,039)
- Degradation is already maximal at epoch 5 and partially recovers by epoch 20.

---

## 3. Hypotheses (not yet confirmed)

The following are plausible explanations given the facts above. Evidence would be needed to confirm any of them.

### H1: The rank-1 adapters distort the pretrained representation

The pretrained rank-8 LoRA was optimised over thousands of volumes across multiple organ types. The new rank-1 adapters modify the same Q and V projections with BTCV-only gradients, on only 24 volumes. With no regularisation (weight_decay = 1e-4, essentially zero), the rank-1 adapters can push the activations in directions that reduce BTCV training loss while moving away from the broader generalisation encoded by the rank-8 weights.

Supporting pattern: the LoRA weight-change L2 norm grows monotonically from 3.87 at epoch 5 to 6.83 at epoch 20 (from `fold5_lora_weight_change.csv`). The per-epoch step size shrinks (3.87 → 1.45 → 1.31 → 1.21) and cosine similarity between consecutive steps increases (0.60 → 0.84 → 0.90), suggesting the optimiser is settling but still moving. The partial test-Dice recovery from epoch 5 to 20 (+0.90 pp) is consistent with settling.

Block-level breakdown (`fold5_lora_dW_by_block.csv`): blocks 11, 10, 9, 2 account for the largest share of total change; block 7 the least. V adapters change more than Q adapters (61% vs 39% of final L2 norm). The later ViT blocks (10, 11) and early blocks (0, 2) move most — these correspond to high-level semantic and low-level feature processing in ViT-B.

**This is plausible but not proven.** A controlled ablation (freeze rank-1, evaluate) would check whether the rank-1 adapter values alone cause degradation or whether the interaction with frozen rank-8 weights is key.

### H2: The dual-loss training signal is miscalibrated for the pretrained model

The pretrained model was fine-tuned with a specific training recipe (paper: LR 2e-4, weight decay 0.1, 20 epochs, 4 GPUs, thousands of volumes). This CV fine-tuning uses 10× lower LR, 1000× lower weight decay, runs both prompt types every step (rather than 50/50), and has only 24 training volumes. The gradient signal may be systematically biased toward the 24-volume BTCV training set, pushing representations away from the pretrained generalisation.

The monotonically falling training loss alongside falling held-out Dice is consistent with overfitting to the training distribution. However, the dataset is small (8,159 samples, 24 volumes), and there is no inner validation to detect this during training.

**This is the most likely macro explanation** but is a combination of the following sub-factors, not a single cause.

### H3: JPEG label compression introduces systematic boundary noise

Binary segmentation masks stored as JPEG may have artefact pixels at boundaries. Over 8,159 samples × 20 epochs, the model receives noisy gradient signals at organ boundaries. This could cause the model to learn boundary artefact patterns rather than true anatomy. It would also make the focal loss respond to JPEG-artefact pixels as if they were true foreground.

**Requires inspection:** check JPEG quality setting used in caching step, and compare a decoded label against the original NIfTI label for a few samples. If quality was set ≥90, the artefacts may be minimal; if ≤75, they could be significant.

### H4: Prompt mismatch between training and evaluation

During training, the prompt is generated from `mask[1]` (middle slice of the window), using the training cache's JPEG-loaded label. During evaluation, the authors' evaluator generates the initial box from the maximum-area foreground slice in the full 3D volume, using the original NIfTI label without JPEG degradation. These are different prompt generation strategies. The model may be tuned to handle prompts derived from JPEG-degraded middle slices, not clean ground-truth bounding boxes on the best slice.

Additionally, `BoxPromptGenerator.enlarge()` has the shrinkage bug (Section 2.7). If `enlarge()` is called anywhere in the training path, training prompts are systematically smaller than intended, creating a mismatch with evaluation prompts.

### H5: The capacity of rank-1 adapters is insufficient to positively adapt

Rank-1 LoRA adds a rank-1 outer product to Q and V in each block: 768 × 1 × 1 × 768 = a single direction per projection. With 12 blocks and Q+V, this is 24 rank-1 matrices. The representational capacity is extremely low. On a small dataset, this may be enough to overfit the training loss without providing a generalisation benefit — but not enough to actually improve the generalisation of the rank-8 pretrained representation.

**Note:** this is a design question, not a bug. Whether rank-1 capacity is sufficient depends on what the BTCV data provides that the pretrained model lacked. This is an open question.

---

## 4. What cannot be concluded from the current evidence

- **Overfitting is not proven.** Lower training loss + lower test Dice is consistent with overfitting, but also with distribution shift, prompt mismatch, or corrupted labels. A genuine overfitting diagnosis requires a held-out set from the same distribution that was never seen during training.
- **The partial recovery (epochs 5→20) is not proven to be meaningful.** It is within the noise of the evaluation and does not have a confidence interval.
- **The LR being 10× below the paper's value is not proven to cause the degradation.** Lower LR means slower change; if the rank-1 adapters are harmful, slower is not necessarily better.
- **The dual-loss (both prompts every step) is not proven to be the primary cause.** It increases the gradient per step relative to 50/50 random selection, but the exact magnitude depends on the scales of the two loss terms. Whether this increase is harmful is not established.

---

## 5. Summary of verified problems

| # | Problem | Source | Severity |
|---|---|---|---|
| P1 | No LR scheduler — LR frozen at 2e-5 for 20 epochs | `learner2.py:configure_optimizers` | Medium — missed warmup/decay |
| P2 | No validation set during training | `ddp_sub_cv.py:ddp_train` | High — no stopping criterion |
| P3 | Only 48 rank-1 trainable params (36,864 values) | `learner_sub1.py:use_lora_sub`, `lora_sam.py` | Design decision — may be insufficient capacity |
| P4 | Both prompt types run every step, losses summed | `learner3.py:training_step` | Medium — larger per-step gradient than 50/50; exact factor depends on loss scales |
| P5 | JPEG-compressed label masks in cache | `cache_dataset3d3` (base of `cache_dataset3d_cv`) | Unknown — depends on JPEG quality |
| P6 | Prompt generated only from middle slice | `data_engine.py:get_prompt` | Low-medium — intentional design, but mismatch with eval |
| P7 | `BoxPromptGenerator.enlarge()` has shrinkage bug | `data_engine.py` | Low for training (not called in training path) |
| P8 | Epoch 0 checkpoint has 1 gradient update | `trainer_ddp.py` (epoch==0 early return) | Cosmetic — confirmed ≈ identical to pretrained |
| P9 | Weight decay 1e-4 vs paper's 0.1 | `fold5.yaml` | Medium — nearly zero regularisation |
| P10 | LR 2e-5 vs paper's 2e-4 | `fold5.yaml` | Medium — 10× slower update rate |

---

## 6. Proposed pilot experiment

### Pilot goal

Establish whether the degradation is primarily due to the training setup (P1–P4, P9–P10) or the data quality (P5–P6), by running a minimal controlled experiment with a corrected training setup. This is a **development pilot**, not a publication result.

### Pilot design constraints (required by the roadmap)

1. Reserve 4 patients from the 24 training patients as an **inner validation set**, disjoint from the 6 held-out test patients. All slices from a patient must stay together (no patient-level leakage).
2. Train on the remaining 20 patients only.
3. Evaluate on inner validation Dice (not test patients) to make stopping and hyperparameter decisions.
4. Keep the official pretrained checkpoint as the untouched reference in every comparison.
5. Change **one factor at a time** across cells.

### Proposed pilot cells

| Cell | Change from baseline | Rationale |
|---|---|---|
| **Baseline** (reproduce fold 5) | No change — rank-1 LoRA, LR 2e-5, WD 1e-4, both prompts | Confirm reproduced result on inner-val |
| **C1** | Add cosine LR schedule (warmup 2 epochs, decay to 1e-6) | Tests P1: whether scheduling matters |
| **C2** | Switch to 50/50 prompt (learner2-style, not learner3) | Tests P4: whether dual-loss doubles gradient too much |
| **C3** | Increase weight decay to 0.01 (between 1e-4 and paper's 0.1) | Tests P9: whether regularisation helps |
| **C4** | LR 5e-5 (midpoint on log scale between 2e-5 and 2e-4) | Tests P10: whether LR is too low |

Run each for 20 epochs. Use inner-validation Dice at each epoch-5 checkpoint to pick the best-performing cell. Do not use test patients to select among cells.

### What to check before launching

1. **Inspect JPEG label quality** (P5): load one cached label `.jpg` and compare it to the original NIfTI label for the same slice. If artefacts are visible at organ boundaries, the cache should be rebuilt with lossless PNG or NPY before any cell training. This is a prerequisite that takes ~30 min to inspect and cannot be deferred.

2. **Verify `enlarge()` call sites** (P7): confirm whether the evaluator or any training code calls `enlarge()`. If the training box prompt uses `enlarge()`, the shrinkage bug should be corrected as a baseline fix before any cell comparison.

3. **Confirm inner-val patient split**: the 24 training patients for fold 5 are `img0001–img0010, img0021–img0034`. A reasonable inner validation set: `img0005, img0010, img0025, img0030` (4 patients, evenly spaced). All must be removed from the training manifest before training.

### Exact commands (to be reviewed before user launches)

```bash
# Step 1: Inspect JPEG label integrity (read-only, ~5 min)
conda activate slidesam
cd /home/jupyter-team2pop/Slide-SAM/Slide-SAM
python - <<'EOF'
import numpy as np
from PIL import Image
import nibabel as nib, glob, os

# Load one cached JPEG label
label_paths = sorted(glob.glob(
    "btcv_cross_validation/cache/all30/01_BCV-Abdomen/label_jpg/label_0000_00000025_*.jpg"
))
for lp in label_paths[:2]:
    img = np.array(Image.open(lp))
    print(f"{os.path.basename(lp)}: dtype={img.dtype}, unique={np.unique(img)[:10]}, "
          f"min={img.min()}, max={img.max()}")
EOF
```

If the label shows non-integer values or unexpected unique values (anything other than 0, 1, or the organ label integers), the cache has JPEG artefacts and should be rebuilt before any training pilot.

```bash
# Step 2: Create inner-val manifests (read-only dry run first)
# Inner-val patients: img0005, img0010, img0025, img0030
# (adjust if these patients have unusually few slices — check sample_counts in cache audit)
grep -E "img0005|img0010|img0025|img0030" \
  btcv_cross_validation/dataset_list/fold5/01_BTCV_train_4to1.txt
```

```bash
# Step 3: Baseline pilot — confirm reproduced degradation on inner-val
# Copy config first, then modify:
cp btcv_cross_validation/configs/fold5.yaml \
   btcv_cross_validation/configs/pilot_baseline.yaml
# Edit pilot_baseline.yaml:
#   training.breakpoint_path: weights/slidesam_b.pth  (unchanged)
#   training.lr: 0.00002   (unchanged)
#   training.weight_decay: 0.0001  (unchanged)
#   training.num_epochs: 21  (unchanged)
#   training.batch_size: 4  (unchanged)
#   dataset.data_txt_path: btcv_cross_validation/dataset_list/pilot_baseline/
#     (new folder with 20-patient train + 4-patient val manifests)
#   base.base_dir: btcv_cross_validation/runs/pilot/
#   base.tag: ''
```

Before launching: confirm GPU access with `nvidia-smi`, record the sha256 of `weights/slidesam_b.pth`, and record source file hashes for `core/ddp_sub_cv.py` and `core/learner_sub1.py`.

The user should approve the inner-val split and the JPEG inspection results before any cell training runs.

---

## 7. Evidence provenance

| Claim | Source file(s) |
|---|---|
| 48 trainable tensors, 36,864 values | `fold5_analysis/delta_pack_verification.log` |
| LR frozen at 2e-5 | `fold5_analysis/fold5_epoch_losses.csv` (lr column), `core/learner2.py` |
| Scheduler = None | `core/learner2.py:configure_optimizers` |
| Dual-loss per step | `core/learner3.py:training_step` |
| validset=None | `core/ddp_sub_cv.py:ddp_train` |
| Epoch 0 debug path | `trans_utils/trainer_ddp.py:train` (epoch==0 early return), `fold5_analysis/recovered_original_iteration_lines.log` |
| JPEG label cache | `datasets/cache_dataset3d_cv.py` → `cache_dataset3d3` base class |
| Prompt from mask[1] | `datasets/data_engine.py:DataEngine.get_prompt` |
| enlarge() bug | `datasets/data_engine.py:BoxPromptGenerator.enlarge` |
| Trajectory Dice scores | `btcv_cross_validation/logs/checkpoint_audits/checkpoint_trajectory.report.md` |
| Weight change L2 norms | `fold5_analysis/fold5_lora_weight_change.csv`, `fold5_lora_dW_by_block.csv` |
| Training log timestamps | `fold5_analysis/recovered_original_iteration_lines.log` |
| Fold 5 config (LR, WD, epochs) | `btcv_cross_validation/configs/fold5.yaml` |

---

## 8. JPEG label inspection — executed 2026-10-01

**Status: JPEG artefacts confirmed as significant. Prerequisite inspection is complete.**

### 8.1 How labels are stored and read

From `datasets/cache_dataset3d.py → save_slice_mask()`:
```python
labeli = masks_data[i].astype(np.uint8) * 255   # binary 0/1 → 0/255
cv2.imwrite(path, rearrange(labeli, "c h w -> h w c"))  # saved as JPEG, no quality param
```

From `datasets/cache_dataset3d3.py → _get_cached_data()`:
```python
mask = np.int32(cv2.imread(mask_path) > 0)   # reads back, thresholds > 0
```

Labels are written as **0 or 255** (binary mask scaled to full uint8 range). JPEG compression creates intermediate values (1–254). On read, **any value > 0 is treated as foreground = 1**. This means:
- JPEG artefact pixels that bleed into the background become **false positive foreground**.
- JPEG artefact pixels that corrupt a true-foreground 255 region down to 0 become **false negative** (undetectable from outside).
- No explicit JPEG quality parameter is passed to `cv2.imwrite`, so OpenCV's default quality of 95 is used.

### 8.2 Measured artefact severity

Sampled 202 files across the 9,897-file cache:

| Statistic | Value |
|---|---|
| Files with artefact pixels | 202 / 202 (100%) |
| Total pixels sampled | 211,812,352 |
| Artefact pixels (values 1–254) | 4,112,447 (1.94%) |
| Unique values per file (mean) | 146.1 |
| Unique values per file (range) | 56 – 193 |

A clean binary mask should have exactly **2 unique values** (0 and 255). All sampled files have 56–193 unique values, confirming pervasive JPEG artefacts.

### 8.3 Artefact severity relative to foreground size

On an example label file with 2,678 true foreground pixels (255) and 2,817 artefact pixels (1–254):

| Region | Pixel count |
|---|---|
| Definite foreground (255) | 2,678 |
| Definite background (0) | 1,043,081 |
| Ambiguous artefact (1–254) | 2,817 |
| Artefact within 5px of foreground edge | 1,794 |
| Artefact further than 5px from edge | 1,023 |

The ratio of ambiguous artefact pixels to true foreground pixels is ~105% for this slice. **This ratio does not establish a 105% false-positive rate.** Artefact pixels (1–254) fall into three categories that cannot be distinguished without the original NIfTI label:
- **(a) False positive:** JPEG bleed into true background — under `> 0` they become foreground when they should be background.
- **(b) Harmless:** JPEG bleed from true 255 to a lower non-zero value — still treated as foreground under `> 0`, no change in outcome.
- **(c) False negative:** JPEG bleed from true 255 all the way down to 0 — becomes background when it should be foreground. Undetectable from the JPEG alone.

A correct FP/FN estimate requires comparing the decoded JPEG mask against the nearest-neighbor-resized NIfTI label for the same slice. That comparison is the verification step required before approving the lossless rebuild (see Section 8.5). What is confirmed: artefact pixels are **pervasive** (100% of sampled files), **concentrated at organ edges** (64% within 5px), and **comparable in count to the true foreground** for small organs — sufficient justification to rebuild the cache losslessly regardless of the exact FP/FN split.

### 8.4 Conclusion

The JPEG label artefact is a **confirmed training data quality problem**. It is not the only contributor to degradation (see Sections 2–4), but it provides a persistent, boundary-concentrated noisy gradient throughout training. This is independent of the capacity, LR, or scheduling issues.

**Recommended action before any training pilot:**

Rebuild the label cache using lossless storage. The simplest fix within the existing code structure:

Option A — **Save labels as `.npy` (lossless, exact)**:
- Change `save_slice_mask()` to `np.save(path + '.npy', labeli)` instead of `cv2.imwrite(...jpg)`.
- Change `_get_cached_data()` to `mask = np.load(path).astype(np.int32)` instead of `cv2.imread(...) > 0`.
- This avoids all artefacts at the cost of ~3–4× larger disk usage per label file.

Option B — **Save labels as `.png` (lossless, compact)**:
- Change `cv2.imwrite(path.replace('.jpg','.png'), ...)` — PNG is lossless by default.
- Requires renaming paths in the read path too.

Option C — **Raise JPEG quality to 100** (near-lossless but not exact):
- Pass `[cv2.IMWRITE_JPEG_QUALITY, 100]` to `cv2.imwrite`. Still introduces minor artefacts.
- Not recommended for binary masks; lossless is strictly better.

**The rebuilt cache must use a different directory** (e.g. `btcv_cross_validation/cache/all30_lossless/`) to preserve the existing cache and all completed evaluations that depend on it. Do not overwrite the existing cache.

The existing evaluations used the authors' code which reads labels from the original NIfTI files (not the JPEG cache), so the completed Dice scores are unaffected. The artefact only affects the training loss computation.

### 8.5 Updated prerequisite checklist before pilot training

1. ✅ **JPEG inspection done** (this section) — artefacts confirmed, severity quantified.
2. ⬜ **Rebuild label cache as lossless** — new directory, do not overwrite existing cache.
3. ⬜ **Update `cache_dataset3d_cv.py`** to point to the new lossless cache (or parameterise it).
4. ⬜ **Verify lossless cache** — load one label and confirm unique values = {0, 255} only.
5. ⬜ **Create inner-val manifests** — hold out 4 patients from the 24 training patients.
6. ⬜ **Confirm `enlarge()` is not in the training prompt path** — check callers.
7. ⬜ **User approves pilot design** before any training launches.

### 8.6 Cache rebuild — deferred (2026-10-01)

**Decision:** cache rebuild is parked until a training pilot is actually planned. It is not needed for the box padding experiment (Stage D / `Novelty_First_Next_Steps_2026-10-01.md`), which reads original NIfTI labels and does not touch the training cache.

**When to revisit:** before running any training cell in the pilot design (Section 6). At that point, implementation plan is:
- Use **PNG** (lossless, ~0.5× JPEG size): estimated ~530 MB total vs 667 MB current.
- Or **NPZ compressed** (~0.3× JPEG size): estimated ~493 MB total.
- New cache directory: `btcv_cross_validation/cache/all30_lossless/`
- Rebuild from **original NIfTI labels** (not existing JPEG masks).
- New isolated loader modules (copied, not modifying existing files).
- Verify exact mask contents against NIfTI-derived nearest-neighbor-resized labels, not just `{0,255}` check.
- Update all path discovery patterns (`label_jpg/*.jpg` → `label_png/*.png` or similar) in both base and CV loaders.
- Keep existing `all30/` cache and all completed results untouched.
