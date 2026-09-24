# Slide-SAM → MARS Integration Roadmap

Date: 24 Sep 2026

This report plans how Slide-SAM becomes the real model behind the MARS Segmentation Agent. It draws on:
- the Slide-SAM working copy at `~/Slide-SAM/Slide-SAM`, including its `KT/` progress notes and `test/volume_eval_cv.py`;
- the MARS mock pipeline in `backend/mars/`.

## 1. Current state

### Slide-SAM
- The working copy is `~/Slide-SAM/Slide-SAM`. The outer `~/Slide-SAM` folder is the original upstream code.
- The model is Slide-SAM-B (`weights/slidesam_b.pth`), fine-tuned with LoRA and LoRA-sub on BTCV, sometimes with WORD added.
- It segments **organs**:

  | Organ | Liver | Stomach | Left Kidney | Spleen |
  |---|---|---|---|---|
  | BTCV label | 6 | 7 | 3 | 1 |

- Best Dice on the fixed 6-volume test set is a mean of **0.791** (word_mini model). The baseline is 0.724. The two were scored with different eval scripts, so they need a like-for-like re-evaluation before we compare them.
- 5-fold cross-validation is in progress. Fold 1 is training, and folds 2–5 will run one at a time on the shared A6000.
- The inference recipe is `learner.use_lora()` → `use_lora_sub()` → `load_well_trained_model(pth)` → `predict_volume(box=...)`. The model takes a box on one slice and carries the mask up and down through the volume.
- **Key gap:** the box prompt is built from the **ground-truth label** (`test/volume_eval_cv.py:85`). A real case has no ground truth, so MARS needs another prompt source.
- Each slice's prediction includes a predicted-IoU score (`pred_iou_thresh` in `core/volume_predictor.py`). This is a built-in confidence signal.
- NSD (surface-distance metric) is not computed yet. It is logged as 0.0.

### MARS
- `agents/segmentation.py` is a placeholder that picks voxels above a brightness threshold. Its output is `SegmentationResult(mask, per_voxel_confidence, aggregate_confidence)`, and `AGENT.md` says to swap in the real model behind that same interface.
- The 15 test cases are synthetic `.npy` volumes, and their metadata uses made-up sites and size ranges. Real data is NIfTI files, and the mock `SITES` map in `agents/validation.py` doesn't know real organs.
- `requirements.md` asks for **lesion** segmentation, but Slide-SAM currently segments **organs**.

## 2. Roadmap

### Phase 0: Lock the model (runs alongside CV)
1. Let fold 1 finish, then run folds 2–5 as planned.
2. Choose one checkpoint for MARS v1: the best fold, or word_mini after a fair re-evaluation.
3. Write down the exact inference recipe: config, checkpoint path, LoRA flags and label ids.

### Phase 1: Wrap Slide-SAM as a standalone inference function
1. Add a thin wrapper inside MARS that imports Slide-SAM from its folder, so the Slide-SAM repo doesn't change:
   `slidesam_infer(volume, box, slice_idx) -> (mask, prob_map, iou_per_slice)`.
2. Load the model once at startup and reuse it, because loading it for every request is too slow.
3. Handle preprocessing the same way `volume_eval_cv.py` does: CT windowing, resizing to 1024, and converting the mask back to the original size.
4. Check that the wrapper reproduces the CV Dice on 1–2 BTCV test volumes before going further.

### Phase 2: Decide where the prompt comes from (main design decision)
- **A. Radiologist box (simplest; matches "supports, doesn't replace"):** the user draws a box on one slice in the frontend, and the frontend sends it to the backend.
- **B. Box from clinical metadata:** use `expected_site` plus a rough organ location to build a coarse box automatically.
- **C. Automatic box from a detector:** a small nnU-Net or other detector proposes the box. This takes the most work, but it also gives a second model to compare against.

Recommendation: start with **A**, and log **B** alongside it as a test of how well automatic prompting works.

### Phase 3: Swap the segmentation agent
1. Replace the thresholding in `agents/segmentation.py` with the Slide-SAM wrapper and keep `SegmentationResult` unchanged.
2. Build `per_voxel_confidence` from the sigmoid of the mask logits (`return_logits=True`).
3. Build `aggregate_confidence` from the mean predicted-IoU across slices, plus the average boundary probability.
4. Add extra signals the uncertainty agent can use:
   - how fast confidence drops as the mask moves away from the prompt slice;
   - whether the mask stopped early or broke apart.
5. Keep the mock available behind a flag so tests and the demo still run on a machine without a GPU.

### Phase 4: Real data and validation
1. Add a NIfTI case loader for the BTCV test volumes, and use real voxel spacing so volumes come out in mm³.
2. Replace the mock `SITES` map with:
   - the BTCV organ labels;
   - typical volume ranges per organ;
   - position checks, e.g. the liver sitting in the upper right of the abdomen.
3. Give the validation agent the organ the user asked for.

### Phase 5: Make the uncertainty scores trustworthy
1. Run the full pipeline on the CV held-out volumes, which already have ground truth.
2. Fit the uncertainty weights on that data. Then measure calibration (ECE) and risk–coverage curves, as `requirements.md` §6 asks.
3. Optionally add stronger signals:
   - running with slightly shifted boxes;
   - test-time augmentation;
   - comparing the five fold models. How much the folds disagree is a useful uncertainty signal.

### Phase 6: Backend API and frontend
1. Add `POST /cases/{id}/run` with a body of `{box, slice_idx, organ}`.
2. Serve real NIfTI slices to the frontend.
3. Add a box-drawing tool to `frontend/src/components/SliceViewer.tsx`.
4. Plan for GPU sharing: one inference at a time, and make sure it doesn't collide with CV training on the same A6000.

### Later: organs → lesions
Slide-SAM is fine-tuned on organs, while the requirements target lesions. For v1, the likely path is:
1. Present MARS v1 as organ segmentation.
2. Fine-tune the same LoRA pipeline on a lesion dataset, such as MSD Liver/Pancreas or KiTS. The data format for this is already documented in the Slide-SAM readme.

## 3. Decisions for the team
1. **Prompt source:** radiologist box (A), metadata box (B) or detector (C).
2. **v1 target:** organs, which work now, or lesions, which need new fine-tuning.
3. **Deployment checkpoint:** best single fold, or an ensemble of all five folds, which is slower but gives uncertainty for free.
