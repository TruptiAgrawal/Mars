# Isolated box-padding evaluation

Created 29 September 2026 while fold 5 training was still in progress. This folder is an experimental copy; the existing training, dataset, and evaluation modules were not edited. No GPU evaluation has been run as part of preparing this code.

## Backups and files

Before changes, four source files were copied into `originals/`, and working copies were created beside them:

| Original source | Experimental working file |
|---|---|
| `test/volume_eval_cv.py` | `evaluate.py` |
| `core/volume_predictor.py` | `predictor.py` |
| `datasets/data_engine.py` | `data_engine.py` |
| `datasets/eval_dataloader/loader_abstract.py` | `loader.py` |

`originals/source_manifest.json` records source hashes. `originals/protected_sha256.json` fingerprints 117 existing code/configuration/manifest files. The evaluator and loader copies were rewritten for explicit inputs and recorded protocols; the predictor and box-generator copies were patched. Shared model architecture and learner code are imported read-only during inference.

`helpers.py` implements image-relative padding, manifest checks, prompt selection, and volume/surface metrics. `test_cpu.py` covers the helpers and propagation wiring without constructing a neural network or loading weights.

## Three padding modes

| Mode | Meaning | CLI |
|---|---|---|
| Historical | Existing `enlarge()` arithmetic, including its known sign/margin issues | `--padding-mode historical` |
| Corrected zero | Tight propagated box with valid image bounds | `--padding-mode none` |
| Image-relative | Add a fraction of image width/height to each side | `--padding-mode image --padding-fraction 0.05` |

Only automatically propagated boxes are padded. All modes receive the same deterministic initial ground-truth-derived prompt. Padding uses floating-point coordinates with no premature integer rounding and clips to inclusive bounds `[0,W−1]`, `[0,H−1]` on the resized model-input grid. A fraction of 0.05 means 5% of the **image** dimensions on **each side**, not 5% of the box dimensions.

Historical mode is evaluated through the corrected experimental evaluator. It is not guaranteed to reproduce old logged scores exactly: slice-boundary handling, orientation handling, and metrics differ. Preserve the untouched original evaluator and results as the historical provenance; compare the three new modes under identical new settings.

## Protocol and metric changes

- Explicit image arrays in SimpleITK z-y-x order. No smallest-dimension axis inference and no z-resampling.
- Image/label geometry checked before evaluation; spacing is reversed from x-y-z to z-y-x for surface metrics.
- Initial slice is the ground-truth maximum-area slice when it is a valid three-slice center. At an endpoint, choose the nearest valid foreground center and record the adjustment. An absent target or target too short for the selected protocol raises an explicit error, rather than silently skipping the case.
- `--protocol gt-cropped` retains ground-truth extent cropping. `--protocol full-volume` evaluates the entire volume but still uses an oracle initial prompt. These protocols must have separate baselines; do not compare padding OFF on one with padding ON on the other.
- Point prompts are deterministic foreground points nearest the tight box center. They differ from the historical random fallback and must be reported as a new point protocol.
- Dice, NSD, area-weighted robust HD95, recall, precision, and false-positive milliliters are recorded per case. The NSD tolerance must be supplied explicitly. An example of 1 mm below is a declared example, not a validated clinical tolerance.
- Empty prediction with nonempty target scores Dice/NSD=0; HD95 is undefined (`null`), with its count exposed in summaries. Both-empty metric inputs score Dice/NSD=1 and HD95=0, but the positive-prompt evaluator rejects absent targets before inference. Undefined HD95 cases must not disappear from interpretation of the mean.
- Constant-intensity three-slice windows are normalized to zeros to avoid division by zero. Empty initial model predictions are returned as empty masks and recorded as failures, not discarded.

## Safe preflight now: no inference by default

Run from the repository root. This command only checks arguments, file existence, and manifest disjointness. It does **not** read image volumes or checkpoint tensors, import model code, create the output directory, or use CUDA.

```sh
python -m btcv_cross_validation.box_padding.evaluate \
  --data-root datasets \
  --manifest datasets/dataset_list/01_BTCV_val.txt \
  --train-manifest datasets/dataset_list/01_BTCV_train.txt \
  --checkpoint ../runs/sam/ddp_sub/btcv_liver_run2/ckpt/model_epoch_3.pth \
  --checkpoint-kind finetuned \
  --label-idx 2 \
  --prompt box \
  --protocol gt-cropped \
  --padding-mode none \
  --nsd-tolerance-mm 1.0 \
  --output btcv_cross_validation/box_padding/results/pilot_right_kidney_none
```

Expected evaluation IDs: `img0035`, `img0036`, `img0037`. Label 2 is right kidney, not liver. Use labels 6, 7, 3, 1 for the four-organ paper-aligned evaluation.

**After fold 5 finishes and GPU availability is checked**, add `--run-inference` to explicitly load the model and run evaluation. Add `--save-predictions` to export working-grid masks as NPZ files. Cropping offsets, original shape, spacing, origin, and direction are recorded in JSON; these NPZ files are not standalone NIfTI exports.

Use separate output directories for each checkpoint, label, protocol, and padding mode. Existing directories are rejected. Compare `none`, `historical`, and `image` with `--padding-fraction 0.05`, changing only padding and output path. Do not change the NSD tolerance or initial prompt protocol between modes.

The supplied checkpoint must actually have been trained with the supplied training manifest. The code checks manifest overlap; it cannot establish checkpoint training provenance from a weight file. The older checkpoint has prior cache/run-log evidence described in `KT/Basic_Audit_and_Padding_Pilot_Plan.md`.

For fold evaluation, explicitly use that fold's `01_BTCV_train_4to1.txt`, `01_BTCV_test_4to1.txt`, and completed checkpoint. Beware fold 5's two differently capitalized folders: `cv_Fold5` is the progressing run found in the audit. Do not use the unfinished latest file while training is writing it.

`--checkpoint-kind finetuned` installs both rank-8 and rank-1 adapters before loading. `--checkpoint-kind official` installs rank-8 only and is intended for the official Slide-SAM-B checkpoint. Raw original SAM weights are not supported by this evaluator.

## Results and reproducibility

Every executed run writes `results.json` incrementally, with arguments, source/manifest/checkpoint hashes, patient IDs, original-grid initial slice, actual initial prompt, propagated boxes, geometry, metrics, and inference latency/peak allocated GPU memory. A failed run retains completed cases and an explicit error status. Timing excludes image loading and metric computation; checkpoint hashing happens only during explicitly requested inference.

The three historical validation cases are part of fold 5's held-out set. Using their scores to select a padding policy makes them development data, regardless of the older checkpoint. Disclose this overlap and distinguish code-mechanics checks from score-driven tuning.

## CPU verification

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/slidesam-padding-mpl \
  python -m unittest btcv_cross_validation.box_padding.test_cpu -v
```

Tests use synthetic arrays/NIfTI files in temporary directories and a dummy CPU model descriptor for prompt wiring. No neural model is constructed. They cover padding coordinates, historical equivalence, clipping, mask metrics and spacing, slice endpoints, deterministic initial prompts, geometry mismatch, manifest leakage, protected originals, and preflight behavior.

These checks establish CPU correctness of the preparation path; they do not replace the pending real-checkpoint GPU pilot or establish improved segmentation performance.
