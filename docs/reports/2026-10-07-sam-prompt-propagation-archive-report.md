# SAM Prompt Placement by Propagation: Archive Report

Date: 7 Oct 2026
Repository archived: `~/SAM-Prompt-Placement-by-Propagation/` (deletion planned after review)
Purpose: keep every finding, implementation detail and result from this repository after the repo is deleted. The report is also source material for documenting why the project moved from plain SAM to Slide-SAM.

Everything below comes from files in the repository: code, configs, logs, NIfTI predictions and prompt JSONs. I computed the statistics directly from the saved prediction files on 7 Oct 2026. Where the repository cannot answer a question, the report says so. Most often this is because the ground-truth data was deleted on 26 Sep 2026.

---

## 1. Summary

- **What it is:** a local working copy of the code for the MIDL paper *Training-free Prompt Placement by Propagation for SAM Predictions in Bone CT Scans* ([OpenReview](https://openreview.net/forum?id=F6rhgpGkAy)). It segments 3D bone CT with the 2D Segment Anything Model (SAM ViT-B) and no training. A single box on one slice is propagated slice by slice through the volume.
- **Four propagation strategies** are implemented: **baseline**, **stochastic**, **nested** and **combined**. There is also a **reference** mode that takes a box from the ground truth on every slice.
- **What we ran:** a TotalSegmentator-derived nnU-Net dataset, `Dataset101_Bone`, with label 1. According to `convert_labels.py`, label 1 is `femur_left`.
- **Results that exist:**
  - All four methods were compared on **one subject (s0000)**. Dice scores were 0.9458 (nested), 0.9429 (baseline), 0.9390 (combined) and **0.8299 (stochastic)**.
  - The **baseline** was run on 77 subjects (s0000–s0168). One output (`s0024.nii.gz`) is a **0-byte corrupt file**. Only s0000 was ever scored against ground truth.
- **`PROJECT_SUMMARY.md` (26 Sep 2026) contains errors.** It says 77 subjects were *evaluated*, but only one was. It also says nested and combined were *not run*, but both were run and scored on s0000. Section 8 lists the corrections.
- **Code defects found.** The HD95/ASSD evaluation script cannot run. The nested and combined look-ahead step uses the wrong variable. The reference script stops after the first case. The resume logic silently skips corrupt outputs. Section 7 has the details.
- **Relevance to Slide-SAM:** see Section 10. The limitations seen here line up with what Slide-SAM adds: purely 2D SAM with no inter-slice context, no medical adaptation, no CT windowing, error drift once propagation starts, and expensive per-slice re-encoding. One gap carries over unchanged: the initial box still comes from the ground truth.

---

## 2. Provenance and repository state

| Item | Value |
|---|---|
| Git remote (`origin`) | `https://github.com/TruptiAgrawal/SAM-Prompt-Placement-by-Propagation.git` (fork) |
| Upstream commits | 3 commits by GitHub user `CarolineMagg`, 16 May 2024: `0944f40` (README, run script), `ccf2911` (code, 1,548 lines), `776783e` (README update) |
| Unpushed commits / stashes | None. Local `main` equals `origin/main` (`776783e`). |
| Checkout layout | The repo is nested twice: `~/SAM-Prompt-Placement-by-Propagation/SAM-Prompt-Placement-by-Propagation/`. In the outer checkout, git sees the top-level tracked files as deleted and the inner folder as untracked. All the work lives in the **inner** folder, which is a complete clone with its own `.git`. |
| Size on disk | 368 MB. Of this, 358 MB is `sam_vit_b_01ec64.pth`, 8.4 MB is `results/` and about 360 KB is `predictions_*_old/`. |
| Local working period | File timestamps run from 19 Jul 2026 (clone) to 17 Aug 2026 (last prediction). `PROJECT_SUMMARY.md` was written on 26 Sep 2026. |

**Lost on deletion: local-only content not on GitHub.** Appendix B reproduces all of it word for word.
- `run_scripts/prompt_propagation.py`: **modified** and uncommitted (see Section 6).
- Untracked: `convert_labels.py`, `scripts/convert_labels.py`, `scripts/untitled.py`, `evaluation/nnunet_evaluation_connector.py`, `evaluation/__init__.py`, `run_all_models.py`, `setup.sh`, `env_setup.sh`, `requirements.txt`, `requirements_new.txt`, `environment.yml`, `sam_prompt_environment.yml`, `PROJECT_SUMMARY.md`.
- Outputs: `results/`, `predictions_*_old/`, `predictions_test/`.
- Junk: an empty file literally named `WARNING:` (probably a stray shell redirect), `.ipynb_checkpoints/` and `__pycache__/`.

**Recoverable elsewhere:**
- Upstream code: on GitHub at the remote above.
- SAM weights: an identical copy is at `~/Mars/Slide-SAM/Slide-SAM/sam_vit_b_01ec64.pth` (MD5 `01ec64d29a2fca3f0661936605ae66f8` for both). They can also be downloaded from `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`. Slide-SAM's setup notes (`KT/00_Progress/past_slide_sam_inital_setup.md`) record that its copy came from this repo. No code in Mars or Slide-SAM refers to this repo's path.
- Datasets: already deleted (see Section 5). They can be regenerated from Zenodo.

---

## 3. Repository inventory (inner folder)

| Path | Origin | Role |
|---|---|---|
| `scr/SAMPromptPropagation.py` (556 lines) | upstream | Core: helper functions plus the classes `SAMPredictorVolumeBaseline`, `…Stochastic`, `…Nested` and `…NestedStochastic` |
| `scr/SAMReferencePrompt.py` (105 lines) | upstream | `SAMPredictorVolumeBasedOnReference`: a box from the ground truth on every slice |
| `run_scripts/prompt_propagation.py` | upstream, **locally modified** | CLI: runs one of the four propagation experiments over an nnU-Net dataset |
| `run_scripts/prompt_reference.py` | upstream | CLI: reference (slice-by-slice) predictions |
| `run_scripts/prompt_propagation_postprocessing.py` | upstream | Post-processing 1: keep the connected component(s) tied to the initial prompt |
| `run_scripts/prompt_propagation_postprocessing2.py` | upstream | Post-processing 2: cut the prediction where boxes shrink below 10% of the initial box |
| `run_scripts/prompt_propagation_evaluation.py` | upstream | Dice, IoU, TP, FP, FN per class, written to `<Dataset>_log.json` |
| `run_scripts/prompt_propagation_evaluation2.py` | upstream | HD95 and ASSD, written to `<Dataset>_log2.json`. **Broken here** (Section 7) |
| `run_scripts/create_box_prompt_visualizations.py` | upstream | Draws the box prompts onto the prediction and writes `<case>_vis.nii.gz` |
| `run_script_example.sh` | upstream | The original author's pipeline example (dataset `500_TotalSegmentatorBoneSimple`, labels 2–5, test split `Ts`) |
| `evaluation/nnunet_evaluation_connector.py` | **local** | Local stand-in for a module that upstream imports but never shipped. Only the Dice/IoU function is defined |
| `convert_labels.py`, `scripts/convert_labels.py` | **local** | TotalSegmentator → `Dataset101_Bone` (femur_left=1, femur_right=2). The two copies differ only in the input folder name |
| `scripts/untitled.py` | **local** | Alternative converter that builds one binary nnU-Net dataset per TotalSegmentator label |
| `run_all_models.py` | **local** | Driver: for each `Dataset*` folder, runs all four experiments and then evaluation, with label 1 |
| `setup.sh`, `env_setup.sh` | **local** | Conda env `sam_prompt` (Python 3.10), PyTorch 2.5.1/cu121, requirements, SAM, nnU-Net environment variables |
| `requirements.txt`, `requirements_new.txt`, `environment.yml` | **local** | Pinned dependencies (Appendix C) |
| `sam_prompt_environment.yml` | **local** | Despite the name, an export of the JupyterHub **base** environment (`prefix: /opt/tljh/user`, Python 3.12). Not useful for this project |
| `PROJECT_SUMMARY.md` | **local** | Earlier archive note. Partly inaccurate (Section 8) |
| `results/Dataset101_Bone/{baseline,stochastic,nested,combined}/` | **local** | Main outputs. `nested/` and `combined/` are empty |
| `predictions_{baseline,stochastic,nested,combined}_old/` | **local** | Early s0000 runs for all four methods, each with a Dice log |
| `predictions_test/` | **local** | Empty |
| `data/` | **local** | Only `.ipynb_checkpoints/` is left. The datasets were deleted |

---

## 4. Method and implementation

### 4.1 Idea (from the README)
SAM is a 2D model prompted per image. For a 3D CT, the method does the following:
1. Takes **one box on one slice**. The paper calls this "two annotated pixels", meaning the box corners.
2. Predicts that slice with SAM.
3. Turns the prediction into a box prompt for the **adjacent** slice.
4. Repeats in both directions until the mask disappears.

No training or fine-tuning is involved. nnU-Net is used only for the dataset layout (`imagesTr/`, `labelsTr/`) and the environment variable `nnUNet_raw`.

### 4.2 Data handling (`scr/SAMPromptPropagation.py`)
- **Loading:** nibabel. The volume is reordered with `np.moveaxis(x, -1, 0)`, so propagation runs along the **last array axis**. For TotalSegmentator NIfTIs this is the axial z axis. The affine is kept for saving.
- **Slice preprocessing** (`process_input_array`, line 88):
  - Each slice is min-max normalised separately to uint8 0–255.
  - It is then passed through `cv2.cvtColor(..., COLOR_BGR2RGB)`, which turns one channel into three. I checked that this works with the pinned `opencv-python-headless 5.0.0.93`.
  - There is **no HU windowing**. The reference mode normalises differently: it divides by the slice maximum (`SAMReferencePrompt.py:75`).
- **Model:**
  - `sam_model_registry["vit_b"]` with the checkpoint taken from `os.getcwd()/sam_vit_b_01ec64.pth`, so scripts must be run from the repo root.
  - The model is hard-wired to `cuda`.
  - `SamPredictor.set_image()` is called for **every slice**, which re-runs the ViT image encoder each time.
  - `predict(box=..., multimask_output=False)` is used.

### 4.3 Initialisation: the "two-pixel" prompt (`run_scripts/prompt_propagation.py`)
1. The binary mask for the class is taken from the **ground-truth label volume**.
2. The first and last z slices that contain the object are found.
3. **Initial slice:**
   - For all experiments, the initial slice is picked at random (`random.seed(1)`, `np.random.seed(1)`) from the central `--percentage` (default **0.5**) of the object's z extent.
   - The baseline can also start from the `first` or `last` slice with `--slice_select`.
   - `get_random_init_slice` picks an offset around the midpoint of the object's extent and adds `idx_first` back.
4. The initial box is the bounding box of the ground-truth mask on that slice, from the same component and box logic as below.

### 4.4 Mask → box prompt (shared by all methods)
- `find_all_disconnected_regions`:
  - Labels connected components with `skimage.measure.label`, `connectivity=2`.
  - If `remove_small_boxes=1` (the default), it drops components smaller than **5%** of the mask or smaller than **5 px**.
  - It sorts the rest by size, largest first.
- `generate_prompt`: returns the tight bounding boxes `[x0, y0, x1, y1]` of **up to the 3 largest** components, with `--margin` padding (default 0).
- Multiple boxes are each predicted separately and **summed**. Any pixel above 0 counts as foreground, so the result is a union.

### 4.5 Propagation loop (`predict_volume` / `predict_volume_one_way`)
- **Direction 1:** from the initial slice to the end. **Direction 2:** from the initial slice back to slice 0. The two halves are stitched together, and the initial slice is predicted in both.
- **Slice 0 of each direction:** the box comes from the ground-truth mask. On later slices, it comes from the **previous prediction**.
- **Stopping:** if the previous mask is empty, the slice gets an empty mask and a `None` prompt. **Once the mask disappears, propagation never recovers in that direction.** All-zero image slices are also skipped.
- **Outputs:**
  - A mask volume.
  - A prompt dictionary keyed by absolute slice index: a list of boxes, or `null`.
  - `"time"` (seconds) and `"idx_init"`.

### 4.6 The four strategies

| Experiment | Class | What changes compared with the baseline | Main parameters (CLI default) |
|---|---|---|---|
| **baseline** | `SAMPredictorVolumeBaseline` | Nothing. One SAM call per box per slice | `margin=0`, `remove_small_boxes=1`, `slice_select=random` |
| **stochastic** | `SAMPredictorVolumeStochastic` | Each box is randomly jittered `iterations` times. Each corner moves by 0 to `max_margin-1` px (0–4 px), with a random sign. A pixel is kept only if it is predicted **more than** `int(iterations*majority_vote)` times | `iterations=10`, `max_margin=5`, `majority_vote=0.55`. This means at least **6 of 10** votes. The class default is 0.8, but the CLI passes 0.55 |
| **nested** | `SAMPredictorVolumeNested` | Look-ahead and re-prediction. After predicting slice *i*, it predicts slice *i+1* and re-predicts slice *i* from that result. With `aggregate=1`, the next prompt comes from the **intersection** of all of slice *i*'s masks. With `aggregate=0` (the CLI default `--merge_results 0`), it comes from the re-predicted mask. **The implementation differs from this description; see Section 7.2.** | `merge_results=0` |
| **combined** | `SAMPredictorVolumeNestedStochastic` | Nested with stochastic box voting for every SAM call | All of the above |
| *reference* | `SAMPredictorVolumeBasedOnReference` | Not propagation. **On every slice**, a box from the ground truth. This is the "traditional slice-by-slice SAM" comparison | `remove_small_boxes=0`, `margin=0` |

Notes on the stochastic vote:
- With several boxes on a slice, votes are **summed across boxes**, but the threshold scales only with `iterations` (`SAMPromptPropagation.py:296`).
- Because the vote threshold is strict, a pixel that only one of two boxes covers can still pass.

### 4.7 Post-processing (upstream, recommended by the README for propagation output)
- **Post-processing 1** (`prompt_propagation_postprocessing.py`):
  - Labels the 3D connected components with `acvl_utils.label_with_component_sizes`.
  - Keeps the component(s) present on the initial slice.
  - If the initial slice has more than one component, it keeps the largest component when it lies inside the initial box. Otherwise it keeps every component in the box.
  - The code also handles legacy `dir1`/`dir2`/`idx_first`/`idx_last` prompt keys that the current propagation code no longer writes.
- **Post-processing 2** (`prompt_propagation_postprocessing2.py`):
  - Computes the total box area on each slice.
  - Finds the closest slice above and the closest slice below the initial slice where the area falls under **10% of the initial box area**.
  - Removes everything beyond those slices.
  - This is a heuristic stopping rule for runaway propagation.
- Neither step rewrites the prompt JSON. Later scripts therefore need both the post-processed folder and the original prompt folder.
- **No post-processed outputs exist in this repository.** Post-processing was never run here.

### 4.8 Evaluation
- `prompt_propagation_evaluation.py`:
  - For each case and class, it binarises the reference and the prediction.
  - It calls `compute_metrics_like_nnunet_from_arrays_simple`, which gives Dice = 2TP/(2TP+FP+FN), IoU = TP/(TP+FP+FN), and TP, FP, FN voxel counts (`evaluation/nnunet_evaluation_connector.py`).
  - It uses a 4-process spawn pool.
  - It writes `<output_folder>/Dataset<ID>_log.json`.
  - It also loads SAM onto the GPU, which it never uses.
- `prompt_propagation_evaluation2.py`: meant to compute HD95 and ASSD with voxel spacing. It **cannot run here** (Section 7.1).

---

## 5. Data

### 5.1 Paper datasets (README)
The paper used three datasets: two internal and one public. The public one was TotalSegmentator ([Zenodo 10.5281/zenodo.6802613](https://doi.org/10.5281/zenodo.6802613), Wasserthal et al. 2023). Two subsets were taken from its **test** set:
- 33 CTs with the left and right humerus and scapula;
- 53 CTs with the left and right hip and femur.

The upstream example uses `Dataset500_TotalSegmentatorBoneSimple`, labels `2 3 4 5`, split `Ts`.

### 5.2 Our dataset: `Dataset101_Bone`
- **Source:** TotalSegmentator, extracted under `data/TotalSegmentator` (or `data/Totalsegmentator_dataset_v201` in the `scripts/` copy). According to `PROJECT_SUMMARY.md`, it was about 31 GB with 1,229 case folders from s0000 to s1371.
- **Conversion** (`convert_labels.py`):
  1. Copies each case's `ct.nii.gz` to `imagesTr/<case>_0000.nii.gz`.
  2. Merges `femur_left`→**1** and `femur_right`→**2** into `labelsTr/<case>.nii.gz`.
  3. A commented-out block holds the shoulder mapping (scapula L/R = 1/2, humerus L/R = 3/4).
  4. A case with neither label gets an image but no label file. Only cases with labels are processed, because the runner lists `labelsTr/`.
- **Size:** about 22 GB, 1,228 subjects (`PROJECT_SUMMARY.md`).
- **Label used:** `--label_order 1` everywhere: `run_all_models.py`, `PROJECT_SUMMARY.md`, and every prompt JSON, whose only key is `"1"`.
  - Under the current `convert_labels.py`, **label 1 = femur_left**.
  - Caveat: that file is dated 24 Jul 2026, after the first s0000 runs on 20 Jul. The repository cannot prove which converter version built the dataset those runs used.
- **Split:** the training split (`nnunet_suffix` default `Tr`), not the TotalSegmentator test subset the paper used. These results are therefore **not directly comparable to the paper's numbers.**
- **Geometry:** every prediction has isotropic spacing of **1.5 × 1.5 × 1.5 mm**, the TotalSegmentator resampled format. Shapes range from 93×93×83 to 499×154×333.
- **Alternative:** `scripts/untitled.py` builds `DatasetNNN_<label>` binary datasets, one per TotalSegmentator structure. There is no sign that it was run.
- **Deleted 26 Sep 2026** (`PROJECT_SUMMARY.md`): `data/TotalSegmentator` (about 31 GB) and `data/nnUNet_raw` (about 22 GB). Ground truth is therefore unavailable, which is why most predictions cannot be scored now.

---

## 6. Local changes to upstream code

The only tracked file that was changed is `run_scripts/prompt_propagation.py`. The full current version is in Appendix B.1. The changes in substance:

1. **The trailing `break` at the end of the case loop was removed.** Upstream `prompt_propagation.py` stops after the **first** case it processes. This explains why every early run (`predictions_*_old`) contains only s0000. **`prompt_reference.py` still has this `break` (line 115)**, so the reference mode still processes one case only.
2. **Resume support:** a case is skipped if both `<case>.nii.gz` and `<case>_prompts.json` already exist.
3. **Prompt JSON durability:** the prompt JSON is written with `flush()` and `os.fsync()`. The NIfTI is not.
4. **Bug fix:** a `NameError` was fixed. Upstream logged `args.slice_select` inside a function, where `args` is not defined.
5. **An unknown `--experiment` now raises `ValueError`.** Upstream used `break`.
6. Logging was cut down to a progress line and print statements. Formatting only (black-style line wrapping).

---

## 7. Defects and limitations found in the code

### 7.1 HD95/ASSD evaluation cannot run
`prompt_propagation_evaluation2.py:13` imports `computer_metrics_like_nnunet_from_arrays_surface` from `evaluation.nnunet_evaluation_connector`. The local module defines only `compute_metrics_like_nnunet_from_arrays_simple`, so the script fails at import. Upstream never committed an `evaluation/` package; the local one is a partial stand-in. **No surface-distance metric was ever computed in this project.**

### 7.2 The nested/combined look-ahead uses the wrong variable (upstream code)
In `SAMPredictorVolumeNested.predict_volume_one_way`, and in the same block of `…NestedStochastic`, two lines are wrong:
- **Line 358/505:** the code checks `np.max(input_array)` (the current slice) where `input_array2` (the next slice) is meant.
- **Line 370/517:**
  - The code sets `preds_mask_single2 = preds[:, :, 0] > 0`, using `preds`, the **current-slice** prediction.
  - It should use `preds2`, the prediction for slice *i+1*.
  - As a result, the "next-slice" mask equals the current mask. The re-prediction (`prompt3`) re-prompts slice *i* with a box built from its **own** prediction.
- **Effect:** in practice, "nested" as run here is *predict, then re-predict the same slice once from its own box*. It is not true look-ahead. The SAM call on slice *i+1* still runs (`preds2`), which costs time, but its result is thrown away.
- The s0000 nested and combined numbers in Section 8 reflect this behaviour. Whether the published paper results used this same code is not something the repository can show.

### 7.3 Reference mode processes only one case
See Section 6, item 1 (`run_scripts/prompt_reference.py:115`).

### 7.4 Resume logic + non-atomic write → silently skipped corrupt case
- `results/Dataset101_Bone/baseline/s0024.nii.gz` is **0 bytes**, but `s0024_prompts.json` is complete: 9,059 bytes, `idx_init=51`, 153 prompted slices, 55.2 s run time. Both are timestamped 31 Jul 2026 16:06:51.
- The script writes the NIfTI first and the JSON second, and fsyncs only the JSON. This pattern fits a run that stopped before the NIfTI reached disk.
- Because the resume check only tests whether the files *exist*, every later run (the 17 Aug session) **skipped s0024 without warning.**

### 7.5 Post-processing 2 CLI fallback is a no-op
At `prompt_propagation_postprocessing2.py:114-115`, `args.input_folder_prompt = args.input_folder_prompt` does nothing. If `--input_folder_prompt` is left out, `os.listdir(None)` lists the current directory, and the count assertion fails. In effect, the option is required.

### 7.6 Method-level limitations visible in the implementation
1. **The initial box comes from the ground truth.** Both the initial slice and its box are computed from the reference label. Nothing in the code takes a box from a user or a detector.
2. **No inter-slice context.** Each slice is segmented independently by 2D SAM. The only link between slices is the box taken from the previous mask, so errors build up.
   - A box that is too large lets the mask spread into neighbouring structures.
   - A mask that disappears ends propagation for good in that direction (Section 4.5).
3. **No medical adaptation.** These are natural-image SAM ViT-B weights, used as is.
4. **No CT windowing.** Per-slice min-max scaling means contrast depends on whatever else is in the slice, such as air, the table or metal. Reference and propagation modes also normalise differently.
5. **Cost:**
   - The ViT image encoder runs once per slice, and the nested and combined methods run it up to three times per slice.
   - Stochastic and combined make `iterations` (10) decoder calls per box.
   - Measured times are in Section 8.
6. **The stopping rule is a heuristic that needs post-processing.** Without post-processing 1 and 2, nothing stops runaway propagation except an empty mask. There is no confidence signal, and SAM's IoU prediction output is thrown away (`predict` returns `_, _`).
7. **Box-only prompts.** There are no point or mask prompts, and `multimask_output=False`.
8. **Single label per run in our setup.** Only label 1 was used, never `1 2`.
9. **Operational:**
   - The model is hard-coded to CUDA.
   - The checkpoint path depends on the working directory.
   - Both evaluation scripts needlessly load SAM onto the GPU.

---

## 8. Experiments and results

### 8.1 Timeline (from file timestamps)

| Date (2026) | What happened | Evidence |
|---|---|---|
| 19 Jul | Repo cloned | Timestamps of tracked files |
| 20 Jul | conda env built. All four methods run on s0000 (upstream script, first case only) and scored | `predictions_*_old/`, logs 12:21–12:33 |
| 22–24 Jul | Combined re-scored. Combined s0000 prediction re-generated on 24 Jul. Converter scripts written | `predictions_combined_old` timestamps, `convert_labels.py` |
| 28 Jul | `run_all_models.py` created. `results/…/baseline` and `stochastic` s0000 produced (identical masks to the 20 Jul runs). Baseline s0000 scored | `results/` timestamps |
| 29–31 Jul | `setup.sh` and requirements rewritten. `prompt_propagation.py` modified for full-dataset and resume runs. Baseline run reaches 14 more cases before stopping at **s0024 (corrupt)** | 14 files dated 31 Jul |
| 17 Aug | Baseline run resumed: 62 more cases up to **s0168**, then stopped | 62 files dated 17 Aug |
| 26 Sep | Datasets deleted (about 53 GB). `PROJECT_SUMMARY.md` written | `PROJECT_SUMMARY.md` |

The baseline run covered **77 of about 1,228** subjects. The 77 are every labelled case from s0000 to s0168 in sorted order. The repository does not record why the run stopped at s0168.

### 8.2 Four-method comparison on s0000 (the only scored comparison)
These Dice/IoU values come from `predictions_*_old/Dataset101_Bone_log.json`. Voxel counts, slice ranges and times come from the saved NIfTI and prompt JSON files. The ground-truth size is TP+FN = **69,349 voxels** in all four logs, so they were scored against the same reference. The initial slice was the same for all four: **slice 59**, initial box `[58, 49, 79, 67]` (21 × 18 px, about 32 × 27 mm).

| Method | Dice | IoU | TP | FP | FN | Predicted voxels | Slices with mask (z range) | Run time (s) |
|---|---|---|---|---|---|---|---|---|
| **nested** | **0.9458** | **0.8972** | 63,556 | 1,487 | 5,793 | 65,043 | 168 (0–167) | 59.6 |
| baseline | 0.9429 | 0.8920 | 63,426 | 1,755 | 5,923 | 65,181 | 171 (0–170) | 21.1 / 21.5 |
| combined | 0.9390 | 0.8850 | 62,357 | 1,107 | 6,992 | 63,464 | 166 (0–165) | 77.8 |
| stochastic | 0.8299 | 0.7092 | 49,500 | **446** | **19,849** | 49,946 | 149 (0–148) | 25.6 / 25.9 |

**Consistency check:** for every method, the logged TP+FP equals the voxel count of the saved mask. The combined log (22 Jul) is older than its saved mask (24 Jul), but the counts still match, so the logged metrics describe the masks that are saved.

What the data shows:
- **Baseline, nested and combined are within 0.007 Dice of each other.** Nested scores highest. On a single subject this gap is not meaningful, especially since the nested look-ahead step is defective (Section 7.2).
- **Stochastic under-segments badly.** It has the fewest false positives (446) but almost 3.4× the false negatives of baseline. Its mask also stops about 22 slices earlier than baseline (slice 148 vs 170). This fits the strict majority vote (≥6/10): each slice comes out a little smaller, so the next box is smaller, and the shrinkage compounds until the mask disappears.
- **Cost:** compared with baseline, nested takes about **2.8×** as long and combined about **3.7×**. Stochastic costs about 1.2×, because the image encoder runs once per slice and only the decoder repeats.
- **Determinism:** baseline and stochastic s0000 were re-run on 28 Jul and gave **byte-identical masks** to the 20 Jul runs (same MD5). Only the recorded `time` differs. With the seeds fixed at 1, results are reproducible.

### 8.3 Baseline over the dataset (77 subjects, s0000–s0168)
None of these 77 cases except s0000 was scored. These are **descriptive statistics of the predictions only.** The per-case table is in Appendix A.

| Statistic | Value (76 readable cases; s0024 corrupt) |
|---|---|
| Predicted volume | median **137.7 mL** (IQR 102.8–187.7), min 12.5 mL (s0009), max **1,071.5 mL (s0091)** |
| Slices containing prediction | median 77, mean 85.2, range 25–405 |
| Run time per case (baseline, RTX A6000) | median **33.9 s**, mean 35.4 s, range 9.8–53.3 s; total of the recorded times about 0.75 h |
| Cases with at least one multi-box slice | 49 of 76 (at most 3 boxes on a slice) |
| Predictions reaching slice 0 (the first z slice of the volume) | 74 of 76 (not reaching it: s0062, which starts at 17, and s0151, which starts at 8) |
| Predictions above 300 mL | 4: s0091 (1,071.5 mL, 405 slices), s0096 (500.4 mL, 221 slices), s0062 (348.5 mL), s0098 (318.6 mL) |

**Notes:**
- The four large cases are far outside the IQR. They are the most likely candidates for propagation leaking into neighbouring structures. This cannot be checked without the ground truth, so they are worth reviewing first if the dataset is ever regenerated.
- 74 of 76 predictions reach the first slice of the volume. This could mean the femur really runs to the edge of the scan, or that propagation ran on. The data cannot tell these apart.

### 8.4 Corrections to `PROJECT_SUMMARY.md`

| `PROJECT_SUMMARY.md` says | What the files show |
|---|---|
| Baseline: "77 subjects evaluated" | 77 predicted, 76 readable, **1 evaluated** (s0000). The `_log.json` file contains only s0000 |
| "Best single-subject result (s0000)" | s0000 is the **only** evaluated subject, not the best of several |
| Stochastic: 1 subject | Correct. Dice 0.8299 (it was evaluated, in `predictions_stochastic_old`) |
| Nested and combined "not run" | **Both were run and evaluated on s0000** (Dice 0.9458 and 0.9390, in `predictions_*_old`). Only the `results/…/nested` and `combined` folders are empty |
| "Bone label 1" | Under `convert_labels.py`, label 1 = **femur_left** |
| `predictions_*_old` = "legacy predictions (3 files)" | Each holds the four-method s0000 comparison above, which is the only cross-method evidence in the repo |
| Reproduce: "Evaluate" with `prompt_propagation_evaluation` | Works. The HD95/ASSD script (`evaluation2`) does not (Section 7.1) |

---

## 9. Environment and reproduction

### 9.1 Environment (as installed)
- **Hardware:** NVIDIA RTX A6000 (48 GB), shared.
- **Conda env:** `sam_prompt`, Python 3.10.20. It still exists at `~/.conda/envs/sam_prompt` and is **separate from the repo**, so deleting the repo does not remove it.
- **Key pins** (`requirements.txt`):
  - `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`
  - `nnunetv2==2.8.1`, `numpy==2.2.6`, `nibabel==5.4.2`, `scikit-image==0.25.2`
  - `opencv-python-headless==5.0.0.93`, `acvl_utils==0.2.6`, `natsort==8.4.0`
  - `segment_anything @ git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf`
- `requirements_new.txt`, which `setup.sh` uses, loosens several pins and adds `monai==1.6.0`, `MedPy==0.5.2`, `tensorboardX` and `gdown`. The full files are in Appendix C.
- **SAM checkpoint:** `sam_vit_b_01ec64.pth`, 375,042,383 bytes, MD5 `01ec64d29a2fca3f0661936605ae66f8`. It must be in the working directory.

### 9.2 Reproduction steps
These are corrected from `PROJECT_SUMMARY.md` and checked against the code.

```bash
# 0. Code: clone the fork, then restore the local-only files from Appendix B
git clone https://github.com/TruptiAgrawal/SAM-Prompt-Placement-by-Propagation.git
cd SAM-Prompt-Placement-by-Propagation
#    restore: run_scripts/prompt_propagation.py (B.1), evaluation/__init__.py (empty) +
#    evaluation/nnunet_evaluation_connector.py (B.2), convert_labels.py (B.3), setup.sh (B.6), requirements_new.txt (C.2)

# 1. Environment (creates conda env sam_prompt, adds nnUNet_* vars to ~/.bashrc)
bash setup.sh && source ~/.bashrc

# 2. SAM ViT-B weights (or copy from ~/Mars/Slide-SAM/Slide-SAM/)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# 3. Data: TotalSegmentator from Zenodo into data/TotalSegmentator/<case>/{ct.nii.gz,segmentations/}
pip install zenodo-get && zenodo_get 6802613 -o data/TotalSegmentator   # extract the archive if needed

# 4. Build Dataset101_Bone (femur_left=1, femur_right=2)
python convert_labels.py

# 5. Propagation (run from the repo root; the checkpoint path is cwd-relative)
python -m run_scripts.prompt_propagation --dataset 101_Bone --label_order 1 \
    --output_folder results/Dataset101_Bone/baseline --experiment baseline
#    --experiment stochastic | nested | combined   (see Section 4.6 for parameters)

# 6. Optional post-processing (writes NIfTIs only; keep the original folder for prompts)
python -m run_scripts.prompt_propagation_postprocessing --label_order 1 \
    --input_folder results/Dataset101_Bone/baseline --output_folder results/Dataset101_Bone/baseline_pp
python -m run_scripts.prompt_propagation_postprocessing2 --label_order 1 \
    --input_folder results/Dataset101_Bone/baseline_pp \
    --input_folder_prompt results/Dataset101_Bone/baseline \
    --output_folder results/Dataset101_Bone/baseline_pp2

# 7. Evaluation (Dice/IoU)
python -m run_scripts.prompt_propagation_evaluation --dataset 101_Bone --label_order 1 \
    --output_folder results/Dataset101_Bone/baseline
```

**`--dataset` value:** the scripts build the path as `"Dataset" + args.dataset`, so `--dataset 101` resolves to `Dataset101`, not `Dataset101_Bone`.
- For our folder name, pass `--dataset 101_Bone`.
- `run_all_models.py` does exactly this: it strips `"Dataset"` from the folder name.
- The `--dataset 101` form shown in `PROJECT_SUMMARY.md` would not find the folder.

**Before re-running at scale:**
- Delete or validate zero-byte outputs (Section 7.4).
- Fix Section 7.1 if surface metrics are needed.
- Fix Section 7.2 if nested or combined should do real look-ahead.

---

## 10. Technical insights for the move from SAM to Slide-SAM

This repository has **no written decision record** for the switch. This section draws on two kinds of evidence:
- what the code and results here show;
- how the Mars documents describe Slide-SAM: `docs/reports/2026-09-24-slide-sam-integration-roadmap.md` and the References section of `requirements.md`.

Statements about Slide-SAM come from those documents, not from this repository.

### 10.1 Limitations of training-free 2D SAM propagation, and how Slide-SAM relates

| Limitation seen here | Evidence in this repo | How Slide-SAM relates (per Mars docs) |
|---|---|---|
| No inter-slice context. Each slice is segmented on its own, and only a box carries over | `predict_volume_one_way`: one 2D `predict` per slice (Section 4.5) | Slide-SAM uses a **three-slice sliding window** (`requirements.md`, ref. 2), so neighbouring slices inform each prediction |
| Errors compound along the propagation path. A disappearing mask ends propagation, and an oversized one can leak | Stochastic s0000 shrank and stopped 22 slices early (Dice 0.83). Four baseline cases above 300 mL, one at 1,071 mL (Section 8.3). Heuristic post-processing exists only to undo this | Slide-SAM still propagates mask-derived prompts. `requirements.md` lists "prompt propagation errors" as a remaining Slide-SAM limitation, so this risk is **reduced, not removed** |
| No medical-domain adaptation (natural-image SAM ViT-B weights) | `sam_model_registry["vit_b"]` with the original checkpoint, no fine-tuning | Slide-SAM-B is fine-tuned with **LoRA** on BTCV (and WORD), which keeps pretrained SAM knowledge (roadmap §1) |
| No CT windowing. Per-slice min-max scaling. Two different normalisations in one repo | `process_input_array`, `SAMReferencePrompt.py:75` | Slide-SAM preprocessing applies CT windowing and resizes to 1024 (roadmap Phase 1) |
| No confidence signal. SAM's IoU prediction is thrown away. Stopping depends on an empty mask or post-processing 2 | `preds, _, _ = self.predictor.predict(...)` everywhere | Slide-SAM produces a **predicted-IoU score per slice** (`pred_iou_thresh`), which the roadmap plans to use for confidence and uncertainty |
| Cost: the image encoder re-runs per slice, and the variants multiply it | Baseline median 33.9 s per volume. Nested 2.8× and combined 3.7× baseline on s0000 | Not measured in this repo. Compare Slide-SAM's runtime before claiming a speed-up |
| Only the Dice/IoU evaluation works. No surface metrics | Section 7.1 | Slide-SAM's NSD also logs as 0.0 (roadmap §1). Surface metrics are still missing in both |
| Evaluated at scale? Only one subject was scored; the baseline outputs for 76 more were never evaluated | Section 8 | Slide-SAM has a fixed 6-volume test set and 5-fold CV underway (roadmap §1) |

### 10.2 What carries over unchanged (do not treat as solved)
- **The initial box still comes from the ground truth.**
  - Here: `get_first_last_slice` and `select_init_slice` on the reference label.
  - In Slide-SAM: `test/volume_eval_cv.py:85` (roadmap §1, "Key gap").
  - Both systems need a real prompt source: a clinician-drawn box, metadata, or a detector. Results from either system are optimistic compared with real use.
- **Same base weights.** Slide-SAM's `sam_vit_b_01ec64.pth` is byte-identical to this repo's copy.
- **Anatomy differs.** This project targeted **bone** (femur), while Slide-SAM in Mars targets **abdominal organs** (liver, stomach, left kidney, spleen). Neither targets the **lesions** that `requirements.md` asks for.
- **No direct comparison is possible.** SAM-propagation and Slide-SAM numbers come from different datasets (TotalSegmentator femur vs BTCV organs), different splits and different evaluation scripts.

### 10.3 Ideas from this repo worth carrying into Slide-SAM work
1. **Box jitter as an uncertainty probe needs care.** Roadmap Phase 5 plans "running with slightly shifted boxes". Here, jitter combined with a strict majority vote cut Dice from 0.94 to 0.83 through steady shrinkage. Use the *spread* across jittered runs as an uncertainty signal, not a strict vote as the prediction.
2. **Component filtering** (post-processing 1: keep the component tied to the initial prompt) is model-agnostic. It can clean Slide-SAM volumes too.
3. **The box-shrink stopping rule** (post-processing 2: cut where the box area falls below 10% of the initial box) is an inexpensive extra check alongside Slide-SAM's IoU-based stopping. It is also a useful feature for the uncertainty agent: early stop or fragmentation (roadmap Phase 3, step 4).
4. **Per-slice prompt logs are useful.** Saving every slice's boxes plus `idx_init` and `time` made this archive analysis possible without ground truth. Keep the same habit for Slide-SAM runs.
5. **Engineering lessons:**
   - Write outputs atomically: write to a temp file, fsync, then rename.
   - Validate files on resume rather than only checking that they exist (s0024).
   - Keep seeds fixed. Determinism was confirmed here.
   - Avoid loading the model inside evaluation scripts.
   - Record which label mapping built each dataset.

---

## 11. Deletion checklist

| Item | Safe to delete? | Where it is preserved |
|---|---|---|
| Upstream code (`scr/`, unmodified `run_scripts/*`, README, `run_script_example.sh`) | Yes | GitHub fork plus upstream. README content summarised in Sections 4–5 |
| `run_scripts/prompt_propagation.py` (modified) | Yes | Appendix B.1 (full text) |
| Local scripts and configs | Yes | Appendices B and C (full text). `environment.yml` (conda export) is not reproduced; its pip pins match `requirements.txt` |
| `sam_vit_b_01ec64.pth` (358 MB) | Yes | `~/Mars/Slide-SAM/Slide-SAM/` (same MD5), plus the public URL |
| `results/` (8.4 MB) and `predictions_*_old/` (about 360 KB) | Yes, if the numbers in this report are enough | Section 8 and Appendix A hold the metrics and per-case statistics. **The prediction masks themselves are not kept.** Keep these about 8.8 MB if you might re-score them against regenerated ground truth |
| `PROJECT_SUMMARY.md` | Yes | Appendix D (full text) plus corrections in Section 8.4 |
| conda env `~/.conda/envs/sam_prompt` | Not part of the repo | Remove separately with `conda env remove -n sam_prompt` if you want the space back |
| Nested outer checkout (`~/SAM-Prompt-Placement-by-Propagation/.git`) | Yes | Same commits as the inner clone |

---

## Appendix A: Baseline per-subject prediction statistics

These were computed on 7 Oct 2026 from `results/Dataset101_Bone/baseline/*.nii.gz` and `*_prompts.json`. Label 1, voxel volume 3.375 mm³. "Initial slice" is the random starting slice (`idx_init`). "Run time" is the propagation time recorded in the prompt JSON. No ground-truth metrics are available except for s0000 (Dice 0.9429).

| Subject | Volume shape (x, y, z) | Initial slice | Slices with prediction (z range) | Predicted voxels | Predicted volume (mL) | Max boxes per slice | Slices with >1 box | Run time (s) |
|---|---|---|---|---|---|---|---|---|
| s0000 | 294×192×179 | 59 | 171 (0–170) | 65,181 | 220.0 | 2 | 5 | 21.5 |
| s0001 | 249×188×213 | 15 | 32 (0–31) | 14,668 | 49.5 | 1 | 0 | 24.0 |
| s0004 | 255×177×440 | 31 | 67 (0–66) | 34,004 | 114.8 | 2 | 6 | 48.1 |
| s0006 | 216×216×217 | 17 | 46 (0–45) | 36,332 | 122.6 | 2 | 5 | 24.3 |
| s0009 | 119×119×185 | 4 | 25 (0–24) | 3,704 | 12.5 | 1 | 0 | 20.6 |
| s0010 | 259×259×283 | 29 | 73 (0–72) | 47,043 | 158.8 | 2 | 7 | 32.1 |
| s0011 | 311×311×431 | 39 | 77 (0–76) | 52,386 | 176.8 | 2 | 38 | 48.8 |
| s0012 | 238×145×315 | 46 | 86 (0–85) | 46,482 | 156.9 | 2 | 1 | 35.4 |
| s0013 | 231×175×301 | 49 | 87 (0–86) | 59,937 | 202.3 | 2 | 6 | 34.0 |
| s0014 | 286×286×308 | 40 | 79 (0–78) | 45,204 | 152.6 | 2 | 12 | 35.0 |
| s0015 | 293×293×344 | 29 | 60 (0–59) | 31,684 | 106.9 | 1 | 0 | 38.6 |
| s0016 | 259×259×294 | 23 | 66 (0–65) | 29,337 | 99.0 | 1 | 0 | 33.2 |
| s0019 | 221×221×407 | 50 | 89 (0–88) | 61,654 | 208.1 | 2 | 2 | 45.8 |
| s0022 | 295×295×250 | 12 | 65 (0–64) | 24,657 | 83.2 | 1 | 0 | 28.3 |
| s0024 | — (0-byte file) | 51 | — | — | — | — | — | 55.2 |
| s0028 | 299×299×394 | 17 | 53 (0–52) | 21,440 | 72.4 | 1 | 0 | 43.9 |
| s0029 | 300×300×422 | 56 | 95 (0–94) | 29,734 | 100.4 | 2 | 54 | 47.6 |
| s0030 | 309×309×317 | 41 | 72 (0–71) | 46,985 | 158.6 | 2 | 25 | 36.0 |
| s0031 | 288×288×280 | 40 | 79 (0–78) | 53,475 | 180.5 | 2 | 28 | 32.0 |
| s0034 | 308×308×283 | 27 | 134 (0–133) | 44,479 | 150.1 | 2 | 3 | 32.9 |
| s0038 | 240×240×333 | 36 | 107 (0–106) | 30,509 | 103.0 | 2 | 9 | 38.1 |
| s0040 | 291×291×421 | 17 | 51 (0–50) | 27,771 | 93.7 | 2 | 10 | 47.2 |
| s0042 | 285×285×332 | 51 | 125 (0–124) | 63,758 | 215.2 | 2 | 4 | 38.2 |
| s0045 | 231×231×306 | 47 | 83 (0–82) | 32,770 | 110.6 | 2 | 11 | 34.8 |
| s0049 | 221×221×455 | 52 | 82 (0–81) | 42,002 | 141.8 | 1 | 0 | 51.2 |
| s0050 | 261×182×399 | 32 | 58 (0–57) | 26,271 | 88.7 | 2 | 6 | 44.7 |
| s0052 | 249×249×294 | 41 | 89 (0–88) | 62,996 | 212.6 | 2 | 7 | 33.7 |
| s0053 | 333×333×346 | 31 | 89 (0–88) | 65,061 | 219.6 | 2 | 14 | 39.6 |
| s0054 | 261×261×273 | 23 | 86 (0–85) | 26,792 | 90.4 | 1 | 0 | 31.2 |
| s0058 | 298×298×325 | 47 | 77 (0–76) | 45,302 | 152.9 | 2 | 5 | 37.0 |
| s0059 | 233×233×449 | 18 | 62 (0–61) | 24,672 | 83.3 | 1 | 0 | 50.5 |
| s0062 | 273×205×273 | 106 | 212 (17–228) | 103,272 | 348.5 | 3 | 21 | 32.3 |
| s0065 | 243×243×409 | 28 | 64 (0–63) | 30,653 | 103.5 | 1 | 0 | 46.1 |
| s0066 | 260×202×217 | 24 | 52 (0–51) | 27,847 | 94.0 | 1 | 0 | 24.6 |
| s0068 | 104×137×189 | 14 | 32 (0–31) | 6,279 | 21.2 | 1 | 0 | 21.2 |
| s0072 | 286×286×269 | 38 | 68 (0–67) | 32,761 | 110.6 | 1 | 0 | 30.6 |
| s0073 | 247×245×265 | 10 | 61 (0–60) | 43,883 | 148.1 | 2 | 1 | 30.0 |
| s0074 | 251×192×180 | 57 | 93 (0–92) | 66,004 | 222.8 | 3 | 26 | 20.9 |
| s0075 | 220×363×317 | 33 | 77 (0–76) | 52,929 | 178.6 | 2 | 7 | 35.8 |
| s0076 | 319×319×441 | 30 | 78 (0–77) | 55,582 | 187.6 | 1 | 0 | 49.9 |
| s0077 | 255×255×272 | 22 | 62 (0–61) | 30,337 | 102.4 | 1 | 0 | 30.9 |
| s0078 | 271×271×266 | 35 | 74 (0–73) | 48,638 | 164.2 | 2 | 7 | 30.3 |
| s0082 | 260×205×311 | 17 | 72 (0–71) | 32,381 | 109.3 | 1 | 0 | 35.1 |
| s0086 | 273×430×430 | 25 | 89 (0–88) | 65,838 | 222.2 | 1 | 0 | 48.4 |
| s0089 | 221×221×263 | 15 | 63 (0–62) | 24,109 | 81.4 | 1 | 0 | 29.8 |
| s0090 | 258×158×275 | 34 | 63 (0–62) | 32,349 | 109.2 | 2 | 11 | 31.0 |
| s0091 | 312×312×437 | 53 | 405 (0–404) | 317,469 | 1071.5 | 3 | 56 | 53.3 |
| s0092 | 243×167×203 | 13 | 45 (0–44) | 23,551 | 79.5 | 2 | 3 | 23.0 |
| s0095 | 221×221×413 | 43 | 74 (0–73) | 38,374 | 129.5 | 2 | 1 | 46.6 |
| s0096 | 259×259×310 | 40 | 221 (0–220) | 148,276 | 500.4 | 2 | 7 | 36.8 |
| s0098 | 231×170×231 | 38 | 132 (0–132) | 94,402 | 318.6 | 2 | 16 | 26.7 |
| s0104 | 295×295×202 | 40 | 104 (0–103) | 37,206 | 125.6 | 1 | 0 | 23.5 |
| s0107 | 499×154×333 | 48 | 88 (0–87) | 37,488 | 126.5 | 1 | 0 | 37.3 |
| s0108 | 288×288×293 | 18 | 89 (0–88) | 55,694 | 188.0 | 2 | 30 | 33.6 |
| s0109 | 303×252×303 | 38 | 54 (0–53) | 33,558 | 113.3 | 2 | 6 | 34.1 |
| s0117 | 259×156×266 | 19 | 59 (0–58) | 38,093 | 128.6 | 2 | 3 | 29.8 |
| s0119 | 293×293×343 | 47 | 93 (0–92) | 66,119 | 223.2 | 2 | 8 | 39.0 |
| s0120 | 261×254×355 | 60 | 117 (0–116) | 68,412 | 230.9 | 3 | 5 | 40.5 |
| s0122 | 93×93×83 | 30 | 57 (0–56) | 24,334 | 82.1 | 1 | 0 | 9.8 |
| s0124 | 431×159×431 | 41 | 58 (0–57) | 37,961 | 128.1 | 1 | 0 | 48.0 |
| s0131 | 265×202×245 | 22 | 56 (0–55) | 38,218 | 129.0 | 2 | 6 | 27.5 |
| s0133 | 307×307×301 | 33 | 92 (0–91) | 62,164 | 209.8 | 2 | 14 | 34.5 |
| s0137 | 247×247×265 | 39 | 88 (0–87) | 46,700 | 157.6 | 2 | 41 | 30.5 |
| s0139 | 280×280×285 | 46 | 80 (0–79) | 49,457 | 166.9 | 3 | 4 | 32.5 |
| s0141 | 406×168×406 | 35 | 93 (0–92) | 31,128 | 105.1 | 1 | 0 | 45.5 |
| s0143 | 279×166×267 | 43 | 70 (0–69) | 28,307 | 95.5 | 2 | 18 | 30.2 |
| s0145 | 222×219×447 | 20 | 89 (0–88) | 63,447 | 214.1 | 2 | 7 | 50.5 |
| s0147 | 220×310×310 | 43 | 132 (0–131) | 68,000 | 229.5 | 1 | 0 | 35.5 |
| s0150 | 247×247×296 | 54 | 91 (0–90) | 54,963 | 185.5 | 2 | 8 | 33.8 |
| s0151 | 292×184×292 | 60 | 80 (8–87) | 45,190 | 152.5 | 2 | 14 | 33.0 |
| s0153 | 248×195×299 | 11 | 49 (0–48) | 31,784 | 107.3 | 2 | 4 | 33.6 |
| s0157 | 314×314×261 | 28 | 66 (0–65) | 25,900 | 87.4 | 1 | 0 | 29.7 |
| s0158 | 253×253×271 | 17 | 50 (0–49) | 24,409 | 82.4 | 1 | 0 | 30.6 |
| s0161 | 246×181×300 | 24 | 55 (0–54) | 39,603 | 133.7 | 1 | 0 | 33.7 |
| s0163 | 291×291×440 | 37 | 105 (0–104) | 64,377 | 217.3 | 2 | 14 | 50.0 |
| s0166 | 200×321×321 | 42 | 68 (0–67) | 44,024 | 148.6 | 2 | 23 | 36.2 |
| s0168 | 229×229×280 | 49 | 91 (0–90) | 44,961 | 151.7 | 2 | 24 | 32.1 |

---

## Appendix B: Local-only source files (verbatim)

These files are not on GitHub and would be lost on deletion.

### B.1 run_scripts/prompt_propagation.py (locally modified version)

Replaces the upstream file. Differences are described in Section 6.

~~~~python
import argparse
import json
import logging
import os
import random

import nibabel as nib
import numpy as np
from natsort import natsorted

from nnunetv2.paths import nnUNet_raw
from scr.SAMPromptPropagation import (
    SAMPredictorVolumeBaseline,
    SAMPredictorVolumeStochastic,
    SAMPredictorVolumeNested,
    SAMPredictorVolumeNestedStochastic,
)

logging.basicConfig(level=logging.INFO)

np.random.seed(1)
random.seed(1)


def get_random_init_slice(idx_first, idx_last, percent_middle):
    r = int((idx_last - idx_first + 1) * percent_middle)
    list_choices = list(
        range(
            (idx_last - idx_first) // 2 - r // 2,
            (idx_last - idx_first) // 2 + r // 2 + 1,
        )
    )
    if len(list_choices) == 0:
        choice = (idx_last - idx_first) // 2
    else:
        choice = random.choice(list_choices)
    return choice + idx_first


def get_first_last_slice(mask_binary):
    idx_first = next(idx for idx, x in enumerate(mask_binary) if np.sum(x) != 0)
    idx_last = len(mask_binary) - 1 - next(
        idx for idx, x in enumerate(reversed(mask_binary)) if np.sum(x) != 0
    )
    return idx_first, idx_last


def propagate_prompt_per_class(
    predictor,
    class_value,
    experiment,
    percent_middle,
    slice_select=None,
):
    mask_binary = np.array(
        predictor.mask_reference_full == class_value,
        dtype=np.uint8,
    )

    if np.sum(mask_binary) == 0:
        logging.info(
            f"no SAM predicts volume for single class, since {class_value} has no pixels"
        )
        return None, None

    idx_first, idx_last = get_first_last_slice(mask_binary)

    if experiment == "baseline":
        if slice_select == "random":
            init_slice = get_random_init_slice(
                idx_first,
                idx_last,
                percent_middle,
            )
        elif slice_select == "first":
            init_slice = idx_first
        elif slice_select == "last":
            init_slice = idx_last
        else:
            logging.info(f"{slice_select} not valid")
            return None, None
    else:
        init_slice = get_random_init_slice(
            idx_first,
            idx_last,
            percent_middle,
        )

    predictor.select_init_slice(init_slice, class_value)

    pred_mask, prompts = predictor.predict_volume()

    pred_mask = np.moveaxis(pred_mask, 0, -1)
    pred_mask[pred_mask == 1] = class_value
    prompts["idx_init"] = init_slice

    return {class_value: pred_mask}, {class_value: prompts}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--label_order", required=True, nargs="+")
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--experiment", required=True)

    parser.add_argument("--margin", default=0, type=int)
    parser.add_argument("--remove_small_boxes", default=1, type=int)
    parser.add_argument("--percentage", default=0.5, type=float)
    parser.add_argument("--iterations", default=10, type=int)
    parser.add_argument("--max_margin", default=5, type=int)
    parser.add_argument("--majority_vote", default=0.55, type=float)
    parser.add_argument("--merge_results", default=0, type=int)
    parser.add_argument("--nnunet_suffix", default="Tr")
    parser.add_argument("--slice_select", default="random")

    args = parser.parse_args()

    dataset = "Dataset" + args.dataset

    init_path = os.path.join(nnUNet_raw, dataset)

    input_img_dir = os.path.join(
        init_path,
        f"images{args.nnunet_suffix}",
    )

    input_seg_dir = os.path.join(
        init_path,
        f"labels{args.nnunet_suffix}",
    )

    output_dir = args.output_folder
    os.makedirs(output_dir, exist_ok=True)

    mask_list = natsorted(os.listdir(input_seg_dir))

    class_labels = [int(x) for x in args.label_order]

    logging.info(f"Dataset: {dataset}")
    logging.info(f"Number of cases: {len(mask_list)}")

    for im_idx, im_name in enumerate(mask_list):

        logging.info(f"[{im_idx+1}/{len(mask_list)}] {im_name}")

        output_file = os.path.join(
            output_dir,
            f"{im_name[:-7]}.nii.gz",
        )

        prompt_file = os.path.join(
            output_dir,
            f"{im_name[:-7]}_prompts.json",
        )

        # Skip already processed cases
        if os.path.exists(output_file) and os.path.exists(prompt_file):
            print(f"Skipping {im_name}")
            continue

        # Predictor
        if args.experiment == "baseline":
            predictor = SAMPredictorVolumeBaseline(
                margin=args.margin,
                remove_small_boxes=args.remove_small_boxes,
            )

        elif args.experiment == "stochastic":
            predictor = SAMPredictorVolumeStochastic(
                margin=args.margin,
                iterations=args.iterations,
                max_margin=args.max_margin,
                remove_small_boxes=args.remove_small_boxes,
                majority_vote=args.majority_vote,
            )

        elif args.experiment == "nested":
            predictor = SAMPredictorVolumeNested(
                margin=args.margin,
                aggregate=args.merge_results,
                remove_small_boxes=args.remove_small_boxes,
            )

        elif args.experiment == "combined":
            predictor = SAMPredictorVolumeNestedStochastic(
                margin=args.margin,
                iterations=args.iterations,
                max_margin=args.max_margin,
                remove_small_boxes=args.remove_small_boxes,
                majority_vote=args.majority_vote,
                aggregate=args.merge_results,
            )

        else:
            raise ValueError("Unknown experiment")

        predictor.read_volume(
            os.path.join(
                input_img_dir,
                im_name.replace(".nii", "_0000.nii"),
            )
        )

        predictor.read_reference(
            os.path.join(
                input_seg_dir,
                im_name,
            )
        )

        if not predictor.check_for_empty_reference(class_labels):
            logging.info("No labels present.")
            continue

        results_mask = []
        results_prompt = []

        for cls in class_labels:

            mask, prompts = propagate_prompt_per_class(
                predictor,
                cls,
                args.experiment,
                args.percentage,
                args.slice_select,
            )

            if mask is not None:
                results_mask.append(mask)
                results_prompt.append(prompts)

        full_output_mask = np.zeros_like(
            np.moveaxis(
                predictor.mask_reference_full,
                0,
                -1,
            ),
            dtype=np.uint8,
        )

        for cls in class_labels:
            for elem in results_mask:
                for c, mask in elem.items():
                    if c == cls:
                        full_output_mask[mask == cls] = cls

        full_prompts = {
            k: v
            for res in results_prompt
            for k, v in res.items()
        }

        nii = nib.Nifti1Image(
            full_output_mask.astype(np.uint8),
            affine=predictor.affine,
        )

        nib.save(nii, output_file)

        with open(prompt_file, "w") as f:
            json.dump(full_prompts, f)
            f.flush()
            os.fsync(f.fileno())

        print(f"Saved {im_name}")

    print("Finished.")
~~~~

### B.2 evaluation/nnunet_evaluation_connector.py

Also create an empty `evaluation/__init__.py`. It defines only the Dice/IoU function (see Section 7.1).

~~~~python
import numpy as np


def compute_metrics_like_nnunet_from_arrays_simple(reference, prediction):
    reference = reference.astype(bool)
    prediction = prediction.astype(bool)

    tp = np.logical_and(reference, prediction).sum()
    fp = np.logical_and(~reference, prediction).sum()
    fn = np.logical_and(reference, ~prediction).sum()

    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return {
        "Dice": float(dice),
        "IoU": float(iou),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
    }
~~~~

### B.3 convert_labels.py

`scripts/convert_labels.py` is identical except for `INPUT_ROOT = Path("data/Totalsegmentator_dataset_v201")`.

~~~~python
"""
Convert TotalSegmentator data to nnU-Net format.

For each patient:
1. Copy ct.nii.gz to imagesTr
2. Merge the required segmentation masks into one label image
3. Save the merged label in labelsTr
"""

import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


# Input and output directories

INPUT_ROOT = Path("data/TotalSegmentator")
OUTPUT_ROOT = Path("data/nnUNet_raw/Dataset101_Bone")


# Labels to include in the merged mask

LABELS = {
    "femur_left": 1,
    "femur_right": 2,
}

# Example for the shoulder dataset:
#
# LABELS = {
#     "scapula_left": 1,
#     "scapula_right": 2,
#     "humerus_left": 3,
#     "humerus_right": 4,
# }


# Create the nnU-Net folders if they don't exist

images_dir = OUTPUT_ROOT / "imagesTr"
labels_dir = OUTPUT_ROOT / "labelsTr"

images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)


# Find all patient folders

cases = sorted([case for case in INPUT_ROOT.iterdir() if case.is_dir()])

print(f"Found {len(cases)} cases")


# Process each patient

for case in cases:

    print(f"\nProcessing {case.name}")

    # Copy the CT image

    ct_file = case / "ct.nii.gz"
    destination_ct = images_dir / f"{case.name}_0000.nii.gz"

    shutil.copy2(ct_file, destination_ct)

    # Load one segmentation file to get the image size

    reference_image = None

    for organ in LABELS:

        organ_file = case / "segmentations" / f"{organ}.nii.gz"

        if organ_file.exists():
            reference_image = nib.load(organ_file)
            break

    if reference_image is None:
        print("No required labels found. Skipping this case.")
        continue

    # Create an empty label image

    merged_label = np.zeros(reference_image.shape, dtype=np.uint8)

    # Add each selected organ to the merged label image

    for organ, label_value in LABELS.items():

        organ_file = case / "segmentations" / f"{organ}.nii.gz"

        if not organ_file.exists():
            print(f"Missing segmentation: {organ}")
            continue

        mask = nib.load(organ_file).get_fdata()

        merged_label[mask > 0] = label_value

        print(f"Added {organ} as label {label_value}")

    # Save the merged label image

    output_label = nib.Nifti1Image(
        merged_label,
        affine=reference_image.affine,
        header=reference_image.header,
    )

    output_path = labels_dir / f"{case.name}.nii.gz"

    nib.save(output_label, output_path)

    print(f"Saved {output_path.name}")

print("\nConversion completed successfully.")
print(f"Images directory : {images_dir}")
print(f"Labels directory : {labels_dir}")
~~~~

### B.4 scripts/untitled.py

Alternative converter: one binary nnU-Net dataset per TotalSegmentator label. Not known to have been run.

~~~~python
import os
import shutil

# Path to the extracted TotalSegmentator dataset
SOURCE_ROOT = "data/Totalsegmentator_dataset_v201"

# Where the nnUNet datasets will be created
OUTPUT_ROOT = "data/nnUNet_raw"

# Find every case folder
case_folders = sorted(
    folder for folder in os.listdir(SOURCE_ROOT)
    if os.path.isdir(os.path.join(SOURCE_ROOT, folder))
)

# Find every unique segmentation name
all_labels = set()

for case in case_folders:
    segmentation_dir = os.path.join(SOURCE_ROOT, case, "segmentations")

    if not os.path.isdir(segmentation_dir):
        continue

    for file in os.listdir(segmentation_dir):
        if file.endswith(".nii.gz"):
            all_labels.add(file)

all_labels = sorted(all_labels)

print(f"Found {len(all_labels)} labels")

# Create one nnUNet dataset for every label
for dataset_id, label_file in enumerate(all_labels, start=1):

    label_name = label_file.replace(".nii.gz", "")
    dataset_name = f"Dataset{dataset_id:03d}_{label_name}"

    print(f"Creating {dataset_name}")

    images_dir = os.path.join(OUTPUT_ROOT, dataset_name, "imagesTr")
    labels_dir = os.path.join(OUTPUT_ROOT, dataset_name, "labelsTr")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    copied_cases = 0

    for case in case_folders:

        ct_path = os.path.join(
            SOURCE_ROOT,
            case,
            "ct.nii.gz"
        )

        label_path = os.path.join(
            SOURCE_ROOT,
            case,
            "segmentations",
            label_file
        )

        if not os.path.exists(label_path):
            continue

        image_output = os.path.join(
            images_dir,
            f"{case}_0000.nii.gz"
        )

        label_output = os.path.join(
            labels_dir,
            f"{case}.nii.gz"
        )

        shutil.copy2(ct_path, image_output)
        shutil.copy2(label_path, label_output)

        copied_cases += 1

    print(f"  Copied {copied_cases} cases")

print("\nDone!")
~~~~

### B.5 run_all_models.py

Runs all four experiments and the Dice/IoU evaluation for every `Dataset*` folder, with label 1.

~~~~python
import os
import subprocess

NNUNET_RAW = "data/nnUNet_raw"

EXPERIMENTS = [
    "baseline",
    "stochastic",
    "nested",
    "combined"
]

dataset_folders = sorted(
    folder for folder in os.listdir(NNUNET_RAW)
    if folder.startswith("Dataset")
)

print(f"Found {len(dataset_folders)} datasets\n")

for dataset in dataset_folders:

    dataset_id = dataset.replace("Dataset", "")

    print("=" * 80)
    print(f"Processing {dataset}")
    print("=" * 80)

    for experiment in EXPERIMENTS:

        output_folder = os.path.join(
            "results",
            dataset,
            experiment
        )

        os.makedirs(output_folder, exist_ok=True)

        print(f"\nRunning {experiment}")

        propagation_cmd = [
            "python",
            "-m",
            "run_scripts.prompt_propagation",
            "--dataset",
            dataset_id,
            "--label_order",
            "1",
            "--output_folder",
            output_folder,
            "--experiment",
            experiment
        ]

        result = subprocess.run(propagation_cmd)

        if result.returncode != 0:
            print(f"Propagation failed for {dataset} ({experiment})")
            continue

        print("Running evaluation")

        evaluation_cmd = [
            "python",
            "-m",
            "run_scripts.prompt_propagation_evaluation",
            "--dataset",
            dataset_id,
            "--label_order",
            "1",
            "--output_folder",
            output_folder
        ]

        result = subprocess.run(evaluation_cmd)

        if result.returncode != 0:
            print(f"Evaluation failed for {dataset} ({experiment})")
            continue

        print("Done")

print("\nAll experiments finished.")
~~~~

### B.6 setup.sh

Creates the `sam_prompt` conda env and appends the nnU-Net variables to `~/.bashrc`.

~~~~bash
#!/bin/bash


ENV_NAME="sam_prompt"

# -------------------------------------------------
# Create environment (only if it doesn't exist)
# -------------------------------------------------

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Creating conda environment: $ENV_NAME"
    conda create -y -n "$ENV_NAME" python=3.10
else
    echo "Conda environment '$ENV_NAME' already exists."
fi

conda activate "$ENV_NAME"

# -------------------------------------------------
# Install PyTorch (only if missing)
# -------------------------------------------------

if ! python -c "import torch" >/dev/null 2>&1; then
    echo "Installing PyTorch..."

    pip install \
        torch==2.5.1 \
        torchvision==0.20.1 \
        torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121
else
    echo "PyTorch already installed."
fi

# -------------------------------------------------
# Install project dependencies
# -------------------------------------------------

echo "Installing project requirements..."

pip install -r requirements_new.txt

# -------------------------------------------------
# Install Segment Anything (only if missing)
# -------------------------------------------------

if ! python -c "import segment_anything" >/dev/null 2>&1; then
    echo "Installing Segment Anything..."

    pip install git+https://github.com/facebookresearch/segment-anything.git
else
    echo "Segment Anything already installed."
fi

# -------------------------------------------------
# nnU-Net environment variables
# -------------------------------------------------

REPO_DIR="$(pwd)"

NNUNET_RAW="$REPO_DIR/data/nnUNet_raw"
NNUNET_PREPROCESSED="$REPO_DIR/data/nnUNet_preprocessed"
NNUNET_RESULTS="$REPO_DIR/data/nnUNet_results"

echo ""
echo "Setting nnU-Net environment variables..."

grep -qxF "export nnUNet_raw=$NNUNET_RAW" ~/.bashrc || \
echo "export nnUNet_raw=$NNUNET_RAW" >> ~/.bashrc

grep -qxF "export nnUNet_preprocessed=$NNUNET_PREPROCESSED" ~/.bashrc || \
echo "export nnUNet_preprocessed=$NNUNET_PREPROCESSED" >> ~/.bashrc

grep -qxF "export nnUNet_results=$NNUNET_RESULTS" ~/.bashrc || \
echo "export nnUNet_results=$NNUNET_RESULTS" >> ~/.bashrc

export nnUNet_raw="$NNUNET_RAW"
export nnUNet_preprocessed="$NNUNET_PREPROCESSED"
export nnUNet_results="$NNUNET_RESULTS"

# -------------------------------------------------
# Verify installation
# -------------------------------------------------

echo ""
echo "Verifying installation..."

python - <<EOF
import torch

print("Torch version :", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version  :", torch.version.cuda)

try:
    import segment_anything
    print("Segment Anything: OK")
except Exception as e:
    print("Segment Anything:", e)

try:
    import nnunetv2
    print("nnUNet: OK")
except Exception as e:
    print("nnUNet:", e)
EOF

echo ""
echo "======================================="
echo "Setup complete!"
echo "======================================="
echo ""
echo "If this is the first setup, run:"
echo "    source ~/.bashrc"
echo ""
echo "Current nnU-Net paths:"
echo "nnUNet_raw          = $nnUNet_raw"
echo "nnUNet_preprocessed = $nnUNet_preprocessed"
echo "nnUNet_results      = $nnUNet_results"
~~~~

### B.7 env_setup.sh



~~~~bash
export nnUNet_raw=$PWD/data/nnUNet_raw
export nnUNet_preprocessed=$PWD/data/nnUNet_preprocessed
export nnUNet_results=$PWD/data/nnUNet_results
~~~~

---

## Appendix C: Dependency pins (verbatim)

### C.1 requirements.txt (frozen from the working env)



~~~~text
acvl_utils==0.2.6
annotated-types==0.7.0
anyio==4.14.2
batchgenerators==0.25.3
batchgeneratorsv2==0.3.5
blosc2==4.3.3
certifi==2026.6.17
charset-normalizer==3.4.9
click==8.4.2
connected-components-3d==4.0.0
contourpy==1.3.2
cuda-bindings==13.3.1
cuda-pathfinder==1.5.6
cuda-toolkit==13.0.3.0
cycler==0.12.1
dynamic_network_architectures==0.4.4
einops==0.8.2
exceptiongroup==1.3.1
filelock==3.31.0
fonttools==4.63.0
fsspec==2026.6.0
future==1.0.0
graphviz==0.21
h11==0.16.0
hf-xet==1.5.2
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.24.0
idna==3.18
imagecodecs==2025.3.30
ImageIO==2.37.3
importlib_resources==7.1.0
Jinja2==3.1.6
joblib==1.5.3
kiwisolver==1.5.0
lazy-loader==0.5
linecache2==1.0.0
MarkupSafe==3.0.3
matplotlib==3.10.9
mpmath==1.3.0
msgpack==1.2.1
natsort==8.4.0
ndindex==1.10.1
networkx==3.4.2
nibabel==5.4.2
nnunetv2==2.8.1
numexpr==2.14.1
numpy==2.2.6
nvidia-cublas==13.1.1.3
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti==13.0.85
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime==13.0.96
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==9.1.0.70
nvidia-cudnn-cu13==9.20.0.48
nvidia-cufft==12.0.0.61
nvidia-cufft-cu12==11.0.2.54
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver==12.0.4.66
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse==12.6.3.3
nvidia-cusparse-cu12==12.1.0.106
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu12==2.21.5
nvidia-nccl-cu13==2.29.7
nvidia-nvjitlink==13.3.33
nvidia-nvjitlink-cu12==12.9.86
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
nvidia-nvtx-cu12==12.1.105
opencv-python-headless==5.0.0.93
packaging 
pandas==2.3.3
pillow==12.3.0
pydantic==2.13.4
pydantic_core==2.46.4
pyparsing==3.3.2
python-dateutil==2.9.0.post0
pytz==2026.2
PyYAML==6.0.3
requests==2.34.2
safetensors==0.8.0
scikit-image==0.25.2
scikit-learn==1.7.2
scipy==1.15.3
seaborn==0.13.2
segment_anything @ git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf
simpleitk==2.5.5
six==1.17.0
sympy==1.13.1
threadpoolctl==3.6.0
tifffile==2025.5.10
timm==1.0.22
torch==2.5.1+cu121
torchaudio==2.5.1+cu121
torchvision==0.20.1+cu121
tqdm==4.69.0
traceback2==1.4.0
triton==3.1.0
typing-inspection==0.4.2
typing_extensions==4.16.0
tzdata==2026.3
unittest2==1.1.0
urllib3==2.7.0
yacs==0.1.8

~~~~

### C.2 requirements_new.txt (used by setup.sh)



~~~~text
acvl_utils==0.2.6
annotated-types==0.7.0
anyio==4.12.0
batchgenerators==0.25.3
batchgeneratorsv2==0.3.5
blosc2==4.9.1
certifi==2026.6.17
charset-normalizer==3.4.9
click==8.4.2
connected-components-3d==4.0.0
contourpy==1.3.3
cycler==0.12.1
dynamic_network_architectures==0.4.4
einops==0.8.2
filelock==3.31.0
fonttools==4.63.0
fsspec==2026.6.0
future==1.0.0
gdown==6.1.0
graphviz==0.21
h11==0.16.0
hf-xet==1.5.2
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.24.0
idna==3.18
imagecodecs==2026.6.26
imageio==2.37.3
Jinja2==3.1.6
joblib==1.5.3
kiwisolver==1.5.0
lazy-loader==0.5
linecache2==1.0.0
MarkupSafe==3.0.3
matplotlib==3.11.1
MedPy==0.5.2
monai==1.6.0
mpmath==1.3.0
msgpack==1.2.1
natsort
ndindex==1.10.1
networkx==3.6.1
nibabel
nnunetv2
numexpr==2.14.2
numpy==2.5.1
opencv-python-headless
packaging
pandas==2.3.3
pillow==12.3.0
pydantic==2.12.5
pydantic_core==2.41.5
pyparsing==3.3.2
python-dateutil==2.9.0.post0
PyYAML==6.0.3
requests==2.34.2
safetensors==0.8.0
scikit-image==0.26.0
scikit-learn==1.9.0
scipy==1.18.0
seaborn==0.13.2
SimpleITK==2.5.5
six==1.17.0
sympy==1.13.1
tensorboardX==2.6.5
threadpoolctl==3.6.0
tifffile==2026.7.14
timm==1.0.22
torch==2.5.1+cu121
torchaudio==2.5.1+cu121
torchvision==0.20.1+cu121
tqdm==4.69.0
traceback2==1.4.0
triton==3.1.0
typing-inspection==0.4.2
typing_extensions==4.15.0
tzdata==2026.3
unittest2==1.1.0
urllib3==2.7.0
yacs==0.1.8
~~~~

---

## Appendix D: Previous archive note, PROJECT_SUMMARY.md (verbatim, 26 Sep 2026)

Kept for the record. **Several claims are wrong; see Section 8.4.**

~~~~markdown
# SAM Box Prompt Placement by Propagation — Project Summary

**Paper:** [Training-free Prompt Placement by Propagation for SAM Predictions in Bone CT Scans](https://openreview.net/forum?id=F6rhgpGkAy)
**Venue:** MIDL (Medical Imaging with Deep Learning)
**Date archived:** 2026-09-26

---

## Overview

This project implements four training-free strategies for applying Segment Anything Model (SAM)
with box prompts to 3D bone CT scans. All methods are initialized with only two annotated pixels,
significantly reducing annotation cost compared to slice-by-slice SAM or fully supervised segmentation.

The pipeline uses SAM (ViT-B) for prediction and nnUNet for data structure and evaluation.

---

## Dataset

### TotalSegmentator (raw source)
- **Size:** ~31 GB (1,229 CT scan folders, s0000–s1371)
- **Format:** Per-subject folders containing `ct.nii.gz` + `segmentations/` directory
- **License:** Public, freely available
- **Download:**
  ```bash
  # Option 1 — via Zenodo (v2, DOI: 10.5281/zenodo.6802613)
  pip install zenodo-get
  zenodo_get 6802613

  # Option 2 — via TotalSegmentator package (downloads on first use)
  pip install TotalSegmentator
  TotalSegmentator -i ct.nii.gz -o seg/

  # Option 3 — direct Zenodo URL
  # https://zenodo.org/record/6802613
  ```

### nnUNet Dataset101_Bone (derived)
- **Size:** ~22 GB (1,228 subjects)
- **Format:** nnUNet v2 structure — `imagesTr/` (22 GB) + `labelsTr/` (96 MB)
- **Source:** Converted from TotalSegmentator using nnUNet dataset conversion
- **Regeneration:** Run the conversion script after re-downloading TotalSegmentator:
  ```bash
  # Set nnUNet environment variables first (see setup.sh)
  export nnUNet_raw="$(pwd)/data/nnUNet_raw"
  export nnUNet_preprocessed="$(pwd)/data/nnUNet_preprocessed"
  export nnUNet_results="$(pwd)/data/nnUNet_results"

  # Convert TotalSegmentator → nnUNet Dataset101_Bone
  # (refer to data conversion scripts in scr/ or scripts/ directory)
  ```

> **Note:** Both TotalSegmentator and nnUNet_raw were deleted on 2026-09-26 to free ~53 GB of disk
> space. They are fully regenerable from the public Zenodo source above.

---

## Experiments

Four propagation strategies were implemented and evaluated on Dataset101_Bone (bone segmentation):

| Experiment | Description | Subjects Evaluated |
|---|---|---|
| **baseline** | Deterministic box prompt propagation from center slice | 77 |
| **stochastic** | Stochastic/randomized prompt placement | 1 |
| **nested** | Nested propagation strategy | 0 (not run) |
| **combined** | Combined strategy | 0 (not run) |

---

## Results

### Baseline — Dataset101_Bone (77 subjects)

Best single-subject result (s0000, label 1 — bone):

| Metric | Value |
|---|---|
| **Dice** | **0.9429** |
| **IoU** | **0.8920** |
| TP | 63,426 |
| FP | 1,755 |
| FN | 5,923 |

Full per-subject results are stored in:
```
results/Dataset101_Bone/baseline/    # 77 x .nii.gz predictions + _prompts.json
results/Dataset101_Bone/stochastic/  # 1 subject (s0000)
```

The `Dataset101_Bone_log.json` in `results/Dataset101_Bone/baseline/` contains the
structured metrics for evaluated subjects.

### Stochastic — Dataset101_Bone (1 subject)
- Only s0000 was evaluated under the stochastic experiment.
- Results stored in `results/Dataset101_Bone/stochastic/`.

### Nested / Combined
- Output directories exist but are empty — these experiments were not run to completion.

---

## Repository Structure (post-cleanup)

```
SAM-Prompt-Placement-by-Propagation/
├── README.md                        # Original project README
├── PROJECT_SUMMARY.md               # This file
├── setup.sh                         # Environment setup script
├── requirements.txt / requirements_new.txt
├── run_script_example.sh            # Example run commands
├── sam_vit_b_01ec64.pth             # SAM ViT-B weights (358 MB, re-downloadable)
├── run_scripts/                     # Main experiment scripts
│   ├── prompt_propagation.py
│   ├── prompt_propagation_evaluation.py
│   ├── prompt_propagation_evaluation2.py
│   ├── prompt_propagation_postprocessing.py
│   ├── prompt_propagation_postprocessing2.py
│   ├── prompt_reference.py
│   └── create_box_prompt_visualizations.py
├── scripts/                         # Utility scripts
├── scr/                             # Additional source files
├── evaluation/                      # Evaluation utilities
├── results/                         # *** KEPT — experiment outputs (8.4 MB) ***
│   └── Dataset101_Bone/
│       ├── baseline/                # 77 subjects: predictions + prompt JSONs
│       ├── stochastic/              # 1 subject
│       ├── nested/                  # empty
│       └── combined/                # empty
├── predictions_baseline_old/        # Legacy predictions (3 files)
├── predictions_stochastic_old/
├── predictions_nested_old/
├── predictions_combined_old/
└── data/                            # *** DELETED — regenerable datasets ***
    ├── TotalSegmentator/            # DELETED (31 GB) — re-download from Zenodo
    └── nnUNet_raw/                  # DELETED (22 GB) — regenerate from TotalSegmentator
```

---

## How to Reproduce

### 1. Setup environment
```bash
bash setup.sh
source ~/.bashrc
```

### 2. Download SAM weights
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 3. Re-download TotalSegmentator data
```bash
pip install zenodo-get
zenodo_get 6802613 -o data/TotalSegmentator
```

### 4. Convert to nnUNet format
```bash
# Follow nnUNet dataset conversion for Dataset101_Bone
# See: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md
```

### 5. Run experiments
```bash
# Baseline
python -m run_scripts.prompt_propagation \
    --dataset 101 \
    --label_order 1 \
    --output_folder results/Dataset101_Bone/baseline \
    --experiment baseline

# Evaluate
python -m run_scripts.prompt_propagation_evaluation \
    --dataset 101 \
    --label_order 1 \
    --output_folder results/Dataset101_Bone/baseline
```

---

## Notes

- SAM model: `vit_b` (ViT-Base), weights: `sam_vit_b_01ec64.pth`
- nnUNet dataset ID: `101`, name: `Dataset101_Bone`
- Bone label order used: `1` (based on `label_order` in run configs)
- The `run_script_example.sh` uses dataset `500_TotalSegmentatorBoneSimple` (original dev reference) — adapt dataset ID to `101` for this setup
- `combined` and `nested` experiments require completing the full pipeline including postprocessing steps (see `run_script_example.sh`)

~~~~

---

## Appendix E: Output file formats

Each case produces `<case>.nii.gz`, a uint8 mask with class values and the input affine, and `<case>_prompts.json`. The JSON is keyed by class (as a string). For each class it holds absolute slice index → list of `[x0, y0, x1, y1]` boxes, or `null` where no prompt was made, plus `"time"` (seconds) and `"idx_init"`. The reference mode also writes `"idx_last"`. Excerpt from baseline s0000, slices 58–60:

~~~~json
{"1": {"58": [[59, 49, 78, 67]], "59": [[58, 49, 79, 67]], "60": [[59, 49, 78, 67]], "...": "...", "time": 21.45920181274414, "idx_init": 59}}
~~~~

The evaluation log `Dataset<ID>_log.json` is shaped `{"<case>.nii.gz": {"<class>": {"Dice", "IoU", "TP", "FP", "FN"}}}`.

### E.1 Upstream example pipeline (run_script_example.sh)

The original author's end-to-end example, showing step order (reference → propagate → evaluate → post-process 1 → evaluate → post-process 2 → evaluate). This file is also on GitHub.

~~~~bash
#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python -m run_scripts.prompt_reference --dataset 500_TotalSegmentatorBoneSimple --label_order 2 3 4 5 \
   --output_folder /home/caroline/data/sam/test/reference --nnunet_suffix Ts
CUDA_VISIBLE_DEVICES=7 python -m run_scripts.prompt_propagation --dataset 500_TotalSegmentatorBoneSimple --label_order 2 3 4 5 \
   --output_folder /home/caroline/data/sam/test/nested_default --experiment nested --nnunet_suffix Ts
CUDA_VISIBLE_DEVICES=7 python -m run_scripts.prompt_propagation_evaluation --dataset 500_TotalSegmentatorBoneSimple --label_order 2 3 4 5 \
   --output_folder /home/caroline/data/sam/test/nested_default --nnunet_suffix Ts
CUDA_VISIBLE_DEVICES=7 python -m run_scripts.prompt_propagation_evaluation2 --dataset 500_TotalSegmentatorBoneSimple --label_order 2 3 4 5 \
   --output_folder /home/caroline/data/sam/test/nested_default --nnunet_suffix Ts
CUDA_VISIBLE_DEVICES=7 python -m run_scripts.create_box_prompt_visualizations --dataset 500_TotalSegmentatorBoneSimple --label_order 2 3 4 5 \
   --output_folder /home/caroline/data/sam/test/nested_default --nnunet_suffix Ts
CUDA_VISIBLE_DEVICES=7 python -m run_scripts.prompt_propagation_postprocessing --dataset 500_TotalSegmentatorBoneSimple --label_order 2 3 4 5 \
   --input_folder /home/caroline/data/sam/test/nested_default --output_folder /home/caroline/data/sam/test/nested_default_pp
CUDA_VISIBLE_DEVICES=7 python -m run_scripts.prompt_propagation_evaluation --dataset 500_TotalSegmentatorBoneSimple --label_order 2 3 4 5 \
   --output_folder /home/caroline/data/sam/test/nested_default_pp --nnunet_suffix Ts
CUDA_VISIBLE_DEVICES=7 python -m run_scripts.prompt_propagation_evaluation2 --dataset 500_TotalSegmentatorBoneSimple --label_order 2 3 4 5 \
   --output_folder /home/caroline/data/sam/test/nested_default_pp --nnunet_suffix Ts
CUDA_VISIBLE_DEVICES=7 python -m run_scripts.prompt_propagation_postprocessing2 --dataset 500_TotalSegmentatorBoneSimple --label_order 2 3 4 5 \
   --input_folder /home/caroline/data/sam/test/nested_default_pp --input_folder_prompt /home/caroline/data/sam/test/nested_default \
   --output_folder /home/caroline/data/sam/test/nested_default_pp2
CUDA_VISIBLE_DEVICES=7 python -m run_scripts.prompt_propagation_evaluation --dataset 500_TotalSegmentatorBoneSimple --label_order 2 3 4 5 \
    --output_folder /home/caroline/data/sam/test/nested_default_pp2 --nnunet_suffix Ts
CUDA_VISIBLE_DEVICES=7 python -m run_scripts.prompt_propagation_evaluation2 --dataset 500_TotalSegmentatorBoneSimple --label_order 2 3 4 5 \
    --output_folder /home/caroline/data/sam/test/nested_default_pp2 --nnunet_suffix Ts

~~~~
