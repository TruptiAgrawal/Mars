# Past Slide-SAM Initial Setup (`SLIDE_SAM.ipynb`)

Report written 7 Oct 2026 about `/home/jupyter-team2pop/SLIDE_SAM.ipynb`.
That notebook is 1.1 MB, has 170 cells and was last modified 2 Sep 2026.

## What the notebook is
It is a working log of the first Slide-SAM setup: cloning, installing, getting the data,
fixing environments and running the first evaluation. It contains no project code that
anything else imports or runs.

## Is anything connected to it?
- **Nothing depends on it.** No `.py`, `.yaml`, shell script or other notebook in the home
  folder references it. The only mentions are its own Jupyter checkpoint copy, JupyterLab
  workspace files and AI-assistant chat logs.
- **It doesn't point to this repo.** Every project path in it uses the old location
  `/home/jupyter-team2pop/Slide-SAM/Slide-SAM/`, which no longer exists. The code now
  lives in `Mars/Slide-SAM/Slide-SAM/`, so the notebook won't re-run as written.
- **What it produced is already in this repo:** `weights/slidesam_b.pth`,
  `datasets/07_WORD/`, `datasets/dataset_list/word_*.txt`, `cached_dataset2/07_WORD/` and
  the config edits.

## Steps it recorded
1. **Clone and inspect:** cloned `github.com/Trupthi1271/Slide-SAM` and looked at the repo
   layout, `core/volume_predictor.py` and the dataset lists.
2. **Weights:** downloaded `weights/slidesam_b.pth` (360 MB) with `gdown` and confirmed it
   loads as a state dict. Copied `sam_vit_b_01ec64.pth` from
   `SAM-Prompt-Placement-by-Propagation/`.
3. **WORD dataset:** downloaded it from Hugging Face (`MedOtter/word`) and arranged it as
   `datasets/07_WORD/WORD-V0.1.0/{imagesTr,labelsTr,imagesVa,labelsVa,imagesTs,labelsTs}`.
4. **Dataset lists:** changed `HOME_PATH` and the save paths in
   `datasets/generate_txt.py`, then ran it, which produced
   `word_train.txt` (100), `word_val.txt` (20) and `word_test.txt` (30).
5. **Configs:** in `configs/vit_b.yaml` and `configs/vit_sub.yaml`, set
   `dataset_list: ["word"]` and pointed `data_root_path`, `cache_data_path` and
   `sam_checkpoint` to local paths.
6. **Cache:** ran `python -m datasets.cache_dataset3d`, which produced
   `cached_dataset2/07_WORD` with 14,159 image slices and 51,369 label slices.
7. **Environment fixes:** see below.
8. **First evaluation:** ran `python -u -m test.volume_eval` in the `slidesam`
   environment. With a box prompt on WORD label 13, the mean Dice was **0.577**.
   Prediction volumes were saved to `visual/07_WORD/`.
9. **Training:** started a first `core.ddp --tag test_train` run at the end.

## Environment problems and fixes
| Problem | Cause | Fix |
|---|---|---|
| `NVIDIA driver on your system is too old (found version 12020)` | Base Python had torch 2.13 + CUDA 13.0, but the driver supports CUDA 12.2 | Use torch built for CUDA 12.1 |
| `undefined symbol: ncclCommResume` | `sam_prompt` env (and base `~/.local`) mixed `nvidia-nccl-cu12` and `nvidia-nccl-cu13` | Stopped using `sam_prompt` and created a clean env |
| `No module named 'core'` | `conda run` didn't run from the repo folder | Run with `bash -c "cd <repo> && ..."` or `conda activate` in a `%%bash` cell |
| `trans-utils` / `tutils` dependency conflicts | pip pulls in conflicting dependencies | `pip install trans-utils==0.4.3.2 --no-deps` |
| `configs/vit_b_103.yaml` not found | `core/volume_predictor.py` `__main__` hard-codes a config that doesn't exist | Use `test.volume_eval` instead |
| `example_train.txt` not found | Config still listed `dataset_list: ["example"]` | Switch to `["word"]` |

### Working environment: `slidesam`
```bash
conda create -n slidesam python=3.10 -y
conda run -n slidesam pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
conda run -n slidesam pip install -r requirements.txt
conda run -n slidesam pip install trans-utils==0.4.3.2 --no-deps
conda run -n slidesam pip install monai==1.4.0 timm einops tensorboardX SimpleITK opencv-python MedPy h5py openpyxl xlwt
```
Checked result: torch 2.5.1+cu121, CUDA available (NVIDIA RTX A6000), MONAI 1.4.0, `tutils` OK.
Set `MPLBACKEND=Agg` when running headless.

## Leftover issue: old hard-coded paths
The config edits from this notebook still point to `/home/jupyter-team2pop/Slide-SAM/Slide-SAM/`,
which no longer exists. These need updating to `/home/jupyter-team2pop/Mars/Slide-SAM/Slide-SAM/`:
- `configs/vit_b.yaml` (lines 125, 131)
- `configs/vit_sub.yaml` (lines 32, 37, 41)
- `configs/vit_sub_btcv_word15.yaml` (lines 31, 36, 40)
- `core/volume_predictor.py` (lines 671, 673)
- `btcv_cross_validation/five_box_padding/predictor.py` (line 685)

## Recommendation
This report captures what's useful in the notebook, so `SLIDE_SAM.ipynb` and
`.ipynb_checkpoints/SLIDE_SAM-checkpoint.ipynb` can be archived or deleted.
