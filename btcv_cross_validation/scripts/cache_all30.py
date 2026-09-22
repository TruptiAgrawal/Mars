"""
Step 3: Build one shared BTCV cache containing all 30 volumes.

CPU/IO-bound only (no CUDA calls in datasets/cache_dataset3d.py's caching path),
so this is safe to run regardless of current GPU load.

Reads raw volumes from the main repo (read-only) using the full 30-volume
manifest generated in Step 1 (dataset_list/01_BTCV_all30_train.txt), and
writes 2D-slice cache (image/label_jpg/meta) only under
btcv_cross_validation/cache/all30/. Nothing in the main repo is modified.

Per the roadmap: caching all 30 volumes into one shared pool is NOT itself
leakage. Leakage would only occur if a fold's training loader (Step 4) pulled
samples belonging to that fold's held-out volumes from this cache. This
script only builds the storage representation.
"""
import os
import sys

sys.path.insert(0, os.getcwd())  # must be run from the Slide-SAM/Slide-SAM repo root

from datasets.cache_dataset3d import Dataset3D

CV_ROOT = "/home/jupyter-team2pop/Slide-SAM/Slide-SAM/btcv_cross_validation"

config = {
    "data_root_path": "/home/jupyter-team2pop/Slide-SAM/Slide-SAM/datasets/",
    "dataset_list": ["01_BTCV_all30"],
    "data_txt_path": os.path.join(CV_ROOT, "dataset_list") + "/",
    "cache_data_path": os.path.join(CV_ROOT, "cache", "all30"),
    "cache_prefix": ["01"],
}

dataset = Dataset3D(config=config, use_cache=False, split="train")
print(f"CACHE BUILD: {len(dataset)} volumes -> {config['cache_data_path']}")
dataset.caching_data()
print("CACHE BUILD DONE.")
