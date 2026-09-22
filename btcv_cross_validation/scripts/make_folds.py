"""
Step 1: Generate 5 BTCV-only, volume-level train/test manifests.

- Reads the real volume list from datasets/01_BCV-Abdomen/Training/img/*.nii.gz
  (read-only; nothing under the main repo is modified).
- Writes manifests only under btcv_cross_validation/dataset_list/.
- BTCV-only: no WORD_50_train.txt is created anywhere in this folder.
"""
import os
import glob

HERE = os.path.dirname(os.path.abspath(__file__))          # .../btcv_cross_validation/scripts
CV_ROOT = os.path.dirname(HERE)                             # .../btcv_cross_validation
MAIN_REPO = os.path.dirname(CV_ROOT)                         # .../Slide-SAM/Slide-SAM
IMG_DIR = os.path.join(MAIN_REPO, "datasets", "01_BCV-Abdomen", "Training", "img")
OUT_DIR = os.path.join(CV_ROOT, "dataset_list")

N_FOLDS = 5

vol_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.nii.gz")))
vol_ids = [os.path.basename(p).replace(".nii.gz", "") for p in vol_paths]  # img0001, img0002, ...
assert len(vol_ids) == 30, f"Expected 30 BTCV volumes, found {len(vol_ids)}: {vol_ids}"

os.makedirs(OUT_DIR, exist_ok=True)

# Full 30-volume pool list (2-column format), used later to build the shared all30 cache.
all30_path = os.path.join(OUT_DIR, "01_BTCV_all30_train.txt")
with open(all30_path, "w") as f:
    for vid in vol_ids:
        num = vid.replace("img", "")
        f.write(f"01_BCV-Abdomen/Training/img/{vid}.nii.gz\t01_BCV-Abdomen/Training/label/label{num}.nii.gz\n")
print(f"Wrote {all30_path} ({len(vol_ids)} volumes)")

fold_size = len(vol_ids) // N_FOLDS  # 6
assert fold_size * N_FOLDS == len(vol_ids), "30 volumes must divide evenly into N_FOLDS folds"

for k in range(N_FOLDS):
    test_ids = vol_ids[k * fold_size: (k + 1) * fold_size]
    train_ids = [v for v in vol_ids if v not in test_ids]
    assert len(train_ids) == 24 and len(test_ids) == 6
    assert set(train_ids).isdisjoint(set(test_ids))

    fold_dir = os.path.join(OUT_DIR, f"fold{k + 1}")
    os.makedirs(fold_dir, exist_ok=True)

    train_path = os.path.join(fold_dir, "01_BTCV_train_4to1.txt")
    with open(train_path, "w") as f:
        f.write("\n".join(f"{vid}.nii.gz" for vid in train_ids) + "\n")

    test_path = os.path.join(fold_dir, "01_BTCV_test_4to1.txt")
    with open(test_path, "w") as f:
        f.write("\n".join(f"{vid}.nii.gz" for vid in test_ids) + "\n")

    print(f"fold{k + 1}: train=24 test={test_ids}")

print("Done ->", OUT_DIR)
print("Note: no WORD_50_train.txt written anywhere (BTCV-only CV).")
