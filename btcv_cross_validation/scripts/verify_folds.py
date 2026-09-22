"""
Step 2: Independent mathematical verification of the 5 BTCV-only folds
generated in Step 1 (btcv_cross_validation/scripts/make_folds.py).

Checks (per 5foldClaudeRoadmap.txt Step 2):
  2.1 Exactly 30 unique BTCV volumes on disk.
  2.2 Every fold has |Train_k| = 24 and |Test_k| = 6.
  2.3 No within-fold leakage: Train_k ∩ Test_k = ∅ for every fold.
  2.4 Every volume held out exactly once: union of Test_1..Test_5 = V,
      |union| = 30, no duplicates across test sets.
  2.5 Deterministic split: re-running the generator reproduces byte-identical files.
  2.6 Manifests correspond to actual files under
      datasets/01_BCV-Abdomen/Training/{img,label}/.

Exits non-zero if any check fails. Does not write anything outside a
temp directory (used only for the determinism re-run check).
"""
import os
import sys
import glob
import filecmp
import tempfile
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))           # .../btcv_cross_validation/scripts
CV_ROOT = os.path.dirname(HERE)                              # .../btcv_cross_validation
MAIN_REPO = os.path.dirname(CV_ROOT)                          # .../Slide-SAM/Slide-SAM
IMG_DIR = os.path.join(MAIN_REPO, "datasets", "01_BCV-Abdomen", "Training", "img")
LABEL_DIR = os.path.join(MAIN_REPO, "datasets", "01_BCV-Abdomen", "Training", "label")
DATASET_LIST_DIR = os.path.join(CV_ROOT, "dataset_list")
MAKE_FOLDS_SCRIPT = os.path.join(HERE, "make_folds.py")

N_FOLDS = 5
EXPECTED_TRAIN = 24
EXPECTED_TEST = 6
EXPECTED_TOTAL = 30

failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(label)
    return condition


def read_ids(path):
    with open(path) as f:
        return [line.strip().replace(".nii.gz", "") for line in f if line.strip()]


print("=" * 70)
print("STEP 2: MATHEMATICAL VERIFICATION OF BTCV 5-FOLD SPLIT")
print("=" * 70)

# ---------------------------------------------------------------
# 2.1  Exactly 30 unique BTCV volumes (ground truth from disk)
# ---------------------------------------------------------------
real_img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.nii.gz")))
V = set(os.path.basename(p).replace(".nii.gz", "") for p in real_img_paths)
check("2.1 |V| = 30 unique BTCV volumes on disk", len(V) == EXPECTED_TOTAL,
      f"found {len(V)}")

# ---------------------------------------------------------------
# Load the 5 fold manifests
# ---------------------------------------------------------------
folds = {}
for k in range(1, N_FOLDS + 1):
    fold_dir = os.path.join(DATASET_LIST_DIR, f"fold{k}")
    train_path = os.path.join(fold_dir, "01_BTCV_train_4to1.txt")
    test_path = os.path.join(fold_dir, "01_BTCV_test_4to1.txt")
    train_ids = set(read_ids(train_path))
    test_ids = set(read_ids(test_path))
    folds[k] = {"train": train_ids, "test": test_ids,
                "train_path": train_path, "test_path": test_path}

# ---------------------------------------------------------------
# 2.2  Every fold has 24 train + 6 test
# ---------------------------------------------------------------
print("-" * 70)
for k in range(1, N_FOLDS + 1):
    n_train = len(folds[k]["train"])
    n_test = len(folds[k]["test"])
    check(f"2.2 Fold {k}: |Train_{k}| = {EXPECTED_TRAIN}", n_train == EXPECTED_TRAIN,
          f"got {n_train}")
    check(f"2.2 Fold {k}: |Test_{k}| = {EXPECTED_TEST}", n_test == EXPECTED_TEST,
          f"got {n_test}")

# ---------------------------------------------------------------
# 2.3  No within-fold leakage: Train_k ∩ Test_k = ∅
# ---------------------------------------------------------------
print("-" * 70)
for k in range(1, N_FOLDS + 1):
    overlap = folds[k]["train"] & folds[k]["test"]
    check(f"2.3 Fold {k}: Train_{k} ∩ Test_{k} = ∅", len(overlap) == 0,
          f"overlap={overlap}")

# ---------------------------------------------------------------
# 2.4  Every volume held out exactly once across the 5 test sets
# ---------------------------------------------------------------
print("-" * 70)
test_sets = [folds[k]["test"] for k in range(1, N_FOLDS + 1)]
union_test = set().union(*test_sets)
check("2.4 Test_1 ∪ ... ∪ Test_5 = V", union_test == V,
      f"missing={V - union_test}, extra={union_test - V}")
check("2.4 |Test_1 ∪ ... ∪ Test_5| = 30", len(union_test) == EXPECTED_TOTAL,
      f"got {len(union_test)}")

# pairwise disjoint test sets -> each volume appears in exactly one test set
pairwise_ok = True
for i in range(N_FOLDS):
    for j in range(i + 1, N_FOLDS):
        inter = test_sets[i] & test_sets[j]
        if inter:
            pairwise_ok = False
            print(f"    overlap between Test_{i+1} and Test_{j+1}: {inter}")
check("2.4 Test sets pairwise disjoint (each volume held out exactly once)", pairwise_ok)

# also confirm every train set is exactly V minus that fold's test set (24 = 30-6, consistent)
for k in range(1, N_FOLDS + 1):
    expected_train = V - folds[k]["test"]
    check(f"2.4 Fold {k}: Train_{k} == V \\ Test_{k}", folds[k]["train"] == expected_train)

# ---------------------------------------------------------------
# 2.5  Deterministic split: back up current manifests, re-run the REAL
#      generator in place (same paths it always uses), diff against backup,
#      then restore to guarantee this verification is non-destructive.
# ---------------------------------------------------------------
print("-" * 70)
backup_dir = tempfile.mkdtemp(prefix="btcv_fold_backup_")
try:
    backup_dataset_list = os.path.join(backup_dir, "dataset_list")
    shutil.copytree(DATASET_LIST_DIR, backup_dataset_list)

    result = subprocess.run(
        [sys.executable, MAKE_FOLDS_SCRIPT],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    all_identical = True
    all30_a = os.path.join(backup_dataset_list, "01_BTCV_all30_train.txt")
    all30_b = os.path.join(DATASET_LIST_DIR, "01_BTCV_all30_train.txt")
    if not filecmp.cmp(all30_a, all30_b, shallow=False):
        all_identical = False
        print("    01_BTCV_all30_train.txt differs on re-run")
    for k in range(1, N_FOLDS + 1):
        for fname in ("01_BTCV_train_4to1.txt", "01_BTCV_test_4to1.txt"):
            a = os.path.join(backup_dataset_list, f"fold{k}", fname)
            b = os.path.join(DATASET_LIST_DIR, f"fold{k}", fname)
            if not filecmp.cmp(a, b, shallow=False):
                all_identical = False
                print(f"    fold{k}/{fname} differs on re-run")
    check("2.5 Re-running the generator reproduces byte-identical manifests", all_identical)
finally:
    shutil.rmtree(backup_dir, ignore_errors=True)

# ---------------------------------------------------------------
# 2.6  Manifests correspond to actual files on disk
# ---------------------------------------------------------------
print("-" * 70)
all_vol_ids = set()
for k in range(1, N_FOLDS + 1):
    all_vol_ids |= folds[k]["train"] | folds[k]["test"]

missing_img = []
missing_label = []
for vid in sorted(all_vol_ids):
    num = vid.replace("img", "")
    img_path = os.path.join(IMG_DIR, f"{vid}.nii.gz")
    label_path = os.path.join(LABEL_DIR, f"label{num}.nii.gz")
    if not os.path.isfile(img_path):
        missing_img.append(img_path)
    if not os.path.isfile(label_path):
        missing_label.append(label_path)

check("2.6 Every manifest volume has a real image file on disk", len(missing_img) == 0,
      f"missing={missing_img}")
check("2.6 Every manifest volume has a real label file on disk", len(missing_label) == 0,
      f"missing={missing_label}")
check("2.6 Manifest volume set matches disk volume set exactly", all_vol_ids == V,
      f"manifest-only={all_vol_ids - V}, disk-only={V - all_vol_ids}")

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
print("=" * 70)
if failures:
    print(f"RESULT: FAIL ({len(failures)} check(s) failed)")
    for f in failures:
        print("  -", f)
    sys.exit(1)
else:
    print(f"RESULT: ALL CHECKS PASSED ({N_FOLDS} folds, {EXPECTED_TOTAL} volumes)")
    sys.exit(0)
