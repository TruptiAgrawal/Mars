"""
btcv_cross_validation/z_spacing/find_eligible_patients.py

Identify all BTCV patients with z-spacing = 3 mm that appear in the
held-out test sets of folds 1-5, and map each to its fold checkpoint.

Eligibility: z_spacing < 3.5 mm (nominal 3 mm or 2.5 mm).
The proposed resampling rule triggers at z > 5.5 mm, so any 3 mm patient
is a valid subject for the B00 vs B01 comparison (trigger fires on their
synthetic 6 mm version). Native 5 mm patients would NOT trigger the rule.

Prints a table and writes:
    btcv_cross_validation/z_spacing/eligible_patients.json

Run from repo root:
    python btcv_cross_validation/z_spacing/find_eligible_patients.py
"""
import json
from pathlib import Path
import SimpleITK as sitk

DATA_ROOT = Path("datasets/01_BCV-Abdomen/Training/img")
FOLDS_DIR = Path("btcv_cross_validation/dataset_list")
CKPT_DIR  = Path("btcv_cross_validation/runs")
OUTPUT    = Path("btcv_cross_validation/z_spacing/eligible_patients.json")

FOLD_TEST_FILES = {
    1: FOLDS_DIR / "fold1" / "01_BTCV_test_4to1.txt",
    2: FOLDS_DIR / "fold2" / "01_BTCV_test_4to1.txt",
    3: FOLDS_DIR / "fold3" / "01_BTCV_test_4to1.txt",
    4: FOLDS_DIR / "fold4" / "01_BTCV_test_4to1.txt",
    5: FOLDS_DIR / "fold5" / "01_BTCV_test_4to1.txt",
}

FOLD_CHECKPOINTS = {
    1: CKPT_DIR / "fold1/ddp_sub_cv/cv_fold1/ckpt/model_epoch_20.pth",
    2: CKPT_DIR / "fold2/ddp_sub_cv/cv_fold2/ckpt/model_epoch_20.pth",
    3: CKPT_DIR / "fold3/ddp_sub_cv/cv_fold3/ckpt/model_epoch_20.pth",
    4: CKPT_DIR / "fold4/ddp_sub_cv/cv_fold4/ckpt/model_epoch_20.pth",
    5: CKPT_DIR / "fold5/ddp_sub_cv/cv_fold5/ckpt/model_epoch_20.pth",
}


def read_ids(path):
    ids = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        token = line.split()[0]
        vid = Path(token).name.replace(".nii.gz", "")
        ids.append(vid)
    return ids


def get_z_spacing(volume_id):
    path = DATA_ROOT / f"{volume_id}.nii.gz"
    if not path.exists():
        return None, None
    itk = sitk.ReadImage(str(path))
    sp  = itk.GetSpacing()   # x, y, z
    sz  = itk.GetSize()
    return float(sp[2]), int(sz[2])


eligible = []
fold_coverage = {}

print(f"{'Fold':<6} {'Patient':<12} {'z_mm':>6} {'slices':>8} {'checkpoint_exists':>18}")
print("-" * 56)

for fold in range(1, 6):
    test_file = FOLD_TEST_FILES[fold]
    ckpt_path = FOLD_CHECKPOINTS[fold]
    ckpt_exists = ckpt_path.exists()
    patients = read_ids(test_file)
    fold_coverage[fold] = {"all": patients, "eligible": []}

    for vid in patients:
        z_mm, n_slices = get_z_spacing(vid)
        if z_mm is None:
            print(f"{fold:<6} {vid:<12} {'MISSING':>6}")
            continue
        is_eligible = z_mm < 3.5   # 2.5 mm or 3.0 mm
        marker = "  <-- ELIGIBLE" if is_eligible else ""
        print(f"{fold:<6} {vid:<12} {z_mm:>6.1f} {n_slices:>8} {str(ckpt_exists):>18}{marker}")
        if is_eligible:
            eligible.append({
                "volume_id": vid,
                "fold": fold,
                "z_spacing_mm": z_mm,
                "n_slices": n_slices,
                "checkpoint": str(ckpt_path),
                "checkpoint_exists": ckpt_exists,
                "test_manifest": str(test_file),
            })
            fold_coverage[fold]["eligible"].append(vid)

print()
print("=" * 56)
print(f"Total eligible patients (z < 3.5 mm): {len(eligible)}")
for fold in range(1, 6):
    n = len(fold_coverage[fold]["eligible"])
    ids = fold_coverage[fold]["eligible"]
    print(f"  Fold {fold}: {n} eligible — {ids}")

print()
missing_ckpts = [e for e in eligible if not e["checkpoint_exists"]]
if missing_ckpts:
    print("WARNING — missing checkpoints:")
    for e in missing_ckpts:
        print(f"  fold {e['fold']} {e['volume_id']}: {e['checkpoint']}")
else:
    print("All checkpoints exist.")

OUTPUT.write_text(json.dumps(eligible, indent=2) + "\n")
print(f"\nWritten: {OUTPUT}")
