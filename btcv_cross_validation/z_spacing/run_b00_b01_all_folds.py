"""
btcv_cross_validation/z_spacing/run_b00_b01_all_folds.py

Build synthetic 6 mm volumes for all 18 eligible patients (z < 3.5 mm,
from test sets of all 5 folds) and run the B00 vs B01 comparison using
the v3 geometry-corrected evaluator.

B00 = control   : segment synthetic 6 mm directly (corrected-zero padding)
B01 = proposed  : resample 6 mm -> ~3 mm, segment, restore to 6 mm grid
                  (trigger > 5.5 mm, target 3 mm, corrected-zero padding)

Each patient is evaluated with the checkpoint from the fold where it was
held out.  Results go to per-fold subdirectories under results_v3/.

Steps:
  1. Run CPU geometry tests (test_geometry_fixes.py)
  2. Build synthetic volumes per fold (separate directories)
  3. Preflight all arms (no GPU)
  4. GPU inference (only when Z_RUN_GPU=1 env var is set)

Run:
    conda activate slidesam
    cd /home/jupyter-team2pop/Slide-SAM/Slide-SAM
    python btcv_cross_validation/z_spacing/run_b00_b01_all_folds.py
    # then, after reviewing setup output:
    Z_RUN_GPU=1 python btcv_cross_validation/z_spacing/run_b00_b01_all_folds.py
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[2]
ELIGIBLE   = ROOT / "btcv_cross_validation/z_spacing/eligible_patients.json"
DATA_ROOT  = ROOT / "datasets"
SYNTH_BASE = ROOT / "btcv_cross_validation/z_spacing/synthetic_6mm_v3"
RESULTS    = ROOT / "btcv_cross_validation/z_spacing/results_v3"
LOGS       = ROOT / "btcv_cross_validation/z_spacing/logs"
CKPT_ROOT  = ROOT / "btcv_cross_validation/runs"

TARGET_Z   = 3.0
TRIGGER    = 5.5
NSD_TOL    = 1.0
ORGANS     = [6, 7, 3, 1]
BRANCHES   = ["control", "proposed"]
RUN_GPU    = os.environ.get("Z_RUN_GPU", "0") == "1"


def run(cmd, **kwargs):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd, **kwargs)
    if r.returncode != 0:
        sys.exit(f"Command failed with code {r.returncode}")
    return r


def python(*args, **kwargs):
    return run([sys.executable, *[str(a) for a in args]], **kwargs)


# ── Step 0: Load eligible patients ───────────────────────────────────────────

with open(ELIGIBLE) as f:
    eligible = json.load(f)

# Group by fold
by_fold = {}
for e in eligible:
    fold = e["fold"]
    by_fold.setdefault(fold, []).append(e)

print(f"Eligible patients: {len(eligible)} across folds {sorted(by_fold.keys())}")
for fold, patients in sorted(by_fold.items()):
    print(f"  Fold {fold}: {[p['volume_id'] for p in patients]}")
print()


# ── Step 1: CPU geometry tests ────────────────────────────────────────────────

print("=== Step 1: CPU geometry tests ===")
LOGS.mkdir(parents=True, exist_ok=True)
setup_log = LOGS / "setup_v3.log"

env = os.environ.copy()
env["PYTHONPATH"]            = str(ROOT)
env["PYTHONDONTWRITEBYTECODE"] = "1"
env["CUDA_VISIBLE_DEVICES"]  = ""
env["OMP_NUM_THREADS"]       = "1"

with open(setup_log, "a") as logf:
    logf.write("=== CPU tests ===\n")
    r = subprocess.run(
        [sys.executable, "-m", "unittest",
         "btcv_cross_validation.z_spacing.test_geometry_fixes", "-v"],
        cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    out = r.stdout.decode()
    logf.write(out)
    print(out)
    if r.returncode != 0:
        sys.exit("CPU geometry tests FAILED. Fix before proceeding.")
print("CPU tests passed.\n")


# ── Step 2: Build synthetic volumes per fold ──────────────────────────────────

print("=== Step 2: Build synthetic 6 mm volumes ===")
SYNTH_BASE.mkdir(parents=True, exist_ok=True)

# Write one temporary manifest per fold (only eligible patients)
for fold, patients in sorted(by_fold.items()):
    synth_dir = SYNTH_BASE / f"fold{fold}"

    if synth_dir.exists():
        print(f"  Fold {fold}: synthetic dir already exists, skipping build.")
        continue

    # Write a temporary manifest for this fold's eligible patients
    tmp_manifest = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir=ROOT)
    for p in patients:
        vid = p["volume_id"]
        tmp_manifest.write(
            f"01_BCV-Abdomen/Training/img/{vid}.nii.gz "
            f"01_BCV-Abdomen/Training/label/label{vid[3:]}.nii.gz\n")
    tmp_manifest.close()

    print(f"  Fold {fold}: building {len(patients)} volumes -> {synth_dir}")
    with open(setup_log, "a") as logf:
        r = subprocess.run(
            [sys.executable, "-m",
             "btcv_cross_validation.z_spacing.make_synthetic",
             "--data-root", str(DATA_ROOT),
             "--manifest", tmp_manifest.name,
             "--output", str(synth_dir),
             "--target-z-mm", "6.0"],
            cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        out = r.stdout.decode()
        logf.write(f"\n=== make_synthetic fold{fold} ===\n{out}")
        print(out)
        if r.returncode != 0:
            sys.exit(f"make_synthetic fold{fold} FAILED.")
    os.unlink(tmp_manifest.name)

print("Synthetic volumes ready.\n")


# ── Step 3: Preflights ────────────────────────────────────────────────────────

print("=== Step 3: Preflights ===")
RESULTS.mkdir(parents=True, exist_ok=True)

preflight_ok = True
for fold, patients in sorted(by_fold.items()):
    synth_dir  = SYNTH_BASE / f"fold{fold}"
    manifest   = synth_dir / "synthetic_manifest.txt"
    ckpt_path  = patients[0]["checkpoint"]  # same for all in fold
    train_manifest = (ROOT / "btcv_cross_validation/dataset_list"
                      / f"fold{fold}" / "01_BTCV_train_4to1.txt")
    results_fold = RESULTS / f"fold{fold}"
    results_fold.mkdir(exist_ok=True)

    for branch in BRANCHES:
        for organ in ORGANS:
            out_dir = results_fold / f"b01_{branch}_label{organ}" if branch == "proposed" \
                      else results_fold / f"b00_{branch}_label{organ}"
            r = subprocess.run(
                [sys.executable, "-c",
                 "import random,runpy,numpy as np,torch; "
                 "random.seed(0); np.random.seed(0); torch.manual_seed(0); "
                 "torch.backends.cudnn.enabled=False; torch.backends.cudnn.benchmark=False; "
                 "runpy.run_module('btcv_cross_validation.z_spacing.evaluate_v3',"
                 "run_name='__main__',alter_sys=True)",
                 "--synthetic-root", str(synth_dir),
                 "--manifest", str(manifest),
                 "--train-manifest", str(train_manifest),
                 "--checkpoint", str(ckpt_path),
                 "--checkpoint-kind", "finetuned",
                 "--label-idx", str(organ),
                 "--protocol", "gt-cropped",
                 "--branch", branch,
                 "--target-z-mm", str(TARGET_Z),
                 "--trigger-mm", str(TRIGGER),
                 "--nsd-tolerance-mm", str(NSD_TOL),
                 "--output", str(out_dir)],
                cwd=ROOT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            status = "OK" if r.returncode == 0 else "FAIL"
            print(f"  fold{fold} {branch} label{organ}: {status}")
            if r.returncode != 0:
                print(r.stdout.decode()[-500:])
                preflight_ok = False

if not preflight_ok:
    sys.exit("Preflight FAILED. Fix errors before GPU inference.")
print("All preflights passed.\n")


# ── Step 4: GPU inference ─────────────────────────────────────────────────────

if not RUN_GPU:
    print("=== Step 4: GPU inference ===")
    print("Skipped (Z_RUN_GPU not set to 1).")
    print("Rerun with:  Z_RUN_GPU=1 python btcv_cross_validation/z_spacing/run_b00_b01_all_folds.py")
    sys.exit(0)

print("=== Step 4: GPU inference ===")
gpu_env = os.environ.copy()
gpu_env["PYTHONPATH"]              = str(ROOT)
gpu_env["PYTHONDONTWRITEBYTECODE"] = "1"
gpu_env["OMP_NUM_THREADS"]         = "4"
gpu_env["OPENBLAS_NUM_THREADS"]    = "1"
gpu_env["MPLCONFIGDIR"]            = "/tmp/slidesam-b00b01-mpl"

total_runs = sum(len(patients) for patients in by_fold.values()) * len(ORGANS) * len(BRANCHES)
completed  = 0

for fold, patients in sorted(by_fold.items()):
    synth_dir  = SYNTH_BASE / f"fold{fold}"
    manifest   = synth_dir / "synthetic_manifest.txt"
    ckpt_path  = patients[0]["checkpoint"]
    train_manifest = (ROOT / "btcv_cross_validation/dataset_list"
                      / f"fold{fold}" / "01_BTCV_train_4to1.txt")
    results_fold = RESULTS / f"fold{fold}"
    log_fold = LOGS / f"inference_v3_fold{fold}.log"

    print(f"\n--- Fold {fold} ({len(patients)} patients) ---")
    with open(log_fold, "a") as logf:
        logf.write(f"=== fold{fold} GPU inference ===\n")
        for branch in BRANCHES:
            for organ in ORGANS:
                out_dir = results_fold / (
                    f"b01_{branch}_label{organ}" if branch == "proposed"
                    else f"b00_{branch}_label{organ}")

                label = f"fold{fold} {branch} label{organ}"
                print(f"  [{completed+1}/{total_runs}] {label} ...", end=" ", flush=True)
                logf.write(f"\n--- {label} ---\n")

                r = subprocess.run(
                    [sys.executable, "-c",
                     "import random,runpy,numpy as np,torch; "
                     "random.seed(0); np.random.seed(0); torch.manual_seed(0); "
                     "torch.backends.cudnn.enabled=False; torch.backends.cudnn.benchmark=False; "
                     "runpy.run_module('btcv_cross_validation.z_spacing.evaluate_v3',"
                     "run_name='__main__',alter_sys=True)",
                     "--synthetic-root", str(synth_dir),
                     "--manifest", str(manifest),
                     "--train-manifest", str(train_manifest),
                     "--checkpoint", str(ckpt_path),
                     "--checkpoint-kind", "finetuned",
                     "--label-idx", str(organ),
                     "--protocol", "gt-cropped",
                     "--branch", branch,
                     "--target-z-mm", str(TARGET_Z),
                     "--trigger-mm", str(TRIGGER),
                     "--nsd-tolerance-mm", str(NSD_TOL),
                     "--output", str(out_dir),
                     "--run-inference"],
                    cwd=ROOT, env=gpu_env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                )
                out = r.stdout.decode()
                logf.write(out)
                status = "done" if r.returncode == 0 else f"FAILED (code {r.returncode})"
                print(status)
                if r.returncode != 0:
                    print("  Last 300 chars:", out[-300:])
                    # Do not exit — record failure, continue with remaining arms
                completed += 1

print(f"\nAll {total_runs} arms attempted.")
print(f"Results: {RESULTS}")
print(f"Logs:    {LOGS}/inference_v3_fold*.log")
