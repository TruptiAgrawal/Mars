#!/usr/bin/env bash
# btcv_cross_validation/z_spacing/run_z_spacing_eval.sh  (v2)
#
# Logs:
#   btcv_cross_validation/z_spacing/logs/setup_v2.log
#   btcv_cross_validation/z_spacing/logs/inference_control_v2.log
#   btcv_cross_validation/z_spacing/logs/inference_proposed_v2.log
#
# Run from repo root:
#   conda activate slidesam
#   bash btcv_cross_validation/z_spacing/run_z_spacing_eval.sh
#
# Add Z_RUN_GPU=1 to also launch GPU inference (step 5):
#   Z_RUN_GPU=1 bash btcv_cross_validation/z_spacing/run_z_spacing_eval.sh

set -euo pipefail

cd /home/jupyter-team2pop/Slide-SAM/Slide-SAM

export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/slidesam-zspacing-mpl

SYNTHETIC_ROOT="btcv_cross_validation/z_spacing/synthetic_6mm_v2"
MANIFEST="btcv_cross_validation/z_spacing/synthetic_6mm_v2/synthetic_manifest.txt"
TRAIN_MANIFEST="btcv_cross_validation/dataset_list/fold5/01_BTCV_train_4to1.txt"
CHECKPOINT="btcv_cross_validation/runs/fold5/ddp_sub_cv/cv_fold5/ckpt/model_epoch_20.pth"
RESULTS_ROOT="btcv_cross_validation/z_spacing/results_v2"
LOGS_DIR="btcv_cross_validation/z_spacing/logs"
Z_RUN_GPU="${Z_RUN_GPU:-0}"

mkdir -p "$LOGS_DIR"
SETUP_LOG="$LOGS_DIR/setup_v2.log"

exec > >(tee -a "$SETUP_LOG") 2>&1

echo "===== Z-SPACING v2 SETUP LOG — $(date '+%Y-%m-%d %H:%M:%S') ====="
echo ""

# ---------------------------------------------------------------------------
# Step 1: CPU tests
# ---------------------------------------------------------------------------
echo "=== Step 1: CPU geometry tests (v2) ==="
CUDA_VISIBLE_DEVICES='' python -m unittest \
  btcv_cross_validation.z_spacing.test_cpu -v
echo "--- CPU tests passed ---"
echo ""

# ---------------------------------------------------------------------------
# Step 2: Dry run
# ---------------------------------------------------------------------------
echo "=== Step 2: Dry-run synthetic volume plan ==="
python -m btcv_cross_validation.z_spacing.make_synthetic \
  --data-root datasets \
  --manifest btcv_cross_validation/dataset_list/fold5/01_BTCV_test_4to1.txt \
  --output "$SYNTHETIC_ROOT" \
  --target-z-mm 6.0 \
  --dry-run
echo "--- Dry run complete ---"
echo ""

# ---------------------------------------------------------------------------
# Step 3: Create synthetic volumes (skip if already built)
# ---------------------------------------------------------------------------
echo "=== Step 3: Create synthetic 6mm volumes (v2) ==="
if [ -d "$SYNTHETIC_ROOT" ]; then
  echo "  Already exists: $SYNTHETIC_ROOT — skipping rebuild."
else
  python -m btcv_cross_validation.z_spacing.make_synthetic \
    --data-root datasets \
    --manifest btcv_cross_validation/dataset_list/fold5/01_BTCV_test_4to1.txt \
    --output "$SYNTHETIC_ROOT" \
    --target-z-mm 6.0
  echo "  Synthetic volumes created."
fi
echo ""

# ---------------------------------------------------------------------------
# Step 4: Preflights
# ---------------------------------------------------------------------------
echo "=== Step 4: Preflights for all 8 arms ==="
mkdir -p "$RESULTS_ROOT"

for branch in control proposed; do
  for organ in 6 7 3 1; do
    echo "  preflight: branch=$branch organ=$organ"
    python -u -c "
import random, runpy
import numpy as np, torch
random.seed(0); np.random.seed(0); torch.manual_seed(0)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
runpy.run_module('btcv_cross_validation.z_spacing.evaluate', run_name='__main__', alter_sys=True)
" \
      --synthetic-root "$SYNTHETIC_ROOT" \
      --manifest "$MANIFEST" \
      --train-manifest "$TRAIN_MANIFEST" \
      --checkpoint "$CHECKPOINT" \
      --checkpoint-kind finetuned \
      --label-idx "$organ" \
      --protocol gt-cropped \
      --branch "$branch" \
      --target-z-mm 3.0 \
      --trigger-mm 5.5 \
      --nsd-tolerance-mm 1.0 \
      --output "${RESULTS_ROOT}/fold5_e20_${branch}_label${organ}"
  done
done
echo "--- All 8 preflights passed ---"
echo ""
echo "===== SETUP COMPLETE ====="
echo ""

# ---------------------------------------------------------------------------
# Step 5: GPU inference
# ---------------------------------------------------------------------------
if [ "$Z_RUN_GPU" != "1" ]; then
  echo "Step 5 skipped. Run with Z_RUN_GPU=1 to launch GPU inference."
  exit 0
fi

run_branch() {
  local branch="$1"
  local log_file="$LOGS_DIR/inference_${branch}_v2.log"

  echo "=== Step 5: GPU inference — branch=$branch ===" | tee -a "$log_file"
  nvidia-smi --query-gpu=name,memory.free --format=csv,noheader | tee -a "$log_file"

  for organ in 6 7 3 1; do
    echo "--- organ=$organ ---" | tee -a "$log_file"
    python -u -c "
import random, runpy
import numpy as np, torch
random.seed(0); np.random.seed(0); torch.manual_seed(0)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
runpy.run_module('btcv_cross_validation.z_spacing.evaluate', run_name='__main__', alter_sys=True)
" \
      --synthetic-root "$SYNTHETIC_ROOT" \
      --manifest "$MANIFEST" \
      --train-manifest "$TRAIN_MANIFEST" \
      --checkpoint "$CHECKPOINT" \
      --checkpoint-kind finetuned \
      --label-idx "$organ" \
      --protocol gt-cropped \
      --branch "$branch" \
      --target-z-mm 3.0 \
      --trigger-mm 5.5 \
      --nsd-tolerance-mm 1.0 \
      --output "${RESULTS_ROOT}/fold5_e20_${branch}_label${organ}" \
      --run-inference \
      2>&1 | tee -a "$log_file"
  done
  echo "--- branch=$branch done ---" | tee -a "$log_file"
}

run_branch control
run_branch proposed

echo ""
echo "All 16 predictions complete (2 patients x 4 organs x 2 branches)."
echo "Results: $RESULTS_ROOT"
echo "Logs: $LOGS_DIR/"
