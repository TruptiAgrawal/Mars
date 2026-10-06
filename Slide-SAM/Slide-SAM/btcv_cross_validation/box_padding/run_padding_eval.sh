#!/usr/bin/env bash
# Run all 12 box-padding evaluation arms.
# Usage: bash btcv_cross_validation/box_padding/run_padding_eval.sh
# Run from repo root: /home/jupyter-team2pop/Slide-SAM/Slide-SAM

set -euo pipefail

cd /home/jupyter-team2pop/Slide-SAM/Slide-SAM

export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/slidesam-padding-mpl

mkdir -p btcv_cross_validation/box_padding/logs
stamp=$(date +%Y%m%d_%H%M%S)

run_padding_arm() {
  local mode="$1"
  local organ="$2"
  local fraction=0
  if [ "$mode" = "image" ]; then fraction=0.05; fi

  echo "=== mode=$mode organ=$organ fraction=$fraction ==="

  python -u -c "
import random, runpy
import numpy as np
import torch
random.seed(0); np.random.seed(0); torch.manual_seed(0)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
runpy.run_module('btcv_cross_validation.box_padding.evaluate', run_name='__main__', alter_sys=True)
" \
    --data-root datasets \
    --manifest btcv_cross_validation/dataset_list/fold5/01_BTCV_test_4to1.txt \
    --train-manifest btcv_cross_validation/dataset_list/fold5/01_BTCV_train_4to1.txt \
    --checkpoint btcv_cross_validation/runs/fold5/ddp_sub_cv/cv_fold5/ckpt/model_epoch_20.pth \
    --checkpoint-kind finetuned \
    --label-idx "$organ" \
    --prompt box \
    --protocol gt-cropped \
    --padding-mode "$mode" \
    --padding-fraction "$fraction" \
    --nsd-tolerance-mm 1.0 \
    --output "btcv_cross_validation/box_padding/results/fold5_e20_${mode}_label${organ}" \
    --run-inference \
    2>&1 | tee "btcv_cross_validation/box_padding/logs/${stamp}_${mode}_label${organ}.log"

  echo "--- done: mode=$mode organ=$organ ---"
}

# 3 modes x 4 organs = 12 runs
# organs: liver=6, stomach=7, left kidney=3, spleen=1
for mode in historical none image; do
  for organ in 6 7 3 1; do
    run_padding_arm "$mode" "$organ"
  done
done

echo ""
echo "All 12 arms complete. Results in btcv_cross_validation/box_padding/results/"
echo "Logs in btcv_cross_validation/box_padding/logs/"
