#!/bin/sh
# Usage: sh tests/run_step9.sh
# Default output: btcv_cross_validation/results/step9/
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$script_dir/aggregate_folds.py" "$@"
