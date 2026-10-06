# Preserve training logs

Start a fresh run inside tmux using the training environment:

```bash
conda activate slidesam
CUDA_VISIBLE_DEVICES=0 python btcv_cross_validation/scripts/train_fold_safe.py 5
```

Replace `5` with the desired fold. This starts training; no shell redirection or
`tee` is needed. The launcher displays output and saves both stdout and stderr.
Every invocation gets a timestamp and random suffix in both the log filename and
run tag. Existing run directories are refused, logs are created exclusively, and
the training exit code is returned. This launcher is for fresh runs, not resuming.

Logs become read-only as soon as they are opened. The launcher can continue
writing through its existing descriptor, while a subsequent `>` or `tee` to that
filename fails with permission denied. This also preserves logs of failed launches.
The existing five training logs have been made read-only and backed up under
`logs/backups/`. Fold 5's backup includes the reconstructed console log and its
surviving original internal log and analysis CSVs.

Read-only permissions prevent accidental file writes, not deletion/replacement,
explicit permission changes, or privileged writes. Keep external backups for those
cases. Do not use old commands that redirect output to a fixed log filename.

For extra protection in each interactive shell, `set -C` enables noclobber:

```bash
set -C
```

That makes ordinary `>` refuse existing files even when writable. It does not
protect against `tee`, `>|`, or deletion, and applies only to that shell.
