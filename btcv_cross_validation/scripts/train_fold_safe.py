"""Launch a fresh fold run with an exclusive, read-only console log."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import shlex
import subprocess
import sys
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]


def run_logged(command, log_path, cwd):
    # Exclusive creation prevents truncation, including through an existing symlink.
    with log_path.open("xb", buffering=0) as log:
        # This descriptor can still write; subsequent `>` / tee opens cannot.
        os.fchmod(log.fileno(), 0o444)
        header = f"Started: {datetime.now().astimezone().isoformat()}\nCommand: {shlex.join(command)}\n"
        log.write(header.encode())
        print(f"Log: {log_path}", flush=True)
        try:
            with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT) as child:
                try:
                    for line in child.stdout:
                        log.write(line)
                        try:
                            sys.stdout.buffer.write(line)
                            sys.stdout.buffer.flush()
                        except BrokenPipeError:
                            pass  # Continue saving even if the display pipe closes.
                    code = child.wait()
                except BaseException:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    raise
        except BaseException as exc:
            log.write(f"\nLauncher stopped: {type(exc).__name__}: {exc}\n".encode())
            raise
        log.write(f"\nFinished: {datetime.now().astimezone().isoformat()} | exit code: {code}\n".encode())
        return code if code >= 0 else 128 - code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold", type=int, choices=range(1, 6))
    args = parser.parse_args()
    config = ROOT / f"btcv_cross_validation/configs/fold{args.fold}.yaml"
    import yaml
    with config.open() as stream:
        settings = yaml.safe_load(stream)
    tag = f"cv_fold{args.fold}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:12]}"
    base = Path(settings["base"]["base_dir"])
    if not base.is_absolute():
        base = ROOT / base
    run_dir = base / "ddp_sub_cv" / tag
    if run_dir.exists() or run_dir.is_symlink():
        parser.error(f"Refusing to reuse run directory: {run_dir}")
    logs = ROOT / "btcv_cross_validation/logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"train_{tag}.log"
    command = [sys.executable, "-u", "-m", "core.ddp_sub_cv",
               "--config", str(config), "--tag", tag]
    print(f"Fresh run: {tag}\nExpected output directory: {run_dir}", flush=True)
    return run_logged(command, log_path, ROOT)


if __name__ == "__main__":
    sys.exit(main())
