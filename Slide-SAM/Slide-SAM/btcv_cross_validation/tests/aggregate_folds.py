"""Step 9: aggregate saved held-out results, using only Python's standard library.

No model loading or inference. Each fold contributes the mean of six volumes;
CV SD is the sample SD of the FIVE fold means (ddof=1).
Additional metrics are supported as per_volume_<metric> dictionaries in every
organ of every fold, optionally accompanied by mean_<metric>/std_<metric>.
Only genuine computed metrics should be supplied; the box evaluator's NSD=0
placeholder is not a measurement and is not included by eval_fold.py.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys


CV_DIR = Path(__file__).resolve().parents[1]
ORGANS = {"Liver": 6, "Stomach": 7, "Left Kidney": 3, "Spleen": 1}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def manifest(path, count):
    ids = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        name = Path(line.split()[0]).name
        require(re.fullmatch(r"img\d+\.nii\.gz", name), f"Invalid volume in {path}: {name}")
        ids.append(name[:-7])
    require(len(ids) == count and len(set(ids)) == count,
            f"{path}: expected {count} unique volumes")
    return set(ids)


def number(value, context):
    require(type(value) in (int, float) and math.isfinite(value),
            f"{context}: expected a finite number, got {value!r}")
    return float(value)


def check_saved(record, key, calculated, context):
    if key in record:
        saved = number(record[key], f"{context}/{key}")
        require(math.isclose(saved, calculated, rel_tol=1e-7, abs_tol=1e-9),
                f"{context}/{key}: saved {saved} != recalculated {calculated}")


def aggregate(cv_dir, input_dir, epoch, prompt):
    all_ids = manifest(cv_dir / "dataset_list/01_BTCV_all30_train.txt", 30)
    seen = set()
    fold_means, volume_rows, sources = {}, [], []
    expected_metrics = None
    for fold in range(1, 6):
        split = cv_dir / "dataset_list" / f"fold{fold}"
        heldout = manifest(split / "01_BTCV_test_4to1.txt", 6)
        training = manifest(split / "01_BTCV_train_4to1.txt", 24)
        require(not heldout & training and heldout | training == all_ids,
                f"Fold {fold}: train/test leakage or incorrect partition")
        require(not seen & heldout, f"Fold {fold}: repeated held-out volumes")
        seen.update(heldout)
        path = input_dir / f"fold{fold}_evaluation.json"
        raw = path.read_bytes()
        data = json.loads(raw, object_pairs_hook=unique_object)
        require(data["fold"] == fold, f"{path}: incorrect fold ID")
        require(data["epoch"] == epoch and data["prompt"] == prompt,
                f"{path}: expected epoch={epoch}, prompt={prompt}")
        require(set(data["organs"]) == set(ORGANS), f"{path}: incorrect organ set")
        sources.append({"fold": fold, "path": str(path),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "checkpoint": data.get("checkpoint"),
                        "heldout_volumes": sorted(heldout)})
        for organ, label in ORGANS.items():
            record = data["organs"][organ]
            context = f"Fold {fold}/{organ}"
            require(record["label_idx"] == label, f"{context}: wrong BTCV label")
            metrics = {k[len("per_volume_"):] for k in record if k.startswith("per_volume_")}
            require("dice" in metrics, f"{context}: missing per-volume Dice")
            if expected_metrics is None:
                expected_metrics = metrics
            require(metrics == expected_metrics, f"{context}: inconsistent metric set")
            summaries = {k[5:] for k in record if k.startswith("mean_")}
            require(summaries <= metrics, f"{context}: metric summary without per-volume data")
            for metric in sorted(metrics):
                scores = record[f"per_volume_{metric}"]
                require(isinstance(scores, dict) and set(scores) == heldout,
                        f"{context}/{metric}: expected exactly the six held-out IDs")
                values = []
                for volume in sorted(scores):
                    value = number(scores[volume], f"{context}/{metric}/{volume}")
                    if metric in {"dice", "iou", "nsd", "foreground_recall", "foreground_precision"}:
                        require(0 <= value <= 1, f"{context}/{metric}: score outside [0, 1]")
                    values.append(value)
                    volume_rows.append(dict(fold=fold, organ=organ, label_idx=label,
                                            volume=volume, metric=metric, value=value))
                mean = statistics.mean(values)
                check_saved(record, f"mean_{metric}", mean, context)
                check_saved(record, f"std_{metric}", statistics.stdev(values), context)
                fold_means.setdefault((organ, metric), []).append(mean)
        for metric in sorted(expected_metrics):
            macro = statistics.mean(fold_means[(organ, metric)][-1] for organ in ORGANS)
            fold_means.setdefault(("Four-organ macro average", metric), []).append(macro)
            check_saved(data, f"mean_{metric}_4_organs", macro, f"Fold {fold}")
    require(seen == all_ids, "Held-out sets do not cover all 30 volumes")
    rows = []
    for (organ, metric), values in fold_means.items():
        rows.append(dict(organ=organ, metric=metric,
                         **{f"fold{i}": v for i, v in enumerate(values, 1)},
                         mean=statistics.mean(values), sd=statistics.stdev(values),
                         n_folds=5, sd_ddof=1))
    return rows, volume_rows, sources


def csv_write(path, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-dir", type=Path, default=CV_DIR)
    parser.add_argument("--input-dir", type=Path, help="Default: <cv-dir>/logs")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: <cv-dir>/results; existing outputs are never overwritten")
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--prompt", default="box")
    args = parser.parse_args()
    try:
        cv_dir = args.cv_dir.resolve()
        args.output_dir = args.output_dir or cv_dir / "results"
        input_dir = (args.input_dir or cv_dir / "logs").resolve()
        require(not args.output_dir.exists(), f"Output directory already exists: {args.output_dir}")
        rows, volumes, sources = aggregate(cv_dir, input_dir, args.epoch, args.prompt)
        report = ["# BTCV five-fold cross-validation", "",
                  f"Epoch {args.epoch}; prompt: {args.prompt}. All 30 held-out volumes validated against manifests.", "",
                  "Each fold score is the unweighted mean over its six held-out volumes. "
                  "Final mean and sample SD (ddof=1) are calculated across the five fold scores. "
                  "The macro row averages four organs within each fold before calculating CV mean/SD.", "",
                  "CSV/JSON retain original metric units. Dice in this table is a percentage; "
                  "other metrics retain their input units.", "",
                  "| Organ | Metric | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± SD |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            scale = 100 if row["metric"] == "dice" else 1
            values = " | ".join(f"{row[f'fold{i}'] * scale:.2f}" for i in range(1, 6))
            report.append(f"| {row['organ']} | {row['metric']} | {values} | "
                          f"{row['mean'] * scale:.2f} ± {row['sd'] * scale:.2f} |")
        report += ["", "Only metrics present as per-volume measurements in all folds are aggregated. "
                   "The current eval_fold.py records Dice only; box-prompt NSD=0 is a placeholder, "
                   "so it must not be reported as measured NSD. "
                   "This script validates saved results and manifests; it does not rerun inference "
                   "or independently verify checkpoint provenance.", ""]
        args.output_dir.mkdir(parents=True, exist_ok=False)
        csv_write(args.output_dir / "fold_summary.csv", rows)
        csv_write(args.output_dir / "per_volume_metrics.csv", volumes)
        payload = dict(epoch=args.epoch, prompt=args.prompt, n_folds=5, n_volumes=30,
                       sd_ddof=1, aggregation="mean and sample SD across fold means",
                       sources=sources, results=rows)
        (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        (args.output_dir / "report.md").write_text("\n".join(report))
        print("\n".join(report))
        print(f"Reports saved to {args.output_dir.resolve()}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
