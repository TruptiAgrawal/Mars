"""
btcv_cross_validation/z_spacing/audit_kidney.py

Read-only audit of left-kidney (label 3) results from both branches.
Compares termination messages, propagation prompt counts, recall/precision,
and geometry metadata between control and proposed branches.

Run from repo root:
    python btcv_cross_validation/z_spacing/audit_kidney.py
"""
import json
import numpy as np
from pathlib import Path

RESULTS_V2 = Path("btcv_cross_validation/z_spacing/results_v2")
LOGS_DIR   = Path("btcv_cross_validation/z_spacing/logs")
LABEL_IDX  = 3
ORGAN      = "Left Kidney"

def load(branch):
    path = RESULTS_V2 / f"fold5_e20_{branch}_label{LABEL_IDX}" / "results.json"
    with open(path) as f:
        return json.load(f)

ctrl = load("control")
prop = load("proposed")

print("=" * 70)
print(f"KIDNEY AUDIT — {ORGAN} (label {LABEL_IDX}), v2 geometry")
print("=" * 70)

for branch, d in [("control", ctrl), ("proposed", prop)]:
    print(f"\n--- branch={branch} ---")
    for c in d["cases"]:
        vid   = c["volume_id"]
        m     = c["metrics"]
        ri    = c.get("resampling_info", {})
        pstat = c.get("prediction_status", "?")
        prompts = c.get("propagation_prompts", {})

        print(f"\n  {vid}:")
        print(f"    Dice      = {m['dice']*100:.4f}%")
        print(f"    Recall    = {m['foreground_recall']*100:.4f}%")
        print(f"    Precision = {m['foreground_precision']*100:.4f}%")
        print(f"    FP vol    = {m['false_positive_ml']:.2f} mL")
        print(f"    NSD@1mm   = {m['nsd']*100:.4f}%")
        print(f"    HD95      = {m['hd95_mm']:.2f} mm")
        print(f"    prediction_status = {pstat}")
        print(f"    crop z    = [{c['crop_start_z']}, {c['crop_stop_z']})  "
              f"working_shape = {c['working_shape_zyx']}")
        print(f"    original_shape = {c['original_shape_zyx']}")
        print(f"    center_slice (original z) = {c['initial_slice_original_z']}  "
              f"adjusted = {c['initial_slice_adjusted']}")

        if ri:
            print(f"    resampling_applied  = {ri.get('resampling_applied')}")
            print(f"    actual_z_mm         = {ri.get('actual_z_mm')}")
            print(f"    resampled_shape     = {ri.get('resampled_shape_zyx')}")
            print(f"    restored_shape      = {ri.get('restored_shape_zyx')}")
            print(f"    center_scoring_grid = {ri.get('center_scoring_grid')}")
            print(f"    center_continuous   = {ri.get('center_continuous_resampled')}")
            print(f"    center_resampled    = {ri.get('center_resampled_actual')}")
            print(f"    displacement_mm     = {ri.get('center_physical_displacement_mm')}")
            warn = ri.get("prediction_shape_warning")
            if warn:
                print(f"    *** shape_warning: {warn}")

        n_prompts = len(prompts)
        prompt_slices = sorted(int(k) for k in prompts.keys()) if prompts else []
        working_z = c["working_shape_zyx"][0]
        print(f"    n_propagation_prompts = {n_prompts}  "
              f"(working grid has {working_z} slices)")
        if prompt_slices:
            print(f"    prompt slice range: {min(prompt_slices)} – {max(prompt_slices)}")
            coverage = (max(prompt_slices) - min(prompt_slices) + 1) / working_z * 100
            print(f"    slice coverage: {coverage:.1f}%")

        t = c.get("inference_seconds", c.get("total_seconds", "?"))
        print(f"    inference_seconds = {t}")

# --- summary comparison ---
print("\n" + "=" * 70)
print("SUMMARY COMPARISON")
print("=" * 70)
print(f"{'Patient':<10} {'Ctrl Dice':>10} {'Prop Dice':>10} {'Diff':>8} "
      f"{'Ctrl Rec':>10} {'Prop Rec':>10} "
      f"{'Ctrl Prec':>10} {'Prop Prec':>10}")
print("-" * 80)

ctrl_cases = {c["volume_id"]: c for c in ctrl["cases"]}
prop_cases = {c["volume_id"]: c for c in prop["cases"]}

for vid in sorted(ctrl_cases.keys()):
    cm = ctrl_cases[vid]["metrics"]
    pm = prop_cases[vid]["metrics"]
    diff = (pm["dice"] - cm["dice"]) * 100
    print(f"{vid:<10} "
          f"{cm['dice']*100:>10.4f} {pm['dice']*100:>10.4f} {diff:>+8.4f} "
          f"{cm['foreground_recall']*100:>10.4f} {pm['foreground_recall']*100:>10.4f} "
          f"{cm['foreground_precision']*100:>10.4f} {pm['foreground_precision']*100:>10.4f}")

print()
print("Propagation coverage (prompt slices / working z slices):")
for vid in sorted(ctrl_cases.keys()):
    for branch, cases in [("control", ctrl_cases), ("proposed", prop_cases)]:
        c = cases[vid]
        prompts = c.get("propagation_prompts", {})
        wz = c["working_shape_zyx"][0]
        n  = len(prompts)
        slices = sorted(int(k) for k in prompts.keys()) if prompts else []
        span   = (max(slices) - min(slices) + 1) if slices else 0
        print(f"  {vid} {branch}: {n} prompts, "
              f"span={span}/{wz} slices ({100*span/wz:.1f}%)")

print()
print("Note: propagation_prompts records boxes per output slice, not per")
print("model call. Missing slices indicate early termination.")
print()

# Check inference logs for 'No masks' lines
print("Scanning inference logs for termination messages (kidney)...")
for logname in ["inference_control_v2.log", "inference_proposed_v2.log"]:
    logpath = LOGS_DIR / logname
    if not logpath.exists():
        print(f"  {logname}: NOT FOUND")
        continue
    lines = logpath.read_text().splitlines()
    # Find section around label 3
    in_label3 = False
    terminations = []
    for line in lines:
        if "organ=3" in line or "label_idx': 3" in line or "label3" in line.lower():
            in_label3 = True
        if in_label3 and ("organ=6" in line or "organ=7" in line or "organ=1" in line):
            in_label3 = False
        if in_label3 and any(kw in line for kw in
                             ["No masks", "Ends for no mask", "empty", "terminate",
                              "break", "stop"]):
            terminations.append(line.strip())
    print(f"  {logname}: {len(terminations)} termination lines near label 3")
    for t in terminations:
        print(f"    {t}")
