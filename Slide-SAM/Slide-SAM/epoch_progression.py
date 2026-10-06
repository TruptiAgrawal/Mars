"""
epoch_progression.py
--------------------
Runs each epoch checkpoint over the SAME validation cases and renders
a rows=cases x cols=epochs comparison figure with Dice under each panel.

IMPORTANT: run from the repo root so the relative config paths resolve:

    cd ~/Slide-SAM/Slide-SAM
    python3 epoch_progression.py --limit 1 --only 0     # smoke test first
    python3 epoch_progression.py --limit 3              # full run

Outputs to ../Visulaizer/figures/
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from tutils.new.manager import ConfigManager
from datasets.eval_dataloader.loader_abstract import AbstractLoader
from datasets.data_engine import DataManager, BoxPromptGenerator
from core.volume_predictor import VolumePredictor
from core.learner3 import SamLearner
from modeling.build_sam3d2 import sam_model_registry
from trans_utils.metrics import compute_dice_np

OUT = "../Visulaizer/figures"
LABEL_IDX = 2          # matches specific_label: [2] used in training

# the four epoch checkpoints, in training order
CKPTS = [
    ("epoch 1", "../runs/sam/ddp/word_label2_fresh_4/ckpt/model_epoch_0.pth"),
    ("epoch 2", "../runs/sam/ddp/word_label2_fresh_4/ckpt/model_epoch_1.pth"),
    ("epoch 3", "../runs/sam/ddp/word_label2_epoch2_3/ckpt/model_epoch_0.pth"),
    ("epoch 4", "../runs/sam/ddp/word_label2_epoch2_3/ckpt/model_epoch_1.pth"),
]


def build_config():
    ex = {
        "dataset": {
            "prompt": "box",
            "dataset_list": ["word"],
            "label_idx": LABEL_IDX,
        },
        "model_type": "vit_b",
        "pth": None,
    }
    cfg = ConfigManager()
    cfg.add_config("configs/vit_b.yaml")
    cfg.add_config(ex)
    return cfg


def load_predictor(cfg, pth):
    """Exactly the sequence test/volume_eval.py uses."""
    sam = sam_model_registry[cfg["model_type"]](checkpoint=None)
    learner = SamLearner(sam_model=sam, config=cfg,
                         data_engine=DataManager(img_size=(1024, 1024)))
    learner.use_lora()                      # LoRA layers must exist first
    learner.load_well_trained_model(pth)
    learner.cuda()
    return VolumePredictor(model=learner.model,
                           use_postprocess=True,
                           use_noise_remove=True)


def prep(data):
    """Replicate the cropping / template-slice logic from Evaluater.eval_step."""
    tsid = int(data["template_slice_id"][0])
    if tsid == 0:
        tsid += 1
    elif tsid == data["img"].shape[0] - 1:
        tsid -= 1
    if data["img"].shape[-1] < 260:
        img = data["img"][0][:, :256, :256]
        label = data["label"][0][:, :256, :256]
    else:
        img = data["img"][0]
        label = data["label"][0]
    return img, label, tsid


def run_ckpt(cfg, pth, cases):
    """Return {case_name: (pred_volume, dice)}"""
    predictor = load_predictor(cfg, pth)
    results = {}
    for name, img, label, tsid in cases:
        lab_np = label.detach().cpu().numpy()
        box = BoxPromptGenerator(size=None).mask_to_bbox(lab_np[tsid])
        box = np.array([box])
        with torch.no_grad():
            pred, _ = predictor.predict_volume(
                x=img, box=box, template_slice_id=tsid,
                return_stability=True)
        d = compute_dice_np(pred, lab_np)
        print(f"    {name}: dice = {d:.4f}")
        results[name] = (np.asarray(pred), float(d))
    del predictor
    torch.cuda.empty_cache()
    return results


def collect_cases(cfg, limit):
    ds = AbstractLoader(cfg["dataset"], split="test")
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    cases = []
    for i, data in enumerate(dl):
        if i >= limit:
            break
        img, label, tsid = prep(data)
        nm = data["name"]
        nm = nm[0] if isinstance(nm, (list, tuple)) else str(nm)
        nm = os.path.basename(str(nm)).replace(".nii.gz", "")
        cases.append((f"{nm}", img, label, tsid))
        print(f"  case {i}: {nm}  shape={tuple(img.shape)}  template z={tsid}")
    return cases


def overlay(ax, base, mask, color, alpha=0.55):
    ax.imshow(base, cmap="gray", interpolation="nearest")
    if mask is not None and mask.any():
        rgba = np.zeros((*mask.shape, 4))
        rgba[mask > 0] = color
        rgba[..., 3] = np.where(mask > 0, alpha, 0)
        ax.imshow(rgba, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])


def figure(cases, all_results, labels):
    ncol = len(labels) + 1
    nrow = len(cases)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(3.0 * ncol, 3.2 * nrow), squeeze=False)
    for r, (name, img, label, tsid) in enumerate(cases):
        base = np.asarray(img[tsid], dtype=np.float32)
        base = np.clip((base - np.percentile(base, 1)) /
                       (np.percentile(base, 99) - np.percentile(base, 1) + 1e-8),
                       0, 1)
        gt = np.asarray(label[tsid]) > 0
        for c, lab in enumerate(labels):
            pred, dice = all_results[lab][name]
            pslice = pred[tsid] > 0 if pred.ndim == 3 else pred > 0
            overlay(axes[r][c], base, pslice, (0.15, 0.55, 1.0, 1.0))
            axes[r][c].set_xlabel(f"Dice {dice:.3f}", fontsize=9)
            if r == 0:
                axes[r][c].set_title(lab, fontsize=11)
        overlay(axes[r][ncol - 1], base, gt, (0.2, 0.85, 0.3, 1.0))
        if r == 0:
            axes[r][ncol - 1].set_title("ground truth", fontsize=11)
        axes[r][0].set_ylabel(name, fontsize=9, rotation=0,
                              ha="right", va="center", labelpad=18)
    fig.suptitle("Spleen segmentation across training epochs "
                 "(box prompt, template slice)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "epoch_progression.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("saved:", p)


def dice_chart(all_results, labels, cases):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, *_ in cases:
        ys = [all_results[l][name][1] for l in labels]
        ax.plot(range(len(labels)), ys, "o-", lw=2, ms=7, label=name)
    means = [np.mean([all_results[l][n][1] for n, *_ in cases]) for l in labels]
    ax.plot(range(len(labels)), means, "k--", lw=2.5, label="mean")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylabel("Dice"); ax.set_ylim(0, 1.02)
    ax.set_title("Validation Dice per epoch (spleen)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "dice_per_epoch.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("saved:", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3, help="how many cases")
    ap.add_argument("--only", type=int, default=None,
                    help="run just one checkpoint index (0-3), for smoke tests")
    args = ap.parse_args()

    cfg = build_config()
    print("collecting cases...")
    cases = collect_cases(cfg, args.limit)
    if not cases:
        print("no cases loaded -- check the test split list")
        return

    ckpts = CKPTS if args.only is None else [CKPTS[args.only]]
    all_results, labels = {}, []
    for lab, pth in ckpts:
        if not os.path.exists(pth):
            print(f"  missing checkpoint, skipping: {pth}")
            continue
        print(f"\n[{lab}] {pth}")
        all_results[lab] = run_ckpt(cfg, pth, cases)
        labels.append(lab)

    if not labels:
        print("no checkpoints ran")
        return

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "dice_results.json"), "w") as f:
        json.dump({l: {n: v[1] for n, v in all_results[l].items()}
                   for l in labels}, f, indent=2)
    print("saved:", os.path.join(OUT, "dice_results.json"))

    figure(cases, all_results, labels)
    if len(labels) > 1:
        dice_chart(all_results, labels, cases)


if __name__ == "__main__":
    main()
