"""
slidesam_viz.py
---------------
Visualization helpers for a Slide-SAM style project.

Drop this next to your notebook (e.g. in ~/Slide-SAM/) and use:

    from slidesam_viz import *

Nothing here assumes a specific dataset layout -- start with `discover()`
to see what you actually have, then feed paths into the plotting helpers.
"""

import os
import re
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --------------------------------------------------------------------------
# 0. Discovery -- run this FIRST so you know what you're working with
# --------------------------------------------------------------------------

def discover(root=".", run_dir="runs", max_show=15):
    """Print what logs / checkpoints / dataset files exist."""
    print("=" * 70)
    print("TENSORBOARD EVENT FILES")
    ev = glob.glob(os.path.join(root, run_dir, "**", "*tfevents*"), recursive=True)
    for p in ev[:max_show]:
        print("  ", p)
    if not ev:
        print("   (none)")

    print("\nTEXT / CSV / JSON LOGS")
    logs = []
    for ext in ("*.log", "*.txt", "*.csv", "*.json"):
        logs += glob.glob(os.path.join(root, run_dir, "**", ext), recursive=True)
    for p in logs[:max_show]:
        print("  ", p, f"({os.path.getsize(p)/1024:.1f} KB)")
    if not logs:
        print("   (none)")

    print("\nCHECKPOINTS")
    ck = glob.glob(os.path.join(root, run_dir, "**", "*.pt*"), recursive=True)
    ck += glob.glob(os.path.join(root, run_dir, "**", "*.ckpt"), recursive=True)
    for p in ck[:max_show]:
        print("  ", p, f"({os.path.getsize(p)/1e6:.1f} MB)")
    if not ck:
        print("   (none)")

    print("\nDATASET FILES (first few of each type)")
    for ext in ("*.npz", "*.npy", "*.h5", "*.hdf5", "*.nii.gz", "*.nii"):
        hits = glob.glob(os.path.join(root, "datasets", "**", ext), recursive=True)
        if hits:
            print(f"   {ext}: {len(hits)} files")
            for p in hits[:3]:
                print("      ", p)
    print("=" * 70)
    return {"events": ev, "logs": logs, "ckpt": ck}


# --------------------------------------------------------------------------
# 1. Training curves
# --------------------------------------------------------------------------

def load_tensorboard(logdir):
    """Return {tag: (steps, values)} from a tensorboard run directory."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(logdir, size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for tag in acc.Tags()["scalars"]:
        ev = acc.Scalars(tag)
        out[tag] = (np.array([e.step for e in ev]),
                    np.array([e.value for e in ev]))
    print("tags found:", list(out))
    return out


def parse_text_log(path,
                   pattern=r"epoch[:\s]*(\d+).*?loss[:\s]*([0-9.]+)"):
    """Fallback: scrape 'epoch ... loss ...' lines out of a plain text log.
    Adjust `pattern` to match however your trainer prints."""
    epochs, losses = [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = re.search(pattern, line, flags=re.I)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return np.array(epochs), np.array(losses)


def plot_loss(steps, values, label="train loss", smooth=0.9, ax=None,
              xlabel="iteration"):
    """Raw loss (faint) + exponentially smoothed loss (bold)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, values, lw=0.8, alpha=0.25, color="tab:blue")
    if smooth and len(values) > 1:
        sm, last = [], values[0]
        for v in values:
            last = last * smooth + v * (1 - smooth)
            sm.append(last)
        ax.plot(steps, sm, lw=2, color="tab:blue", label=f"{label} (smoothed)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("loss")
    ax.set_title("Training loss")
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


def plot_epoch_metric(epochs, values, name="Dice", ax=None):
    """Few-epoch metric: markers + value labels, so 4 points read as honest."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ax.plot(epochs, values, "o-", lw=2, ms=8, color="tab:green")
    for e, v in zip(epochs, values):
        ax.annotate(f"{v:.3f}", (e, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9)
    ax.set_xticks(list(epochs))
    ax.set_xlabel("epoch")
    ax.set_ylabel(name)
    ax.set_title(f"Validation {name} ({len(epochs)} epochs completed)")
    ax.grid(alpha=0.3)
    ax.margins(y=0.18)
    return ax


# --------------------------------------------------------------------------
# 2. Dataset samples
# --------------------------------------------------------------------------

# up to 4 foreground objects + background
CLASS_COLORS = ["#00000000", "#e6194b", "#3cb44b", "#4363d8", "#ffe119"]


def mask_cmap(n_classes=5):
    return ListedColormap(CLASS_COLORS[:n_classes])


def load_case(path):
    """Load one case as (image, label). Handles npz / npy / h5 / nifti."""
    ext = path.lower()
    if ext.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        keys = list(d.keys())
        img_k = next((k for k in keys if k.lower() in
                      ("image", "img", "data", "arr_0")), keys[0])
        lab_k = next((k for k in keys if k.lower() in
                      ("label", "mask", "seg", "gt", "arr_1")), None)
        return d[img_k], (d[lab_k] if lab_k else None)
    if ext.endswith(".npy"):
        return np.load(path), None
    if ext.endswith((".h5", ".hdf5")):
        import h5py
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            img_k = next((k for k in keys if k.lower() in
                          ("image", "img", "data")), keys[0])
            lab_k = next((k for k in keys if k.lower() in
                          ("label", "mask", "seg")), None)
            return f[img_k][:], (f[lab_k][:] if lab_k else None)
    if ext.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        return nib.load(path).get_fdata(), None
    raise ValueError(f"unsupported file type: {path}")


def norm(img):
    img = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(img, [0.5, 99.5])
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)


def overlay(ax, img, mask=None, alpha=0.45, n_classes=5, title=None):
    ax.imshow(norm(img), cmap="gray")
    if mask is not None:
        m = np.ma.masked_where(mask == 0, mask)
        ax.imshow(m, cmap=mask_cmap(n_classes), vmin=0, vmax=n_classes - 1,
                  alpha=alpha, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def show_slice_triplet(volume, label, z, dataset_name=""):
    """The three adjacent slices Slide-SAM actually consumes.
    Assumes volume shaped (Z, H, W)."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for j, off in enumerate((-1, 0, 1)):
        zz = np.clip(z + off, 0, volume.shape[0] - 1)
        tag = "prompted slice" if off == 0 else f"z{off:+d}"
        overlay(axes[0, j], volume[zz], None, title=f"{tag} (image)")
        overlay(axes[1, j], volume[zz],
                label[zz] if label is not None else None,
                title=f"{tag} (+ mask)")
    fig.suptitle(f"{dataset_name}  |  3-slice input window at z={z}",
                 fontsize=12)
    fig.tight_layout()
    return fig


def dataset_grid(samples, n_classes=5):
    """samples: list of dicts {'name', 'image' (2D), 'label' (2D)}.
    One row per dataset: image | mask | overlay."""
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for i, s in enumerate(samples):
        overlay(axes[i, 0], s["image"], None,
                title="image" if i == 0 else None)
        axes[i, 1].imshow(s["label"], cmap=mask_cmap(n_classes),
                          vmin=0, vmax=n_classes - 1, interpolation="nearest")
        axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])
        if i == 0:
            axes[i, 1].set_title("ground-truth mask")
        overlay(axes[i, 2], s["image"], s["label"], n_classes=n_classes,
                title="overlay" if i == 0 else None)
        axes[i, 0].set_ylabel(s["name"], fontsize=10, rotation=0,
                              ha="right", va="center", labelpad=10)
    fig.tight_layout()
    return fig


def class_balance(labels, class_names=None, dataset_name=""):
    """labels: array (or list) of integer masks. Shows pixel fraction per class
    on a log scale -- this is how you explain poor Dice on small organs."""
    lab = np.concatenate([np.asarray(l).ravel() for l in np.atleast_1d(labels)]) \
        if isinstance(labels, (list, tuple)) else np.asarray(labels).ravel()
    ids, counts = np.unique(lab, return_counts=True)
    frac = counts / counts.sum()
    names = class_names or [f"class {i}" for i in ids]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar([str(n) for n in names], frac,
           color=[CLASS_COLORS[min(int(i), 4)] if i else "#999999" for i in ids])
    ax.set_yscale("log")
    ax.set_ylabel("pixel fraction (log)")
    ax.set_title(f"Class balance {dataset_name}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def intensity_hist(volumes, names):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for v, n in zip(volumes, names):
        ax.hist(np.asarray(v).ravel(), bins=120, histtype="step",
                density=True, lw=1.6, label=n)
    ax.set_xlabel("voxel intensity")
    ax.set_ylabel("density")
    ax.set_title("Intensity distribution per dataset")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 3. Results
# --------------------------------------------------------------------------

def dice(pred, gt, cls):
    p, g = (pred == cls), (gt == cls)
    denom = p.sum() + g.sum()
    return 1.0 if denom == 0 else 2.0 * (p & g).sum() / denom


def dice_bars(dice_by_class, class_names=None, title="Dice per class"):
    ids = list(range(len(dice_by_class)))
    names = class_names or [f"class {i+1}" for i in ids]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, dice_by_class,
                  color=[CLASS_COLORS[min(i + 1, 4)] for i in ids])
    for b, v in zip(bars, dice_by_class):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9)
    ax.axhline(np.mean(dice_by_class), ls="--", c="k", lw=1,
               label=f"mean = {np.mean(dice_by_class):.3f}")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Dice")
    ax.set_title(title); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def dice_distribution(dice_per_case_per_class, class_names=None):
    """dice_per_case_per_class: list of lists, one inner list per class.
    A box plot shows spread, which a single mean hides."""
    names = class_names or [f"class {i+1}"
                            for i in range(len(dice_per_case_per_class))]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.boxplot(dice_per_case_per_class, labels=names, showmeans=True)
    for i, vals in enumerate(dice_per_case_per_class, start=1):
        ax.scatter(np.random.normal(i, 0.05, len(vals)), vals,
                   s=14, alpha=0.5, color="tab:blue", zorder=3)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Dice")
    ax.set_title("Per-case Dice distribution")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def pred_vs_gt(image, gt, pred, n_classes=5, case=""):
    """The panel your professor will look at first."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    overlay(axes[0], image, None, title="image")
    overlay(axes[1], image, gt, n_classes=n_classes, title="ground truth")
    overlay(axes[2], image, pred, n_classes=n_classes, title="prediction")
    err = np.zeros_like(gt)
    err[(gt > 0) & (pred == 0)] = 1        # missed
    err[(gt == 0) & (pred > 0)] = 2        # false positive
    err[(gt > 0) & (pred > 0) & (gt != pred)] = 3   # wrong class
    axes[3].imshow(norm(image), cmap="gray")
    axes[3].imshow(np.ma.masked_where(err == 0, err),
                   cmap=ListedColormap(["#00000000", "#e6194b",
                                        "#4363d8", "#ffe119"]),
                   vmin=0, vmax=3, alpha=0.6, interpolation="nearest")
    axes[3].set_xticks([]); axes[3].set_yticks([])
    axes[3].set_title("error  (red=missed, blue=FP, yellow=wrong class)",
                      fontsize=9)
    fig.suptitle(f"Qualitative result {case}", fontsize=12)
    fig.tight_layout()
    return fig


def propagation_curve(offsets, dice_values, title="Dice vs. distance from prompted slice"):
    """The signature Slide-SAM plot: how far the mask survives propagation."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(offsets, dice_values, "o-", lw=2, color="tab:purple")
    ax.axvline(0, ls="--", c="k", lw=1, label="prompted slice")
    ax.set_xlabel("slice offset from prompt")
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def save(fig, name, outdir="figures", dpi=200):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print("saved:", path)
    return path
