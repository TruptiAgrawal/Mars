"""
load_tb_recursive.py
--------------------
Slide-SAM's trainer writes each scalar into its own subdirectory
(tb/train/loss/, tb/train/dl/, ...), so a single EventAccumulator on tb/
finds nothing. This walks every leaf directory that contains an event file
and merges the results.

Usage:
    python3 load_tb_recursive.py ../runs/sam/ddp/word_label2_fresh_4/tb
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def event_dirs(root):
    """Every directory under root that directly contains a tfevents file."""
    out = []
    for dirpath, _, files in os.walk(root):
        if any("tfevents" in f for f in files):
            out.append(dirpath)
    return sorted(out)


def load_all(root):
    """Return {name: (steps, values)} merged across all subdirectories."""
    data = {}
    for d in event_dirs(root):
        acc = EventAccumulator(d, size_guidance={"scalars": 0})
        try:
            acc.Reload()
        except Exception as e:
            print(f"  skip {d}: {e}")
            continue
        rel = os.path.relpath(d, root).replace(os.sep, "/")
        for tag in acc.Tags()["scalars"]:
            ev = acc.Scalars(tag)
            steps = np.array([e.step for e in ev])
            vals = np.array([e.value for e in ev])
            # name it by folder when the tag itself is generic
            name = rel if rel != "." else tag
            if name in data:                      # merge duplicates
                s0, v0 = data[name]
                order = np.argsort(np.concatenate([s0, steps]))
                data[name] = (np.concatenate([s0, steps])[order],
                              np.concatenate([v0, vals])[order])
            else:
                data[name] = (steps, vals)
    return data


def report(data):
    if not data:
        print("still nothing -- check the path points at a tb/ folder")
        return
    print(f"{len(data)} scalar series found:\n")
    for k in sorted(data):
        s, v = data[k]
        print(f"  {k:28s} {len(s):7d} pts   step {s.min()}-{s.max()}   "
              f"first={v[0]:.5g}  last={v[-1]:.5g}")


def smooth(v, w=0.9):
    out, last = [], v[0]
    for x in v:
        last = last * w + x * (1 - w)
        out.append(last)
    return np.array(out)


def plot_curves(data, keys=None, outdir="figures", w=0.95):
    os.makedirs(outdir, exist_ok=True)
    wanted = keys or [k for k in data
                      if any(t in k for t in ("loss", "/dl", "/fl", "lr"))
                      and "time" not in k]
    if not wanted:
        wanted = list(data)[:6]

    # one combined figure
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for k in wanted:
        if "lr" in k:
            continue
        s, v = data[k]
        ax.plot(s, v, lw=0.7, alpha=0.2)
        ax.plot(s, smooth(v, w), lw=2, label=k.split("/")[-1])
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title("Training loss components")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, "tb_loss_components.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("saved:", p)

    # small multiples -- one panel per series, easier to read
    n = len(wanted)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow),
                             squeeze=False)
    for i, k in enumerate(wanted):
        ax = axes[i // ncol][i % ncol]
        s, v = data[k]
        ax.plot(s, v, lw=0.7, alpha=0.25, color="tab:blue")
        ax.plot(s, smooth(v, w), lw=1.8, color="tab:blue")
        ax.set_title(k.split("/")[-1], fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlabel("iter", fontsize=8)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.tight_layout()
    p = os.path.join(outdir, "tb_small_multiples.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("saved:", p)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else \
        "../runs/sam/ddp/word_label2_fresh_4/tb"
    print("scanning:", root)
    for d in event_dirs(root):
        print("   event dir:", d)
    data = load_all(root)
    report(data)
    if data:
        plot_curves(data)
