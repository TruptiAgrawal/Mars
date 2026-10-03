"""
CPU-only analysis of one CV fold's training: loss curves from the training log and
LoRA weight-change curves from the packed deltas (see fold_lora_pack.py).

Reads : btcv_cross_validation/logs/train_foldN.log, logs/foldN_analysis/foldN_lora_deltas.npz
Writes: logs/foldN_analysis/foldN_epoch_losses.csv, foldN_iter_losses.csv,
        foldN_lora_weight_change.csv, foldN_lora_dW_by_block.csv,
        foldN_loss_curves.png, foldN_lora_weight_change.png,
        foldN_validation_dice.png and .csv (when evaluation JSON exists)

    python btcv_cross_validation/scripts/fold_analysis.py --fold 1
"""
import argparse
import ast
import csv
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

CV_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, BLUE_LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#86b6ef"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.solid_capstyle": "round",
})


def title(ax, text, sub=None):
    ax.set_title(text, pad=22 if sub else 8)
    if sub:
        ax.text(0, 1.03, sub, transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")


def parse_log(path):
    raw = open(path, "rb").read().replace(b"\0", b"").decode("utf-8", "replace")
    epochs, iters = {}, {}
    ep_re = re.compile(r"Epoch (\d+): loss:([\d.]+) point_loss:([\d.]+) box_loss:([\d.]+) fl:([\d.]+) dl:([\d.]+) "
                       r"min_losses:([\d.]+) selected_losses:([\d.]+).*?time_total:([\d.]+) lr:([\d.eE+-]+)")
    it_re = re.compile(r"Epoch: (\d+) \| batch:(\d+)/(\d+), Iteration:(\d+), results: (\{.*\})\s*$")
    ts_re = re.compile(r"^\[([\d\- :]+)\]")
    per_epoch_batches = None
    for line in raw.splitlines():
        m = ep_re.search(line)
        if m:
            g = m.groups()
            t = ts_re.search(line)
            epochs[int(g[0])] = dict(epoch=int(g[0]), timestamp=t.group(1) if t else "", loss=float(g[1]),
                                     point_loss=float(g[2]), box_loss=float(g[3]), fl=float(g[4]), dl=float(g[5]),
                                     min_losses=float(g[6]), selected_losses=float(g[7]),
                                     time_total_s=float(g[8]), lr=float(g[9]))
            continue
        m = it_re.search(line)
        if m:
            d = ast.literal_eval(m.group(5))
            per_epoch_batches = int(m.group(3))
            iters[int(m.group(4))] = dict(iteration=int(m.group(4)), epoch=int(m.group(1)), batch=int(m.group(2)),
                                          loss=d["loss"], point_loss=d["point_loss"], box_loss=d["box_loss"])
    return epochs, iters, per_epoch_batches


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def moving_avg(y, w):
    k = np.ones(w) / w
    return np.convolve(y, k, mode="valid")


def loss_figure(epochs, iters, n_per_epoch, out, fold=1):
    e = np.array(sorted(epochs))
    tot = np.array([epochs[i]["loss"] for i in e])
    pt = np.array([epochs[i]["point_loss"] for i in e])
    bx = np.array([epochs[i]["box_loss"] for i in e])
    it = np.array(sorted(iters))
    x_it = it / n_per_epoch
    y_it = np.array([iters[i]["loss"] for i in it])

    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.9), dpi=150)
    for y, c, name in ((tot, BLUE, "total"), (pt, ORANGE, "point"), (bx, AQUA, "box")):
        a.plot(e, y, color=c, lw=2, marker="o", ms=4.5, mfc=c, mec=SURFACE, mew=1, label=name)
        a.text(e[-1] + 0.4, y[-1], f"{name} {y[-1]:.3f}", color=INK, va="center", fontsize=9)
    a.set_xlim(0.5, 23.2)
    a.set_ylim(0, 0.6)
    a.set_xticks([1, 5, 10, 15, 20])
    a.set_xlabel("epoch")
    a.set_ylabel("mean loss over the epoch")
    a.legend(loc="upper right", bbox_to_anchor=(0.93, 0.97))
    title(a, "Training loss per epoch", f"total = point + box; fold{fold}, 24 training volumes, batch size 4")

    w = 10
    ma = moving_avg(y_it, w)
    b.plot(x_it, y_it, color=BLUE_LIGHT, lw=1, label="every 100 iterations")
    b.plot(x_it[w - 1:], ma, color=BLUE, lw=2, label="moving average (10 samples)")
    b.text(x_it[-1] + 0.15, ma[-1], f"{ma[-1]:.3f}", color=INK, va="center", fontsize=9)
    b.set_xlim(0, 20.9)
    b.set_ylim(0, max(y_it) * 1.05)
    b.set_xticks([0, 5, 10, 15, 20])
    b.set_xlabel("epoch")
    b.set_ylabel("total loss (one batch)")
    b.legend(loc="lower center", ncol=2)
    title(b, "Total loss every 100 iterations",
          "single-batch values are noisy; the 10-sample moving average (about half an epoch) shows the trend")
    fig.text(0.01, 0.005, f"Source: train_fold{fold}.log. Values: fold{fold}_epoch_losses.csv, fold{fold}_iter_losses.csv",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def weight_analysis(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    keys = list(z["_lora_keys"])
    tags = [t for t in z["_tags"] if t.startswith("epoch_")]
    theta = {int(t[6:]): np.concatenate([z[f"{t}|{k}"].astype(np.float64).ravel() for k in keys]) for t in tags}
    saved_epochs = sorted(theta)
    ep = saved_epochs[1:]
    previous = saved_epochs[:-1]
    n_ep = len(ep)
    d0 = np.array([np.linalg.norm(theta[n] - theta[0]) for n in ep])
    step = np.array([np.linalg.norm(theta[n] - theta[p]) for p, n in zip(previous, ep)])
    updates = [theta[n] - theta[p] for p, n in zip(previous, ep)]
    cos = np.array([float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                    if np.linalg.norm(a) * np.linalg.norm(b) else np.nan
                    for a, b in zip(updates, updates[1:])])

    pat = re.compile(r"blocks\.(\d+)\.attn\.qkv\.linear_([ab])_([qv])\.weight")
    idx = {}
    for k in keys:
        m = pat.search(k)
        if not m:
            sys.exit(f"unexpected LoRA tensor name: {k}")
        idx[(int(m.group(1)), m.group(3), m.group(2))] = k
    blocks = sorted({b for b, _, _ in idx})

    def ab(tag, b, s):
        return z[f"{tag}|{idx[(b, s, 'a')]}"].astype(np.float64), z[f"{tag}|{idx[(b, s, 'b')]}"].astype(np.float64)

    A0, B0 = ab("epoch_00", blocks[0], "q")
    rank = A0.shape[0]
    assert B0.shape[1] == rank, f"unexpected LoRA shapes {A0.shape} {B0.shape}"
    dW = np.zeros((len(blocks), n_ep, 2))          # block x epoch(1..) x {q, v}
    init_norm = np.zeros((len(blocks), 2))
    for bi, b in enumerate(blocks):
        for si, s in enumerate(("q", "v")):
            a0, b0 = ab("epoch_00", b, s)
            w0 = b0 @ a0
            init_norm[bi, si] = np.linalg.norm(w0)
            for ni, n in enumerate(ep):
                an, bn = ab(f"epoch_{n:02d}", b, s)
                dW[bi, ni, si] = np.linalg.norm(bn @ an - w0)
    return dict(epochs=ep, previous_epochs=previous, keys=keys, rank=rank, shape_a=A0.shape, shape_b=B0.shape, n_ep=n_ep, d0=d0, step=step, cos=cos,
                blocks=blocks, dW=dW, init_norm=init_norm, n_params=sum(theta[0].shape))


def weight_figure(W, out, fold=1):
    ep = np.array(W["epochs"])
    interval = "epoch" if np.all(np.diff([0, *ep]) == 1) else "saved checkpoint interval"
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.8), dpi=150)
    (a, b), (c, d) = ax

    a.plot(ep, W["d0"], color=BLUE, lw=2, marker="o", ms=4.5, mfc=BLUE, mec=SURFACE, mew=1)
    a.text(ep[-1] + 0.3, W["d0"][-1], f"{W['d0'][-1]:.2f}", va="center", fontsize=9)
    a.set_xlim(0.5, 21.6); a.set_ylim(bottom=0); a.set_xticks([1, 5, 10, 15, 20])
    a.set_xlabel("epoch"); a.set_ylabel("L2 distance of all 48 LoRA tensors")
    title(a, "How far training moved the weights", "distance from the epoch-0 (starting) LoRA weights")

    b.plot(ep, W["step"], color=BLUE, lw=2, marker="o", ms=4.5, mfc=BLUE, mec=SURFACE, mew=1)
    b.text(ep[-1] + 0.3, W["step"][-1], f"{W['step'][-1]:.2f}", va="center", fontsize=9)
    b.set_xlim(0.5, 21.6); b.set_ylim(bottom=0); b.set_xticks([1, 5, 10, 15, 20])
    b.set_xlabel("epoch"); b.set_ylabel(f"L2 change during the {interval}")
    title(b, f"Update size per {interval}", "change since the preceding saved checkpoint")

    c.axhline(0, color=INK2, lw=1)
    c.plot(ep[1:], W["cos"], color=BLUE, lw=2, marker="o", ms=4.5, mfc=BLUE, mec=SURFACE, mew=1)
    c.set_xlim(1.5, 20.6); c.set_ylim(-1, 1); c.set_xticks([2, 5, 10, 15, 20])
    c.set_xlabel("epoch (end of later checkpoint interval)"); c.set_ylabel("cosine similarity")
    title(c, "Do consecutive updates point the same way?", "near 1: steady progress; near 0 or below: noise or oscillation")

    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    comb = np.sqrt((W["dW"] ** 2).sum(axis=2))       # block x epoch
    im = d.imshow(comb, aspect="auto", cmap=cmap, origin="upper")
    d.grid(False)
    tick_idx = np.unique(np.linspace(0, len(ep) - 1, min(5, len(ep)), dtype=int))
    d.set_xticks(tick_idx, ep[tick_idx]); d.set_yticks(range(len(W["blocks"])), W["blocks"])
    d.set_xlabel("epoch"); d.set_ylabel("encoder block")
    for s in d.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=d, fraction=0.046, pad=0.02)
    cb.set_label("effective weight change (q and v combined)", fontsize=9)
    cb.outline.set_visible(False)
    title(d, "Which blocks changed, and when",
          "darker = larger change from the starting model (Frobenius norm of the B·A change)")

    fig.text(0.01, 0.005, f"Source: fold{fold}_lora_deltas.npz (48 LoRA tensors per checkpoint). Values: fold{fold}_lora_weight_change.csv, "
             f"fold{fold}_lora_dW_by_block.csv", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out)
    plt.close(fig)
    return comb


def validation_figure(path, out, csv_path):
    with open(path) as source:
        result = json.load(source)
    organs = list(result["organs"])
    volumes = sorted(result["organs"][organs[0]]["per_volume_dice"])
    scores = np.array([[result["organs"][o]["per_volume_dice"][v] for v in volumes] for o in organs])
    means, stds = scores.mean(axis=1), scores.std(axis=1, ddof=1)
    assert np.allclose(means, [result["organs"][o]["mean_dice"] for o in organs])
    assert np.allclose(stds, [result["organs"][o]["std_dice"] for o in organs])
    assert np.isclose(means.mean(), result["mean_dice_4_organs"])
    write_csv(csv_path, [dict(organ=o, volume=v, dice=scores[i, j])
                         for i, o in enumerate(organs) for j, v in enumerate(volumes)])
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.9), dpi=150)
    y = np.arange(len(organs))
    a.barh(y, 100 * means, xerr=100 * stds, color=BLUE, height=0.55,
           error_kw=dict(ecolor=INK2, capsize=4))
    for i, mean in enumerate(means):
        a.text(2, i, f"{100 * mean:.2f}%", va="center", color="white", weight="bold")
    a.axvline(100 * means.mean(), color=ORANGE, ls="--", label=f"Overall mean: {100 * means.mean():.2f}%")
    a.set_yticks(y, organs)
    a.invert_yaxis()
    a.set_xlim(0, 100)
    a.set_xlabel("Dice score (%)")
    a.set_ylim(len(organs) - 0.15, -0.65)
    a.legend(loc="lower right", fontsize=8)
    title(a, "Validation Dice by organ", f"Epoch {result['epoch']}; {result['prompt']} prompts; mean ± SD across {len(volumes)} volumes")
    im = b.imshow(100 * scores, vmin=0, vmax=100, cmap=LinearSegmentedColormap.from_list("dice_blue", BLUE_RAMP), aspect="auto")
    b.grid(False)
    b.set_xticks(range(len(volumes)), volumes, rotation=35, ha="right")
    b.set_yticks(y, organs)
    for i in y:
        for j in range(len(volumes)):
            b.text(j, i, f"{100 * scores[i, j]:.1f}", ha="center", va="center",
                   color="white" if scores[i, j] >= 0.6 else INK)
    fig.colorbar(im, ax=b, fraction=0.046, pad=0.03, label="Dice score (%)")
    title(b, "Validation Dice by volume", "Each cell is one organ in one held-out volume")
    fig.suptitle(f"Fold {result['fold']} · validation segmentation performance", fontsize=14, weight="bold")
    fig.text(0.01, 0.012, f"Source: {os.path.basename(path)}. Dice measures mask overlap, not voxel accuracy. Only the final epoch was evaluated.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.savefig(out)
    plt.close(fig)
    print("\n=== validation Dice summary ===")
    print(f"epoch {result['epoch']}, {result['prompt']} prompts, {len(volumes)} volumes; four-organ macro mean: {means.mean():.6f}")
    for organ, mean, std in zip(organs, means, stds):
        print(f"{organ}: {100 * mean:.2f}% ± {100 * std:.2f}% (sample SD across volumes)")
    print("Dice is segmentation overlap, not voxel accuracy; no per-epoch validation curve is available.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    f = ap.parse_args().fold
    out = os.path.join(CV_DIR, "logs", f"fold{f}_analysis")
    os.makedirs(out, exist_ok=True)
    P = lambda name: os.path.join(out, f"fold{f}_{name}")

    epochs, iters, npe = parse_log(os.path.join(CV_DIR, "logs", f"train_fold{f}.log"))
    print("=== log parse integrity ===")
    miss_e = sorted(set(range(1, max(epochs) + 1)) - set(epochs))
    print(f"epoch summary lines: {len(epochs)} (epochs {min(epochs)}..{max(epochs)}), missing: {miss_e or 'none'}")
    expected = set(range(0, max(epochs) * npe, 100))
    got = set(iters)
    print(f"batches per epoch: {npe} | per-100-iteration lines: {len(got)} (expected {len(expected)}), "
          f"missing: {len(expected - got)}, unexpected: {len(got - expected)}")
    write_csv(P("epoch_losses.csv"), [epochs[k] for k in sorted(epochs)])
    write_csv(P("iter_losses.csv"), [iters[k] for k in sorted(iters)])
    loss_figure(epochs, iters, npe, P("loss_curves.png"), fold=f)

    evaluation = os.path.join(CV_DIR, "logs", f"fold{f}_evaluation.json")
    if os.path.isfile(evaluation):
        validation_figure(evaluation, P("validation_dice.png"), P("validation_dice.csv"))

    W = weight_analysis(P("lora_deltas.npz"))
    comb = weight_figure(W, P("lora_weight_change.png"), fold=f)
    write_csv(P("lora_weight_change.csv"), [dict(epoch=W["epochs"][n], previous_epoch=W["previous_epochs"][n], l2_from_epoch0=W["d0"][n], l2_step=W["step"][n],
                                                cos_with_previous_step=(W["cos"][n - 1] if n >= 1 else ""))
                                            for n in range(W["n_ep"])])
    write_csv(P("lora_dW_by_block.csv"), [dict(block=b, epoch=W["epochs"][n], dW_q=W["dW"][bi, n, 0], dW_v=W["dW"][bi, n, 1],
                                               dW_combined=comb[bi, n])
                                          for bi, b in enumerate(W["blocks"]) for n in range(W["n_ep"])])

    e = sorted(epochs)
    L = np.array([epochs[i]["loss"] for i in e])
    print("\n=== loss summary ===")
    print(f"epoch 1 -> {e[-1]}: {L[0]:.4f} -> {L[-1]:.4f}  ({100 * (L[-1] / L[0] - 1):+.1f}%)")
    print(f"point {epochs[1]['point_loss']:.4f} -> {epochs[e[-1]]['point_loss']:.4f} | box {epochs[1]['box_loss']:.4f} -> {epochs[e[-1]]['box_loss']:.4f}")
    print(f"lowest epoch loss: epoch {e[int(np.argmin(L))]} ({L.min():.4f})")
    print(f"mean gain per epoch, epochs 1-5 : {(L[0] - L[4]) / 4:.4f}   | epochs 16-20: {(L[-5] - L[-1]) / 4:.4f}")
    print(f"average epoch time: {np.mean([epochs[i]['time_total_s'] for i in e]) / 60:.1f} min")
    yi = np.array([iters[i]["loss"] for i in sorted(iters)])
    print(f"per-100-iteration loss: median {np.median(yi):.3f}, 95th pct {np.percentile(yi, 95):.3f}, max {yi.max():.3f}")
    diffs = [abs(epochs[i]["loss"] - np.mean([v["loss"] for v in iters.values() if v["epoch"] == i])) for i in e]
    print(f"epoch mean vs mean of sampled batches: mean abs diff {np.mean(diffs):.3f} (sampling noise, cross-check only)")

    print("\n=== LoRA weight-change summary ===")
    print(f"48 tensors; per site A{W['shape_a']} B{W['shape_b']} (rank {W['rank']}), {W['n_params']:,} trainable values")
    print(f"starting ||B0 A0|| across blocks: min {W['init_norm'].min():.4f}, max {W['init_norm'].max():.4f}")
    print(f"distance from start: epoch {W['epochs'][0]} {W['d0'][0]:.2f} -> epoch {W['epochs'][-1]} {W['d0'][-1]:.2f}")
    print(f"checkpoint update size: epoch {W['epochs'][0]} {W['step'][0]:.2f}, epoch {W['epochs'][1]} {W['step'][1]:.2f}, epoch {W['epochs'][-1]} {W['step'][-1]:.2f}")
    print(f"saved weight epochs: {[0, *W['epochs']]} (updates span consecutive saved checkpoints)")
    print(f"consecutive checkpoint-update cosine: mean {W['cos'].mean():.2f}")
    tot = comb[:, -1]
    order = np.argsort(tot)[::-1]
    print("blocks by final change (share of total): " +
          ", ".join(f"{W['blocks'][i]} ({100 * tot[i] / tot.sum():.0f}%)" for i in order[:4]) +
          f" | lowest: block {W['blocks'][order[-1]]} ({100 * tot[order[-1]] / tot.sum():.0f}%)")
    qv = W["dW"][:, -1, :].sum(axis=0)
    print(f"q vs v share of final change: q {100 * qv[0] / qv.sum():.0f}%, v {100 * qv[1] / qv.sum():.0f}%")


if __name__ == "__main__":
    main()
