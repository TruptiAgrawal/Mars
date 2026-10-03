"""
Lossless LoRA-delta packing for one BTCV CV fold's checkpoints.

Every checkpoint stores the full 422-tensor state dict (~377 MB), but training only
changes the 48 LoRA A/B tensors (~0.15 MB); every other tensor stays bit-identical to
model_epoch_0.pth. `pack` stores those 48 tensors for every epoch and iteration
checkpoint in one small .npz, then proves the pack is lossless by rebuilding each
checkpoint from (model_epoch_0.pth + pack) and comparing it BIT FOR BIT with the
original file. Only after `verify` passes is it safe to delete full snapshots.

    python btcv_cross_validation/scripts/fold_lora_pack.py pack    --fold 1
    python btcv_cross_validation/scripts/fold_lora_pack.py verify  --fold 1
    python btcv_cross_validation/scripts/fold_lora_pack.py rebuild --fold 1 --epoch 12 --out /tmp/model_epoch_12.pth

Keep model_epoch_0.pth: it is the base every rebuild starts from.
"""
import argparse
import json
import os
import re
import sys
import warnings
from collections import OrderedDict

import numpy as np
import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

CV_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ckpt_dir(fold):
    return os.path.join(CV_DIR, "runs", f"fold{fold}", "ddp_sub_cv", f"cv_fold{fold}", "ckpt")


def out_dir(fold):
    d = os.path.join(CV_DIR, "logs", f"fold{fold}_analysis")
    os.makedirs(d, exist_ok=True)
    return d


def pack_path(fold):
    return os.path.join(out_dir(fold), f"fold{fold}_lora_deltas.npz")


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def list_ckpts(fold):
    """[(tag, path)] sorted: epoch_00..epoch_NN then iter_00000... ; tags are zero-padded."""
    d = ckpt_dir(fold)
    epochs, iters = [], []
    for name in os.listdir(d):
        m = re.fullmatch(r"model_epoch_(\d+)\.pth", name)
        if m:
            epochs.append((int(m.group(1)), name))
        m = re.fullmatch(r"model_iter_(\d+)\.pth", name)
        if m:
            iters.append((int(m.group(1)), name))
    out = [(f"epoch_{n:02d}", os.path.join(d, nm)) for n, nm in sorted(epochs)]
    out += [(f"iter_{n:05d}", os.path.join(d, nm)) for n, nm in sorted(iters)]
    return out


def same_bits(a, b):
    """Bit-for-bit equality (same shape, dtype and raw bytes)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    try:
        x = np.ascontiguousarray(a.numpy()).reshape(-1).view(np.uint8)
        y = np.ascontiguousarray(b.numpy()).reshape(-1).view(np.uint8)
        return np.array_equal(x, y)
    except (TypeError, RuntimeError):  # dtype numpy cannot represent (e.g. bfloat16)
        return torch.equal(a, b)


def rebuild_state(base, lora_keys, arrays):
    """Full state dict: base tensors, with the LoRA tensors replaced from `arrays` (key -> ndarray)."""
    lk = set(lora_keys)
    return OrderedDict((k, torch.from_numpy(np.array(arrays[k])) if k in lk else base[k]) for k in base)


def cmd_pack(fold):
    cks = list_ckpts(fold)
    if not cks or cks[0][0] != "epoch_00":
        sys.exit("model_epoch_0.pth not found: it is the required base.")
    base = load(cks[0][1])
    last_epoch = [c for c in cks if c[0].startswith("epoch_")][-1]
    final = load(last_epoch[1])
    lora_keys = [k for k in base if not same_bits(base[k], final[k])]
    if not lora_keys:
        sys.exit("No tensors differ between epoch 0 and the final epoch; nothing to pack.")
    lk = set(lora_keys)
    assert all(base[k].dtype == torch.float32 for k in lora_keys), "LoRA tensors must be float32 for .npz"
    print(f"fold {fold}: {len(base)} tensors per checkpoint, {len(lora_keys)} LoRA tensors "
          f"({sum(base[k].numel() * 4 for k in lora_keys) / 1e6:.3f} MB) vs "
          f"{sum(v.numel() * v.element_size() for v in base.values()) / 1e6:.1f} MB full")
    del final

    arrays, tags = {}, []
    for tag, path in cks:
        sd = load(path)
        if list(sd.keys()) != list(base.keys()):
            sys.exit(f"{tag}: key set/order differs from epoch 0")
        bad = [k for k in base if k not in lk and not same_bits(base[k], sd[k])]
        if bad:
            sys.exit(f"{tag}: {len(bad)} non-LoRA tensors differ from epoch 0 (e.g. {bad[:2]}); "
                     "the frozen-weights assumption is broken, refusing to pack")
        for k in lora_keys:
            arrays[f"{tag}|{k}"] = sd[k].numpy().copy()
        tags.append(tag)
        print(f"  packed {tag:<12} ({os.path.basename(path)})")

    latest = os.path.join(ckpt_dir(fold), "..", "ckpt_v", "model_latest.pth")
    if os.path.isfile(latest):
        lat, fin = load(latest), load(last_epoch[1])
        same = list(lat.keys()) == list(fin.keys()) and all(same_bits(lat[k], fin[k]) for k in fin)
        print(f"ckpt_v/model_latest.pth is bit-identical to {last_epoch[0]}: {same}")

    meta = {"fold": fold, "base": "model_epoch_0.pth", "n_tensors": len(base), "n_lora": len(lora_keys)}
    dest = pack_path(fold)
    np.savez(dest, _lora_keys=np.array(lora_keys), _tags=np.array(tags), _meta=np.array(json.dumps(meta)), **arrays)
    print(f"wrote {dest} ({os.path.getsize(dest) / 1e6:.2f} MB, {len(tags)} checkpoints)")
    return cmd_verify(fold)


def cmd_verify(fold):
    z = np.load(pack_path(fold), allow_pickle=False)
    lora_keys, tags = list(z["_lora_keys"]), list(z["_tags"])
    paths = dict(list_ckpts(fold))
    base = load(paths["epoch_00"])
    n_bad = 0
    for tag in tags:
        orig = load(paths[tag])
        rebuilt = rebuild_state(base, lora_keys, {k: z[f"{tag}|{k}"] for k in lora_keys})
        ok = list(orig.keys()) == list(rebuilt.keys()) and all(same_bits(orig[k], rebuilt[k]) for k in orig)
        n_bad += (not ok)
        print(f"  {'OK  ' if ok else 'FAIL'} {tag:<12} rebuilt from pack == {os.path.basename(paths[tag])}")
    print(f"\nRESULT: {len(tags) - n_bad}/{len(tags)} checkpoints rebuild bit-exactly from "
          f"model_epoch_0.pth + {os.path.basename(pack_path(fold))}")
    if n_bad:
        sys.exit(1)


def cmd_rebuild(fold, epoch, it, out):
    tag = f"epoch_{epoch:02d}" if epoch is not None else f"iter_{it:05d}"
    if os.path.exists(out):
        sys.exit(f"Refusing to overwrite {out}")
    z = np.load(pack_path(fold), allow_pickle=False)
    lora_keys = list(z["_lora_keys"])
    base = load(dict(list_ckpts(fold))["epoch_00"])
    state = rebuild_state(base, lora_keys, {k: z[f"{tag}|{k}"] for k in lora_keys})
    torch.save(state, out)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB) rebuilt as {tag}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["pack", "verify", "rebuild"])
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--epoch", type=int)
    p.add_argument("--iter", type=int, dest="it")
    p.add_argument("--out")
    a = p.parse_args()
    if a.command == "pack":
        cmd_pack(a.fold)
    elif a.command == "verify":
        cmd_verify(a.fold)
    else:
        if (a.epoch is None) == (a.it is None) or not a.out:
            sys.exit("rebuild needs exactly one of --epoch/--iter, and --out")
        cmd_rebuild(a.fold, a.epoch, a.it, a.out)
