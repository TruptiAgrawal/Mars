"""
Evaluate one BTCV cross-validation fold on its 6 held-out volumes.

Loads the fold's final checkpoint once, then runs the existing box-prompt
evaluation (test.volume_eval_cv.Evaluater.eval_step, unchanged) for each of
the four paper-aligned organs:

    6 = Liver, 7 = Stomach, 3 = Left Kidney, 1 = Spleen

Test volumes come from FoldTestLoader, which reads exactly
dataset_list/foldN/01_BTCV_test_4to1.txt.

Checkpoint policy: the final epoch (model_epoch_20.pth) is used for every
fold. There is no validation split, so choosing an epoch by test Dice would
leak the held-out volumes into model selection.

Outputs (never overwritten unless --overwrite is passed):
    btcv_cross_validation/logs/foldN_evaluation.log   full stdout/stderr
    btcv_cross_validation/logs/foldN_evaluation.json  per-volume + mean Dice

Usage (from the repo root, slidesam env):
    python btcv_cross_validation/tests/eval_fold.py --fold 1
"""
import argparse
import json
import os
import sys
import time

CV_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(CV_DIR)
sys.path.insert(0, REPO_ROOT)

ORGANS = {6: "Liver", 7: "Stomach", 3: "Left Kidney", 1: "Spleen"}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--prompt", default="box", choices=["box"],
                        help="Only box is supported: eval_step_point in test/volume_eval_cv.py "
                             "calls an undefined compute_surface_dice.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    log_path = os.path.join(CV_DIR, "logs", f"fold{args.fold}_evaluation.log")
    json_path = os.path.join(CV_DIR, "logs", f"fold{args.fold}_evaluation.json")
    if not args.overwrite and (os.path.exists(log_path) or os.path.exists(json_path)):
        sys.exit(f"Refusing to overwrite {log_path} / {json_path} (pass --overwrite).")

    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tutils.new.manager import ConfigManager

    from core.learner_sub1 import SamLearner
    from core.volume_predictor import VolumePredictor
    from datasets.data_engine import DataManager
    from datasets.eval_dataloader.loader_cv import FoldTestLoader
    from modeling.build_sam3d2 import sam_model_registry
    from test.volume_eval_cv import Evaluater

    torch.manual_seed(0)
    np.random.seed(0)

    config_path = os.path.join(CV_DIR, "configs", f"fold{args.fold}.yaml")
    ckpt_path = os.path.join(CV_DIR, "runs", f"fold{args.fold}", "ddp_sub_cv",
                             f"cv_fold{args.fold}", "ckpt", f"model_epoch_{args.epoch}.pth")
    if not os.path.isfile(ckpt_path):
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    print("=" * 70)
    print(f"BTCV CV EVALUATION  fold={args.fold}  epoch={args.epoch}  prompt={args.prompt}")
    print(f"started:    {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"config:     {config_path}")
    print(f"checkpoint: {ckpt_path}")
    print("=" * 70)

    config = ConfigManager()
    config.add_config(config_path)
    config.add_config({"dataset": {"prompt": args.prompt, "label_idx": 0}, "pth": ckpt_path})

    sam = sam_model_registry["vit_b"](checkpoint=None)
    learner = SamLearner(sam_model=sam, config=config, data_engine=DataManager(img_size=(1024, 1024)))
    learner.use_lora()
    learner.use_lora_sub()
    learner.load_well_trained_model(ckpt_path)
    learner.cuda()
    learner.model.eval()
    predictor = VolumePredictor(model=learner.model, use_postprocess=True, use_noise_remove=True)

    results = {"fold": args.fold, "epoch": args.epoch, "prompt": args.prompt,
               "checkpoint": ckpt_path, "organs": {}}

    for label_idx, organ in ORGANS.items():
        print("\n" + "-" * 70)
        print(f"Organ: {organ} (label_idx={label_idx})")
        print("-" * 70)
        config["dataset"]["label_idx"] = label_idx
        solver = Evaluater(config)
        solver.predictor = predictor
        dataset = FoldTestLoader(config["dataset"], split="test")
        assert len(dataset) == 6, f"Expected 6 held-out volumes, got {len(dataset)}"

        per_volume = {}
        t0 = time.time()
        with torch.no_grad():
            for i, data in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):
                res = solver.eval_step(data, batch_idx=i)
                vol = os.path.basename(data["name"][0]).replace(".nii.gz", "")
                per_volume[vol] = float(res["dice"])

        dice = np.array(list(per_volume.values()))
        results["organs"][organ] = {
            "label_idx": label_idx,
            "per_volume_dice": per_volume,
            "mean_dice": float(dice.mean()),
            "std_dice": float(dice.std(ddof=1)),
        }
        print(f"[{organ}] mean Dice = {dice.mean():.4f} +/- {dice.std(ddof=1):.4f}  "
              f"({time.time() - t0:.0f}s)")

    print("\n" + "=" * 70)
    print(f"FOLD {args.fold} SUMMARY (Dice over 6 held-out volumes, mean +/- SD)")
    print("=" * 70)
    for organ, r in results["organs"].items():
        print(f"  {organ:<12} {r['mean_dice']:.4f} +/- {r['std_dice']:.4f}")
    avg = np.mean([r["mean_dice"] for r in results["organs"].values()])
    results["mean_dice_4_organs"] = float(avg)
    print(f"  {'4-organ avg':<12} {avg:.4f}")
    print(f"finished:   {time.strftime('%Y-%m-%d %H:%M:%S')}")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {json_path}")
    print(f"Saved {log_path}")


if __name__ == "__main__":
    main()
