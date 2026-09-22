"""
Fold-aware volume evaluation for BTCV 5-fold cross-validation.

Identical evaluation logic to volume_eval_sublora.py (Evaluater / eval_step /
eval_step_point / eval_step_slice are unchanged), but:

  - Uses datasets.eval_dataloader.loader_cv.FoldTestLoader instead of
    AbstractLoader, so it reads exactly that fold's 01_BTCV_test_4to1.txt
    (the 6 held-out volumes) rather than a generic <dataset_list>_test.txt
    that could silently resolve to the wrong (or a stale, fixed) split.
  - Takes --config / --pth / --label_idx from the CLI instead of a
    hand-edited EX_CONFIG dict, so it can be re-run per fold and per organ
    without touching this file. No default label_idx is provided on purpose:
    volume_eval_sublora.py's own EX_CONFIG example used label_idx=2 with a
    comment "2 = liver", which is WRONG for this repo's BTCV mapping. The
    verified mapping is:
        1 = Spleen, 2 = Right Kidney, 3 = Left Kidney, 6 = Liver, 7 = Stomach

Example:
    python -m test.volume_eval_cv \\
        --config btcv_cross_validation/configs/fold1.yaml \\
        --pth btcv_cross_validation/runs/fold1/ddp_sub_cv/cv_fold1/ckpt/model_latest.pth \\
        --label_idx 6   # Liver
"""
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader
from tutils.new.manager import ConfigManager
from datasets.eval_dataloader.loader_cv import FoldTestLoader

from core.volume_predictor import VolumePredictor
from datasets.data_engine import DataManager, BoxPromptGenerator, PointPromptGenerator

from tutils import tfilename
from tutils.new.trainer.recorder import Recorder
from trans_utils.metrics import compute_dice_np
from trans_utils.data_utils import Data3dSolver

import surface_distance as surfdist
from tutils.tutils.ttimer import timer


class Evaluater:
    def __init__(self, config) -> None:
        self.config = config
        self.recorder = Recorder()

    def solve(self, model, dataset):
        self.predictor = model
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        for i, data in enumerate(dataloader):
            if self.config['dataset']['prompt'] == 'box':
                res = self.eval_step(data, batch_idx=i)
            if self.config['dataset']['prompt'] == 'point':
                res = self.eval_step_point(data, batch_idx=i)
            self.recorder.record(res)
        res = self.recorder.cal_metrics()
        print(res)
        print("prompt:", self.config['dataset']['prompt'], " class_idx:", self.config['dataset']['label_idx'])
        return res

    def eval_step(self, data, batch_idx=0):
        name = data['name']
        dataset_name = data['dataset_name'][0]
        label_idx = data['label_idx'][0]
        template_slice_id = data['template_slice_id'][0]

        assert data['img'].shape[1] >= 3, f" Got img.shape {data['img'].shape}"
        if template_slice_id == 0:
            template_slice_id += 1
        elif template_slice_id == (data['img'].shape[0] - 1):
            template_slice_id -= 1

        spacing = data['spacing'].numpy().tolist()[0]
        if data['img'].shape[-1] < 260:
            img = data['img'][0][:, :256, :256]
            label = data['label'][0][:, :256, :256]
        else:
            img = data['img'][0]
            label = data['label'][0]

        box = BoxPromptGenerator(size=None).mask_to_bbox(label[template_slice_id].detach().cpu().numpy())
        box = np.array([box])
        pred, stability = self.predictor.predict_volume(
            x=img,
            box=box,
            template_slice_id=template_slice_id,
            return_stability=True,
        )
        dice = compute_dice_np(pred, label.detach().cpu().numpy())
        nsd = 0

        print(dataset_name, name, dice, nsd)
        return {"dice": dice, "nsd": nsd}

    def eval_step_point(self, data, batch_idx=0):
        name = data['name']
        dataset_name = data['dataset_name'][0]
        label_idx = data['label_idx'][0]
        template_slice_id = data['template_slice_id'][0]

        assert data['img'].shape[1] >= 3, f" Got img.shape {data['img'].shape}"
        if template_slice_id == 0:
            template_slice_id += 1
        elif template_slice_id == (data['img'].shape[0] - 1):
            template_slice_id -= 1

        if data['img'].shape[-1] < 260:
            img = data['img'][0][:, :256, :256]
            label = data['label'][0][:, :256, :256]
        else:
            img = data['img'][0]
            label = data['label'][0]

        box = BoxPromptGenerator(size=None).mask_to_bbox(label[template_slice_id].detach().cpu().numpy())
        point = (box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5
        point = np.array([point]).astype(int)
        if label[template_slice_id][point[0, 1], point[0, 0]] == 0:
            print("Use random point instead !!!")
            point = PointPromptGenerator().get_prompt_point(label[template_slice_id])
            point = np.array([point]).astype(int)
        pred = self.predictor.predict_volume(
            x=img,
            point_coords=point,
            point_labels=np.ones_like(point)[:, :1],
            template_slice_id=template_slice_id,
        )
        dice = compute_dice_np(pred, label.detach().cpu().numpy())
        nsd = compute_surface_dice(pred, label.detach().cpu().numpy())
        print(dataset_name, name, dice)
        return {"dice": dice, "nsd": nsd}


if __name__ == "__main__":
    from core.learner_sub1 import SamLearner
    from modeling.build_sam3d2 import sam_model_registry

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                         help="Fold training config, e.g. btcv_cross_validation/configs/fold1.yaml")
    parser.add_argument("--pth", required=True,
                         help="Path to this fold's trained checkpoint")
    parser.add_argument("--label_idx", type=int, required=True,
                         help="BTCV organ label id: 1=Spleen, 2=Right Kidney, 3=Left Kidney, 6=Liver, 7=Stomach")
    parser.add_argument("--prompt", default="box", choices=["box", "point"])
    args = parser.parse_args()

    EX_CONFIG = {
        "dataset": {
            "prompt": args.prompt,
            "label_idx": args.label_idx,
        },
        "pth": args.pth,
    }

    config = ConfigManager()
    config.add_config(args.config)
    config.add_config(EX_CONFIG)

    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=None)
    learner = SamLearner(sam_model=sam, config=config, data_engine=DataManager(img_size=(1024, 1024)))
    learner.use_lora()
    learner.use_lora_sub()
    learner.load_well_trained_model(args.pth)
    learner.cuda()
    predictor = VolumePredictor(
        model=learner.model,
        use_postprocess=True,
        use_noise_remove=True,
    )

    solver = Evaluater(config)
    dataset = FoldTestLoader(config['dataset'], split="test")
    tt = timer()
    solver.solve(predictor, dataset)

    print("Time: ", tt())
