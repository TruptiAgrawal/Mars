"""
    from ddp_sub.py

    BTCV-only 5-fold cross-validation variant.

    Identical to ddp_sub.py in every respect except the dataset import:
    this uses datasets.cache_dataset3d_cv (BTCV-only, fold-aware, leakage-
    audited) instead of datasets.cache_dataset3d_e2 (BTCV + WORD-50 E2
    experiment). See datasets/cache_dataset3d_cv.py for details.

    Run exactly like ddp_sub.py, just as this module instead:
        python -m core.ddp_sub_cv --config btcv_cross_validation/configs/fold1.yaml --tag cv_fold1
"""

import os
import torch
torch.backends.cudnn.enabled = False
import torch.distributed as dist
import torch.multiprocessing as mp

from tutils import tfilename, tdir

from datasets.dataset3d_2dmask import Dataset2D
# BTCV-only CV loader (no WORD-50 dependency) -- see datasets/cache_dataset3d_cv.py
from datasets.cache_dataset3d_cv import Dataset3D
from datasets.dataset_merged import DatasetMerged, TestsetMerged
from datasets.data_engine import DataEngine
from modeling.build_sam3d2 import sam_model_registry

from .learner_sub1 import SamLearner
from trans_utils.trainer_ddp import DDPTrainer

import warnings
warnings.filterwarnings("ignore")


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def ddp_train(rank, world_size, config):
    setup(rank, world_size)

    model_type = "vit_b"
    device = rank

    config_data = config['dataset']
    data_type = config_data.get("types", ["3d", "2d"])
    data_type = [data_type] if isinstance(data_type, str) else data_type
    dataset = Dataset3D(config_data, split='train')

    data_engine = DataEngine(dataset=dataset, img_size=(1024,1024))
    sam = sam_model_registry[model_type](checkpoint=None)

    learner = SamLearner(sam_model=sam, config=config, data_engine=data_engine)

    checkpoint_path = config['training']['breakpoint_path']

    # Official Slide-SAM-B checkpoint already contains rank-8 LoRA.
    if checkpoint_path.endswith("slidesam_b.pth"):
        learner.use_lora()
        learner.load_well_trained_model(checkpoint_path)
        learner.use_lora_sub()

    # Recovery checkpoint already contains nested rank-8 + rank-1 LoRA.
    else:
        learner.use_lora()
        learner.use_lora_sub()
        learner.load_well_trained_model(checkpoint_path)

    ddp_trainer = DDPTrainer(config=config, rank=rank, world_size=world_size)
    ddp_trainer.fit(learner, trainset=data_engine, validset=None)

    cleanup()


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def run_demo(demo_fn, world_size, config):
    mp.spawn(demo_fn,
             args=(world_size,config),
             nprocs=world_size,
             join=True)

from collections import OrderedDict
import yaml
import yamlloader
def _ordereddict_to_dict(d):
    if not isinstance(d, dict):
        return d
    for k, v in d.items():
        if isinstance(v, OrderedDict):
            v = _ordereddict_to_dict(v)
            d[k] = dict(v)
        elif type(v) == list:
            d[k] = _ordereddict_to_dict(v)
        elif isinstance(v, dict):
            d[k] = _ordereddict_to_dict(v)
    return d

# python -m core.ddp_sub_cv --tag cv_fold1 --config btcv_cross_validation/configs/fold1.yaml

if __name__ == "__main__":
    import argparse
    from tutils.new.manager import trans_args, trans_init, ConfigManager

    n_gpus = torch.cuda.device_count()
    if n_gpus == 1:
        print("Warning! Running on only 1 GPU! just for debug")
    world_size = n_gpus

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/vit_sub.yaml")
    parser.add_argument("--func", default="train")
    parser.add_argument("--reuse", action="store_true")

    args = trans_args(parser=parser)
    config = ConfigManager()
    config.auto_init(file=__file__, args=args, ex_config=None)
    path = tfilename(config['base']['runs_dir'], "config.yaml")
    with open(path, "w") as f:
        yaml.dump(_ordereddict_to_dict(config), f)
        print("Save config file to ", path)

    if n_gpus < 1: exit(0)
    run_demo(ddp_train, world_size, config)
