"""Five-fold ten-box evaluation, preserving the authors' correction/merge rule."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/slidesam-tenbox-mpl')
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PROMPTS = 10
# Fold evaluation logs record checkpoints under the repository's former location.
OLD_ROOT = Path('/home/jupyter-team2pop/Slide-SAM/Slide-SAM')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def save(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); tmp.replace(path)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold',type=int,choices=range(1,6),required=True)
    parser.add_argument('--run-inference',action='store_true')
    args=parser.parse_args()
    from .helpers import checked_manifests
    from .loader import case_paths
    manifest=ROOT/f'btcv_cross_validation/dataset_list/fold{args.fold}'
    train,patients=checked_manifests(manifest/'01_BTCV_train_4to1.txt',manifest/'01_BTCV_test_4to1.txt')
    previous=json.loads((ROOT/f'btcv_cross_validation/logs/fold{args.fold}_evaluation.json').read_text())
    recorded_checkpoint=Path(previous['checkpoint'])
    checkpoint=ROOT/recorded_checkpoint.relative_to(OLD_ROOT)
    assert checkpoint.is_file() and len(patients)==6
    for patient in patients:
        assert all(p.is_file() for p in case_paths(ROOT/'datasets',patient))
    out=HERE/'results'/f'fold{args.fold}'
    assert not out.exists(), f'Preserve existing output: {out}'
    print('PREFLIGHT',args.fold,patients,checkpoint,flush=True)
    if not args.run_inference:return

    import torch
    from core.learner_sub1 import SamLearner
    from modeling.build_sam3d2 import sam_model_registry
    from .predictor import VolumePredictor
    from .data_engine import DataManager
    from .loader import read_case
    from .helpers import initial_prompt
    from .author_mbox import Evaluater, compute_dice_np
    assert torch.cuda.is_available()
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.benchmark=False
    torch.set_num_threads(4)
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    out.mkdir(parents=True,exist_ok=False)
    report={'status':'running','fold':args.fold,'training_ids':train,'evaluation_ids':patients,
            'checkpoint':str(checkpoint),'recorded_checkpoint':str(recorded_checkpoint),'checkpoint_sha256':sha(checkpoint),
            'protocol':'gt-cropped; corrected-zero propagated boxes; initial GT max-area box plus nine lowest-Dice unused interior-slice boxes; unchanged authors confidence merge',
            'prompt_budget':PROMPTS,'cudnn_enabled':False,'torch_version':torch.__version__,
            'gpu':torch.cuda.get_device_name(),'cases':[],
            'source_sha256':{f.name:sha(f) for f in HERE.glob('*.py')},
            'manifest_sha256':{f.name:sha(f) for f in manifest.glob('*.txt')}}
    path=out/'results.json';save(path,report)
    try:
        learner=SamLearner(sam_model=sam_model_registry['vit_b'](checkpoint=None),config={},data_engine=DataManager(img_size=(1024,1024)))
        learner.use_lora();learner.use_lora_sub();learner.load_well_trained_model(str(checkpoint));learner.cuda().eval()
        predictor=VolumePredictor(learner.model,use_postprocess=True,use_noise_remove=True,padding_mode='none',padding_fraction=0)
        evaluator=Evaluater({});evaluator.predictor=predictor
        for organ,label in [('Liver',6),('Stomach',7),('Left kidney',3),('Spleen',1)]:
            for patient in patients:
                image,target,center,spacing,info=read_case(ROOT/'datasets',patient,label,'gt-cropped')
                # The unmodified authors' ranking assumes foreground on every
                # interior slice and at least ten distinct valid centers.
                assert len(target)-2>=PROMPTS, 'Fewer than ten valid prompt slices'
                assert np.all(target[1:-1].sum(axis=(1,2))>0), 'Authors ranking undefined on empty GT slice'
                prompt=initial_prompt(target[center],'box')
                torch.cuda.synchronize();start=time.monotonic()
                with torch.inference_mode():
                    pred,_=predictor.predict_volume(image,template_slice_id=center,return_stability=True,**prompt)
                    scores=[float(compute_dice_np(pred,target))];used=[int(center)]
                    boxes=[prompt['box'].tolist()]
                    for _ in range(PROMPTS-1):
                        dice,pred,_,idx=evaluator.finetune_with_more_prompt(pred,target,exclude_slide_id=used)
                        idx=int(idx);assert idx not in used
                        used.append(idx);scores.append(float(dice));boxes.append(initial_prompt(target[idx],'box')['box'].tolist())
                torch.cuda.synchronize()
                case={**info,'organ':organ,'dice_by_prompt_count':scores,'one_box_dice':scores[0],
                      'five_box_dice':scores[4],'ten_box_dice':scores[-1],'prompt_slices_cropped':used,
                      'prompt_slices_original':[i+info['crop_start_z'] for i in used],
                      'boxes_original_xyxy':boxes,'total_seconds':time.monotonic()-start}
                assert len(used)==len(set(used))==PROMPTS and len(scores)==PROMPTS and all(0<=x<=1 for x in scores)
                report['cases'].append(case);save(path,report)
                print('TENBOX_CASE',args.fold,organ,patient,json.dumps(scores),flush=True)
        assert len(report['cases'])==24
        report['status']='complete';save(path,report)
        print('TENBOX_COMPLETE',args.fold,flush=True)
    except Exception as exc:
        report['status']='failed';report['error']=repr(exc);save(path,report);raise

if __name__=='__main__':main()
