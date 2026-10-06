"""Small CPU checks: no model creation, weights, real image arrays, or CUDA."""
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .helpers import (expand_box, checked_manifests, manifest_ids, prepare_arrays,
                      initial_prompt, segmentation_metrics, fingerprint)
from .evaluate import build_parser, preflight, main
from .loader import read_case


class PaddingTests(unittest.TestCase):
    def test_predictor_propagation_wiring_cpu(self):
        from types import SimpleNamespace
        import torch
        from .predictor import VolumePredictor
        model=SimpleNamespace(image_encoder=SimpleNamespace(img_size=1024),device=torch.device('cpu'))
        mask=np.zeros((1,3,20,40),dtype=np.float32);mask[:,:,4:12,5:15]=1
        previous={'masks':mask,'tags':np.array(['organ'])}
        boxes={}
        for mode,p in [('historical',0),('none',0),('image',.05)]:
            predictor=VolumePredictor(model,padding_mode=mode,padding_fraction=p)
            boxes[mode],tags=predictor.generate_prompts_from_previous_masks(previous,'down')
            self.assertEqual(tags,['organ'])
            normalized=predictor.batched_to_RGB(torch.ones(1,3,4,4))
            self.assertTrue(torch.isfinite(normalized).all())
        np.testing.assert_allclose(boxes['image'][0],expand_box(boxes['none'][0],(1024,1024),.05))
        np.testing.assert_allclose(boxes['historical'][0],[116,193,363,594])

    def test_zero_is_identity(self):
        self.assertEqual(expand_box((20,30,80,90),(200,400),0),(20,30,80,90))

    def test_non_square_image_uses_correct_axes(self):
        self.assertEqual(expand_box((20,30,80,90),(200,400),.05),(0,20,100,100))

    def test_fractional_bounds_and_clipping(self):
        self.assertEqual(expand_box((0,0,511,255),(256,512),.05),(0,0,511,255))
        np.testing.assert_allclose(expand_box((100,100,200,200),(512,512),.05),
                                   [74.4,74.4,225.6,225.6])

    def test_invalid_inputs(self):
        for box,shape,p in [((5,0,1,5),(10,10),.05),((0,0,1,1),(10,10),-.1),
                            ((0,0,np.nan,1),(10,10),.05),((20,0,30,1),(10,10),.05)]:
            with self.assertRaises(ValueError): expand_box(box,shape,p)

    def test_copied_helper_routes_modes_and_preserves_legacy(self):
        def load_method(path):
            tree=ast.parse(path.read_text())
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BoxPromptGenerator')
            fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='enlarge')
            ns={'__package__':'btcv_cross_validation.box_padding'}
            exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
            return ns['enlarge']
        here=Path(__file__).parent
        current,old=load_method(here/'data_engine.py'),load_method(here/'originals/data_engine.py')
        obj=type('Bounds',(),{'size':(1024,1024),'padding_mode':'historical','padding_fraction':0})()
        box=(100,200,300,300)
        self.assertEqual(current(obj,box),old(obj,box))
        self.assertEqual(current(obj,box),(90,190,295,295))
        obj.padding_mode='none'
        self.assertEqual(current(obj,box),box)
        obj.padding_mode='image';obj.padding_fraction=.05
        np.testing.assert_allclose(current(obj,box),[48.8,148.8,351.2,351.2])


class ProtocolTests(unittest.TestCase):
    def arrays(self):
        image=np.arange(8*10*12).reshape(8,10,12)-300
        mask=np.zeros_like(image,dtype=bool);mask[2:6,3:7,4:9]=True
        mask[3,2:8,3:10]=True
        return image,mask

    def test_cropped_and_full_keep_same_physical_prompt(self):
        image,mask=self.arrays()
        a=prepare_arrays(image,mask,'gt-cropped');b=prepare_arrays(image,mask,'full-volume')
        self.assertEqual(a[3]['initial_slice_original_z'],b[3]['initial_slice_original_z'])
        self.assertEqual(a[0].shape,(4,10,12))
        self.assertEqual(b[0].shape,(8,10,12))
        np.testing.assert_array_equal(initial_prompt(a[1][a[2]],'box')['box'],
                                      initial_prompt(b[1][b[2]],'box')['box'])

    def test_initial_slice_never_first_or_last(self):
        for index in [0,7]:
            image=np.ones((8,10,12));mask=np.zeros_like(image,dtype=bool)
            mask[:,3:5,4:6]=True;mask[index,2:7,3:9]=True
            _,target,center,metadata=prepare_arrays(image,mask,'gt-cropped')
            self.assertTrue(1 <= center < len(target)-1)
            self.assertTrue(metadata['initial_slice_adjusted'])
            self.assertTrue(target[center].any())

    def test_absent_and_short_target_explicit_failure(self):
        image=np.ones((8,10,12));mask=np.zeros_like(image,dtype=bool)
        with self.assertRaises(ValueError):prepare_arrays(image,mask,'gt-cropped')
        mask[3,2:7,3:9]=True
        with self.assertRaises(ValueError):prepare_arrays(image,mask,'gt-cropped')

    def test_point_is_deterministic_foreground(self):
        mask=np.zeros((12,12),dtype=bool);mask[2:10,2]=True;mask[2:10,9]=True
        a=initial_prompt(mask,'point');b=initial_prompt(mask,'point')
        np.testing.assert_array_equal(a['point_coords'],b['point_coords'])
        x,y=a['point_coords'][0].astype(int)
        self.assertTrue(mask[y,x])

    def test_synthetic_nifti_geometry_and_spacing(self):
        import SimpleITK as sitk
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'01_BCV-Abdomen/Training'
            (base/'img').mkdir(parents=True);(base/'label').mkdir()
            image,mask=self.arrays()
            img=sitk.GetImageFromArray(image.astype(np.float32));img.SetSpacing((.7,.8,5))
            lab=sitk.GetImageFromArray(mask.astype(np.uint8)*2);lab.CopyInformation(img)
            sitk.WriteImage(img,str(base/'img/img0001.nii.gz'))
            sitk.WriteImage(lab,str(base/'label/label0001.nii.gz'))
            _,target,_,spacing,info=read_case(tmp,'img0001',2,'gt-cropped')
            np.testing.assert_allclose(spacing,[5,.8,.7],rtol=1e-6)
            self.assertEqual(target.shape,(4,10,12))
            lab.SetOrigin((1,0,0));sitk.WriteImage(lab,str(base/'label/label0001.nii.gz'))
            with self.assertRaises(ValueError):read_case(tmp,'img0001',2,'gt-cropped')


class MetricTests(unittest.TestCase):
    def masks(self):
        a=np.zeros((12,13,14),dtype=bool);a[2:6,3:8,4:9]=True
        b=np.zeros_like(a);b[4:8,3:8,4:9]=True
        return a,b

    def test_identical(self):
        a,_=self.masks();m=segmentation_metrics(a,a,(3,.8,.7),1)
        self.assertEqual(m['dice'],1);self.assertEqual(m['nsd'],1);self.assertEqual(m['hd95_mm'],0)

    def test_distances_scale_in_mm(self):
        a,b=self.masks()
        unit=segmentation_metrics(a,b,(1,1,1),.5)
        scaled=segmentation_metrics(a,b,(3,3,3),1.5)
        self.assertAlmostEqual(scaled['hd95_mm'],3*unit['hd95_mm'])
        self.assertAlmostEqual(scaled['nsd'],unit['nsd'])

    def test_axis_permutation_preserves_surface_metrics(self):
        a,b=self.masks()
        first=segmentation_metrics(a,b,(5,.8,.7),1)
        permuted=segmentation_metrics(a.transpose(2,1,0),b.transpose(2,1,0),(.7,.8,5),1)
        for metric in ('dice','nsd','hd95_mm'):
            self.assertAlmostEqual(first[metric],permuted[metric])

    def test_empty_masks_are_json_safe(self):
        a,_=self.masks();empty=np.zeros_like(a)
        result=segmentation_metrics(empty,a,(1,1,1),1)
        self.assertEqual(result['dice'],0);self.assertEqual(result['nsd'],0)
        self.assertIsNone(result['hd95_mm'])
        json.dumps(result,allow_nan=False)
        self.assertEqual(segmentation_metrics(empty,empty,(1,1,1),1)['dice'],1)


class ManifestTests(unittest.TestCase):
    def test_duplicates_and_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'train.txt',Path(tmp)/'val.txt'
            a.write_text('img0001\nimg0002\n');b.write_text('img0002\n')
            with self.assertRaises(ValueError):checked_manifests(a,b)
            b.write_text('img0003\nimg0003\n')
            with self.assertRaises(ValueError):manifest_ids(b)

    def test_both_manifest_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a.txt';a.write_text('01_BCV-Abdomen/Training/img/img0035.nii.gz\tlabels/label0035.nii.gz\n')
            self.assertEqual(manifest_ids(a),['img0035'])
            a.write_text('img0035\n');self.assertEqual(manifest_ids(a),['img0035'])

    def test_preflight_does_not_read_checkpoint_or_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'train.txt';b=root/'val.txt';ckpt=root/'not-real.pth'
            a.write_text('img0001\n');b.write_text('img0002\n');ckpt.write_text('not a checkpoint')
            base=root/'01_BCV-Abdomen/Training';(base/'img').mkdir(parents=True);(base/'label').mkdir()
            (base/'img/img0002.nii.gz').touch();(base/'label/label0002.nii.gz').touch()
            output=root/'results'
            argv=['--manifest',str(b),'--train-manifest',str(a),'--checkpoint',str(ckpt),
                  '--checkpoint-kind','finetuned','--label-idx','2','--protocol','gt-cropped',
                  '--padding-mode','none','--nsd-tolerance-mm','1','--data-root',str(root),'--output',str(output)]
            with redirect_stdout(io.StringIO()):main(argv)
            self.assertFalse(output.exists())
            output.mkdir()
            with self.assertRaises(FileExistsError):preflight(build_parser().parse_args(argv))

    def test_original_sources_and_backups_unchanged(self):
        here=Path(__file__).parent;root=here.parents[1]
        manifest=json.loads((here/'originals/source_manifest.json').read_text())
        for source,item in manifest.items():
            self.assertEqual(fingerprint(root/source),item['sha256'])
            self.assertEqual(fingerprint(here/'originals'/item['copy']),item['sha256'])


if __name__=='__main__':
    unittest.main()
