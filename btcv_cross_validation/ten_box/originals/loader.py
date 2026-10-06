"""Explicit BTCV manifest loader, replacing only the experimental copy.

SimpleITK arrays are z-y-x. No resampling, canonicalization or implicit axis swap.
"""
from pathlib import Path
import numpy as np
from .helpers import prepare_arrays


def case_paths(data_root, volume_id):
    base = Path(data_root) / '01_BCV-Abdomen/Training'
    return base/'img'/f'{volume_id}.nii.gz', base/'label'/f'label{volume_id[3:]}.nii.gz'


def read_case(data_root, volume_id, label_idx, protocol):
    import SimpleITK as sitk
    image_path, label_path = case_paths(data_root, volume_id)
    image_itk, label_itk = sitk.ReadImage(str(image_path)), sitk.ReadImage(str(label_path))
    if image_itk.GetDimension() != 3 or label_itk.GetDimension() != 3:
        raise ValueError('Expected scalar 3D NIfTI images.')
    for getter in ('GetSize','GetSpacing','GetOrigin','GetDirection'):
        if not np.allclose(getattr(image_itk,getter)(),getattr(label_itk,getter)(),rtol=0,atol=1e-5):
            raise ValueError(f'Image/label geometry mismatch: {volume_id}, {getter}')
    image = sitk.GetArrayFromImage(image_itk)
    label = sitk.GetArrayFromImage(label_itk) == label_idx
    working, target, center, metadata = prepare_arrays(image,label,protocol)
    spacing_zyx = tuple(reversed(image_itk.GetSpacing()))
    metadata.update(volume_id=volume_id, label_idx=label_idx,
                    spacing_zyx_mm=list(spacing_zyx), origin_xyz=list(image_itk.GetOrigin()),
                    direction=list(image_itk.GetDirection()), image_path=str(image_path),
                    label_path=str(label_path))
    return working, target, center, spacing_zyx, metadata
