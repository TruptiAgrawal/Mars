"""
btcv_cross_validation/z_spacing/loader.py  (v2)

Loads synthetic thick-spacing NIfTI volumes.  Now returns the full SimpleITK
geometry (origin, direction, spacing) so that evaluate.py can pass it to
the SimpleITK resampler without re-reading the file.
"""
from pathlib import Path
import json
import numpy as np
from .helpers import prepare_arrays


def case_paths(synthetic_root, volume_id):
    root = Path(synthetic_root)
    return (root / 'img'   / f'{volume_id}.nii.gz',
            root / 'label' / f'{volume_id}.nii.gz')


def read_synthetic_case(synthetic_root, volume_id, label_idx, protocol):
    """
    Returns
    -------
    working      : (Z', Y, X) float32  — clipped/cropped working image
    target       : (Z', Y, X) bool     — cropped binary label (scoring grid)
    center       : int                  — prompt slice index on working grid
    spacing_zyx  : (z_mm, y_mm, x_mm) of the synthetic volume
    origin_xyz   : (ox, oy, oz) SimpleITK physical origin (x-y-z)
    direction    : 9-tuple direction cosines (SimpleITK x-y-z order)
    metadata     : dict
    """
    import SimpleITK as sitk
    image_path, label_path = case_paths(synthetic_root, volume_id)

    image_itk = sitk.ReadImage(str(image_path))
    label_itk = sitk.ReadImage(str(label_path))

    for getter in ('GetSize', 'GetSpacing', 'GetOrigin', 'GetDirection'):
        iv = getattr(image_itk, getter)()
        lv = getattr(label_itk, getter)()
        if not np.allclose(iv, lv, rtol=0, atol=1e-4):
            raise ValueError(
                f'Geometry mismatch for {volume_id}: {getter}: '
                f'img={iv}  label={lv}')

    image = sitk.GetArrayFromImage(image_itk)        # z-y-x
    label_full = sitk.GetArrayFromImage(label_itk)
    label = (label_full == label_idx)

    working, target, center, meta = prepare_arrays(image, label, protocol)

    spacing_xyz = image_itk.GetSpacing()              # SimpleITK: x, y, z
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    origin_xyz  = tuple(image_itk.GetOrigin())
    direction   = tuple(image_itk.GetDirection())

    meta.update(
        volume_id=volume_id,
        label_idx=label_idx,
        spacing_zyx_mm=list(spacing_zyx),
        origin_xyz=list(origin_xyz),
        direction=list(direction),
        image_path=str(image_path),
        label_path=str(label_path),
        scoring_grid='synthetic_thick_spacing_v2',
    )

    meta_path = Path(synthetic_root) / 'meta.json'
    if meta_path.exists():
        with open(meta_path) as f:
            synthetic_meta = json.load(f)
        vm = synthetic_meta.get(volume_id, {})
        meta['synthetic_source_spacing_z_mm'] = vm.get('source_z_mm')
        meta['synthetic_target_spacing_z_mm'] = vm.get('target_z_mm')
        meta['synthetic_actual_spacing_z_mm']  = vm.get('actual_z_mm')

    return working, target, center, spacing_zyx, origin_xyz, direction, meta
