"""
btcv_cross_validation/z_spacing/loader.py

Load synthetic thick-spacing NIfTI volumes from the synthetic directory.
Synthetic volumes were created by make_synthetic.py; they are the ONLY
input to both branches. The fine-spacing originals are never read during
evaluation.

File layout under synthetic_root:
    img/<volume_id>.nii.gz   — thick-spacing image
    label/<volume_id>.nii.gz — thick-spacing label (nearest-neighbour resampled)
    meta.json                — records source volume, source spacing, target spacing,
                               method and sha256 of source files

The scoring grid is always the synthetic (thick-spacing) grid, for both branches.
"""
from pathlib import Path
import json
import numpy as np
from .helpers import prepare_arrays


def case_paths(synthetic_root, volume_id):
    root = Path(synthetic_root)
    return (root / 'img'  / f'{volume_id}.nii.gz',
            root / 'label' / f'{volume_id}.nii.gz')


def read_synthetic_case(synthetic_root, volume_id, label_idx, protocol):
    """
    Read a synthetic thick-spacing volume and prepare arrays for evaluation.

    Returns
    -------
    working  : np.ndarray (Z', Y, X) float32 — clipped/cropped image
    target   : np.ndarray (Z', Y, X) bool    — cropped binary label
    center   : int                             — initial prompt slice (working grid)
    spacing  : tuple (z_mm, y_mm, x_mm)       — physical voxel spacing
    metadata : dict                            — provenance and geometry info
    """
    import SimpleITK as sitk
    image_path, label_path = case_paths(synthetic_root, volume_id)

    image_itk = sitk.ReadImage(str(image_path))
    label_itk = sitk.ReadImage(str(label_path))

    if image_itk.GetDimension() != 3 or label_itk.GetDimension() != 3:
        raise ValueError(f'Expected 3-D NIfTI for {volume_id}.')

    for getter in ('GetSize', 'GetSpacing', 'GetOrigin', 'GetDirection'):
        img_val = getattr(image_itk, getter)()
        lbl_val = getattr(label_itk, getter)()
        if not np.allclose(img_val, lbl_val, rtol=0, atol=1e-4):
            raise ValueError(
                f'Geometry mismatch between image and label for {volume_id}: '
                f'{getter}: img={img_val}  label={lbl_val}'
            )

    image = sitk.GetArrayFromImage(image_itk)   # z-y-x
    label_full = sitk.GetArrayFromImage(label_itk)
    label = (label_full == label_idx)

    working, target, center, meta = prepare_arrays(image, label, protocol)

    spacing_xyz = image_itk.GetSpacing()           # SimpleITK: x, y, z
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])

    meta.update(
        volume_id=volume_id,
        label_idx=label_idx,
        spacing_zyx_mm=list(spacing_zyx),
        origin_xyz=list(image_itk.GetOrigin()),
        direction=list(image_itk.GetDirection()),
        image_path=str(image_path),
        label_path=str(label_path),
        scoring_grid='synthetic_thick_spacing',
    )

    # Attach synthetic meta if available
    meta_path = Path(synthetic_root) / 'meta.json'
    if meta_path.exists():
        with open(meta_path) as f:
            synthetic_meta = json.load(f)
        vol_meta = synthetic_meta.get(volume_id, {})
        meta['synthetic_source_spacing_z_mm'] = vol_meta.get('source_z_mm')
        meta['synthetic_target_spacing_z_mm'] = vol_meta.get('target_z_mm')
        meta['synthetic_actual_spacing_z_mm'] = vol_meta.get('actual_z_mm')

    return working, target, center, spacing_zyx, meta
