"""
btcv_cross_validation/z_spacing/make_synthetic.py

Downsample fine-spacing BTCV volumes (3 mm z) to synthetic thick-spacing
(6 mm z) to create a test bed for the z-spacing experiment.

Rules:
- Uses only fine-spacing source volumes (z=3.0 mm). Volumes already at
  5 mm are skipped (they exist natively but don't exceed the 5.5 mm trigger).
- Image: linear interpolation (scipy.ndimage.zoom order=1).
- Label: nearest-neighbour interpolation (order=0) — preserves integer values.
- Origin, x/y spacing, direction are preserved exactly in the output NIfTI.
- Output goes to a SEPARATE directory (never overwrites source volumes).
- Records provenance in meta.json.

Usage:
    conda activate slidesam
    cd /home/jupyter-team2pop/Slide-SAM/Slide-SAM
    python -m btcv_cross_validation.z_spacing.make_synthetic \\
        --data-root datasets \\
        --manifest btcv_cross_validation/dataset_list/fold5/01_BTCV_test_4to1.txt \\
        --output btcv_cross_validation/z_spacing/synthetic_6mm \\
        --target-z-mm 6.0

Dry-run (no files written):
    python -m btcv_cross_validation.z_spacing.make_synthetic \\
        --data-root datasets \\
        --manifest btcv_cross_validation/dataset_list/fold5/01_BTCV_test_4to1.txt \\
        --output btcv_cross_validation/z_spacing/synthetic_6mm \\
        --target-z-mm 6.0 \\
        --dry-run
"""
import argparse
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from .helpers import (
    resample_image_z, resample_label_z, manifest_ids, fingerprint
)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True,
                   help='BTCV datasets root (contains 01_BCV-Abdomen/).')
    p.add_argument('--manifest', type=Path, required=True,
                   help='Manifest listing volume IDs to process.')
    p.add_argument('--output', type=Path, required=True,
                   help='Output directory (must not exist; will be created).')
    p.add_argument('--target-z-mm', type=float, default=6.0,
                   help='Target z spacing in mm (default 6.0).')
    p.add_argument('--dry-run', action='store_true',
                   help='Print plan without writing any files.')
    return p


def source_paths(data_root, volume_id):
    base = Path(data_root) / '01_BCV-Abdomen' / 'Training'
    img_path = base / 'img'   / f'{volume_id}.nii.gz'
    lbl_path = base / 'label' / f'label{volume_id[3:]}.nii.gz'
    return img_path, lbl_path


def save_nifti_from_array(array, reference_itk, new_spacing_xyz, output_path):
    """
    Write array as NIfTI, preserving origin and direction from reference_itk
    but replacing spacing with new_spacing_xyz.
    """
    out_itk = sitk.GetImageFromArray(array)
    out_itk.SetOrigin(reference_itk.GetOrigin())
    out_itk.SetDirection(reference_itk.GetDirection())
    out_itk.SetSpacing(new_spacing_xyz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_itk, str(output_path))


def main(argv=None):
    args = build_parser().parse_args(argv)

    if not np.isfinite(args.target_z_mm) or args.target_z_mm <= 0:
        raise ValueError('target-z-mm must be positive.')

    volume_ids = manifest_ids(args.manifest)
    print(f'Manifest: {len(volume_ids)} volumes')
    print(f'Target z spacing: {args.target_z_mm} mm')
    print(f'Output: {args.output}')
    if args.dry_run:
        print('DRY RUN — no files will be written.')
    print()

    if not args.dry_run:
        if args.output.exists():
            raise FileExistsError(
                f'Output directory already exists: {args.output}\n'
                f'Delete or rename it before re-running.')
        args.output.mkdir(parents=True)

    meta = {}

    for vid in volume_ids:
        img_path, lbl_path = source_paths(args.data_root, vid)

        if not img_path.exists():
            raise FileNotFoundError(img_path)
        if not lbl_path.exists():
            raise FileNotFoundError(lbl_path)

        img_itk = sitk.ReadImage(str(img_path))
        lbl_itk = sitk.ReadImage(str(lbl_path))

        spacing_xyz = img_itk.GetSpacing()   # x, y, z
        source_z_mm = spacing_xyz[2]

        # Skip volumes not at 3 mm (only downsample fine-spacing ones).
        if abs(source_z_mm - 3.0) > 0.1:
            print(f'  SKIP {vid}: z={source_z_mm:.3f}mm (not 3mm)')
            continue

        if abs(args.target_z_mm - source_z_mm) < 0.1:
            print(f'  SKIP {vid}: already at target spacing {args.target_z_mm}mm')
            continue

        image_np = sitk.GetArrayFromImage(img_itk).astype(np.float32)  # z-y-x
        label_np = sitk.GetArrayFromImage(lbl_itk)                      # z-y-x

        print(f'  {vid}: source z={source_z_mm}mm  shape={image_np.shape}', end='')

        image_resampled, actual_z = resample_image_z(
            image_np, source_z_mm, args.target_z_mm)
        label_resampled, _ = resample_label_z(
            label_np, source_z_mm, args.target_z_mm)

        # Verify label values unchanged
        src_vals = set(np.unique(label_np).tolist())
        out_vals = set(np.unique(label_resampled).tolist())
        spurious = out_vals - src_vals
        if spurious:
            raise ValueError(
                f'{vid}: spurious label values after resampling: {spurious}')

        new_spacing_xyz = (spacing_xyz[0], spacing_xyz[1], actual_z)
        print(f'  -> z={actual_z:.4f}mm  shape={image_resampled.shape}')

        meta[vid] = {
            'source_z_mm': source_z_mm,
            'target_z_mm': args.target_z_mm,
            'actual_z_mm': actual_z,
            'source_shape_zyx': list(image_np.shape),
            'output_shape_zyx': list(image_resampled.shape),
            'source_img_sha256': fingerprint(img_path),
            'source_lbl_sha256': fingerprint(lbl_path),
            'xy_spacing_mm': list(spacing_xyz[:2]),
        }

        if not args.dry_run:
            # Clamp image to int16 range (BTCV images are CT HU values).
            image_out = np.clip(image_resampled, -32768, 32767).astype(np.int16)
            label_out = label_resampled.astype(np.uint8)

            img_out_path = args.output / 'img'   / f'{vid}.nii.gz'
            lbl_out_path = args.output / 'label' / f'{vid}.nii.gz'

            save_nifti_from_array(image_out, img_itk, new_spacing_xyz, img_out_path)
            save_nifti_from_array(label_out, lbl_itk, new_spacing_xyz, lbl_out_path)

    print()
    print(f'Processed: {len(meta)} volumes')

    if not args.dry_run:
        meta_path = args.output / 'meta.json'
        meta_path.write_text(json.dumps(meta, indent=2) + '\n')
        print(f'Meta written: {meta_path}')

        # Write a manifest for the synthetic volumes.
        manifest_out = args.output / 'synthetic_manifest.txt'
        manifest_out.write_text('\n'.join(
            f'01_BCV-Abdomen/Training/img/{vid}.nii.gz '
            f'01_BCV-Abdomen/Training/label/label{vid[3:]}.nii.gz'
            for vid in sorted(meta.keys())
        ) + '\n')
        print(f'Manifest written: {manifest_out}')
        print()
        print('Next: run CPU tests, then run_z_spacing_eval.sh')
    else:
        print('(dry run — no files written)')
        print()
        print('Planned volumes:')
        for vid, m in meta.items():
            print(f'  {vid}: {m["source_z_mm"]}mm -> {m["actual_z_mm"]:.4f}mm  '
                  f'{m["source_shape_zyx"]} -> {m["output_shape_zyx"]}')


if __name__ == '__main__':
    main()
