"""
btcv_cross_validation/z_spacing/make_synthetic.py  (v2 — geometry corrected)

Downsample fine-spacing BTCV volumes (3 mm z) to synthetic thick-spacing
using SimpleITK with an explicit reference grid.  Output goes to a new
directory (default: synthetic_6mm_v2/) — never overwrites synthetic_6mm/.

Changes from v1:
  - Uses SimpleITK.Resample with explicit origin/direction/spacing/size
    instead of scipy.ndimage.zoom.  Physical voxel centres are correctly
    aligned between source and output grids.
  - Verifies that resampled label values are a subset of source label values.
  - Records actual achieved spacing in meta.json (may differ from target
    by <1 voxel due to integer output-size rounding).

Usage:
    conda activate slidesam
    cd /home/jupyter-team2pop/Slide-SAM/Slide-SAM
    python -m btcv_cross_validation.z_spacing.make_synthetic \\
        --data-root datasets \\
        --manifest btcv_cross_validation/dataset_list/fold5/01_BTCV_test_4to1.txt \\
        --output btcv_cross_validation/z_spacing/synthetic_6mm_v2 \\
        --target-z-mm 6.0

Dry-run:
    python -m btcv_cross_validation.z_spacing.make_synthetic ... --dry-run
"""
import argparse
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from .helpers import manifest_ids, fingerprint, resample_image_z, resample_label_z


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--target-z-mm', type=float, default=6.0)
    p.add_argument('--dry-run', action='store_true')
    return p


def source_paths(data_root, volume_id):
    base = Path(data_root) / '01_BCV-Abdomen' / 'Training'
    return (base / 'img'   / f'{volume_id}.nii.gz',
            base / 'label' / f'label{volume_id[3:]}.nii.gz')


def main(argv=None):
    args = build_parser().parse_args(argv)

    if not np.isfinite(args.target_z_mm) or args.target_z_mm <= 0:
        raise ValueError('target-z-mm must be positive.')

    volume_ids = manifest_ids(args.manifest)
    print(f'Manifest       : {len(volume_ids)} volumes')
    print(f'Target z spacing: {args.target_z_mm} mm')
    print(f'Output          : {args.output}')
    if args.dry_run:
        print('DRY RUN — no files will be written.')
    print()

    if not args.dry_run:
        if args.output.exists():
            raise FileExistsError(
                f'Output already exists: {args.output}\n'
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

        if abs(source_z_mm - 3.0) > 0.1:
            print(f'  SKIP {vid}: z={source_z_mm:.3f} mm (not 3 mm)')
            continue
        if abs(args.target_z_mm - source_z_mm) < 0.1:
            print(f'  SKIP {vid}: already at target {args.target_z_mm} mm')
            continue

        image_np = sitk.GetArrayFromImage(img_itk).astype(np.float32)  # z-y-x
        label_np = sitk.GetArrayFromImage(lbl_itk)                      # z-y-x

        spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
        origin_xyz  = tuple(img_itk.GetOrigin())
        direction   = tuple(img_itk.GetDirection())

        print(f'  {vid}: z={source_z_mm} mm  shape={image_np.shape}', end='  ')

        # Use SimpleITK resampling (v2 geometry)
        image_resampled, out_sp = resample_image_z(
            image_np, spacing_zyx, origin_xyz, direction, args.target_z_mm)
        label_resampled, _      = resample_label_z(
            label_np, spacing_zyx, origin_xyz, direction, args.target_z_mm)
        actual_z_mm = out_sp[0]

        # Verify label values
        src_vals = set(np.unique(label_np).tolist())
        out_vals = set(np.unique(label_resampled).tolist())
        spurious = out_vals - src_vals
        if spurious:
            raise ValueError(f'{vid}: spurious label values after resampling: {spurious}')

        print(f'-> z={actual_z_mm:.4f} mm  shape={image_resampled.shape}')

        meta[vid] = {
            'source_z_mm': source_z_mm,
            'target_z_mm': args.target_z_mm,
            'actual_z_mm': actual_z_mm,
            'source_shape_zyx': list(image_np.shape),
            'output_shape_zyx': list(image_resampled.shape),
            'source_img_sha256': fingerprint(img_path),
            'source_lbl_sha256': fingerprint(lbl_path),
            'xy_spacing_mm': list(spacing_xyz[:2]),
            'resampler': 'SimpleITK v2',
        }

        if not args.dry_run:
            new_spacing_xyz = (float(spacing_xyz[0]),
                               float(spacing_xyz[1]),
                               float(actual_z_mm))

            # Image: clamp to int16 (CT HU values)
            image_out = np.clip(image_resampled, -32768, 32767).astype(np.int16)
            label_out = label_resampled.astype(label_np.dtype)

            for arr, out_name in [(image_out, f'img/{vid}.nii.gz'),
                                  (label_out, f'label/{vid}.nii.gz')]:
                out_itk = sitk.GetImageFromArray(arr)
                out_itk.SetOrigin(origin_xyz)
                out_itk.SetDirection(direction)
                out_itk.SetSpacing(new_spacing_xyz)
                out_path = args.output / out_name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(out_itk, str(out_path))

    print()
    print(f'Processed: {len(meta)} volumes')

    if not args.dry_run:
        meta_path = args.output / 'meta.json'
        meta_path.write_text(json.dumps(meta, indent=2) + '\n')
        print(f'Meta written   : {meta_path}')

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
            print(f'  {vid}: {m["source_z_mm"]} mm -> {m["actual_z_mm"]:.4f} mm  '
                  f'{m["source_shape_zyx"]} -> {m["output_shape_zyx"]}')


if __name__ == '__main__':
    main()
