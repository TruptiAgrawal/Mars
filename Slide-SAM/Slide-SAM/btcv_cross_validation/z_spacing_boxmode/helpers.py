"""
btcv_cross_validation/z_spacing/helpers.py  (v2 — geometry corrected)

Key fix over v1:
  v1 used scipy.ndimage.zoom, which places output sample centres at
  evenly spaced fractions of the array extent [0, N-1].  This shifts
  the physical origin of the output grid by half a voxel relative to the
  input and makes the declared spacing inconsistent with where samples
  actually sit.

  v2 uses SimpleITK.Resample with an explicit reference grid:
    - origin     : preserved from input (z-component unchanged)
    - direction  : preserved from input
    - spacing    : set to target_z_mm along z, input x/y spacing elsewhere
    - output size: computed from physical extent so coverage is identical
    - interpolator: sitkLinear for images, sitkNearestNeighbor for labels

  This guarantees that:
    * Voxel centre of output[i] maps to physical position
      origin_z + i * target_z_mm.
    * The same SimpleITK transform maps the scoring-grid label and
      predictions back without any half-voxel drift.
    * No manual crop/pad is needed after prediction restoration because
      the output size is declared explicitly.

Conditional trigger:
  needs_resampling(z_spacing_mm, trigger_mm=5.5) returns True only when
  z_spacing_mm > trigger_mm.  This implements the proposed >5.5 mm rule.
  The experiment uses synthetic 6 mm inputs, which exceed the threshold.
"""
from pathlib import Path
import hashlib
import re
import numpy as np


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def manifest_ids(path):
    ids = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        token = line.split()[0]
        match = re.fullmatch(r'(img\d+)(?:\.nii\.gz)?', Path(token).name)
        if not match:
            raise ValueError(f'Unrecognized BTCV image entry: {line}')
        ids.append(match[1])
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f'Empty manifest or duplicate IDs: {path}')
    return ids


def checked_manifests(train_manifest, eval_manifest):
    training = manifest_ids(train_manifest)
    evaluation = manifest_ids(eval_manifest)
    overlap = set(training) & set(evaluation)
    if overlap:
        raise ValueError(f'Training/evaluation overlap: {sorted(overlap)}')
    return training, evaluation


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Conditional trigger
# ---------------------------------------------------------------------------

TRIGGER_MM = 5.5  # proposed rule: resample only when z spacing exceeds this


def needs_resampling(z_spacing_mm, trigger_mm=TRIGGER_MM):
    """Return True iff z spacing exceeds the trigger threshold."""
    return float(z_spacing_mm) > float(trigger_mm)


# ---------------------------------------------------------------------------
# SimpleITK reference-grid resampling
# ---------------------------------------------------------------------------

def _build_sitk_volume(array_zyx, spacing_zyx, origin_xyz, direction):
    """
    Wrap a numpy z-y-x array into a SimpleITK image with the given geometry.

    SimpleITK stores arrays in x-y-z order; GetArrayFromImage returns z-y-x.
    We therefore pass the array directly; SetSpacing/SetOrigin use x-y-z order.
    """
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(array_zyx)
    img.SetSpacing((float(spacing_zyx[2]),   # x
                    float(spacing_zyx[1]),   # y
                    float(spacing_zyx[0])))  # z
    img.SetOrigin(origin_xyz)
    img.SetDirection(direction)
    return img


def _output_size_z(z_in, source_z_mm, target_z_mm):
    """
    Compute output z size so physical coverage is identical to the input.

    Physical extent of input (centre-to-centre):
        (z_in - 1) * source_z_mm

    Minimum z_out such that (z_out - 1) * target_z_mm >= that extent:
        z_out = ceil((z_in - 1) * source_z_mm / target_z_mm) + 1
    """
    if z_in < 2:
        return z_in
    extent = (z_in - 1) * source_z_mm
    return int(np.ceil(extent / target_z_mm)) + 1


def resample_volume_z(array_zyx, source_spacing_zyx, origin_xyz, direction,
                      target_z_mm, interpolator_str):
    """
    Resample a 3-D array along z using an explicit SimpleITK reference grid.

    Parameters
    ----------
    array_zyx        : np.ndarray (Z, Y, X)
    source_spacing_zyx: (z_mm, y_mm, x_mm) of the input
    origin_xyz       : (ox, oy, oz) physical origin (SimpleITK order)
    direction        : 9-element direction cosine tuple (SimpleITK order)
    target_z_mm      : desired z spacing in mm
    interpolator_str : 'linear' | 'nearest'

    Returns
    -------
    out_array : np.ndarray same dtype as input, shape (Z', Y, X)
    out_spacing_zyx : (actual_z_mm, y_mm, x_mm) of the output
    """
    import SimpleITK as sitk

    array_zyx = np.asarray(array_zyx)
    sz, sy, sx = source_spacing_zyx
    z_in, y_in, x_in = array_zyx.shape

    # Identity shortcut
    if abs(sz - target_z_mm) < 1e-6:
        return array_zyx.copy(), tuple(source_spacing_zyx)

    # Build input image
    src = _build_sitk_volume(array_zyx, source_spacing_zyx, origin_xyz, direction)

    # Compute output size and actual achieved spacing
    z_out = _output_size_z(z_in, sz, target_z_mm)
    actual_z_mm = (z_in - 1) * sz / (z_out - 1) if z_out > 1 else sz

    # Reference grid: same origin/direction/x/y, new z spacing and z size
    ref = sitk.Image(x_in, y_in, z_out, sitk.sitkFloat32)
    ref.SetSpacing((float(sx), float(sy), float(actual_z_mm)))
    ref.SetOrigin(origin_xyz)
    ref.SetDirection(direction)

    interp = (sitk.sitkLinear if interpolator_str == 'linear'
              else sitk.sitkNearestNeighbor)

    # Cast source to float32 for resampling
    src_f = sitk.Cast(src, sitk.sitkFloat32)
    resampled = sitk.Resample(src_f, ref, sitk.Transform(), interp, 0.0)

    out = sitk.GetArrayFromImage(resampled)

    # Restore original dtype for labels
    if array_zyx.dtype != np.float32:
        out = out.round().astype(array_zyx.dtype)

    out_spacing_zyx = (float(actual_z_mm), float(sy), float(sx))
    return out, out_spacing_zyx


def resample_image_z(array_zyx, source_spacing_zyx, origin_xyz, direction, target_z_mm):
    """Linear resampling along z for intensity images."""
    return resample_volume_z(array_zyx.astype(np.float32), source_spacing_zyx,
                             origin_xyz, direction, target_z_mm, 'linear')


def resample_label_z(array_zyx, source_spacing_zyx, origin_xyz, direction, target_z_mm):
    """Nearest-neighbour resampling along z for discrete labels/masks."""
    return resample_volume_z(array_zyx, source_spacing_zyx,
                             origin_xyz, direction, target_z_mm, 'nearest')


# ---------------------------------------------------------------------------
# Prompt slice mapping through physical coordinates
# ---------------------------------------------------------------------------

def map_slice_to_resampled(z_idx_source, source_spacing_zyx, out_spacing_zyx):
    """
    Map a z index from the source grid to the resampled grid using
    physical voxel-centre positions.

    Physical centre of source slice i:  origin_z + i * source_z_mm
    (origin_z cancels — it is shared between both grids.)

    Continuous index in output grid:
        z_out_cont = i * source_z_mm / actual_z_mm

    Returns nearest integer, unclamped (caller clamps to valid range).
    """
    source_z_mm = float(source_spacing_zyx[0])
    actual_z_mm = float(out_spacing_zyx[0])
    return z_idx_source * source_z_mm / actual_z_mm


def map_slice_to_original(z_idx_resampled, source_spacing_zyx, out_spacing_zyx):
    """Inverse of map_slice_to_resampled."""
    source_z_mm = float(source_spacing_zyx[0])
    actual_z_mm = float(out_spacing_zyx[0])
    return z_idx_resampled * actual_z_mm / source_z_mm


# ---------------------------------------------------------------------------
# Prepare arrays (gt-crop + center-slice selection)
# ---------------------------------------------------------------------------

def prepare_arrays(image, label, protocol):
    """
    Clip, crop and find the oracle initial-prompt center slice.
    Identical contract to box_padding/helpers.py — copied for isolation.
    """
    image = np.asarray(image, dtype=np.float32)
    label = np.asarray(label, dtype=bool)
    if image.ndim != 3 or image.shape != label.shape or min(image.shape) < 3:
        raise ValueError('Image/label must share a 3-D grid with all dims >= 3.')
    if not np.isfinite(image).all():
        raise ValueError('Non-finite image intensity.')

    areas = label.sum(axis=(1, 2))
    occupied = np.flatnonzero(areas)
    if not len(occupied):
        raise ValueError('Target absent: cannot generate a positive oracle prompt.')

    if protocol == 'gt-cropped':
        start, stop = int(occupied[0]), int(occupied[-1]) + 1
    elif protocol == 'full-volume':
        start, stop = 0, image.shape[0]
    else:
        raise ValueError(f'Unknown evaluation protocol: {protocol!r}')

    clip = (-200, 400) if image.min() < -10 else (0, 600)
    working = np.clip(image, *clip)[start:stop].astype(np.float32)
    target = label[start:stop]

    if len(target) < 3:
        raise ValueError('Fewer than three slices in evaluation grid.')

    candidate = int(np.argmax(areas)) - start
    valid = np.flatnonzero(target.sum(axis=(1, 2)))
    valid = valid[(valid >= 1) & (valid < len(target) - 1)]
    if not len(valid):
        raise ValueError('No foreground slice can center a three-slice window.')

    center = (candidate if candidate in valid
              else int(valid[np.argmin(abs(valid - candidate))]))

    return working, target, center, {
        'crop_start_z': start,
        'crop_stop_z': stop,
        'original_shape_zyx': list(image.shape),
        'working_shape_zyx': list(target.shape),
        'initial_slice_original_z': center + start,
        'initial_slice_adjusted': center != candidate,
        'intensity_clip': list(clip),
        'protocol': protocol,
        'prompt_policy': (
            'ground-truth maximum-area slice; '
            'nearest valid foreground center at endpoints'
        ),
    }


# ---------------------------------------------------------------------------
# Initial prompt
# ---------------------------------------------------------------------------

def initial_prompt(mask2d):
    """Deterministic oracle tight box from the maximum-area foreground slice."""
    yy, xx = np.nonzero(mask2d)
    if not len(xx):
        raise ValueError('Cannot prompt an empty slice.')
    box = np.array([[xx.min(), yy.min(), xx.max(), yy.max()]], dtype=np.float32)
    return {'box': box}


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------

def segmentation_metrics(prediction, target, spacing_zyx, tolerance_mm):
    import surface_distance as sd
    pred = np.asarray(prediction, dtype=bool)
    gt = np.asarray(target, dtype=bool)
    spacing = np.asarray(spacing_zyx, dtype=float)
    if pred.ndim != 3 or pred.shape != gt.shape:
        raise ValueError('Metric masks must share a 3-D grid.')
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Provide positive physical spacing in z-y-x order.')
    if not np.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError('NSD tolerance must be positive millimetres.')

    npred, ngt = int(pred.sum()), int(gt.sum())
    if not npred or not ngt:
        both_empty = not npred and not ngt
        return {
            'dice': 1.0 if both_empty else 0.0,
            'nsd': 1.0 if both_empty else 0.0,
            'hd95_mm': 0.0 if both_empty else None,
            'surface_status': 'both_empty' if both_empty else 'one_empty_undefined_hd95',
            'foreground_recall': None if not ngt else 0.0,
            'foreground_precision': None if not npred else 0.0,
            'false_positive_ml': float(
                np.count_nonzero(pred & ~gt) * np.prod(spacing) / 1000),
        }

    intersection = int(np.count_nonzero(pred & gt))
    distances = sd.compute_surface_distances(gt, pred, spacing.tolist())
    return {
        'dice': 2 * intersection / (npred + ngt),
        'nsd': float(sd.compute_surface_dice_at_tolerance(distances, tolerance_mm)),
        'hd95_mm': float(sd.compute_robust_hausdorff(distances, 95)),
        'surface_status': 'valid',
        'foreground_recall': intersection / ngt,
        'foreground_precision': intersection / npred,
        'false_positive_ml': float(
            np.count_nonzero(pred & ~gt) * np.prod(spacing) / 1000),
    }
