"""
btcv_cross_validation/z_spacing/helpers.py

Geometry-preserving z-axis resampling helpers for the spacing experiment.

Rules:
- Images: linear interpolation (scipy.ndimage.zoom with order=1).
- Discrete labels: nearest-neighbour interpolation (order=0).
- Origin, direction, x/y spacing are preserved exactly.
- Only the z dimension and z spacing change.
- Prompt physical coordinates are mapped through the same grid.
- Scoring always happens on the DECLARED scoring grid, not the working grid.
- No information from the fine-spacing original leaks into the thick-spacing branch.
"""
from pathlib import Path
import hashlib
import re
import numpy as np


# ---------------------------------------------------------------------------
# Manifest helpers (identical to box_padding/helpers.py for isolation)
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
# Resampling
# ---------------------------------------------------------------------------

def resample_image_z(image_zyx, source_z_mm, target_z_mm):
    """
    Resample a 3-D float32 array along axis 0 (z) using linear interpolation.

    Parameters
    ----------
    image_zyx   : np.ndarray, shape (Z, Y, X), float32
    source_z_mm : float, physical spacing of the input along z
    target_z_mm : float, desired spacing along z

    Returns
    -------
    resampled   : np.ndarray, shape (Z', Y, X), float32
    actual_z_mm : float, actual z spacing achieved
                  (may differ slightly from target due to integer rounding)
    """
    from scipy.ndimage import zoom
    image_zyx = np.asarray(image_zyx, dtype=np.float32)
    if image_zyx.ndim != 3:
        raise ValueError('Expected 3-D array.')
    if source_z_mm <= 0 or target_z_mm <= 0:
        raise ValueError('Spacings must be positive.')
    if not np.isfinite(image_zyx).all():
        raise ValueError('Non-finite image values.')

    zoom_factor = source_z_mm / target_z_mm
    if abs(zoom_factor - 1.0) < 1e-6:
        return image_zyx.copy(), float(source_z_mm)

    # Compute output size to determine the actual achieved spacing.
    z_out = max(2, int(round(image_zyx.shape[0] * zoom_factor)))
    actual_zoom = z_out / image_zyx.shape[0]
    actual_z_mm = source_z_mm / actual_zoom

    resampled = zoom(image_zyx, (actual_zoom, 1.0, 1.0), order=1, prefilter=False)
    return resampled.astype(np.float32), float(actual_z_mm)


def resample_label_z(label_zyx, source_z_mm, target_z_mm):
    """
    Resample a 3-D boolean/integer label array along axis 0 using
    nearest-neighbour interpolation.

    Returns the resampled array (same dtype as input) and actual z spacing.
    """
    from scipy.ndimage import zoom
    label_zyx = np.asarray(label_zyx)
    if label_zyx.ndim != 3:
        raise ValueError('Expected 3-D label array.')
    if source_z_mm <= 0 or target_z_mm <= 0:
        raise ValueError('Spacings must be positive.')

    zoom_factor = source_z_mm / target_z_mm
    if abs(zoom_factor - 1.0) < 1e-6:
        return label_zyx.copy(), float(source_z_mm)

    z_out = max(2, int(round(label_zyx.shape[0] * zoom_factor)))
    actual_zoom = z_out / label_zyx.shape[0]
    actual_z_mm = source_z_mm / actual_zoom

    resampled = zoom(label_zyx.astype(np.float32), (actual_zoom, 1.0, 1.0),
                     order=0, prefilter=False)
    return resampled.astype(label_zyx.dtype), float(actual_z_mm)


def map_z_coord_to_resampled(z_orig, source_z_mm, target_z_mm, z_out_size):
    """
    Map a z-index from the source grid to the resampled grid.

    Uses the same zoom factor as resample_image_z / resample_label_z:
        zoom_factor = source_z_mm / target_z_mm
        actual_zoom = z_out_size / z_in_size  (computed inside resample_*)

    This function accepts the already-known z_out_size so callers don't
    need to recompute it.

    Returns the nearest valid integer index in [0, z_out_size - 1].
    """
    if source_z_mm <= 0 or target_z_mm <= 0 or z_out_size < 1:
        raise ValueError('Invalid spacing or output size.')
    # Physical position of the source index centre:
    #   pos_mm = z_orig * source_z_mm + source_z_mm/2
    # Corresponding index in the output grid:
    #   z_new = (pos_mm - target_z_mm/2) / target_z_mm
    # This is the continuous center-to-center mapping.
    pos_mm = z_orig * source_z_mm + source_z_mm / 2.0
    z_new = (pos_mm - target_z_mm / 2.0) / target_z_mm
    return int(np.clip(round(z_new), 0, z_out_size - 1))


def map_z_coord_to_original(z_resampled, source_z_mm, target_z_mm, z_orig_size):
    """
    Inverse of map_z_coord_to_resampled: map a resampled index back to
    the original grid.
    """
    if source_z_mm <= 0 or target_z_mm <= 0 or z_orig_size < 1:
        raise ValueError('Invalid spacing or output size.')
    pos_mm = z_resampled * target_z_mm + target_z_mm / 2.0
    z_orig = (pos_mm - source_z_mm / 2.0) / source_z_mm
    return int(np.clip(round(z_orig), 0, z_orig_size - 1))


# ---------------------------------------------------------------------------
# Prepare arrays (same contract as box_padding/helpers.py)
# ---------------------------------------------------------------------------

def prepare_arrays(image, label, protocol):
    """
    Clip, crop and find the initial-prompt center slice.
    Identical to box_padding/helpers.py::prepare_arrays — copied for isolation.
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

    center = candidate if candidate in valid else int(valid[np.argmin(abs(valid - candidate))])

    return working, target, center, {
        'crop_start_z': start,
        'crop_stop_z': stop,
        'original_shape_zyx': list(image.shape),
        'working_shape_zyx': list(target.shape),
        'initial_slice_original_z': center + start,
        'initial_slice_adjusted': center != candidate,
        'intensity_clip': list(clip),
        'protocol': protocol,
        'prompt_policy': 'ground-truth maximum-area slice; nearest valid foreground center at endpoints',
    }


# ---------------------------------------------------------------------------
# Initial prompt (same contract as box_padding/helpers.py)
# ---------------------------------------------------------------------------

def initial_prompt(mask2d):
    """Deterministic oracle tight box from the maximum-area foreground slice."""
    yy, xx = np.nonzero(mask2d)
    if not len(xx):
        raise ValueError('Cannot prompt an empty slice.')
    box = np.array([[xx.min(), yy.min(), xx.max(), yy.max()]], dtype=np.float32)
    return {'box': box}


# ---------------------------------------------------------------------------
# Segmentation metrics (identical to box_padding/helpers.py)
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
            'false_positive_ml': float(np.count_nonzero(pred & ~gt) * np.prod(spacing) / 1000),
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
        'false_positive_ml': float(np.count_nonzero(pred & ~gt) * np.prod(spacing) / 1000),
    }
