"""CPU-only geometry, manifests and metrics for the padding experiment."""
from pathlib import Path
import hashlib
import re
import numpy as np


def expand_box(box, image_hw, fraction):
    """Image-relative padding PER SIDE, float coordinates, inclusive upper bounds."""
    h, w = image_hw
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError('Box must have four finite coordinates.')
    if h < 1 or w < 1 or not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Invalid dimensions or padding fraction.')
    x0, y0, x1, y1 = box
    if x0 > x1 or y0 > y1 or x1 < 0 or y1 < 0 or x0 > w-1 or y0 > h-1:
        raise ValueError('Invalid box or box outside image.')
    return (float(np.clip(x0-fraction*w, 0, w-1)),
            float(np.clip(y0-fraction*h, 0, h-1)),
            float(np.clip(x1+fraction*w, 0, w-1)),
            float(np.clip(y1+fraction*h, 0, h-1)))


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
    training, evaluation = manifest_ids(train_manifest), manifest_ids(eval_manifest)
    overlap = set(training) & set(evaluation)
    if overlap:
        raise ValueError(f'Training/evaluation overlap: {sorted(overlap)}')
    return training, evaluation


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_arrays(image, label, protocol):
    """Retain z-y-x orientation and explicitly record the oracle prompt/crop."""
    image, label = np.asarray(image), np.asarray(label, dtype=bool)
    if image.ndim != 3 or image.shape != label.shape or min(image.shape) < 3:
        raise ValueError('Image/label must share a 3D grid with dimensions >= 3.')
    if not np.isfinite(image).all():
        raise ValueError('Nonfinite image intensity.')
    areas = label.sum(axis=(1, 2))
    occupied = np.flatnonzero(areas)
    if not len(occupied):
        raise ValueError('Target absent: cannot generate a positive oracle prompt.')
    if protocol == 'gt-cropped':
        start, stop = int(occupied[0]), int(occupied[-1])+1
    elif protocol == 'full-volume':
        start, stop = 0, image.shape[0]
    else:
        raise ValueError('Unknown evaluation protocol.')
    # Match the historical volume-wide clipping decision BEFORE cropping.
    clip = (-200, 400) if image.min() < -10 else (0, 600)
    working = np.clip(image, *clip)[start:stop].astype(np.float32)
    target = label[start:stop]
    if len(target) < 3:
        raise ValueError('Fewer than three slices in evaluation grid.')
    candidate = int(np.argmax(areas))-start
    valid = np.flatnonzero(target.sum(axis=(1, 2)))
    valid = valid[(valid >= 1) & (valid < len(target)-1)]
    if not len(valid):
        raise ValueError('No foreground slice can center a three-slice window.')
    # Preserve the original maximum-area prompt whenever it is a valid center.
    center = candidate if candidate in valid else int(valid[np.argmin(abs(valid-candidate))])
    return working, target, center, {'crop_start_z':start, 'crop_stop_z':stop,
        'original_shape_zyx':list(image.shape), 'working_shape_zyx':list(target.shape),
        'initial_slice_original_z':center+start, 'initial_slice_adjusted':center != candidate,
        'intensity_clip':list(clip), 'protocol':protocol,
        'prompt_policy':'ground-truth maximum-area slice; nearest valid foreground center at endpoints'}


def initial_prompt(mask2d, kind):
    """Deterministic oracle box or a foreground point nearest the box center."""
    yy, xx = np.nonzero(mask2d)
    if not len(xx):
        raise ValueError('Cannot prompt an empty slice.')
    box = np.array([[xx.min(), yy.min(), xx.max(), yy.max()]], dtype=np.float32)
    if kind == 'box':
        return {'box':box}
    if kind != 'point':
        raise ValueError('Unsupported prompt type.')
    cx, cy = (box[0,0]+box[0,2])/2, (box[0,1]+box[0,3])/2
    best = np.argmin((xx-cx)**2+(yy-cy)**2)
    return {'point_coords':np.array([[xx[best], yy[best]]], dtype=np.float32),
            'point_labels':np.ones((1,1), dtype=np.int64)}


def segmentation_metrics(prediction, target, spacing_zyx, tolerance_mm):
    import surface_distance as sd
    pred, gt = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    spacing = np.asarray(spacing_zyx, dtype=float)
    if pred.ndim != 3 or pred.shape != gt.shape:
        raise ValueError('Metric masks must share a 3D grid.')
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Provide positive physical spacing in array z-y-x order.')
    if not np.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError('NSD tolerance must be positive millimeters.')
    npred, ngt = int(pred.sum()), int(gt.sum())
    if not npred or not ngt:
        both_empty = not npred and not ngt
        return {'dice':1.0 if both_empty else 0.0, 'nsd':1.0 if both_empty else 0.0,
                'hd95_mm':0.0 if both_empty else None,
                'surface_status':'both_empty' if both_empty else 'one_empty_undefined_hd95',
                'foreground_recall':None if not ngt else 0.0,
                'foreground_precision':None if not npred else 0.0,
                'false_positive_ml':float(np.count_nonzero(pred & ~gt)*np.prod(spacing)/1000)}
    intersection = int(np.count_nonzero(pred & gt))
    distances = sd.compute_surface_distances(gt, pred, spacing.tolist())
    return {'dice':2*intersection/(npred+ngt),
            'nsd':float(sd.compute_surface_dice_at_tolerance(distances,tolerance_mm)),
            'hd95_mm':float(sd.compute_robust_hausdorff(distances,95)),
            'surface_status':'valid', 'foreground_recall':intersection/ngt,
            'foreground_precision':intersection/npred,
            'false_positive_ml':float(np.count_nonzero(pred & ~gt)*np.prod(spacing)/1000)}
