r"""Run Slide-SAM volume inference and save inspection images.

This follows the test pipeline used by ``test.volume_eval``: the input is a
NIfTI volume, the box prompt is taken from the selected label on the template
slice when a ground-truth label is supplied, and ``VolumePredictor`` performs
the slice propagation and post-processing.

Examples (run from ``MARS``):
    python visualize_slidesam.py --checkpoint path\to\model_epoch_0.pth \
        --input path\to\image.nii.gz --label path\to\label.nii.gz \
        --output visualizations\case_001 --label-index 2

For prediction-only input, provide both ``--box`` and ``--template-slice``.
"""

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch

from core.learner_sub1 import SamLearner
from tutils.new.manager import ConfigManager
from test.volume_eval_sublora import VolumePredictor
from modeling.build_sam3d2 import sam_model_registry
from datasets.data_engine import BoxPromptGenerator, DataManager


def _read_nifti(path):
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image).astype(np.float32), image


def _checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a state-dict mapping: {path}")
    if not checkpoint or not all(torch.is_tensor(value) for value in checkpoint.values()):
        keys = ", ".join(str(key) for key in checkpoint)
        raise ValueError(
            "This repository saves a plain model state_dict. "
            f"Unsupported checkpoint structure in {path}; top-level keys: {keys}"
        )
    return checkpoint


def _checkpoint_iteration(path):
    match = re.search(r"(?:iter|epoch)[_-]?(\d+)", Path(path).name)
    return int(match.group(1)) if match else None


def _load_predictor(checkpoint, model_type, device, config):
    print("Creating model...")

    model = sam_model_registry[model_type](
        checkpoint=None
    )

    learner = SamLearner(
        sam_model=model,
        config=config,
        data_engine=DataManager(img_size=(1024, 1024))
    )

    print("Applying LoRA...")
    learner.use_lora()

    print("Applying LoRA-sub...")
    learner.use_lora_sub()

    print("Loading checkpoint...")
    learner.load_well_trained_model(checkpoint)

    learner.model.to(device)
    learner.model.eval()

    return VolumePredictor(
        model=learner.model,
        use_postprocess=True,
        use_noise_remove=True,
    )
def _prepare_volume(image_path, label_path, label_index, template_slice, box):
    volume, image_meta = _read_nifti(image_path)
    label = None
    crop_start = 0

    if label_path:
        label_raw, _ = _read_nifti(label_path)
        if label_raw.shape != volume.shape:
            raise ValueError(
                f"Image/label shape mismatch: {volume.shape} vs {label_raw.shape}"
            )
        label_raw = label_raw == label_index
        occupied = np.flatnonzero(label_raw.sum(axis=(1, 2)))
        if occupied.size == 0:
            raise ValueError(
                f"Label index {label_index} is absent from the supplied label volume"
            )
        crop_start = int(occupied[0])
        volume = volume[crop_start : int(occupied[-1]) + 1]
        label = label_raw[crop_start : int(occupied[-1]) + 1]

    if volume.shape[0] < 3:
        raise ValueError(f"Volume must contain at least 3 slices, got {volume.shape}")

    volume = np.clip(volume, -200, 400) if volume.min() < -10 else np.clip(volume, 0, 600)
    if template_slice is None:
        if label is None:
            raise ValueError("--template-slice is required when --label is omitted")
        template_slice = int(label.sum(axis=(1, 2)).argmax())
    else:
        template_slice = int(template_slice) - crop_start
    if not 1 <= template_slice < volume.shape[0] - 1:
        raise ValueError(
            f"Template slice must be between 1 and {volume.shape[0] - 2}; "
            f"got {template_slice}"
        )

    if box is None:
        if label is None:
            raise ValueError("Provide --label or --box to create the required box prompt")
        template_mask = label[template_slice]
        box = BoxPromptGenerator(size=None).mask_to_bbox(template_mask)
    else:
        box = tuple(float(value) for value in box)
    return volume, label, template_slice, np.asarray([box], dtype=np.float32), image_meta, crop_start


def _representative_slices(mask, label):
    reference = label if label is not None and label.any() else mask
    indices = np.flatnonzero(reference.sum(axis=(1, 2)) > 0)
    if indices.size == 0:
        indices = np.asarray([reference.shape[0] // 2])
    selected = [int(indices[0]), int(indices[len(indices) // 2]), int(indices[-1])]
    if label is not None and label.any():
        selected.append(int(np.argmax(label.sum(axis=(1, 2)))))
    return list(dict.fromkeys(selected))


def _normalise_slice(image):
    low, high = np.percentile(image, [1, 99])
    return np.clip((image - low) / (high - low + 1e-8), 0, 1)


def _save_slice(output_dir, image, prediction, label, index):
    base = _normalise_slice(image[index])
    prediction = prediction[index].astype(bool)
    fig, axes = plt.subplots(1, 4 if label is not None else 3, figsize=(16, 4))
    axes = np.atleast_1d(axes)
    axes[0].imshow(base, cmap="gray")
    axes[0].set_title("CT slice")
    axes[1].imshow(base, cmap="gray")
    axes[1].imshow(prediction, cmap="autumn", alpha=0.55)
    axes[1].set_title("CT + Slide-SAM prediction")
    axes[2].imshow(prediction, cmap="gray")
    axes[2].set_title("Predicted mask")
    if label is not None:
        axes[3].imshow(base, cmap="gray")
        axes[3].imshow(label[index].astype(bool), cmap="winter", alpha=0.45)
        axes[3].imshow(prediction, cmap="autumn", alpha=0.45)
        axes[3].set_title("GT + prediction")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    path = output_dir / f"slice_{index:04d}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def run_one(checkpoint, args, device):
    output_dir = Path(args.output) / Path(checkpoint).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    volume, label, template_slice, box, image_meta, crop_start = _prepare_volume(
        args.input, args.label, args.label_index, args.template_slice, args.box
    )

    # Build the same configuration used by test/volume_eval_sublora.py
    config = ConfigManager()
    config.add_config("configs/vit_sub.yaml")
    config.add_config({
        "dataset": {
            "prompt": "box",
            "dataset_list": ["01_BTCV"],
            "label_idx": args.label_index,
        },
        "pth": checkpoint,
    })

    predictor = _load_predictor(
        checkpoint,
        args.model_type,
        device,
        config,
    )
    with torch.no_grad():
        prediction, stability = predictor.predict_volume(
            x=volume,
            box=box,
            template_slice_id=template_slice,
            return_stability=True,
        )
    prediction = np.asarray(prediction).astype(bool)
    if prediction.shape != volume.shape:
        raise RuntimeError(f"Prediction shape {prediction.shape} != input shape {volume.shape}")

    mask_image = sitk.GetImageFromArray(prediction.astype(np.uint8))
    mask_image.SetSpacing(image_meta.GetSpacing())
    mask_image.SetDirection(image_meta.GetDirection())
    if crop_start:
        mask_image.SetOrigin(image_meta.TransformIndexToPhysicalPoint((0, 0, crop_start)))
    else:
        mask_image.SetOrigin(image_meta.GetOrigin())
    sitk.WriteImage(mask_image, str(output_dir / "predicted_mask.nii.gz"))
    slices = _representative_slices(prediction, label)
    image_paths = [_save_slice(output_dir, volume, prediction, label, index) for index in slices]
    metadata = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_iteration_or_epoch": _checkpoint_iteration(checkpoint),
        "input": str(Path(args.input).resolve()),
        "label": str(Path(args.label).resolve()) if args.label else None,
        "label_index": args.label_index if args.label else None,
        "template_slice_after_crop": template_slice,
        "crop_start_slice": crop_start,
        "box_prompt": box[0].tolist(),
        "volume_shape_after_crop": list(volume.shape),
        "prediction_voxels": int(prediction.sum()),
        "device": str(device),
        "preprocessing": "clip [-200, 400] when CT-like values are detected; otherwise [0, 600]; repository VolumePredictor resize/postprocess",
        "slices": [int(index) for index in slices],
        "images": [str(path.name) for path in image_paths],
        "stability_available": stability is not None,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    print(f"Saved {len(image_paths)} visualizations and predicted_mask.nii.gz to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoints = parser.add_mutually_exclusive_group(required=True)
    checkpoints.add_argument("--checkpoint")
    checkpoints.add_argument("--checkpoint-dir")
    parser.add_argument("--input", required=True, help="Input CT NIfTI volume")
    parser.add_argument("--label", help="Optional ground-truth NIfTI volume")
    parser.add_argument("--output", required=True)
    parser.add_argument("--label-index", type=int, default=2)
    parser.add_argument("--template-slice", type=int)
    parser.add_argument("--box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--model-type", choices=sorted(sam_model_registry), default="vit_b")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    for path, name in ((args.input, "input"), (args.label, "label")):
        if path and not os.path.isfile(path):
            parser.error(f"{name} path does not exist or is not a file: {path}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(args.device)

    if args.checkpoint:
        checkpoint_paths = [args.checkpoint]
    else:
        checkpoint_paths = sorted(
            str(path)
            for path in Path(args.checkpoint_dir).glob("*.pth")
            if path.is_file()
        )
        if not checkpoint_paths:
            parser.error(f"No .pth checkpoints found in {args.checkpoint_dir}")
    for checkpoint in checkpoint_paths:
        if not os.path.isfile(checkpoint):
            parser.error(f"checkpoint path does not exist or is not a file: {checkpoint}")
        print(f"Running {checkpoint} on {args.input} using {device}")
        run_one(checkpoint, args, device)


if __name__ == "__main__":
    main()