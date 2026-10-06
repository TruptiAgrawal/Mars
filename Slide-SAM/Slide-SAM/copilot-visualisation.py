
r"""Run Slide-SAM volume inference for all BTCV checkpoints, cases and organs.

For each checkpoint:
    for each BTCV CT volume:
        for each BTCV organ:
            create prediction
            save mask
            save representative visualizations
            save metadata

Expected BTCV structure:

datasets/01_BCV-Abdomen/
├── Training/
│   ├── img/
│   │   ├── img0001.nii.gz
│   │   ├── img0002.nii.gz
│   │   └── ...
│   └── label/
│       ├── label0001.nii.gz
│       ├── label0002.nii.gz
│       └── ...

Example:

python3 copilot_visualisation.py \
    --checkpoint-dir "../runs/sam/ddp_sub/btcv_liver_run2/ckpt" \
    --input-dir "datasets/01_BCV-Abdomen/Training/img" \
    --label-dir "datasets/01_BCV-Abdomen/Training/label" \
    --output "visualizations_copilot"
"""
#~/Slide-SAM/runs/sam/ddp_sub/btcv_4to1_20epoch/ckpt_v
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


BTCV_ORGANS = {
    1: "spleen",
    2: "right_kidney",
    3: "left_kidney",
    4: "gallbladder",
    5: "esophagus",
    6: "liver",
    7: "stomach",
    8: "aorta",
    9: "ivc",
    10: "portal_vein",
    11: "pancreas",
    12: "right_adrenal",
    13: "left_adrenal",
}


def _read_nifti(path):
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image).astype(np.float32), image


def _checkpoint_iteration(path):
    match = re.search(
        r"(?:iter|epoch)[_-]?(\d+)",
        Path(path).name
    )

    return int(match.group(1)) if match else None


def _load_predictor(checkpoint, model_type, device, config):
    print("Creating model...")

    model = sam_model_registry[model_type](
        checkpoint=None
    )

    learner = SamLearner(
        sam_model=model,
        config=config,
        data_engine=DataManager(
            img_size=(1024, 1024)
        )
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


def _prepare_volume(
    image_path,
    label_path,
    label_index,
    template_slice=None,
    box=None
):
    volume, image_meta = _read_nifti(
        image_path
    )

    label = None
    crop_start = 0

    if label_path:

        label_raw, _ = _read_nifti(
            label_path
        )

        if label_raw.shape != volume.shape:
            raise ValueError(
                f"Image/label shape mismatch: "
                f"{volume.shape} vs {label_raw.shape}"
            )

        label_raw = (
            label_raw == label_index
        )

        occupied = np.flatnonzero(
            label_raw.sum(axis=(1, 2))
        )

        if occupied.size == 0:
            raise ValueError(
                f"Label index {label_index} "
                f"is absent from the supplied "
                f"label volume"
            )

        crop_start = int(
            occupied[0]
        )

        volume = volume[
            crop_start:int(occupied[-1]) + 1
        ]

        label = label_raw[
            crop_start:int(occupied[-1]) + 1
        ]

    if volume.shape[0] < 3:
        raise ValueError(
            f"Volume must contain at least "
            f"3 slices, got {volume.shape}"
        )

    if volume.min() < -10:
        volume = np.clip(
            volume,
            -200,
            400
        )
    else:
        volume = np.clip(
            volume,
            0,
            600
        )

    if template_slice is None:

        if label is None:
            raise ValueError(
                "Template slice cannot be determined "
                "without a label volume"
            )

        template_slice = int(
            label.sum(axis=(1, 2)).argmax()
        )

    else:
        template_slice = (
            int(template_slice)
            - crop_start
        )

    if not 1 <= template_slice < volume.shape[0] - 1:
        raise ValueError(
            f"Template slice must be between "
            f"1 and {volume.shape[0] - 2}; "
            f"got {template_slice}"
        )

    if box is None:

        if label is None:
            raise ValueError(
                "Provide label or box"
            )

        template_mask = label[
            template_slice
        ]

        box = BoxPromptGenerator(
            size=None
        ).mask_to_bbox(
            template_mask
        )

    else:
        box = tuple(
            float(value)
            for value in box
        )

    return (
        volume,
        label,
        template_slice,
        np.asarray(
            [box],
            dtype=np.float32
        ),
        image_meta,
        crop_start,
    )


def _representative_slices(
    mask,
    label
):
    reference = (
        label
        if label is not None
        and label.any()
        else mask
    )

    indices = np.flatnonzero(
        reference.sum(axis=(1, 2)) > 0
    )

    if indices.size == 0:
        indices = np.asarray([
            reference.shape[0] // 2
        ])

    selected = [
        int(indices[0]),
        int(
            indices[len(indices) // 2]
        ),
        int(indices[-1]),
    ]

    if label is not None and label.any():
        selected.append(
            int(
                np.argmax(
                    label.sum(axis=(1, 2))
                )
            )
        )

    return list(
        dict.fromkeys(selected)
    )


def _normalise_slice(image):
    low, high = np.percentile(
        image,
        [1, 99]
    )

    return np.clip(
        (image - low)
        / (high - low + 1e-8),
        0,
        1
    )


def _save_slice(
    output_dir,
    image,
    prediction,
    label,
    index
):
    base = _normalise_slice(
        image[index]
    )

    prediction_slice = (
        prediction[index]
        .astype(bool)
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4)
    )

    axes = np.atleast_1d(
        axes
    )

    axes[0].imshow(
        base,
        cmap="gray"
    )

    axes[0].set_title(
        "CT slice"
    )

    axes[1].imshow(
        base,
        cmap="gray"
    )

    axes[1].imshow(
        prediction_slice,
        cmap="autumn",
        alpha=0.55
    )

    axes[1].set_title(
        "CT + Slide-SAM prediction"
    )

    axes[2].imshow(
        prediction_slice,
        cmap="gray"
    )

    axes[2].set_title(
        "Predicted mask"
    )

    axes[3].imshow(
        base,
        cmap="gray"
    )

    axes[3].imshow(
        label[index].astype(bool),
        cmap="winter",
        alpha=0.45
    )

    axes[3].imshow(
        prediction_slice,
        cmap="autumn",
        alpha=0.45
    )

    axes[3].set_title(
        "GT + prediction"
    )

    for axis in axes:
        axis.axis("off")

    fig.tight_layout()

    path = (
        output_dir /
        f"slice_{index:04d}.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


def _find_case_number(filename):
    match = re.search(
        r"(\d+)",
        filename
    )

    if match:
        return match.group(1)

    return None


def _find_btcv_cases(
    input_dir,
    label_dir
):
    input_dir = Path(
        input_dir
    )

    label_dir = Path(
        label_dir
    )

    input_paths = sorted(
        input_dir.glob("*.nii.gz")
    )

    cases = []

    for input_path in input_paths:

        case_number = _find_case_number(
            input_path.name
        )

        if case_number is None:

            print(
                f"Skipping input with no "
                f"case number: "
                f"{input_path.name}"
            )

            continue

        label_path = (
            label_dir /
            f"label{case_number}.nii.gz"
        )

        if not label_path.exists():

            print(
                f"Skipping {input_path.name}: "
                f"label not found: "
                f"{label_path}"
            )

            continue

        case_name = (
            f"case_{int(case_number):04d}"
        )

        cases.append(
            {
                "case_name": case_name,
                "input": input_path,
                "label": label_path,
            }
        )

    return cases


def run_one(
    checkpoint,
    predictor,
    device,
    input_path,
    label_path,
    output_root,
    label_index,
    organ_name,
    case_name,
    template_slice=None,
    box=None,
):
    checkpoint_name = (
        Path(checkpoint).stem
    )

    output_dir = (
        Path(output_root)
        / checkpoint_name
        / case_name
        / organ_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("-" * 80)

    print(
        f"Checkpoint : {checkpoint_name}"
    )

    print(
        f"Case       : {case_name}"
    )

    print(
        f"Organ      : {organ_name}"
    )

    print(
        f"Label      : {label_index}"
    )

    print("-" * 80)

    (
        volume,
        label,
        template_slice,
        box_array,
        image_meta,
        crop_start,
    ) = _prepare_volume(
        image_path=input_path,
        label_path=label_path,
        label_index=label_index,
        template_slice=template_slice,
        box=box,
    )

    print(
        f"Template slice: "
        f"{template_slice}"
    )

    print(
        f"Box prompt: "
        f"{box_array[0].tolist()}"
    )

    with torch.no_grad():

        prediction, stability = (
            predictor.predict_volume(
                x=volume,
                box=box_array,
                template_slice_id=template_slice,
                return_stability=True,
            )
        )

    prediction = np.asarray(
        prediction
    ).astype(bool)

    if prediction.shape != volume.shape:
        raise RuntimeError(
            f"Prediction shape "
            f"{prediction.shape} "
            f"!= input shape "
            f"{volume.shape}"
        )

    mask_image = sitk.GetImageFromArray(
        prediction.astype(
            np.uint8
        )
    )

    mask_image.SetSpacing(
        image_meta.GetSpacing()
    )

    mask_image.SetDirection(
        image_meta.GetDirection()
    )

    if crop_start:

        mask_image.SetOrigin(
            image_meta.TransformIndexToPhysicalPoint(
                (0, 0, crop_start)
            )
        )

    else:

        mask_image.SetOrigin(
            image_meta.GetOrigin()
        )

    predicted_mask_path = (
        output_dir /
        "predicted_mask.nii.gz"
    )

    sitk.WriteImage(
        mask_image,
        str(predicted_mask_path)
    )

    slices = _representative_slices(
        prediction,
        label
    )

    image_paths = []

    for index in slices:

        path = _save_slice(
            output_dir,
            volume,
            prediction,
            label,
            index,
        )

        image_paths.append(
            path
        )

    metadata = {
        "checkpoint": str(
            Path(checkpoint).resolve()
        ),

        "checkpoint_iteration_or_epoch":
            _checkpoint_iteration(
                checkpoint
            ),

        "case": case_name,

        "input": str(
            Path(input_path).resolve()
        ),

        "label": str(
            Path(label_path).resolve()
        ),

        "organ": organ_name,

        "label_index": label_index,

        "template_slice_after_crop":
            int(template_slice),

        "crop_start_slice":
            int(crop_start),

        "box_prompt":
            box_array[0].tolist(),

        "volume_shape_after_crop":
            list(volume.shape),

        "prediction_voxels":
            int(prediction.sum()),

        "device":
            str(device),

        "slices":
            [int(index) for index in slices],

        "images":
            [
                str(path.name)
                for path in image_paths
            ],

        "predicted_mask":
            predicted_mask_path.name,

        "stability_available":
            stability is not None,
    }

    metadata_path = (
        output_dir /
        "metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8"
    ) as stream:

        json.dump(
            metadata,
            stream,
            indent=2
        )

    print(
        f"Saved {len(image_paths)} "
        f"visualizations"
    )

    print(
        f"Output: {output_dir}"
    )


def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing all .pth checkpoints"
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing all BTCV CT volumes"
    )

    parser.add_argument(
        "--label-dir",
        required=True,
        help="Directory containing all BTCV label volumes"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output visualization directory"
    )

    parser.add_argument(
        "--model-type",
        choices=sorted(
            sam_model_registry
        ),
        default="vit_b"
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Validate directories
    # ---------------------------------------------------------

    if not os.path.isdir(
        args.checkpoint_dir
    ):
        parser.error(
            "Checkpoint directory does not exist: "
            f"{args.checkpoint_dir}"
        )

    if not os.path.isdir(
        args.input_dir
    ):
        parser.error(
            "Input directory does not exist: "
            f"{args.input_dir}"
        )

    if not os.path.isdir(
        args.label_dir
    ):
        parser.error(
            "Label directory does not exist: "
            f"{args.label_dir}"
        )

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        parser.error(
            "CUDA was requested but "
            "is unavailable"
        )

    device = torch.device(
        args.device
    )

    # ---------------------------------------------------------
    # Find all checkpoints
    # ---------------------------------------------------------

    checkpoint_paths = sorted(
        str(path)
        for path in Path(
            args.checkpoint_dir
        ).glob("*.pth")
        if path.is_file()
    )

    if not checkpoint_paths:

        parser.error(
            "No .pth checkpoints found in "
            f"{args.checkpoint_dir}"
        )

    # ---------------------------------------------------------
    # Find all BTCV input/label pairs
    # ---------------------------------------------------------

    cases = _find_btcv_cases(
        args.input_dir,
        args.label_dir
    )

    if not cases:

        parser.error(
            "No matching BTCV "
            "input/label pairs found"
        )

    # ---------------------------------------------------------
    # Calculate total work
    # ---------------------------------------------------------

    number_of_checkpoints = (
        len(checkpoint_paths)
    )

    number_of_cases = len(
        cases
    )

    number_of_organs = len(
        BTCV_ORGANS
    )

    total_runs = (
        number_of_checkpoints
        * number_of_cases
        * number_of_organs
    )

    print()
    print("=" * 80)
    print("SLIDE-SAM BTCV VISUALIZATION")
    print("=" * 80)

    print(
        f"Checkpoints : "
        f"{number_of_checkpoints}"
    )

    print(
        f"Cases       : "
        f"{number_of_cases}"
    )

    print(
        f"Organs      : "
        f"{number_of_organs}"
    )

    print(
        f"Total runs  : "
        f"{total_runs}"
    )

    print(
        f"Device      : "
        f"{device}"
    )

    print("=" * 80)

    # ---------------------------------------------------------
    # Print checkpoints
    # ---------------------------------------------------------

    print()
    print("Checkpoints:")

    for checkpoint in checkpoint_paths:

        print(
            f"  {Path(checkpoint).name}"
        )

    # ---------------------------------------------------------
    # Print cases
    # ---------------------------------------------------------

    print()
    print(
        f"Found {len(cases)} cases:"
    )

    for case in cases:

        print(
            f"  {case['case_name']} -> "
            f"{case['input'].name} -> "
            f"{case['label'].name}"
        )

    # ---------------------------------------------------------
    # Main processing
    # ---------------------------------------------------------

    completed = 0
    failed = 0
    skipped = 0

    for checkpoint_number, checkpoint in enumerate(
        checkpoint_paths,
        start=1
    ):

        checkpoint_name = (
            Path(checkpoint).stem
        )

        print()
        print()
        print("#" * 80)

        print(
            f"CHECKPOINT "
            f"{checkpoint_number}/"
            f"{number_of_checkpoints}"
        )

        print(
            checkpoint_name
        )

        print("#" * 80)

        # -----------------------------------------------------
        # Create configuration
        # -----------------------------------------------------

        config = ConfigManager()

        config.add_config(
            "configs/vit_sub.yaml"
        )

        config.add_config(
            {
                "dataset": {
                    "prompt": "box",
                    "dataset_list": [
                        "01_BTCV"
                    ],
                    "label_idx": 2,
                },
                "pth": checkpoint,
            }
        )

        # -----------------------------------------------------
        # Load checkpoint ONCE
        # -----------------------------------------------------

        try:

            predictor = _load_predictor(
                checkpoint,
                args.model_type,
                device,
                config,
            )

        except Exception as error:

            print()
            print(
                f"FAILED TO LOAD CHECKPOINT: "
                f"{checkpoint}"
            )

            print(
                f"Error: {error}"
            )

            failed += (
                number_of_cases
                * number_of_organs
            )

            continue

        # -----------------------------------------------------
        # Cases
        # -----------------------------------------------------

        for case_number, case in enumerate(
            cases,
            start=1
        ):

            case_name = case[
                "case_name"
            ]

            input_path = case[
                "input"
            ]

            label_path = case[
                "label"
            ]

            print()
            print(
                "=" * 80
            )

            print(
                f"CASE "
                f"{case_number}/"
                f"{number_of_cases}"
            )

            print(
                f"{case_name}"
            )

            print(
                f"Input: "
                f"{input_path.name}"
            )

            print(
                f"Label: "
                f"{label_path.name}"
            )

            print(
                "=" * 80
            )

            # -------------------------------------------------
            # All 13 organs
            # -------------------------------------------------

            for label_index, organ_name in (
                BTCV_ORGANS.items()
            ):

                completed += 1

                print()
                print(
                    f"Overall progress: "
                    f"{completed}/"
                    f"{total_runs}"
                )

                try:

                    run_one(
                        checkpoint=checkpoint,
                        predictor=predictor,
                        device=device,
                        input_path=str(
                            input_path
                        ),
                        label_path=str(
                            label_path
                        ),
                        output_root=args.output,
                        label_index=label_index,
                        organ_name=organ_name,
                        case_name=case_name,
                    )

                except ValueError as error:

                    skipped += 1

                    print()
                    print(
                        f"SKIPPED: "
                        f"{checkpoint_name} / "
                        f"{case_name} / "
                        f"{organ_name}"
                    )

                    print(
                        f"Reason: {error}"
                    )

                except Exception as error:

                    failed += 1

                    print()
                    print(
                        f"FAILED: "
                        f"{checkpoint_name} / "
                        f"{case_name} / "
                        f"{organ_name}"
                    )

                    print(
                        f"Error: {error}"
                    )

                    print(
                        "Continuing..."
                    )

        # -----------------------------------------------------
        # Release checkpoint
        # -----------------------------------------------------

        del predictor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print()
        print(
            f"Finished checkpoint: "
            f"{checkpoint_name}"
        )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print()
    print()
    print("=" * 80)
    print("ALL VISUALIZATIONS FINISHED")
    print("=" * 80)

    print(
        f"Total expected runs : "
        f"{total_runs}"
    )

    print(
        f"Completed/attempted : "
        f"{completed}"
    )

    print(
        f"Skipped             : "
        f"{skipped}"
    )

    print(
        f"Failed              : "
        f"{failed}"
    )

    print(
        f"Output directory    : "
        f"{args.output}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
