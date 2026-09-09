import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from tutils.new.manager import ConfigManager
from datasets.eval_dataloader.loader_abstract import AbstractLoader
from test.volume_eval_sublora import VolumePredictor
from core.learner_sub1 import SamLearner
from modeling.build_sam3d2 import sam_model_registry


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_DIR = "../runs/sam/ddp_sub/btcv_liver_run2/ckpt"

OUTPUT_DIR = "visualizations_btcv"

# Which BTCV test case to visualize
CASE_INDEX = 0


# ============================================================
# BTCV CONFIGURATION
# ============================================================

EX_CONFIG = {
    "dataset": {
        "prompt": "box",
        "dataset_list": ["01_BTCV"],
        "label_idx": 1,
    }
}


# ============================================================
# CHECK CUDA
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. "
        "This script currently expects a GPU."
    )


# ============================================================
# FIND MODEL CHECKPOINTS
# ============================================================

print("=" * 70)
print("Searching for BTCV model checkpoints...")
print("=" * 70)

if not os.path.isdir(CHECKPOINT_DIR):
    raise FileNotFoundError(
        f"Checkpoint directory does not exist:\n{CHECKPOINT_DIR}"
    )


# Only take model checkpoints.
# Do NOT include optim_*.pth files.

checkpoint_files = [
    f
    for f in os.listdir(CHECKPOINT_DIR)
    if f.endswith(".pth")
    and f.startswith("model_")
]


# Sort them in a sensible order
def checkpoint_sort_key(filename):

    # Iteration checkpoints first
    if "model_iter_" in filename:

        try:
            number = int(
                filename
                .replace("model_iter_", "")
                .replace(".pth", "")
            )

            return (0, number)

        except ValueError:
            pass

    # Epoch checkpoints second
    if "model_epoch_" in filename:

        try:
            number = int(
                filename
                .replace("model_epoch_", "")
                .replace(".pth", "")
            )

            return (1, number)

        except ValueError:
            pass

    return (2, filename)


checkpoint_files.sort(
    key=checkpoint_sort_key
)


if len(checkpoint_files) == 0:

    raise RuntimeError(
        f"No model_*.pth checkpoints found in:\n"
        f"{CHECKPOINT_DIR}"
    )


print("\nCheckpoints found:")

for checkpoint in checkpoint_files:
    print("  ", checkpoint)

print("\nTotal model checkpoints:", len(checkpoint_files))


# ============================================================
# LOAD CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("Loading BTCV configuration...")
print("=" * 70)

config = ConfigManager()

config.add_config("configs/vit_sub.yaml")
config.add_config(EX_CONFIG)


# ============================================================
# LOAD BTCV TEST DATASET
# ============================================================

print("\nLoading BTCV test dataset...")

dataset = AbstractLoader(
    config["dataset"],
    split="test"
)

print("BTCV test cases:", len(dataset))


if len(dataset) == 0:

    raise RuntimeError(
        "BTCV test dataset is empty."
    )


if CASE_INDEX >= len(dataset):

    raise IndexError(
        f"CASE_INDEX={CASE_INDEX}, "
        f"but dataset contains only {len(dataset)} cases."
    )


# ============================================================
# SELECT BTCV TEST CASE
# ============================================================

print("\n" + "=" * 70)
print("Selecting BTCV test case...")
print("=" * 70)

data = dataset[CASE_INDEX]

print("Case name:", data["name"])


# ============================================================
# INSPECT DATA
# ============================================================

img = data["img"]

label = data["label"]

print("\nInput information:")

print(
    "Image type:",
    type(img)
)

print(
    "Image shape:",
    img.shape
)

print(
    "Label type:",
    type(label)
)

print(
    "Label shape:",
    label.shape
)

print(
    "Label unique values:",
    np.unique(label)
)


# ============================================================
# TEMPLATE SLICE
# ============================================================

if "template_slice_id" not in data:

    raise KeyError(
        "template_slice_id was not found "
        "in the BTCV dataset item."
    )


template_slice_id = data["template_slice_id"]

print(
    "Template slice:",
    template_slice_id
)


# ============================================================
# GET TEMPLATE SLICE MASK
# ============================================================

template_mask = label[template_slice_id]

print(
    "Template mask shape:",
    template_mask.shape
)

print(
    "Template mask unique values:",
    np.unique(template_mask)
)


# ============================================================
# CREATE BOX PROMPT
# ============================================================

ys, xs = np.where(
    template_mask > 0
)


if len(xs) == 0:

    raise RuntimeError(
        "No target-organ mask was found "
        "on the template slice.\n"
        "Check the BTCV label_idx and "
        "template_slice_id."
    )


box = np.array([
    xs.min(),
    ys.min(),
    xs.max(),
    ys.max()
])


print("\nGenerated bounding box:")
print(box)


# ============================================================
# CREATE SLIDE-SAM MODEL
# ============================================================

print("\n" + "=" * 70)
print("Creating ViT-B Slide-SAM model...")
print("=" * 70)

sam = sam_model_registry["vit_b"](
    checkpoint=None
)


learner = SamLearner(
    sam_model=sam,
    config=config,
    data_engine=None
)


# ============================================================
# APPLY LoRA
# ============================================================

print("\nApplying LoRA (rank=8)...")

learner.use_lora()


# ============================================================
# APPLY LoRA-SUB
# ============================================================

print("Applying LoRA-sub (rank=1)...")

learner.use_lora_sub()


# ============================================================
# MOVE MODEL TO GPU
# ============================================================

print("\nMoving model to GPU...")

learner.cuda()

learner.model.eval()


# ============================================================
# CREATE VOLUME PREDICTOR
# ============================================================

print("\nCreating VolumePredictor...")

predictor = VolumePredictor(
    model=learner.model,
    use_postprocess=True,
    use_noise_remove=True
)


# ============================================================
# PREPARE CT SLICE
# ============================================================

slice_img = img[template_slice_id]


if torch.is_tensor(slice_img):

    slice_img = (
        slice_img
        .detach()
        .cpu()
        .numpy()
    )


slice_img = np.squeeze(
    slice_img
)


# ============================================================
# PREPARE CASE NAME
# ============================================================

case_name = str(
    data["name"]
).replace(
    "/",
    "_"
).replace(
    " ",
    "_"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# PROCESS EVERY CHECKPOINT
# ============================================================

print("\n" + "=" * 70)
print("STARTING CHECKPOINT VISUALIZATION")
print("=" * 70)


for checkpoint_number, checkpoint_name in enumerate(
    checkpoint_files,
    start=1
):

    print("\n")
    print("=" * 70)

    print(
        f"CHECKPOINT {checkpoint_number}"
        f"/{len(checkpoint_files)}"
    )

    print(
        f"Processing: {checkpoint_name}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # CHECKPOINT PATH
    # --------------------------------------------------------

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        checkpoint_name
    )


    print(
        "Checkpoint path:",
        checkpoint_path
    )


    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    print(
        "\nLoading checkpoint..."
    )

    learner.load_well_trained_model(
        checkpoint_path
    )


    # Make sure we are in evaluation mode
    learner.model.eval()


    # --------------------------------------------------------
    # RUN SLIDE-SAM
    # --------------------------------------------------------

    print(
        "\nRunning Slide-SAM inference..."
    )


    with torch.no_grad():

        pred, stability = predictor.predict_volume(
            x=img,
            box=np.array([box]),
            template_slice_id=template_slice_id,
            return_stability=True
        )


    # --------------------------------------------------------
    # CONVERT PREDICTION TO NUMPY
    # --------------------------------------------------------

    if torch.is_tensor(pred):

        pred = (
            pred
            .detach()
            .cpu()
            .numpy()
        )


    pred = np.asarray(
        pred
    )


    print(
        "\nPrediction information:"
    )

    print(
        "Prediction shape:",
        pred.shape
    )

    print(
        "Prediction dtype:",
        pred.dtype
    )

    print(
        "Prediction min/max:",
        pred.min(),
        pred.max()
    )


    # --------------------------------------------------------
    # GET TEMPLATE SLICE PREDICTION
    # --------------------------------------------------------

    pred_slice = pred[
        template_slice_id
    ]


    if pred_slice.ndim > 2:

        pred_slice = np.squeeze(
            pred_slice
        )


    pred_mask = (
        pred_slice > 0
    )


    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    gt_slice = (
        template_mask > 0
    )


    # --------------------------------------------------------
    # MASK STATISTICS
    # --------------------------------------------------------

    gt_pixels = np.sum(
        gt_slice
    )

    pred_pixels = np.sum(
        pred_mask
    )


    print(
        "\nMask statistics:"
    )

    print(
        "Ground truth pixels:",
        gt_pixels
    )

    print(
        "Prediction pixels:",
        pred_pixels
    )


    # ========================================================
    # CREATE CHECKPOINT OUTPUT DIRECTORY
    # ========================================================

    checkpoint_output_name = (
        checkpoint_name
        .replace(".pth", "")
    )


    checkpoint_output_dir = os.path.join(
        OUTPUT_DIR,
        checkpoint_output_name
    )


    os.makedirs(
        checkpoint_output_dir,
        exist_ok=True
    )


    # ========================================================
    # SAVE 4-PANEL COMPARISON
    # ========================================================

    print(
        "\nCreating comparison visualization..."
    )


    plt.figure(
        figsize=(16, 4)
    )


    # --------------------------------------------------------
    # 1. CT
    # --------------------------------------------------------

    plt.subplot(
        1,
        4,
        1
    )

    plt.imshow(
        slice_img,
        cmap="gray"
    )

    plt.title(
        "BTCV CT Slice"
    )

    plt.axis(
        "off"
    )


    # --------------------------------------------------------
    # 2. Ground Truth
    # --------------------------------------------------------

    plt.subplot(
        1,
        4,
        2
    )

    plt.imshow(
        slice_img,
        cmap="gray"
    )

    plt.imshow(
        gt_slice,
        alpha=0.45
    )

    plt.title(
        "Ground Truth"
    )

    plt.axis(
        "off"
    )


    # --------------------------------------------------------
    # 3. Prediction
    # --------------------------------------------------------

    plt.subplot(
        1,
        4,
        3
    )

    plt.imshow(
        slice_img,
        cmap="gray"
    )

    plt.imshow(
        pred_mask,
        alpha=0.45
    )

    plt.title(
        "Slide-SAM Prediction"
    )

    plt.axis(
        "off"
    )


    # --------------------------------------------------------
    # 4. GT + Prediction
    # --------------------------------------------------------

    plt.subplot(
        1,
        4,
        4
    )

    plt.imshow(
        slice_img,
        cmap="gray"
    )

    plt.imshow(
        gt_slice,
        alpha=0.35
    )

    plt.imshow(
        pred_mask,
        alpha=0.35
    )

    plt.title(
        "GT + Prediction"
    )

    plt.axis(
        "off"
    )


    plt.tight_layout()


    # --------------------------------------------------------
    # SAVE COMPARISON
    # --------------------------------------------------------

    comparison_path = os.path.join(
        checkpoint_output_dir,
        f"{case_name}_comparison.png"
    )


    plt.savefig(
        comparison_path,
        dpi=200,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "Comparison saved:",
        comparison_path
    )


    # ========================================================
    # SAVE MASKED IMAGE
    # ========================================================

    print(
        "Creating masked image..."
    )


    plt.figure(
        figsize=(6, 6)
    )


    plt.imshow(
        slice_img,
        cmap="gray"
    )


    plt.imshow(
        pred_mask,
        alpha=0.50
    )


    plt.title(
        f"Slide-SAM - {checkpoint_name}"
    )


    plt.axis(
        "off"
    )


    plt.tight_layout()


    masked_path = os.path.join(
        checkpoint_output_dir,
        f"{case_name}_masked.png"
    )


    plt.savefig(
        masked_path,
        dpi=200,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "Masked image saved:",
        masked_path
    )


    # ========================================================
    # SAVE RAW MASK
    # ========================================================

    raw_mask_path = os.path.join(
        checkpoint_output_dir,
        f"{case_name}_mask.npy"
    )


    np.save(
        raw_mask_path,
        pred_mask.astype(np.uint8)
    )


    print(
        "Raw mask saved:",
        raw_mask_path
    )


    # ========================================================
    # FINISHED CHECKPOINT
    # ========================================================

    print(
        "\nFinished:",
        checkpoint_name
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("ALL CHECKPOINTS PROCESSED")
print("=" * 70)

print(
    "\nVisualizations are stored in:"
)

print(
    os.path.abspath(OUTPUT_DIR)
)

print("\nGenerated checkpoint folders:")

for checkpoint_name in checkpoint_files:

    folder_name = checkpoint_name.replace(
        ".pth",
        ""
    )

    print(
        "  ",
        os.path.join(
            OUTPUT_DIR,
            folder_name
        )
    )

print("\nDone.")