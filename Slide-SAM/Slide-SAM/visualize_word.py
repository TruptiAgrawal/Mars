import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from tutils.new.manager import ConfigManager
from datasets.eval_dataloader.loader_abstract import AbstractLoader
from test.volume_eval_sublora import VolumePredictor
from core.learner_sub1 import SamLearner
from modeling.build_sam3d2 import sam_model_registry
from trans_utils.data_utils import Data3dSolver


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHECKPOINT = "../runs/sam/ddp/word_label2_fresh_4/ckpt_v/model_latest.pth"

EX_CONFIG = {
    "dataset": {
        "prompt": "box",
        "dataset_list": ["word"],
        "label_idx": 2,          # WORD: 2 = spleen
    }
}


# --------------------------------------------------
# Load configuration
# --------------------------------------------------

config = ConfigManager()
config.add_config("configs/vit_sub.yaml")
config.add_config(EX_CONFIG)

print("Loading WORD dataset...")

dataset = AbstractLoader(
    config["dataset"],
    split="test"
)

print("WORD test cases:", len(dataset))


# --------------------------------------------------
# Load Slide-SAM-B
# --------------------------------------------------

print("Creating ViT-B model...")

sam = sam_model_registry["vit_b"](checkpoint=None)

learner = SamLearner(
    sam_model=sam,
    config=config,
    data_engine=None
)

print("Applying LoRA...")
learner.use_lora()

print("Loading trained WORD checkpoint...")
learner.load_well_trained_model(CHECKPOINT)

print("Applying LoRA-sub...")
learner.use_lora_sub()

learner.cuda()
learner.model.eval()

predictor = VolumePredictor(
    model=learner.model,
    use_postprocess=True,
    use_noise_remove=True
)


# --------------------------------------------------
# Select first WORD test case
# --------------------------------------------------

data = dataset[0]

img = data["img"]
label = data["label"]
template_slice_id = data["template_slice_id"]

print("\nCase:", data["name"])
print("Image shape:", img.shape)
print("Label shape:", label.shape)
print("Template slice:", template_slice_id)


# --------------------------------------------------
# Create box prompt from GT spleen mask
# --------------------------------------------------

template_mask = label[template_slice_id]

ys, xs = np.where(template_mask > 0)

if len(xs) == 0:
    raise RuntimeError("No spleen mask found on template slice.")

box = np.array([
    xs.min(),
    ys.min(),
    xs.max(),
    ys.max()
])

print("Spleen box:", box)


# --------------------------------------------------
# Run Slide-SAM volume prediction
# --------------------------------------------------

print("\nRunning Slide-SAM...")

with torch.no_grad():

    pred, stability = predictor.predict_volume(
        x=img,
        box=np.array([box]),
        template_slice_id=template_slice_id,
        return_stability=True
    )


# --------------------------------------------------
# Convert prediction to numpy
# --------------------------------------------------

if torch.is_tensor(pred):
    pred = pred.detach().cpu().numpy()

pred = np.asarray(pred)

print("Prediction shape:", pred.shape)


# --------------------------------------------------
# Get center slice
# --------------------------------------------------

pred_slice = pred[template_slice_id]

if pred_slice.ndim > 2:
    pred_slice = np.squeeze(pred_slice)

gt_slice = template_mask.astype(bool)


# --------------------------------------------------
# Display result
# --------------------------------------------------

slice_img = img[template_slice_id]

plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.imshow(slice_img, cmap="gray")
plt.title("CT Slice")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(slice_img, cmap="gray")
plt.imshow(gt_slice, alpha=0.45)
plt.title("Ground Truth Spleen")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(slice_img, cmap="gray")
plt.imshow(pred_slice > 0, alpha=0.45)
plt.title("Slide-SAM Prediction")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(slice_img, cmap="gray")
plt.imshow(gt_slice, alpha=0.35)
plt.imshow(pred_slice > 0, alpha=0.35)
plt.title("GT + Prediction")
plt.axis("off")

plt.tight_layout()

os.makedirs("visualizations", exist_ok=True)

output_path = "visualizations/word_spleen_slidesam.png"

plt.savefig(output_path, dpi=200, bbox_inches="tight")

print("\nSaved visualization to:")
print(output_path)

plt.show()
