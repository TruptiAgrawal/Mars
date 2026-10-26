# MARS Plan 2 — Segmentation Agent (Agent 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** CT in → one boolean mask per organ out, using an automatic prompt source (TotalSegmentator) plus the LoRA Slide-SAM refiner, with a per-organ choice of which mask is deployed.

**Architecture:** Two small interfaces hide the GPU models: `PromptSource.predict(ct_path)` (coarse masks) and `Refiner.refine(ct, box)` (Slide-SAM mask + probabilities). `SegmentationAgent` boxes each coarse mask, refines it, saves both masks, and `apply_deploy` points `mask_path` at whichever one `configs/deploy.json` says is better. Offline scripts produce the dev scores and that decision.

**Tech Stack:** numpy, nibabel, pandas, pytest; TotalSegmentator and your Slide-SAM code (GPU, lab only).

**Spec:** `docs/superpowers/specs/2026-10-26-mars-uncertainty-design.md` (§3 data roles, §4 Agent 1, §7.6 ablations) and `2026-10-26-mars-backend-design.md`.

**Prerequisite:** Plan 1 Tasks 1–2 done (`mars.contracts`, `mars.volumes`, `mars.metrics`).

## Global Constraints

- Organs: `liver`, `spleen`, `kidney_left`, `stomach`. Arrays are `(z, y, x)`; spacing in mm.
- Box format: `[z0, y0, x0, z1, y1, x1]`, end-exclusive, always inside the volume.
- The prompt source's own mask (`auto`) is the baseline; Slide-SAM (`refined`) is deployed for an organ only if it is significantly better on the development set (paired patient-clustered bootstrap, 95% CI of mean Dice difference above 0). Otherwise `auto` is deployed.
- An organ with no prompt scores Dice 0 in the dev table (a failure, never skipped).
- BTCV label ids: spleen 1, left kidney 3, liver 6, stomach 7 (verify against your label files).
- Calibration and evaluation use automatic prompts only; oracle-prompt Dice is an ablation, not a result.

## Review Focus

1. Prompt source returns an all-False mask for an organ → treated exactly like "organ absent" (no files, `deployed="none"`) (Task 4 test).
2. Coarse mask touches the volume edge → box is clipped, never out of range (Task 1 test).
3. Slide-SAM raises on one organ → that organ keeps its `auto` mask, other organs are unaffected (Task 4 test).
4. `deploy.json` lacks an organ or says `"refined"` when no refined mask exists → falls back to `auto` (Task 3 test).
5. Missing prompt in dev data → Dice 0 row, not a dropped row (Task 5 test).

---

### Task 1: Boxes and boundary entropy

**Files:**
- Create: `mars/agents/__init__.py`, `mars/agents/segmentation/__init__.py`, `mars/agents/segmentation/boxes.py`
- Test: `tests/agents/segmentation/test_boxes.py` (create the empty `tests/agents/__init__.py` and `tests/agents/segmentation/__init__.py` too)

**Interfaces:**
- Produces: `boxes.box_from_mask(mask: np.ndarray, margin: int = 2) -> list[int] | None`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from mars.agents.segmentation.boxes import box_from_mask


def test_box_with_margin():
    m = np.zeros((20, 20, 20), bool); m[5:8, 6:9, 7:10] = True
    assert box_from_mask(m, margin=2) == [3, 4, 5, 10, 11, 12]


def test_box_is_clipped_to_volume():
    m = np.zeros((10, 10, 10), bool); m[0:2, 8:10, 0:10] = True
    box = box_from_mask(m, margin=3)
    assert box == [0, 5, 0, 5, 10, 10]


def test_empty_mask_has_no_box():
    assert box_from_mask(np.zeros((4, 4, 4), bool)) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/agents/segmentation/test_boxes.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement** `mars/agents/segmentation/boxes.py`

```python
import numpy as np


def box_from_mask(mask: np.ndarray, margin: int = 2) -> list[int] | None:
    """Tight [z0,y0,x0,z1,y1,x1] box (end-exclusive) plus margin, clipped to the volume."""
    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    lo = np.maximum(idx.min(axis=0) - margin, 0)
    hi = np.minimum(idx.max(axis=0) + 1 + margin, mask.shape)
    return [int(v) for v in (*lo, *hi)]
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/agents/segmentation/test_boxes.py -v` → 3 PASS

- [ ] **Step 5: Commit**

```bash
git add mars/agents tests/agents
git commit -m "feat(seg): box from mask"
```

---

### Task 2: PromptSource and Refiner interfaces

**Files:**
- Create: `mars/agents/segmentation/prompt_source.py`, `mars/agents/segmentation/refiner.py`, `mars/agents/segmentation/slide_sam_infer.py`
- Test: `tests/agents/segmentation/test_refiner.py`, `tests/gpu/test_slide_sam_smoke.py` (create `tests/gpu/__init__.py`)

**Interfaces:**
- Produces:
  - `PromptSource` protocol: `name: str`; `predict(ct_path: str) -> dict[str, np.ndarray]` (organ → bool zyx mask; organs it did not find may be missing or all-False)
  - `TotalSegmentatorPromptSource(device: str = "gpu")`, `.name == "totalseg"`
  - `Refiner` protocol: `refine(ct_zyx: np.ndarray, box: list[int]) -> tuple[np.ndarray, np.ndarray]` (bool mask, float32 probability volume, both same shape as `ct_zyx`)
  - `refiner.boundary_entropy(prob: np.ndarray) -> float`
  - `refiner.SlideSamRefiner(model, infer_fn)`, `refiner.load_refiner(ckpt: str, device: str) -> SlideSamRefiner`
  - In `slide_sam_infer.py`: `load_model(ckpt_path: str, device: str) -> object` and `infer(model, ct_zyx: np.ndarray, box_zyx: list[int]) -> np.ndarray` (float32 probability volume in [0, 1], same shape as `ct_zyx`)

- [ ] **Step 1: Write the failing test** — `tests/agents/segmentation/test_refiner.py`

```python
import numpy as np
import pytest

from mars.agents.segmentation.refiner import SlideSamRefiner, boundary_entropy


def test_refiner_thresholds_probabilities():
    ct = np.zeros((4, 4, 4), np.float32)

    def infer(model, ct_zyx, box):
        p = np.zeros_like(ct_zyx); p[1:3] = 0.8
        return p

    mask, prob = SlideSamRefiner(model=None, infer_fn=infer).refine(ct, [0, 0, 0, 4, 4, 4])
    assert mask.dtype == bool and mask[1:3].all() and not mask[0].any()
    assert prob.dtype == np.float32


def test_boundary_entropy_zero_when_confident():
    assert boundary_entropy(np.array([0.0, 1.0, 0.999, 0.001])) == 0.0


def test_boundary_entropy_one_bit_at_half():
    assert boundary_entropy(np.full((2, 2, 2), 0.5)) == pytest.approx(1.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/agents/segmentation/test_refiner.py -v` → FAIL

- [ ] **Step 3: Implement**

`mars/agents/segmentation/refiner.py`:
```python
from typing import Protocol

import numpy as np


class Refiner(Protocol):
    def refine(self, ct_zyx: np.ndarray, box: list[int]) -> tuple[np.ndarray, np.ndarray]: ...


def boundary_entropy(prob: np.ndarray) -> float:
    """Mean binary entropy (bits) over uncertain voxels (0.02 < p < 0.98); 0 if none."""
    band = (prob > 0.02) & (prob < 0.98)
    if not band.any():
        return 0.0
    p = np.clip(prob[band], 1e-6, 1 - 1e-6)
    return float(np.mean(-(p * np.log2(p) + (1 - p) * np.log2(1 - p))))


class SlideSamRefiner:
    def __init__(self, model, infer_fn):
        self.model = model
        self.infer_fn = infer_fn

    def refine(self, ct_zyx, box):
        prob = np.asarray(self.infer_fn(self.model, ct_zyx, box), dtype=np.float32)
        return prob > 0.5, prob


def load_refiner(ckpt: str, device: str) -> SlideSamRefiner:
    from mars.agents.segmentation import slide_sam_infer
    return SlideSamRefiner(slide_sam_infer.load_model(ckpt, device), slide_sam_infer.infer)
```

`mars/agents/segmentation/prompt_source.py`:
```python
import tempfile
from pathlib import Path
from typing import Protocol

import nibabel as nib
import numpy as np

from mars.contracts import ORGANS


class PromptSource(Protocol):
    name: str

    def predict(self, ct_path: str) -> dict[str, np.ndarray]: ...


class TotalSegmentatorPromptSource:
    """Runs TotalSegmentator for the four organs and returns bool zyx masks."""
    name = "totalseg"

    def __init__(self, device: str = "gpu"):
        self.device = device

    def predict(self, ct_path: str) -> dict[str, np.ndarray]:
        from totalsegmentator.python_api import totalsegmentator
        out: dict[str, np.ndarray] = {}
        with tempfile.TemporaryDirectory() as tmp:
            totalsegmentator(ct_path, tmp, roi_subset=list(ORGANS),
                             device=self.device, quiet=True)
            for organ in ORGANS:
                f = Path(tmp) / f"{organ}.nii.gz"
                if f.exists():
                    out[organ] = np.transpose(np.asarray(nib.load(str(f)).dataobj) > 0, (2, 1, 0))
        return out
```

- [ ] **Step 4: Move your Slide-SAM code into `mars/agents/segmentation/slide_sam_infer.py`**

This is the one place the plan cannot show the body: the code lives in your notebook. Create the file with exactly these two public functions and move your existing cells into them:

```python
"""Slide-SAM (LoRA) inference. Moved from the training/eval notebook."""
import numpy as np


def load_model(ckpt_path: str, device: str):
    """Build the LoRA-fine-tuned Slide-SAM exactly as in your eval notebook,
    load the fold weights from ckpt_path, move to `device`, set eval(), return it."""
    ...  # <- paste your model-construction + weight-loading cells here


def infer(model, ct_zyx: np.ndarray, box_zyx: list[int]) -> np.ndarray:
    """Return a float32 probability volume in [0, 1] with the same shape as ct_zyx.

    Use the SAME preprocessing (HU window, resize, slice sampling) and the SAME
    3D-box prompting as your oracle-prompt evaluation. The only change: the box
    comes from `box_zyx` = [z0, y0, x0, z1, y1, x1] (end-exclusive) instead of the
    ground truth. Voxels outside the box slices stay 0.
    """
    ...  # <- paste your inference cell here, returning probabilities (not a thresholded mask)
```
Remove the `...` lines when you paste. Keep both signatures unchanged; everything else in the repo depends on them.

- [ ] **Step 5: Write the GPU smoke test** — `tests/gpu/test_slide_sam_smoke.py`

```python
import os

import numpy as np
import pytest

from mars.agents.segmentation.boxes import box_from_mask
from mars.agents.segmentation.prompt_source import TotalSegmentatorPromptSource
from mars.agents.segmentation.refiner import load_refiner
from mars.volumes import load_ct

CKPT, CT = os.environ.get("MARS_TEST_CKPT"), os.environ.get("MARS_TEST_CT")


@pytest.mark.gpu
@pytest.mark.skipif(not (CKPT and CT), reason="set MARS_TEST_CKPT and MARS_TEST_CT")
def test_end_to_end_on_one_scan():
    masks = TotalSegmentatorPromptSource().predict(CT)
    assert set(masks) <= {"liver", "spleen", "kidney_left", "stomach"}, "unexpected filenames"
    assert "liver" in masks
    ct, _ = load_ct(CT)
    box = box_from_mask(masks["liver"])
    mask, prob = load_refiner(CKPT, "cuda").refine(ct, box)
    assert prob.shape == ct.shape and 0.0 <= prob.min() and prob.max() <= 1.0
    z0, y0, x0, z1, y1, x1 = box
    inside = prob[z0:z1, y0:y1, x0:x1].mean()
    assert inside > prob.mean(), "refiner ignored the box"
    assert mask.sum() > 0
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/agents/segmentation/test_refiner.py -v` → 3 PASS
On the lab: `MARS_TEST_CKPT=... MARS_TEST_CT=... pytest -m gpu tests/gpu -v` → PASS. If TotalSegmentator's output filenames differ from the organ names, fix `prompt_source.py` here.

- [ ] **Step 7: Commit**

```bash
git add mars/agents tests
git commit -m "feat(seg): prompt source and Slide-SAM refiner interfaces"
```

---

### Task 3: Deploy rule

**Files:**
- Create: `mars/agents/segmentation/deploy.py`
- Test: `tests/agents/segmentation/test_deploy.py`

**Interfaces:**
- Consumes: `contracts.SegmentationOutput`, `volumes.load_mask`
- Produces: `deploy.apply_deploy(seg: SegmentationOutput, deploy: dict[str, str]) -> SegmentationOutput`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from mars.agents.segmentation.deploy import apply_deploy
from mars.contracts import OrganMask, SegmentationOutput
from mars.volumes import save_mask


def _seg(tmp_path, refined_exists=True):
    auto = np.zeros((10, 10, 10), bool); auto[2:4, 2:4, 2:4] = True      # 8 voxels
    refined = np.zeros((10, 10, 10), bool); refined[2:6, 2:6, 2:6] = True  # 64 voxels
    save_mask(tmp_path / "liver_auto.npy", auto)
    if refined_exists:
        save_mask(tmp_path / "liver_refined.npy", refined)
    om = OrganMask(prompt_source="totalseg", deployed="auto",
                   mask_path=str(tmp_path / "liver_auto.npy"),
                   auto_mask_path=str(tmp_path / "liver_auto.npy"),
                   refined_mask_path=str(tmp_path / "liver_refined.npy") if refined_exists else None)
    return SegmentationOutput(case_id="c", ct_path="x", spacing=(1.0, 1.0, 1.0),
                              shape=(10, 10, 10),
                              organs={"liver": om, "spleen": OrganMask(), "kidney_left": OrganMask(),
                                      "stomach": OrganMask()})


def test_refined_choice_switches_mask_and_volume(tmp_path):
    out = apply_deploy(_seg(tmp_path), {"liver": "refined"})
    assert out.organs["liver"].deployed == "refined"
    assert out.organs["liver"].mask_path.endswith("liver_refined.npy")
    assert out.organs["liver"].volume_ml == 64 / 1000


def test_missing_organ_in_deploy_defaults_to_auto(tmp_path):
    out = apply_deploy(_seg(tmp_path), {})
    assert out.organs["liver"].deployed == "auto"
    assert out.organs["liver"].volume_ml == 8 / 1000


def test_refined_without_refined_mask_falls_back_to_auto(tmp_path):
    out = apply_deploy(_seg(tmp_path, refined_exists=False), {"liver": "refined"})
    assert out.organs["liver"].deployed == "auto"


def test_absent_organs_are_untouched(tmp_path):
    out = apply_deploy(_seg(tmp_path), {"spleen": "refined"})
    assert out.organs["spleen"].deployed == "none" and out.organs["spleen"].mask_path is None
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/agents/segmentation/test_deploy.py -v` → FAIL

- [ ] **Step 3: Implement** `mars/agents/segmentation/deploy.py`

```python
from mars.contracts import SegmentationOutput
from mars.volumes import load_mask


def apply_deploy(seg: SegmentationOutput, deploy: dict[str, str]) -> SegmentationOutput:
    """Point each organ's mask_path at 'refined' or 'auto' per deploy.json (default auto)."""
    voxel_ml = seg.spacing[0] * seg.spacing[1] * seg.spacing[2] / 1000.0
    organs = {}
    for organ, om in seg.organs.items():
        if om.deployed == "none" or om.auto_mask_path is None:
            organs[organ] = om
            continue
        choice = deploy.get(organ, "auto")
        if choice == "refined" and om.refined_mask_path is None:
            choice = "auto"
        path = om.refined_mask_path if choice == "refined" else om.auto_mask_path
        organs[organ] = om.model_copy(update={
            "deployed": choice, "mask_path": path,
            "volume_ml": float(load_mask(path).sum() * voxel_ml),
        })
    return seg.model_copy(update={"organs": organs})
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/agents/segmentation/test_deploy.py -v` → 4 PASS

- [ ] **Step 5: Commit**

```bash
git add mars tests
git commit -m "feat(seg): deploy rule (refined vs auto)"
```

---

### Task 4: SegmentationAgent

**Files:**
- Create: `mars/agents/segmentation/agent.py`
- Test: `tests/agents/segmentation/test_agent.py`

**Interfaces:**
- Consumes: `PromptSource`, `Refiner`, `box_from_mask`, `boundary_entropy`, `apply_deploy`, `volumes.*`, `metrics.dice`
- Produces: `SegmentationAgent(prompt_source: PromptSource, refiner: Refiner, deploy: dict[str, str])` with `run(case_id: str, ct_path: str, out_dir) -> SegmentationOutput`. Files written: `{out_dir}/masks/{organ}_auto.npy` and `{organ}_refined.npy` (the latter only if refinement succeeded).

- [ ] **Step 1: Write the failing test**

```python
import nibabel as nib
import numpy as np
import pytest

from mars.agents.segmentation.agent import SegmentationAgent
from mars.volumes import load_mask


@pytest.fixture
def ct_path(tmp_path):
    nib.save(nib.Nifti1Image(np.zeros((16, 16, 16), np.float32), np.eye(4)), str(tmp_path / "ct.nii.gz"))
    return str(tmp_path / "ct.nii.gz")


class FakePrompt:
    name = "totalseg"

    def predict(self, ct_path):
        liver = np.zeros((16, 16, 16), bool); liver[4:10, 4:10, 4:10] = True   # 216 voxels
        return {"liver": liver, "spleen": np.zeros((16, 16, 16), bool)}      # spleen empty, others absent


class BoxRefiner:
    """Refined mask = the whole box (bigger than the auto cube)."""
    def refine(self, ct, box):
        z0, y0, x0, z1, y1, x1 = box
        prob = np.zeros(ct.shape, np.float32); prob[z0:z1, y0:y1, x0:x1] = 0.9
        return prob > 0.5, prob


class CrashRefiner:
    def refine(self, ct, box):
        raise RuntimeError("cuda oom")


def test_auto_deployed_by_default(tmp_path, ct_path):
    seg = SegmentationAgent(FakePrompt(), BoxRefiner(), {}).run("c1", ct_path, tmp_path / "out")
    liver = seg.organs["liver"]
    assert liver.deployed == "auto" and liver.volume_ml == pytest.approx(0.216)
    assert liver.box_prompt == [2, 2, 2, 12, 12, 12]
    assert 0.0 < liver.prompt_mask_dice < 1.0
    assert load_mask(liver.refined_mask_path).sum() == 1000


def test_refined_when_deploy_says_so(tmp_path, ct_path):
    seg = SegmentationAgent(FakePrompt(), BoxRefiner(), {"liver": "refined"}).run("c1", ct_path, tmp_path / "out")
    assert seg.organs["liver"].deployed == "refined"
    assert seg.organs["liver"].volume_ml == pytest.approx(1.0)


def test_absent_and_empty_organs_have_no_mask(tmp_path, ct_path):
    seg = SegmentationAgent(FakePrompt(), BoxRefiner(), {}).run("c1", ct_path, tmp_path / "out")
    for organ in ("spleen", "kidney_left", "stomach"):   # empty mask, absent, absent
        om = seg.organs[organ]
        assert om.deployed == "none" and om.mask_path is None and om.box_prompt is None
    assert not (tmp_path / "out" / "masks" / "spleen_auto.npy").exists()


def test_refiner_crash_keeps_auto_mask(tmp_path, ct_path):
    seg = SegmentationAgent(FakePrompt(), CrashRefiner(), {"liver": "refined"}).run("c1", ct_path, tmp_path / "out")
    liver = seg.organs["liver"]
    assert liver.deployed == "auto" and liver.refined_mask_path is None
    assert liver.prompt_mask_dice is None and liver.boundary_entropy is None
    assert liver.volume_ml == pytest.approx(0.216)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/agents/segmentation/test_agent.py -v` → FAIL

- [ ] **Step 3: Implement** `mars/agents/segmentation/agent.py`

```python
import logging
from pathlib import Path

from mars.agents.segmentation.boxes import box_from_mask
from mars.agents.segmentation.deploy import apply_deploy
from mars.agents.segmentation.refiner import boundary_entropy
from mars.contracts import ORGANS, OrganMask, SegmentationOutput
from mars.metrics import dice
from mars.volumes import load_ct, save_mask

log = logging.getLogger("mars.segmentation")


class SegmentationAgent:
    def __init__(self, prompt_source, refiner, deploy: dict[str, str]):
        self.prompt_source = prompt_source
        self.refiner = refiner
        self.deploy = deploy

    def run(self, case_id: str, ct_path: str, out_dir) -> SegmentationOutput:
        ct, spacing = load_ct(ct_path)
        auto_masks = self.prompt_source.predict(ct_path)
        masks_dir = Path(out_dir) / "masks"
        voxel_ml = spacing[0] * spacing[1] * spacing[2] / 1000.0
        organs: dict[str, OrganMask] = {}
        for organ in ORGANS:
            auto = auto_masks.get(organ)
            box = box_from_mask(auto) if auto is not None else None
            if box is None:
                organs[organ] = OrganMask(prompt_source=self.prompt_source.name)
                continue
            auto_path = masks_dir / f"{organ}_auto.npy"
            save_mask(auto_path, auto)
            om = OrganMask(
                prompt_source=self.prompt_source.name, deployed="auto",
                mask_path=str(auto_path), auto_mask_path=str(auto_path),
                box_prompt=box, volume_ml=float(auto.sum() * voxel_ml),
            )
            try:
                refined, prob = self.refiner.refine(ct, box)
                refined_path = masks_dir / f"{organ}_refined.npy"
                save_mask(refined_path, refined)
                om.refined_mask_path = str(refined_path)
                om.prompt_mask_dice = dice(auto, refined)
                om.boundary_entropy = boundary_entropy(prob)
            except Exception:  # noqa: BLE001 - keep the auto mask, other organs unaffected
                log.exception("refinement failed for %s in %s", organ, case_id)
            organs[organ] = om
        seg = SegmentationOutput(case_id=case_id, ct_path=str(ct_path), spacing=spacing,
                                 shape=tuple(ct.shape), organs=organs)
        return apply_deploy(seg, self.deploy)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/agents/segmentation -v` → all PASS

- [ ] **Step 5: Commit**

```bash
git add mars tests
git commit -m "feat(seg): SegmentationAgent with per-organ failure isolation"
```

---

### Task 5: Dev dataset config, ground truth, scoring

**Files:**
- Create: `mars/eval/__init__.py`, `mars/eval/dev_config.py`, `configs/dev_btcv.json` (template below)
- Test: `tests/eval/test_dev_config.py` (create `tests/eval/__init__.py`)

**Interfaces:**
- Produces:
  - `dev_config.DevCase(patient: str, fold: int | None, ct: str, gt: str)`, `dev_config.DevConfig(cases: list[DevCase], ckpts: dict[int, str], gt_format: "btcv" | "totalseg")` with `DevConfig.from_file(path)`
  - `dev_config.BTCV_LABELS = {"spleen": 1, "kidney_left": 3, "liver": 6, "stomach": 7}`
  - `dev_config.load_gt(case: DevCase, gt_format: str) -> dict[str, np.ndarray]`
  - `dev_config.score_case(seg: SegmentationOutput, gt: dict[str, np.ndarray], patient: str, fold: int | None) -> list[dict]` — one row per organ with keys `patient, fold, organ, dice_auto, dice_refined`

- [ ] **Step 1: Write the failing test**

```python
import nibabel as nib
import numpy as np

from mars.contracts import OrganMask, SegmentationOutput
from mars.eval.dev_config import BTCV_LABELS, DevCase, load_gt, score_case
from mars.volumes import save_mask


def test_case_key_defaults_to_patient_and_prefers_case_id():
    assert DevCase(patient="a", ct="x", gt="y").key == "a"
    assert DevCase(patient="a", ct="x", gt="y", case_id="a_f1").key == "a_f1"


def test_btcv_labels_to_organ_masks(tmp_path):
    lab = np.zeros((4, 4, 4), np.int16)      # x, y, z on disk
    lab[0, 0, 0] = 6; lab[1, 0, 0] = 7; lab[2, 0, 0] = 3; lab[3, 0, 0] = 1; lab[3, 3, 3] = 9
    nib.save(nib.Nifti1Image(lab, np.eye(4)), str(tmp_path / "gt.nii.gz"))
    gt = load_gt(DevCase(patient="p", ct="x", gt=str(tmp_path / "gt.nii.gz")), "btcv")
    assert set(gt) == set(BTCV_LABELS)
    assert gt["liver"].sum() == 1 and gt["liver"][0, 0, 0]
    assert gt["spleen"].dtype == bool


def test_totalseg_gt_reads_per_organ_files(tmp_path):
    for organ in BTCV_LABELS:
        a = np.zeros((4, 4, 4), np.uint8); a[0, 0, 0] = 1
        nib.save(nib.Nifti1Image(a, np.eye(4)), str(tmp_path / f"{organ}.nii.gz"))
    gt = load_gt(DevCase(patient="p", ct="x", gt=str(tmp_path)), "totalseg")
    assert all(m.sum() == 1 for m in gt.values())


def test_score_case_rows_and_missing_prompt_scores_zero(tmp_path):
    gt_liver = np.zeros((6, 6, 6), bool); gt_liver[1:4, 1:4, 1:4] = True
    save_mask(tmp_path / "a.npy", gt_liver)                                  # auto = perfect
    half = gt_liver.copy(); half[1:2] = False
    save_mask(tmp_path / "r.npy", half)                                      # refined = worse
    seg = SegmentationOutput(
        case_id="p", ct_path="x", spacing=(1.0, 1.0, 1.0), shape=(6, 6, 6),
        organs={"liver": OrganMask(deployed="auto", auto_mask_path=str(tmp_path / "a.npy"),
                                   refined_mask_path=str(tmp_path / "r.npy"), mask_path=str(tmp_path / "a.npy")),
                "spleen": OrganMask(), "kidney_left": OrganMask(), "stomach": OrganMask()})
    gt = {o: gt_liver for o in BTCV_LABELS}
    rows = {r["organ"]: r for r in score_case(seg, gt, "p", 2)}
    assert len(rows) == 4
    assert rows["liver"]["dice_auto"] == 1.0 and rows["liver"]["dice_refined"] < 1.0
    assert rows["spleen"]["dice_auto"] == 0.0 and rows["spleen"]["dice_refined"] == 0.0
    assert rows["liver"]["patient"] == "p" and rows["liver"]["fold"] == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/eval/test_dev_config.py -v` → FAIL

- [ ] **Step 3: Implement** `mars/eval/dev_config.py`

```python
import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from mars.contracts import ORGANS, SegmentationOutput
from mars.metrics import dice
from mars.volumes import load_ct, load_mask

BTCV_LABELS = {"spleen": 1, "kidney_left": 3, "liver": 6, "stomach": 7}


class DevCase(BaseModel):
    patient: str
    fold: int | None = None
    ct: str
    gt: str  # BTCV: label NIfTI path. TotalSegmentator: directory with <organ>.nii.gz
    case_id: str | None = None  # set when one patient appears once per fold checkpoint (external set)

    @property
    def key(self) -> str:
        """Directory/case name. Bootstraps still cluster on `patient`."""
        return self.case_id or self.patient


class DevConfig(BaseModel):
    cases: list[DevCase]
    ckpts: dict[int, str] = {}   # fold -> Slide-SAM checkpoint
    gt_format: Literal["btcv", "totalseg"] = "btcv"

    @classmethod
    def from_file(cls, path) -> "DevConfig":
        return cls.model_validate(json.loads(Path(path).read_text()))


def load_gt(case: DevCase, gt_format: str) -> dict[str, np.ndarray]:
    if gt_format == "btcv":
        labels = np.rint(load_ct(case.gt)[0]).astype(int)
        return {organ: labels == value for organ, value in BTCV_LABELS.items()}
    return {organ: load_ct(Path(case.gt) / f"{organ}.nii.gz")[0] > 0 for organ in ORGANS}


def score_case(seg: SegmentationOutput, gt: dict[str, np.ndarray],
               patient: str, fold: int | None) -> list[dict]:
    """Dice of the auto and refined masks per organ; a missing prompt scores 0 (a failure)."""
    rows = []
    for organ in ORGANS:
        om = seg.organs[organ]
        d_auto = dice(load_mask(om.auto_mask_path), gt[organ]) if om.auto_mask_path else 0.0
        if om.refined_mask_path:
            d_ref = dice(load_mask(om.refined_mask_path), gt[organ])
        else:
            d_ref = d_auto if om.auto_mask_path else 0.0
        rows.append({"patient": patient, "fold": fold, "organ": organ,
                     "dice_auto": d_auto, "dice_refined": d_ref})
    return rows
```
Note: if refinement crashed but auto exists, the refined score equals auto (no benefit, no penalty); this keeps the paired comparison conservative.

`configs/dev_btcv.json` (template; fill in with your real paths):
```json
{
  "gt_format": "btcv",
  "ckpts": {"0": "/path/fold0.pt", "1": "/path/fold1.pt", "2": "/path/fold2.pt", "3": "/path/fold3.pt", "4": "/path/fold4.pt"},
  "cases": [
    {"patient": "img0001", "fold": 0, "ct": "datasets/btcv/img/img0001.nii.gz", "gt": "datasets/btcv/label/label0001.nii.gz"}
  ]
}
```
Add one entry per BTCV patient with the fold in which that patient was held out.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/eval/test_dev_config.py -v` → 3 PASS

- [ ] **Step 5: Commit**

```bash
git add mars configs tests
git commit -m "feat(seg): dev dataset config, GT loaders, per-organ scoring"
```

---

### Task 6: Dev segmentation runner (offline, per fold)

**Files:**
- Create: `mars/eval/run_dev.py`, `scripts/make_dev_seg.py`
- Test: `tests/eval/test_run_dev.py`

**Interfaces:**
- Consumes: `DevConfig`, `load_gt`, `score_case`, `store`-style JSON writing
- Produces: `run_dev.run_dev_seg(cfg: DevConfig, make_agent, out_dir) -> pd.DataFrame` where `make_agent(fold: int | None) -> SegmentationAgent`. Writes `{out_dir}/seg/{patient}/seg.json` (a `SegmentationOutput`) and `{out_dir}/dev_scores.csv`.

- [ ] **Step 1: Write the failing test**

```python
import nibabel as nib
import numpy as np

from mars.eval.dev_config import DevCase, DevConfig
from mars.eval.run_dev import run_dev_seg
from mars.contracts import SegmentationOutput, OrganMask
from mars.volumes import save_mask


class FakeAgent:
    def run(self, case_id, ct_path, out_dir):
        m = np.zeros((4, 4, 4), bool); m[1:3, 1:3, 1:3] = True
        save_mask(out_dir / "masks" / "liver_auto.npy", m)
        om = OrganMask(deployed="auto", auto_mask_path=str(out_dir / "masks" / "liver_auto.npy"),
                       mask_path=str(out_dir / "masks" / "liver_auto.npy"))
        return SegmentationOutput(case_id=case_id, ct_path=ct_path, spacing=(1.0, 1.0, 1.0),
                                  shape=(4, 4, 4),
                                  organs={"liver": om, "spleen": OrganMask(), "kidney_left": OrganMask(), "stomach": OrganMask()})


def test_run_dev_seg_writes_seg_json_and_scores(tmp_path):
    lab = np.zeros((4, 4, 4), np.int16); lab[1:3, 1:3, 1:3] = 6
    nib.save(nib.Nifti1Image(lab, np.eye(4)), str(tmp_path / "gt.nii.gz"))
    cfg = DevConfig(cases=[DevCase(patient="p1", fold=0, ct="ct.nii.gz", gt=str(tmp_path / "gt.nii.gz"))])
    calls = []
    df = run_dev_seg(cfg, lambda fold: calls.append(fold) or FakeAgent(), tmp_path / "out")
    assert calls == [0]
    assert (tmp_path / "out" / "seg" / "p1" / "seg.json").exists()
    assert (tmp_path / "out" / "dev_scores.csv").exists()
    liver = df[df.organ == "liver"].iloc[0]
    assert liver.dice_auto == 1.0 and len(df) == 4
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/eval/test_run_dev.py -v` → FAIL

- [ ] **Step 3: Implement** `mars/eval/run_dev.py`

```python
from pathlib import Path

import pandas as pd

from mars.eval.dev_config import DevConfig, load_gt, score_case
from mars.volumes import write_json_atomic


def run_dev_seg(cfg: DevConfig, make_agent, out_dir) -> pd.DataFrame:
    """Segment every dev case with the agent built for its fold; score against GT."""
    out_dir = Path(out_dir)
    rows = []
    agents: dict = {}
    for case in cfg.cases:
        if case.fold not in agents:
            agents[case.fold] = make_agent(case.fold)
        case_dir = out_dir / "seg" / case.key
        seg = agents[case.fold].run(case.key, case.ct, case_dir)
        write_json_atomic(case_dir / "seg.json", seg.model_dump_json(indent=2))
        rows += score_case(seg, load_gt(case, cfg.gt_format), case.patient, case.fold)
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "dev_scores.csv", index=False)
    return df
```
Note: the agent here is built with `deploy={}` so both masks are saved; the deploy decision is made afterwards in Task 7.

`scripts/make_dev_seg.py`:
```python
import argparse

from mars.agents.segmentation.agent import SegmentationAgent
from mars.agents.segmentation.prompt_source import TotalSegmentatorPromptSource
from mars.agents.segmentation.refiner import load_refiner
from mars.eval.dev_config import DevConfig
from mars.eval.run_dev import run_dev_seg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cfg = DevConfig.from_file(a.config)
    prompt = TotalSegmentatorPromptSource()
    make_agent = lambda fold: SegmentationAgent(prompt, load_refiner(cfg.ckpts[fold], "cuda"), {})
    df = run_dev_seg(cfg, make_agent, a.out)
    print(df.groupby("organ")[["dice_auto", "dice_refined"]].mean().round(3))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/eval/test_run_dev.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add mars scripts tests
git commit -m "feat(seg): offline dev segmentation runner"
```

---

### Task 7: Refined-vs-auto decision

**Files:**
- Create: `mars/eval/deploy_decision.py`, `scripts/compare_refined_vs_auto.py`
- Test: `tests/eval/test_deploy_decision.py`

**Interfaces:**
- Consumes: `metrics.cluster_bootstrap_ci`
- Produces: `deploy_decision.decide_deploy(scores: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> tuple[dict[str, str], pd.DataFrame]` — deploy map (`"refined"` iff the 95% CI of mean(dice_refined − dice_auto) has lower bound > 0) and a table with columns `organ, mean_auto, mean_refined, diff, lo, hi, deploy`.

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd

from mars.eval.deploy_decision import decide_deploy


def _scores():
    rows = []
    for i in range(10):
        p = f"p{i}"
        rows += [
            {"patient": p, "organ": "liver", "dice_auto": 0.90, "dice_refined": 0.95},    # clearly better
            {"patient": p, "organ": "spleen", "dice_auto": 0.90, "dice_refined": 0.90},   # no difference
            {"patient": p, "organ": "stomach", "dice_auto": 0.80, "dice_refined": 0.70},  # worse
            {"patient": p, "organ": "kidney_left", "dice_auto": 0.90 - 0.01 * i,
             "dice_refined": 0.90 - 0.01 * i + (0.02 if i % 2 else -0.02)},               # noisy, inconclusive
        ]
    return pd.DataFrame(rows)


def test_refined_only_when_ci_excludes_zero():
    deploy, table = decide_deploy(_scores(), n_boot=300)
    assert deploy == {"liver": "refined", "spleen": "auto", "stomach": "auto", "kidney_left": "auto"}
    row = table.set_index("organ").loc["liver"]
    assert row["diff"] > 0 and row["lo"] > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/eval/test_deploy_decision.py -v` → FAIL

- [ ] **Step 3: Implement** `mars/eval/deploy_decision.py`

```python
import pandas as pd

from mars.contracts import ORGANS
from mars.metrics import cluster_bootstrap_ci


def decide_deploy(scores: pd.DataFrame, n_boot: int = 2000, seed: int = 0):
    deploy, rows = {}, []
    for organ in ORGANS:
        g = scores[scores.organ == organ]
        diff, lo, hi = cluster_bootstrap_ci(
            g, lambda d: (d.dice_refined - d.dice_auto).mean(), n_boot=n_boot, seed=seed)
        deploy[organ] = "refined" if lo > 0 else "auto"
        rows.append({"organ": organ, "mean_auto": g.dice_auto.mean(),
                     "mean_refined": g.dice_refined.mean(), "diff": diff,
                     "lo": lo, "hi": hi, "deploy": deploy[organ]})
    return deploy, pd.DataFrame(rows)
```

`scripts/compare_refined_vs_auto.py`:
```python
import argparse
import json
from pathlib import Path

import pandas as pd

from mars.eval.deploy_decision import decide_deploy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--out", default="configs/deploy.json")
    a = p.parse_args()
    deploy, table = decide_deploy(pd.read_csv(a.scores))
    print(table.round(3).to_string(index=False))
    Path(a.out).write_text(json.dumps(deploy, indent=2))
    print(f"\nwrote {a.out}. Organs where refinement is not clearly better use the prompt-source mask.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/eval -v` → all PASS

- [ ] **Step 5: Commit**

```bash
git add mars scripts tests
git commit -m "feat(seg): paired-bootstrap refined-vs-auto deploy decision"
```

---

### Task 8: NSD secondary metric and oracle-prompt ablation source

**Files:**
- Modify: `mars/metrics.py` (add `nsd`), `mars/eval/dev_config.py` (`score_case`), `mars/agents/segmentation/prompt_source.py` (add `OraclePromptSource`)
- Test: `tests/test_nsd.py`, `tests/eval/test_dev_config.py` (extend), `tests/agents/segmentation/test_oracle_source.py`

**Interfaces:**
- Produces:
  - `metrics.nsd(a, b, spacing, tol_mm: float = 2.0) -> float` — normalised surface Dice: fraction of surface voxels (of both masks) within `tol_mm` of the other mask's surface. Both empty → 1.0; one empty → 0.0. The tolerance is **2 mm for every organ**, fixed before any test result.
  - `score_case` rows gain `nsd_auto` and `nsd_refined` (same missing-prompt rule: 0.0).
  - `OraclePromptSource(gt_loader)` with `name = "oracle"`; `predict(ct_path)` returns `gt_loader(ct_path)` (organ → bool mask). Used only for the oracle-vs-auto ablation.

- [ ] **Step 1: Write the failing tests**

`tests/test_nsd.py`:
```python
import numpy as np

from mars.metrics import nsd


def _cube(lo, hi, n=20):
    m = np.zeros((n, n, n), bool); m[lo:hi, lo:hi, lo:hi] = True
    return m


def test_identical_masks_score_one_and_far_apart_masks_zero():
    assert nsd(_cube(2, 6), _cube(2, 6), (1.0, 1.0, 1.0)) == 1.0
    assert nsd(_cube(2, 6), _cube(12, 16), (1.0, 1.0, 1.0)) == 0.0


def test_small_shift_within_tolerance_scores_one_and_spacing_matters():
    a = _cube(2, 6)
    b = np.roll(a, 1, axis=0)                                  # shifted by one voxel
    assert nsd(a, b, (1.0, 1.0, 1.0), tol_mm=2.0) == 1.0
    assert nsd(a, b, (5.0, 1.0, 1.0), tol_mm=2.0) < 1.0        # same shift is 5 mm along z


def test_empty_cases():
    z = np.zeros((4, 4, 4), bool)
    assert nsd(z, z, (1.0, 1.0, 1.0)) == 1.0
    assert nsd(z, _cube(1, 3, 4), (1.0, 1.0, 1.0)) == 0.0
```

`tests/agents/segmentation/test_oracle_source.py`:
```python
import numpy as np

from mars.agents.segmentation.prompt_source import OraclePromptSource


def test_oracle_returns_ground_truth_masks():
    gt = {"liver": np.ones((2, 2, 2), bool)}
    src = OraclePromptSource(lambda ct_path: gt)
    assert src.name == "oracle" and src.predict("x") is gt
```

Extend `tests/eval/test_dev_config.py::test_score_case_rows_and_missing_prompt_scores_zero` with:
```python
    assert rows["liver"]["nsd_auto"] == 1.0 and 0.0 <= rows["liver"]["nsd_refined"] <= 1.0
    assert rows["spleen"]["nsd_auto"] == 0.0 and rows["spleen"]["nsd_refined"] == 0.0
```
(The fixture's refined mask is only one voxel off, well inside the 2 mm tolerance, so `nsd_refined` is not asserted to be below 1; NSD behaviour itself is pinned by `tests/test_nsd.py`.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_nsd.py tests/agents/segmentation/test_oracle_source.py -v` → FAIL

- [ ] **Step 3: Implement**

Append to `mars/metrics.py`:
```python
from scipy import ndimage


def nsd(a: np.ndarray, b: np.ndarray, spacing, tol_mm: float = 2.0) -> float:
    a, b = a.astype(bool), b.astype(bool)
    if not a.any() and not b.any():
        return 1.0
    if not a.any() or not b.any():
        return 0.0
    surf_a = a & ~ndimage.binary_erosion(a)
    surf_b = b & ~ndimage.binary_erosion(b)
    dist_to_b = ndimage.distance_transform_edt(~surf_b, sampling=spacing)
    dist_to_a = ndimage.distance_transform_edt(~surf_a, sampling=spacing)
    close = (dist_to_b[surf_a] <= tol_mm).sum() + (dist_to_a[surf_b] <= tol_mm).sum()
    return float(close / (surf_a.sum() + surf_b.sum()))
```

Replace `score_case` in `mars/eval/dev_config.py` with:
```python
def score_case(seg: SegmentationOutput, gt: dict[str, np.ndarray],
               patient: str, fold: int | None) -> list[dict]:
    """Dice and NSD (2 mm) of the auto and refined masks; a missing prompt scores 0 (a failure)."""
    rows = []
    for organ in ORGANS:
        om = seg.organs[organ]
        auto = load_mask(om.auto_mask_path) if om.auto_mask_path else None
        refined = load_mask(om.refined_mask_path) if om.refined_mask_path else auto  # crashed refiner: no benefit, no penalty
        def both(m):
            if m is None:
                return 0.0, 0.0
            return dice(m, gt[organ]), nsd(m, gt[organ], seg.spacing)
        (d_auto, n_auto), (d_ref, n_ref) = both(auto), both(refined)
        rows.append({"patient": patient, "fold": fold, "organ": organ,
                     "dice_auto": d_auto, "dice_refined": d_ref,
                     "nsd_auto": n_auto, "nsd_refined": n_ref})
    return rows
```
and change the import to `from mars.metrics import dice, nsd`.

Append to `mars/agents/segmentation/prompt_source.py`:
```python
class OraclePromptSource:
    """Ablation only: ground-truth masks as the 'automatic' prompt (reproduces the oracle-prompt Dice)."""
    name = "oracle"

    def __init__(self, gt_loader):
        self.gt_loader = gt_loader   # callable(ct_path) -> dict[organ, bool zyx mask]

    def predict(self, ct_path: str) -> dict[str, np.ndarray]:
        return self.gt_loader(ct_path)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_nsd.py tests/agents/segmentation tests/eval -v` → all PASS

- [ ] **Step 5: Commit**

```bash
git add mars tests
git commit -m "feat(seg): NSD (2 mm) secondary metric and oracle prompt source for the ablation"
```

---

After this plan: run Runbook steps A1–A2 on the lab; commit the resulting `configs/deploy.json` and report the table in your write-up (including organs where Slide-SAM did not beat TotalSegmentator).
