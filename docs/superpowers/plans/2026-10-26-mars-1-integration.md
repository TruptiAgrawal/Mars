# MARS Plan 1 — Foundations and Pipeline (how everything connects)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared contract, helpers, pipeline, store, API and wiring that connect the four agents.

**Architecture:** One pydantic contract module (`mars/contracts.py`) is the only thing agents share. `pipeline.run_case` calls the four agents in order and converts any agent exception into FLAG results, so a case always yields a report. FastAPI runs cases in a background thread and serves JSON and PNG slices.

**Tech Stack:** Python 3.10+, pydantic v2, numpy, scipy, nibabel, Pillow, pandas, scikit-learn, FastAPI, pytest.

**Spec:** `docs/superpowers/specs/2026-10-26-mars-uncertainty-design.md`, `docs/superpowers/specs/2026-10-26-mars-backend-design.md`

**Build order:** Tasks 1–2 first (every other plan imports them). Tasks 3–4 can run in parallel with Plans 2–5 (they use stub agents). Task 5 runs last, after Plans 2–5 exist.

## Global Constraints

- Organs are exactly `liver`, `spleen`, `kidney_left`, `stomach` (TotalSegmentator v2 names).
- Array axis order is `(z, y, x)`; spacing is millimetres `(sz, sy, sx)`.
- Any agent failure degrades to FLAG with an `AGENT_ERROR` reason; a case always yields a report.
- Decisions are per organ, never per scan.
- `p_ok` (outline reliability) and `relevance` (clinical consistency) stay separate; never combined.
- ACCEPT never says an organ is normal/healthy.
- Splits and bootstraps are by patient.
- Python 3.10+; no new runtime dependencies beyond those in `pyproject.toml` (Task 1).

## Review Focus

1. CT file missing/unreadable → case still returns a report, all organs FLAG with `AGENT_ERROR`, no HTTP 500 (Task 3 test).
2. Organ has no mask → slice endpoint returns a plain CT slice, not an error (Task 4 test).
3. No clinical metadata → `provenance="none"` flows through; report carries a banner (Task 5 golden test).
4. `case_id` or `ct_path` with path traversal (`../`, absolute path outside data root) → HTTP 400 (Task 4 test).
5. Same case submitted twice while running → one job, second call returns `running`; `result.json` never half-written (Tasks 2 and 4 tests).

---

### Task 1: Project scaffold and contract

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `mars/__init__.py`, `mars/contracts.py`, `tests/__init__.py`, `tests/factories.py`, `tests/test_contracts.py`

**Interfaces:**
- Produces (used by every plan): everything in `mars/contracts.py` below, and `tests/factories.py: make_organ_mask, make_seg, make_clin, make_unc, make_report`.

- [ ] **Step 1: Create scaffold files**

`pyproject.toml`:
```toml
[project]
name = "mars"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "numpy", "scipy", "pandas", "scikit-learn", "nibabel", "pillow",
  "pydantic>=2", "fastapi", "uvicorn",
]

[project.optional-dependencies]
dev = ["pytest", "httpx"]
llm = ["anthropic"]
gpu = ["torch", "TotalSegmentator"]

[tool.setuptools.packages.find]
include = ["mars*"]

[tool.pytest.ini_options]
pythonpath = ["."]
addopts = "--import-mode=importlib -m 'not gpu'"
markers = ["gpu: needs GPU and model weights; run manually on the lab"]
```

`.gitignore`:
```
__pycache__/
*.pyc
.venv/
outputs/
datasets/
frontend/node_modules/
frontend/dist/
```

Create empty `mars/__init__.py` and `tests/__init__.py`.

- [ ] **Step 2: Write the failing test** — `tests/test_contracts.py`

```python
import pytest
from pydantic import ValidationError

from mars.contracts import (
    CaseResult, ORGANS, REQUIRED_EVIDENCE, Reason, ReasonCode,
)
from tests.factories import make_clin, make_report, make_seg, make_unc


def test_severity_must_be_between_0_and_1():
    with pytest.raises(ValidationError):
        Reason(id="x", code="JITTER_HIGH", organ="liver", severity=1.5)


def test_unknown_organ_rejected():
    with pytest.raises(ValidationError):
        Reason(id="x", code="JITTER_HIGH", organ="lung", severity=0.5)


def test_every_reason_code_has_required_evidence_entry():
    from typing import get_args
    assert set(get_args(ReasonCode)) == set(REQUIRED_EVIDENCE)


def test_case_result_round_trips_through_json():
    result = CaseResult(
        segmentation=make_seg(), clinical=make_clin(),
        uncertainty=make_unc(), report=make_report(),
    )
    again = CaseResult.model_validate_json(result.model_dump_json())
    assert again == result
    assert set(again.uncertainty.organs) == set(ORGANS)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pip install -e ".[dev]" && pytest tests/test_contracts.py -v`
Expected: FAIL (`ModuleNotFoundError: mars.contracts`)

- [ ] **Step 4: Write `mars/contracts.py`**

```python
"""The JSON contract. Every agent, the API and the frontend types come from here."""
from typing import Literal

from pydantic import BaseModel, Field

ORGANS = ("liver", "spleen", "kidney_left", "stomach")
Organ = Literal["liver", "spleen", "kidney_left", "stomach"]
ORGAN_LABELS = {
    "liver": "Liver", "spleen": "Spleen",
    "kidney_left": "Left kidney", "stomach": "Stomach",
}

ReasonCode = Literal[
    "JITTER_HIGH", "DISAGREE_AUTOMODEL", "HIGH_BOUNDARY_ENTROPY", "FRAGMENTED",
    "LOW_RELIABILITY", "VOLUME_IMPLAUSIBLE", "PATHOLOGY_LOC_MISMATCH",
    "FOV_TRUNCATED", "NO_PROMPT", "NO_CLINICAL_CONTEXT", "NOT_CALIBRATABLE",
    "AGENT_ERROR",
]

# Evidence keys each reason code must carry (the report templates print them).
REQUIRED_EVIDENCE: dict[str, tuple[str, ...]] = {
    "JITTER_HIGH": ("jitter_dice", "expected_min"),
    "DISAGREE_AUTOMODEL": ("prompt_mask_dice", "expected_min"),
    "HIGH_BOUNDARY_ENTROPY": ("boundary_entropy", "expected_max"),
    "FRAGMENTED": ("n_components",),
    "LOW_RELIABILITY": ("p_ok", "tau"),
    "VOLUME_IMPLAUSIBLE": ("volume_ml", "z"),
    "PATHOLOGY_LOC_MISMATCH": ("volume_ml", "n_components"),
    "FOV_TRUNCATED": (),
    "NO_PROMPT": (),
    "NO_CLINICAL_CONTEXT": (),
    "NOT_CALIBRATABLE": (),
    "AGENT_ERROR": (),
}


class Reason(BaseModel):
    id: str  # "<source>:<organ>:<code>", e.g. "unc:liver:JITTER_HIGH"
    code: ReasonCode
    organ: Organ
    severity: float = Field(ge=0.0, le=1.0)
    evidence: dict[str, float | str] = Field(default_factory=dict)
    slices: list[int] = Field(default_factory=list)


class OrganMask(BaseModel):
    prompt_source: Literal["totalseg", "nnunet", "oracle", "none"] = "none"  # "oracle" = ablation only
    deployed: Literal["refined", "auto", "none"] = "none"
    mask_path: str | None = None          # the deployed mask (.npy, bool, zyx)
    auto_mask_path: str | None = None     # prompt-source mask
    refined_mask_path: str | None = None  # Slide-SAM mask
    box_prompt: list[int] | None = None   # [z0, y0, x0, z1, y1, x1], end-exclusive
    prompt_mask_dice: float | None = None
    boundary_entropy: float | None = None
    volume_ml: float | None = None


class SegmentationOutput(BaseModel):
    case_id: str
    ct_path: str
    spacing: tuple[float, float, float]
    shape: tuple[int, int, int]
    organs: dict[Organ, OrganMask]


class OrganClinical(BaseModel):
    relevance: float | None = Field(default=None, ge=0.0, le=1.0)
    findings: list[Reason] = Field(default_factory=list)


class ClinicalCheck(BaseModel):
    metadata_provenance: Literal["real", "synthetic", "none"]
    organs: dict[Organ, OrganClinical]


class OrganUncertainty(BaseModel):
    p_ok: float | None = None
    tau: float | None = None
    d_star: float | None = None
    status: Literal["ACCEPT", "FLAG"]
    signals: dict[str, float] = Field(default_factory=dict)
    reasons: list[Reason] = Field(default_factory=list)


class UncertaintyReport(BaseModel):
    calibration_version: str
    targets_hash: str
    organs: dict[Organ, OrganUncertainty]


class ReportSentence(BaseModel):
    text: str
    claim: str        # a ReasonCode, "ACCEPT", or "SUMMARY"
    cites: list[str]  # "reason:<id>" or "field:<organ>.<name>"


class OrganSection(BaseModel):
    organ: Organ
    status: Literal["ACCEPT", "FLAG"]
    sentences: list[ReportSentence]
    used_llm: bool = False


class Report(BaseModel):
    case_id: str
    summary: str
    banner: str | None = None
    disclaimer: str = "Draft for radiologist review; not a diagnosis."
    sections: list[OrganSection]


class CaseResult(BaseModel):
    segmentation: SegmentationOutput
    clinical: ClinicalCheck
    uncertainty: UncertaintyReport
    report: Report
```

- [ ] **Step 5: Write `tests/factories.py`**

```python
"""Small builders so every plan's tests use the same sample objects."""
from mars.contracts import (
    ClinicalCheck, ORGANS, OrganClinical, OrganMask, OrganSection,
    OrganUncertainty, Reason, Report, ReportSentence, SegmentationOutput,
    UncertaintyReport,
)


def make_organ_mask(deployed="refined", volume_ml=1500.0) -> OrganMask:
    return OrganMask(
        prompt_source="totalseg", deployed=deployed, mask_path="m.npy",
        auto_mask_path="a.npy", refined_mask_path="r.npy",
        box_prompt=[0, 0, 0, 10, 10, 10], prompt_mask_dice=0.95,
        boundary_entropy=0.1, volume_ml=volume_ml,
    )


def make_seg(case_id="c1") -> SegmentationOutput:
    return SegmentationOutput(
        case_id=case_id, ct_path="ct.nii.gz", spacing=(1.0, 1.0, 1.0),
        shape=(20, 64, 64), organs={o: make_organ_mask() for o in ORGANS},
    )


def make_clin(provenance="real") -> ClinicalCheck:
    return ClinicalCheck(
        metadata_provenance=provenance,
        organs={o: OrganClinical(relevance=1.0, findings=[]) for o in ORGANS},
    )


def make_unc(flag=("stomach",)) -> UncertaintyReport:
    organs = {}
    for o in ORGANS:
        flagged = o in flag
        reasons = []
        if flagged:
            reasons = [
                Reason(id=f"unc:{o}:LOW_RELIABILITY", code="LOW_RELIABILITY",
                       organ=o, severity=0.8, evidence={"p_ok": 0.42, "tau": 0.90}),
                Reason(id=f"unc:{o}:JITTER_HIGH", code="JITTER_HIGH", organ=o,
                       severity=0.7, evidence={"jitter_dice": 0.71, "expected_min": 0.88}),
            ]
        organs[o] = OrganUncertainty(
            p_ok=0.42 if flagged else 0.96, tau=0.90, d_star=0.90,
            status="FLAG" if flagged else "ACCEPT",
            signals={"jitter_dice": 0.71 if flagged else 0.97}, reasons=reasons,
        )
    return UncertaintyReport(calibration_version="test-v1", targets_hash="abc123", organs=organs)


def make_report(case_id="c1") -> Report:
    sections = [
        OrganSection(
            organ=o, status="ACCEPT",
            sentences=[ReportSentence(text="ok", claim="ACCEPT", cites=[f"field:{o}.p_ok"])],
        )
        for o in ORGANS
    ]
    return Report(case_id=case_id, summary="4/4 organs auto-accepted.", sections=sections)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_contracts.py -v`
Expected: 4 PASS

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore mars tests
git commit -m "feat: project scaffold and JSON contract"
```

---

### Task 2: Shared helpers (config, volumes, metrics, mask stats)

**Files:**
- Create: `mars/config.py`, `mars/volumes.py`, `mars/metrics.py`, `mars/maskstats.py`
- Test: `tests/test_helpers.py`

**Interfaces:**
- Produces:
  - `config.load_targets(path: Path | None = None) -> dict[str, float]`, `config.targets_hash(targets: dict[str, float]) -> str`, `config.CONFIG_DIR: Path`
  - `volumes.load_ct(path) -> tuple[np.ndarray, tuple[float, float, float]]` (zyx float32, spacing zyx mm), `volumes.load_mask(path) -> np.ndarray[bool]`, `volumes.save_mask(path, mask) -> None`, `volumes.write_json_atomic(path, text) -> None`
  - `metrics.dice(a, b) -> float`, `metrics.cluster_bootstrap_ci(df, stat_fn, n_boot=1000, seed=0, level=0.95) -> tuple[float, float, float]` (df needs a `patient` column)
  - `maskstats.OrganStats(volume_ml, n_components, touches_z_edge)`, `maskstats.mask_stats(mask, spacing, min_frac=0.01) -> OrganStats`

- [ ] **Step 1: Write the failing tests** — `tests/test_helpers.py`

```python
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from mars.config import load_targets, targets_hash
from mars.maskstats import mask_stats
from mars.metrics import cluster_bootstrap_ci, dice
from mars.volumes import load_ct, load_mask, save_mask, write_json_atomic


def test_load_targets_requires_all_organs(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"liver": 0.9}))
    with pytest.raises(ValueError, match="spleen"):
        load_targets(p)


def test_load_targets_missing_file_has_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="feasibility"):
        load_targets(tmp_path / "nope.json")


def test_targets_hash_is_stable_and_order_independent():
    a = {"liver": 0.9, "spleen": 0.9, "kidney_left": 0.9, "stomach": 0.8}
    b = dict(reversed(list(a.items())))
    assert targets_hash(a) == targets_hash(b)
    assert targets_hash(a) != targets_hash({**a, "stomach": 0.7})


def test_load_ct_returns_zyx_and_spacing(tmp_path):
    arr = np.zeros((8, 6, 4), dtype=np.float32)  # x, y, z
    nib.save(nib.Nifti1Image(arr, np.diag([2.0, 1.0, 0.5, 1.0])), str(tmp_path / "ct.nii.gz"))
    vol, spacing = load_ct(tmp_path / "ct.nii.gz")
    assert vol.shape == (4, 6, 8)
    assert spacing == (0.5, 1.0, 2.0)


def test_mask_round_trip_and_atomic_json(tmp_path):
    m = np.zeros((3, 3, 3), bool); m[1, 1, 1] = True
    save_mask(tmp_path / "m.npy", m)
    assert load_mask(tmp_path / "m.npy").dtype == bool
    write_json_atomic(tmp_path / "r.json", "{}")
    assert (tmp_path / "r.json").read_text() == "{}"
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_dice_values():
    a = np.zeros((4, 4, 4), bool); a[:2] = True
    assert dice(a, a) == 1.0
    assert dice(a, ~a) == 0.0
    assert dice(np.zeros_like(a), np.zeros_like(a)) == 1.0  # both empty


def test_cluster_bootstrap_constant_has_zero_width():
    df = pd.DataFrame({"patient": ["a", "b", "c"], "v": [5.0, 5.0, 5.0]})
    point, lo, hi = cluster_bootstrap_ci(df, lambda d: d.v.mean(), n_boot=50)
    assert point == lo == hi == 5.0


def test_cluster_bootstrap_resamples_patients_not_rows():
    df = pd.DataFrame({"patient": ["a", "a", "b", "b"], "v": [1.0, 1.0, 3.0, 3.0]})
    point, lo, hi = cluster_bootstrap_ci(df, lambda d: d.v.mean(), n_boot=200)
    assert point == 2.0 and 1.0 <= lo <= 2.0 <= hi <= 3.0


def test_mask_stats_volume_components_and_edges():
    m = np.zeros((10, 10, 10), bool)
    m[2:7, 2:7, 2:7] = True                      # 125 voxels
    s = mask_stats(m, (2.0, 1.0, 1.0))
    assert s.volume_ml == pytest.approx(0.25)    # 125 * 2*1*1 mm3 / 1000
    assert s.n_components == 1 and s.touches_z_edge is False
    m[9, 0, 0] = True                            # speck < 1% of mask -> ignored, but touches z edge
    s2 = mask_stats(m, (2.0, 1.0, 1.0))
    assert s2.n_components == 1 and s2.touches_z_edge is True
    m[0:2, 8:10, 8:10] = True                    # second real piece
    assert mask_stats(m, (1.0, 1.0, 1.0)).n_components == 2


def test_mask_stats_empty():
    s = mask_stats(np.zeros((3, 3, 3), bool), (1.0, 1.0, 1.0))
    assert (s.volume_ml, s.n_components, s.touches_z_edge) == (0.0, 0, False)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_helpers.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement**

`mars/config.py`:
```python
import hashlib
import json
from pathlib import Path

from mars.contracts import ORGANS

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_targets(path: Path | None = None) -> dict[str, float]:
    path = Path(path) if path else CONFIG_DIR / "targets.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Create it after the feasibility check "
            "(Plan 4, Task 6) and commit it before touching the external set."
        )
    raw = json.loads(path.read_text())
    missing = [o for o in ORGANS if o not in raw]
    if missing:
        raise ValueError(f"targets file lacks organs: {', '.join(missing)}")
    targets = {o: float(raw[o]) for o in ORGANS}
    for organ, value in targets.items():
        if not 0.0 < value <= 1.0:
            raise ValueError(f"target for {organ} must be in (0, 1], got {value}")
    return targets


def targets_hash(targets: dict[str, float]) -> str:
    blob = json.dumps({k: float(v) for k, v in targets.items()}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
```

`mars/volumes.py`:
```python
import os
from pathlib import Path

import nibabel as nib
import numpy as np


def load_ct(path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """NIfTI -> (volume[z, y, x] float32, spacing (sz, sy, sx) in mm)."""
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    zx = img.header.get_zooms()[:3]
    return np.transpose(data, (2, 1, 0)), (float(zx[2]), float(zx[1]), float(zx[0]))


def load_mask(path) -> np.ndarray:
    return np.load(str(path)).astype(bool)


def save_mask(path, mask: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), mask.astype(np.uint8))


def write_json_atomic(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
```
Note: `np.save` appends `.npy` only if missing, so callers must pass names ending in `.npy`.

`mars/metrics.py`:
```python
import numpy as np
import pandas as pd


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    total = a.sum() + b.sum()
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / total)


def cluster_bootstrap_ci(df: pd.DataFrame, stat_fn, n_boot: int = 1000,
                         seed: int = 0, level: float = 0.95):
    """Resample whole patients (df['patient']), return (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby("patient")}
    keys = list(groups)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), size=len(keys))
        stats.append(stat_fn(pd.concat([groups[keys[i]] for i in pick])))
    lo, hi = np.nanpercentile(stats, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(stat_fn(df)), float(lo), float(hi)
```

`mars/maskstats.py`:
```python
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class OrganStats:
    volume_ml: float
    n_components: int
    touches_z_edge: bool


def mask_stats(mask: np.ndarray, spacing, min_frac: float = 0.01) -> OrganStats:
    """Volume, number of real pieces (>= min_frac of the mask), z-edge contact."""
    mask = mask.astype(bool)
    total = int(mask.sum())
    if total == 0:
        return OrganStats(0.0, 0, False)
    labels, _ = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())[1:]
    n_components = int((sizes >= min_frac * total).sum())
    voxel_ml = spacing[0] * spacing[1] * spacing[2] / 1000.0
    zs = np.where(mask.any(axis=(1, 2)))[0]
    touches = bool(zs.min() == 0 or zs.max() == mask.shape[0] - 1)
    return OrganStats(float(total * voxel_ml), n_components, touches)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_helpers.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add mars tests
git commit -m "feat: config, volume IO, dice, bootstrap, mask stats"
```

---

### Task 3: Fallbacks and pipeline

**Files:**
- Create: `mars/fallbacks.py`, `mars/pipeline.py`, `mars/store.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `mars.contracts.*`, `mars.volumes.write_json_atomic`
- Produces:
  - `fallbacks.reason(source: str, organ: str, code: str, severity: float = 1.0, **evidence) -> Reason`
  - `fallbacks.empty_segmentation(case_id, ct_path) -> SegmentationOutput`
  - `fallbacks.empty_clinical(code: str, detail: str) -> ClinicalCheck`
  - `fallbacks.flagged_uncertainty(code: str, detail: str) -> UncertaintyReport`
  - `fallbacks.minimal_report(seg, clin, unc) -> Report`
  - `pipeline.run_case(case_id, ct_path, meta, provenance, out_dir, *, seg, clin, unc, rep) -> CaseResult`
  - Agent call signatures the pipeline requires: `seg.run(case_id, ct_path, out_dir) -> SegmentationOutput`; `clin.run(seg, meta: dict | None, provenance: str) -> ClinicalCheck`; `unc.run(seg) -> UncertaintyReport`; `rep.run(seg, clin, unc) -> Report`
  - `store.save_result(out_dir, result) -> None`, `store.load_result(out_dir) -> CaseResult`

- [ ] **Step 1: Write the failing tests** — `tests/test_pipeline.py`

```python
import pytest

from mars.contracts import ORGANS
from mars.fallbacks import minimal_report
from mars.pipeline import run_case
from mars.store import load_result, save_result
from tests.factories import make_clin, make_seg, make_unc


class Boom:
    def run(self, *a, **k):
        raise RuntimeError("boom")


class StubSeg:
    def run(self, case_id, ct_path, out_dir):
        return make_seg(case_id)


class StubClin:
    def run(self, seg, meta, provenance):
        return make_clin(provenance)


class StubUnc:
    def run(self, seg):
        return make_unc(flag=())


class StubRep:
    def run(self, seg, clin, unc):
        return minimal_report(seg, clin, unc)


def _run(tmp_path, **over):
    agents = dict(seg=StubSeg(), clin=StubClin(), unc=StubUnc(), rep=StubRep())
    agents.update(over)
    return run_case("c1", "ct.nii.gz", None, "none", tmp_path, **agents)


def test_happy_path_returns_all_parts(tmp_path):
    r = _run(tmp_path)
    assert all(r.uncertainty.organs[o].status == "ACCEPT" for o in ORGANS)
    assert len(r.report.sections) == 4


def test_segmentation_crash_flags_everything_but_still_reports(tmp_path):
    r = _run(tmp_path, seg=Boom())
    for o in ORGANS:
        assert r.uncertainty.organs[o].status == "FLAG"
        assert r.uncertainty.organs[o].reasons[0].code == "AGENT_ERROR"
        assert r.segmentation.organs[o].deployed == "none"
    assert r.report.sections


def test_uncertainty_crash_flags_everything(tmp_path):
    r = _run(tmp_path, unc=Boom())
    assert all(r.uncertainty.organs[o].status == "FLAG" for o in ORGANS)


def test_clinical_crash_leaves_uncertainty_alone(tmp_path):
    r = _run(tmp_path, clin=Boom())
    assert r.clinical.metadata_provenance == "none"
    assert r.clinical.organs["liver"].relevance is None
    assert r.clinical.organs["liver"].findings[0].code == "AGENT_ERROR"
    assert all(r.uncertainty.organs[o].status == "ACCEPT" for o in ORGANS)


def test_report_crash_falls_back_to_minimal_report(tmp_path):
    r = _run(tmp_path, rep=Boom())
    assert [s.organ for s in r.report.sections] == list(ORGANS)


def test_store_round_trip(tmp_path):
    r = _run(tmp_path)
    save_result(tmp_path, r)
    assert load_result(tmp_path) == r
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL (`ModuleNotFoundError: mars.fallbacks`)

- [ ] **Step 3: Implement**

`mars/fallbacks.py`:
```python
from mars.contracts import (
    ClinicalCheck, ORGANS, ORGAN_LABELS, OrganClinical, OrganMask, OrganSection,
    OrganUncertainty, Reason, Report, ReportSentence, SegmentationOutput,
    UncertaintyReport,
)


def reason(source: str, organ: str, code: str, severity: float = 1.0, **evidence) -> Reason:
    return Reason(id=f"{source}:{organ}:{code}", code=code, organ=organ,
                  severity=severity, evidence=evidence)


def empty_segmentation(case_id: str, ct_path) -> SegmentationOutput:
    return SegmentationOutput(
        case_id=case_id, ct_path=str(ct_path), spacing=(1.0, 1.0, 1.0), shape=(0, 0, 0),
        organs={o: OrganMask() for o in ORGANS},
    )


def empty_clinical(code: str, detail: str) -> ClinicalCheck:
    return ClinicalCheck(
        metadata_provenance="none",
        organs={o: OrganClinical(relevance=None,
                                 findings=[reason("clin", o, code, 0.0, detail=detail)])
                for o in ORGANS},
    )


def flagged_uncertainty(code: str, detail: str) -> UncertaintyReport:
    return UncertaintyReport(
        calibration_version="none", targets_hash="none",
        organs={o: OrganUncertainty(status="FLAG",
                                    reasons=[reason("unc", o, code, 1.0, detail=detail)])
                for o in ORGANS},
    )


def minimal_report(seg: SegmentationOutput, clin: ClinicalCheck,
                   unc: UncertaintyReport) -> Report:
    sections = []
    for o in ORGANS:
        status = unc.organs[o].status
        text = (f"{ORGAN_LABELS[o]}: {status}. Full report text is unavailable; "
                "see the structured results.")
        sections.append(OrganSection(
            organ=o, status=status,
            sentences=[ReportSentence(text=text, claim="SUMMARY", cites=[])]))
    flagged = sum(1 for o in ORGANS if unc.organs[o].status == "FLAG")
    return Report(case_id=seg.case_id,
                  summary=f"{len(ORGANS) - flagged}/{len(ORGANS)} organs auto-accepted.",
                  banner="Report text generation failed; showing statuses only.",
                  sections=sections)
```

`mars/pipeline.py`:
```python
import logging

from mars.contracts import CaseResult
from mars.fallbacks import (
    empty_clinical, empty_segmentation, flagged_uncertainty, minimal_report,
)

log = logging.getLogger("mars.pipeline")


def run_case(case_id, ct_path, meta, provenance, out_dir, *, seg, clin, unc, rep) -> CaseResult:
    """Run the four agents. Never raises because of an agent failure."""
    try:
        s = seg.run(case_id, ct_path, out_dir)
        seg_failed = None
    except Exception as e:  # noqa: BLE001 - degrade, don't crash
        log.exception("segmentation failed for %s", case_id)
        s, seg_failed = empty_segmentation(case_id, ct_path), str(e)

    if seg_failed:
        c = empty_clinical("AGENT_ERROR", seg_failed)
        u = flagged_uncertainty("AGENT_ERROR", seg_failed)
    else:
        try:
            c = clin.run(s, meta, provenance)
        except Exception as e:  # noqa: BLE001
            log.exception("clinical agent failed for %s", case_id)
            c = empty_clinical("AGENT_ERROR", str(e))
        try:
            u = unc.run(s)
        except Exception as e:  # noqa: BLE001
            log.exception("uncertainty agent failed for %s", case_id)
            u = flagged_uncertainty("AGENT_ERROR", str(e))

    try:
        r = rep.run(s, c, u)
    except Exception:  # noqa: BLE001
        log.exception("report agent failed for %s", case_id)
        r = minimal_report(s, c, u)
    return CaseResult(segmentation=s, clinical=c, uncertainty=u, report=r)
```

`mars/store.py`:
```python
from pathlib import Path

from mars.contracts import CaseResult
from mars.volumes import write_json_atomic


def save_result(out_dir, result: CaseResult) -> None:
    write_json_atomic(Path(out_dir) / "result.json", result.model_dump_json(indent=2))


def load_result(out_dir) -> CaseResult:
    return CaseResult.model_validate_json((Path(out_dir) / "result.json").read_text())
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add mars tests
git commit -m "feat: fallbacks, pipeline with failure degradation, result store"
```

---

### Task 4: Slice renderer and API

**Files:**
- Create: `mars/render.py`, `mars/api.py`
- Test: `tests/test_render.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: `store.save_result/load_result`, `volumes.load_ct/load_mask`, `contracts.ORGANS`
- Produces:
  - `render.render_slice_png(ct_zyx, z, mask_zyx=None, flagged=False, window=(-160.0, 240.0)) -> bytes` (raises `IndexError` if `z` out of range)
  - `api.create_app(run_fn, out_root: Path, data_root: Path, background: bool = True) -> FastAPI` where `run_fn(case_id: str, ct_path: str, image_id: str | None, synthetic_meta: dict | None) -> CaseResult`

- [ ] **Step 1: Write the failing tests**

`tests/test_render.py`:
```python
import numpy as np
import pytest

from mars.render import render_slice_png


def test_png_header_and_overlay_changes_pixels():
    ct = np.full((3, 8, 8), 40.0, dtype=np.float32)
    mask = np.zeros((3, 8, 8), bool); mask[1, 2:5, 2:5] = True
    plain = render_slice_png(ct, 1)
    over = render_slice_png(ct, 1, mask, flagged=True)
    assert plain[:8] == b"\x89PNG\r\n\x1a\n"
    assert plain != over


def test_z_out_of_range():
    with pytest.raises(IndexError):
        render_slice_png(np.zeros((3, 4, 4), np.float32), 3)
```

`tests/test_api.py`:
```python
import threading
import time

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from mars.api import create_app
from mars.contracts import CaseResult
from mars.store import save_result
from mars.volumes import save_mask
from tests.factories import make_clin, make_report, make_seg, make_unc


@pytest.fixture
def env(tmp_path):
    data = tmp_path / "data"; data.mkdir()
    arr = np.zeros((8, 8, 4), np.float32)
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(data / "ct.nii.gz"))
    out = tmp_path / "out"; out.mkdir()
    return data, out


def _result(case_id, ct_path, mask_path=None):
    seg = make_seg(case_id)
    seg.ct_path = str(ct_path)
    for om in seg.organs.values():
        om.mask_path = None
    if mask_path:
        seg.organs["liver"].mask_path = str(mask_path)
    return CaseResult(segmentation=seg, clinical=make_clin(), uncertainty=make_unc(),
                      report=make_report(case_id))


def _client(env, run_fn, background=False):
    data, out = env
    return TestClient(create_app(run_fn, out, data, background=background))


def test_run_then_get(env):
    data, _ = env
    client = _client(env, lambda cid, ct, img, meta: _result(cid, ct))
    r = client.post("/cases/c1/run", json={"ct_path": "ct.nii.gz"})
    assert r.status_code == 200 and r.json()["status"] == "done"
    body = client.get("/cases/c1").json()
    assert body["status"] == "done" and body["result"]["report"]["case_id"] == "c1"
    assert client.get("/cases").json()[0]["case_id"] == "c1"


@pytest.mark.parametrize("bad", ["../x", "a b", "x" * 65])
def test_bad_case_id_rejected(env, bad):
    client = _client(env, lambda *a: None)
    assert client.post(f"/cases/{bad}/run", json={"ct_path": "ct.nii.gz"}).status_code in (400, 404)


@pytest.mark.parametrize("path", ["../outside.nii.gz", "/etc/passwd", "missing.nii.gz"])
def test_ct_path_must_be_under_data_root(env, path):
    client = _client(env, lambda *a: None)
    assert client.post("/cases/c1/run", json={"ct_path": path}).status_code == 400


def test_duplicate_run_starts_one_job(env):
    calls, gate = [], threading.Event()

    def slow(cid, ct, img, meta):
        calls.append(cid); gate.wait(2)
        return _result(cid, ct)

    client = _client(env, slow, background=True)
    a = client.post("/cases/c1/run", json={"ct_path": "ct.nii.gz"}).json()
    b = client.post("/cases/c1/run", json={"ct_path": "ct.nii.gz"}).json()
    assert a["status"] == b["status"] == "running"
    gate.set()
    for _ in range(40):
        if client.get("/cases/c1").json()["status"] == "done":
            break
        time.sleep(0.05)
    assert client.get("/cases/c1").json()["status"] == "done"
    assert calls == ["c1"]


def test_failed_run_reports_error(env):
    def bad(*a):
        raise RuntimeError("nope")
    client = _client(env, bad)
    client.post("/cases/c1/run", json={"ct_path": "ct.nii.gz"})
    assert client.get("/cases/c1").json()["status"] == "error"


def test_slice_with_and_without_mask(env):
    data, out = env
    mask = np.zeros((4, 8, 8), bool); mask[1, 2:4, 2:4] = True
    save_mask(out / "liver.npy", mask)
    result = _result("c1", data / "ct.nii.gz", out / "liver.npy")
    (out / "c1").mkdir()
    save_result(out / "c1", result)
    client = _client(env, lambda *a: result)
    with_mask = client.get("/cases/c1/slice", params={"organ": "liver", "z": 1})
    no_mask = client.get("/cases/c1/slice", params={"organ": "spleen", "z": 1})  # no mask
    assert with_mask.status_code == no_mask.status_code == 200
    assert with_mask.content != no_mask.content
    assert client.get("/cases/c1/slice", params={"organ": "lung", "z": 1}).status_code == 404
    assert client.get("/cases/c1/slice", params={"organ": "liver", "z": 99}).status_code == 404
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_render.py tests/test_api.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement**

`mars/render.py`:
```python
import io

import numpy as np
from PIL import Image


def render_slice_png(ct_zyx, z, mask_zyx=None, flagged=False,
                     window=(-160.0, 240.0)) -> bytes:
    if not 0 <= z < ct_zyx.shape[0]:
        raise IndexError(f"z={z} out of range 0..{ct_zyx.shape[0] - 1}")
    lo, hi = window
    img = np.clip((ct_zyx[z] - lo) / (hi - lo), 0.0, 1.0)
    rgb = np.stack([img] * 3, axis=-1)
    if mask_zyx is not None:
        m = mask_zyx[z].astype(bool)
        colour = np.array([1.0, 0.45, 0.0]) if flagged else np.array([0.0, 0.7, 0.3])
        rgb[m] = 0.55 * rgb[m] + 0.45 * colour
    buf = io.BytesIO()
    Image.fromarray((rgb * 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()
```

`mars/api.py`:
```python
import logging
import re
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from mars.contracts import ORGANS
from mars.render import render_slice_png
from mars.store import load_result, save_result
from mars.volumes import load_ct, load_mask

log = logging.getLogger("mars.api")
CASE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class RunRequest(BaseModel):
    ct_path: str
    image_id: str | None = None
    synthetic_meta: dict | None = None


def create_app(run_fn, out_root: Path, data_root: Path, background: bool = True) -> FastAPI:
    out_root, data_root = Path(out_root), Path(data_root).resolve()
    app = FastAPI(title="MARS")
    status: dict[str, str] = {}
    lock = threading.Lock()

    def check_id(case_id: str):
        if not CASE_ID.match(case_id):
            raise HTTPException(400, "case_id must match [A-Za-z0-9_-]{1,64}")

    def work(case_id, ct, image_id, meta):
        try:
            save_result(out_root / case_id, run_fn(case_id, ct, image_id, meta))
            status[case_id] = "done"
        except Exception:  # noqa: BLE001
            log.exception("case %s failed", case_id)
            status[case_id] = "error"

    def current_status(case_id):
        if case_id in status:
            return status[case_id]
        return "done" if (out_root / case_id / "result.json").exists() else None

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/cases")
    def list_cases():
        ids = {p.name for p in out_root.iterdir() if (p / "result.json").exists()} if out_root.exists() else set()
        ids |= set(status)
        return [{"case_id": i, "status": current_status(i)} for i in sorted(ids)]

    @app.post("/cases/{case_id}/run")
    def run(case_id: str, body: RunRequest):
        check_id(case_id)
        ct = (data_root / body.ct_path).resolve()
        if data_root not in ct.parents or not ct.is_file():
            raise HTTPException(400, "ct_path must be an existing file under the data root")
        with lock:
            if status.get(case_id) == "running":
                return {"status": "running"}
            status[case_id] = "running"
        args = (case_id, str(ct), body.image_id, body.synthetic_meta)
        if background:
            threading.Thread(target=work, args=args, daemon=True).start()
        else:
            work(*args)
        return {"status": status[case_id]}

    @app.get("/cases/{case_id}")
    def get_case(case_id: str):
        check_id(case_id)
        st = current_status(case_id)
        if st is None:
            raise HTTPException(404, "unknown case")
        result = load_result(out_root / case_id) if st == "done" else None
        return {"status": st, "result": result.model_dump() if result else None}

    @app.get("/cases/{case_id}/slice")
    def slice_png(case_id: str, organ: str, z: int):
        check_id(case_id)
        if organ not in ORGANS:
            raise HTTPException(404, "unknown organ")
        if current_status(case_id) != "done":
            raise HTTPException(404, "case not finished")
        result = load_result(out_root / case_id)
        ct, _ = load_ct(result.segmentation.ct_path)
        om = result.segmentation.organs[organ]
        mask = load_mask(om.mask_path) if om.mask_path else None
        flagged = result.uncertainty.organs[organ].status == "FLAG"
        try:
            png = render_slice_png(ct, z, mask, flagged)
        except IndexError as e:
            raise HTTPException(404, str(e))
        return Response(content=png, media_type="image/png")

    return app
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_render.py tests/test_api.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add mars tests
git commit -m "feat: slice renderer and FastAPI app with path and duplicate-run guards"
```

---

### Task 5: Wiring, CLI, golden tests, runbook (after Plans 2–5 are done)

**Files:**
- Create: `mars/wiring.py`, `mars/run.py`, `docs/RUNBOOK.md`
- Test: `tests/test_golden.py`

**Interfaces:**
- Consumes (from the other plans, exact constructors):
  - `SegmentationAgent(prompt_source, refiner, deploy: dict[str, str])` — Plan 2
  - `ClinicalAgent(norms: VolumeNorms, llm_fn=None)`, `VolumeNorms.from_csv(path)`, `load_meta_row(csv_path, image_id) -> dict | None` — Plan 3
  - `UncertaintyAgent(refiner, calibrator: Calibrator, targets: dict, n_jitter=8, seed=0)`, `Calibrator.load(path)` — Plan 4
  - `ReportAgent(llm_fn=None)`, `anthropic_llm_fn()` — Plan 5
  - `TotalSegmentatorPromptSource()`, `load_refiner(ckpt, device)` — Plan 2
- Produces: `wiring.Settings.from_env()`, `wiring.make_run_fn(settings) -> run_fn` for `create_app`, CLI `python -m mars.run`, `uvicorn mars.wiring:app`.

- [ ] **Step 1: Write the failing golden tests** — `tests/test_golden.py`

These use the real `ReportAgent` with stubbed ML agents (CPU only).

```python
from mars.agents.report.agent import ReportAgent
from mars.pipeline import run_case
from tests.factories import make_clin, make_seg, make_unc


class Seg:
    def run(self, case_id, ct_path, out_dir):
        return make_seg(case_id)


class Clin:
    def __init__(self, provenance):
        self.provenance = provenance

    def run(self, seg, meta, provenance):
        return make_clin(self.provenance)


class Unc:
    def __init__(self, flag):
        self.flag = flag

    def run(self, seg):
        return make_unc(flag=self.flag)


def _run(tmp_path, flag, provenance):
    return run_case("c1", "ct.nii.gz", None, provenance, tmp_path,
                    seg=Seg(), clin=Clin(provenance), unc=Unc(flag), rep=ReportAgent())


def test_clean_case(tmp_path):
    r = _run(tmp_path, flag=(), provenance="real").report
    assert r.summary == "4/4 organs auto-accepted."
    assert r.banner is None
    text = " ".join(s.text for sec in r.sections for s in sec.sentences).lower()
    assert "normal" not in text and "healthy" not in text


def test_flagged_case_explains_why(tmp_path):
    r = _run(tmp_path, flag=("stomach",), provenance="real").report
    assert "stomach flagged" in r.summary
    stomach = next(s for s in r.sections if s.organ == "stomach")
    assert stomach.status == "FLAG"
    assert any("0.71" in s.text for s in stomach.sentences)  # jitter evidence is printed


def test_missing_metadata_is_stated(tmp_path):
    r = _run(tmp_path, flag=(), provenance="none").report
    assert "No clinical metadata" in r.banner
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_golden.py -v`
Expected: FAIL until Plan 5's `ReportAgent` exists (then these pass unchanged).

- [ ] **Step 3: Write `mars/wiring.py`**

```python
import json
import os
from dataclasses import dataclass
from pathlib import Path

from mars.agents.clinical.agent import ClinicalAgent
from mars.agents.clinical.meta import load_meta_row
from mars.agents.clinical.norms import VolumeNorms
from mars.agents.report.agent import ReportAgent
from mars.agents.report.llm import anthropic_llm_fn
from mars.agents.segmentation.agent import SegmentationAgent
from mars.agents.segmentation.prompt_source import TotalSegmentatorPromptSource
from mars.agents.segmentation.refiner import load_refiner
from mars.agents.uncertainty.agent import UncertaintyAgent
from mars.agents.uncertainty.calibrator import Calibrator
from mars.config import CONFIG_DIR, load_targets
from mars.pipeline import run_case


@dataclass
class Settings:
    data_root: Path
    out_root: Path
    meta_csv: Path
    ckpt: str
    llm: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_root=Path(os.environ.get("MARS_DATA_ROOT", "datasets")),
            out_root=Path(os.environ.get("MARS_OUT_ROOT", "outputs/cases")),
            meta_csv=Path(os.environ.get("MARS_META_CSV", "datasets/meta.csv")),
            ckpt=os.environ["MARS_SLIDESAM_CKPT"],
            llm=os.environ.get("MARS_LLM", "none"),
        )


def make_run_fn(s: Settings):
    refiner = load_refiner(s.ckpt, device="cuda")
    llm_fn = anthropic_llm_fn() if s.llm == "anthropic" else None
    deploy = json.loads((CONFIG_DIR / "deploy.json").read_text())
    seg = SegmentationAgent(TotalSegmentatorPromptSource(), refiner, deploy)
    clin = ClinicalAgent(VolumeNorms.from_csv(CONFIG_DIR / "volume_norms.csv"), llm_fn)
    unc = UncertaintyAgent(refiner, Calibrator.load(CONFIG_DIR / "calibration.json"),
                           load_targets())
    rep = ReportAgent(llm_fn)

    def run_fn(case_id, ct_path, image_id, synthetic_meta):
        if synthetic_meta is not None:
            meta, provenance = synthetic_meta, "synthetic"
        elif image_id and s.meta_csv.exists():
            meta = load_meta_row(s.meta_csv, image_id)
            provenance = "real" if meta else "none"
        else:
            meta, provenance = None, "none"
        return run_case(case_id, ct_path, meta, provenance, s.out_root / case_id,
                        seg=seg, clin=clin, unc=unc, rep=rep)

    return run_fn


def build_app():
    from mars.api import create_app
    s = Settings.from_env()
    return create_app(make_run_fn(s), s.out_root, s.data_root)
```
Serve with: `uvicorn mars.wiring:build_app --factory --host 0.0.0.0 --port 8000`.

- [ ] **Step 4: Write `mars/run.py` (CLI)**

```python
import argparse
import json
from pathlib import Path

from mars.store import save_result
from mars.wiring import Settings, make_run_fn


def main():
    p = argparse.ArgumentParser(description="Run one MARS case")
    p.add_argument("--case-id", required=True)
    p.add_argument("--ct", required=True, help="path to CT NIfTI")
    p.add_argument("--image-id", help="row in meta.csv (real metadata)")
    p.add_argument("--synthetic-meta", help="JSON file with fake metadata (demo only)")
    a = p.parse_args()
    s = Settings.from_env()
    synthetic = json.loads(Path(a.synthetic_meta).read_text()) if a.synthetic_meta else None
    result = make_run_fn(s)(a.case_id, a.ct, a.image_id, synthetic)
    save_result(s.out_root / a.case_id, result)
    print(result.report.summary)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write `docs/RUNBOOK.md`** (the order of operations; copy exactly)

````markdown
# MARS Runbook — order of operations

Run on the remote lab. `export MARS_DATA_ROOT=...` as needed.

## A. Offline (once, in this order)

1. **Dev segmentation + scores** (Plan 2 Task 6):
   `python scripts/make_dev_seg.py --config configs/dev_btcv.json --out outputs/dev`
2. **Refined vs auto decision** (Plan 2 Task 7):
   `python scripts/compare_refined_vs_auto.py --scores outputs/dev/dev_scores.csv --out configs/deploy.json`
3. **Feature table** (Plan 4 Task 5):
   `python scripts/make_feature_table.py --config configs/dev_btcv.json --seg-dir outputs/dev/seg --deploy configs/deploy.json --out outputs/dev/feature_table.csv`
4. **Feasibility check** (Plan 4 Task 6):
   `python scripts/feasibility.py --table outputs/dev/feature_table.csv --d-star 0.9`
   Edit `configs/targets.json` by hand (rule: min(0.90, cited inter-observer Dice)), **commit it with a message that says why**.
5. **Fit and evaluate calibration** (Plan 4 Task 6):
   `python scripts/fit_calibration.py --table outputs/dev/feature_table.csv --alpha 0.9 --out configs/calibration.json`
   `python scripts/evaluate_nested.py --table outputs/dev/feature_table.csv --alpha 0.9 --out outputs/dev/nested`
6. **Freeze:** commit `configs/` (deploy, targets, calibration, volume_norms). Tag `config-frozen`.
7. **External set (once):** build `configs/dev_totalseg.json` (`gt_format: "totalseg"`). List every external patient **once per fold checkpoint**, e.g. `{"patient": "s0123", "case_id": "s0123_f2", "fold": 2, ...}`, so each of the 5 fold models predicts every external scan. Repeat steps 1 and 3 with `--config configs/dev_totalseg.json --out outputs/external`, then evaluate once with the frozen calibration:
   `python scripts/evaluate_external.py --table outputs/external/feature_table.csv --out outputs/external/eval`
   Do not change any file in `configs/` after seeing these results.

## B. Online demo

```
export MARS_DATA_ROOT=datasets MARS_SLIDESAM_CKPT=/path/fold0.pt MARS_LLM=none
uvicorn mars.wiring:build_app --factory --host 0.0.0.0 --port 8000
```
Then the frontend (see frontend spec) proxies `/api` to port 8000.

## C. Checks before showing anyone

- `pytest` passes on CPU.
- `pytest -m gpu` passes on the lab.
- One BTCV case with `--synthetic-meta` shows the "synthetic metadata" banner.
- One TotalSegmentator test-split case with `--image-id` shows no banner.
````

- [ ] **Step 6: Run the whole suite**

Run: `pytest -v`
Expected: everything PASS (golden tests now pass because Plan 5 exists)

- [ ] **Step 7: Commit**

```bash
git add mars tests docs/RUNBOOK.md
git commit -m "feat: wiring, CLI, golden tests, runbook"
```
