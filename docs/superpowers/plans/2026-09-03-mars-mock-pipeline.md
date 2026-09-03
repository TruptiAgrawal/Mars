# MARS Mock Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working, test-first (TDD) mock version of the MARS 4-agent
pipeline (segmentation → validation → uncertainty → routing/reporting) on
synthetic data, with a FastAPI backend and a React frontend that visualizes
cases and their routing decisions.

**Architecture:** Backend is a plain Python package (`backend/mars`) with one
module per agent, a `pipeline.py` orchestrator, and a thin FastAPI layer
(`api.py`) serving pre-generated synthetic case fixtures. Frontend is a Vite +
React + TypeScript app that lists cases, renders a 2D slice viewer with mask
overlay, and shows either an auto-generated report or an escalation panel
depending on the pipeline's routing decision.

**Tech Stack:** Python 3.11+, NumPy, SciPy, Pydantic, FastAPI, pytest,
httpx (for `TestClient`); React 18 + TypeScript, Vite, Jest, React Testing
Library.

**Spec:** `docs/superpowers/specs/2026-09-03-mars-mock-pipeline-design.md`

## Global Constraints

- No real ML models (no PyTorch/SAM/nnU-Net) — segmentation uses
  thresholding + connected-component labeling on synthetic volumes.
- No DICOM/NIfTI — cases are `.npy` volumes + `.json` metadata sidecars.
- No auth, no database — cases are static fixture files under
  `backend/mars/data/cases/`.
- `τ` (tau) defaults to `0.5` and must be overridable per-request (NFR-6).
- Every backend module gets a one-line module docstring stating its single
  responsibility (NFR-2/NFR-4).
- Every agent's output must be JSON-serializable via Pydantic models defined
  in `backend/mars/models.py` — no ad-hoc dicts crossing module boundaries.
- Fixed random seeds for all synthetic data generation (NFR-5).

---

## Task 1: Backend scaffold + core data models

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/mars/__init__.py`
- Create: `backend/mars/models.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/test_models.py`

**Interfaces:**
- Produces: `ClinicalMetadata`, `SegmentationResult`, `ValidationResult`,
  `UncertaintyResult`, `Report`, `Escalation`, `CaseResult` — all Pydantic
  `BaseModel` subclasses in `backend/mars/models.py`, used by every
  subsequent task.

- [ ] **Step 1: Create `backend/pyproject.toml`**

```toml
[project]
name = "mars"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "pydantic>=2.5",
    "fastapi>=0.110",
    "uvicorn>=0.27",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "httpx>=0.27"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["mars*"]
```

- [ ] **Step 2: Create `backend/mars/__init__.py`**

```python
"""MARS: multi-agent mock pipeline for CT lesion segmentation, validation, uncertainty routing, and reporting."""
```

- [ ] **Step 3: Create `backend/tests/__init__.py`** (empty file)

- [ ] **Step 4: Write the failing test `backend/tests/test_models.py`**

```python
from mars.models import (
    ClinicalMetadata,
    SegmentationResult,
    ValidationResult,
    UncertaintyResult,
    Report,
    Escalation,
    CaseResult,
)


def test_clinical_metadata_roundtrip():
    meta = ClinicalMetadata(
        case_id="case_001",
        expected_site="anterior",
        expected_size_range_mm3=(100.0, 5000.0),
        prior_history="none",
    )
    assert meta.case_id == "case_001"
    assert meta.expected_size_range_mm3 == (100.0, 5000.0)


def test_case_result_auto_path_requires_report_not_escalation():
    result = CaseResult(
        case_id="case_001",
        tau=0.5,
        segmentation=SegmentationResult(
            mask=[[[0, 1], [0, 0]]],
            per_voxel_confidence=[[[0.0, 0.9], [0.0, 0.0]]],
            aggregate_confidence=0.9,
        ),
        validation=ValidationResult(consistency_score=0.95, conflicts=[]),
        uncertainty=UncertaintyResult(
            value=0.1, dominant_factor="confidence", breakdown={"confidence_term": 0.05, "consistency_term": 0.05}
        ),
        decision="auto",
        report=Report(
            case_id="case_001",
            findings="single lesion detected",
            measurements={"voxel_count": 1},
            confidence=0.9,
            narrative="Lesion detected with high confidence.",
        ),
        escalation=None,
    )
    assert result.decision == "auto"
    assert result.report is not None
    assert result.escalation is None
```

- [ ] **Step 5: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mars.models'` (or similar import error)

- [ ] **Step 6: Write `backend/mars/models.py`**

```python
"""Pydantic schemas shared across all MARS agents and the API layer."""

from typing import Optional
from pydantic import BaseModel


class ClinicalMetadata(BaseModel):
    case_id: str
    expected_site: str
    expected_size_range_mm3: tuple[float, float]
    prior_history: str


class SegmentationResult(BaseModel):
    mask: list
    per_voxel_confidence: list
    aggregate_confidence: float


class ValidationResult(BaseModel):
    consistency_score: float
    conflicts: list[str]


class UncertaintyResult(BaseModel):
    value: float
    dominant_factor: str
    breakdown: dict[str, float]


class Report(BaseModel):
    case_id: str
    findings: str
    measurements: dict
    confidence: float
    narrative: str


class Escalation(BaseModel):
    case_id: str
    uncertainty: float
    reasons: list[str]


class CaseResult(BaseModel):
    case_id: str
    tau: float
    segmentation: SegmentationResult
    validation: ValidationResult
    uncertainty: UncertaintyResult
    decision: str
    report: Optional[Report] = None
    escalation: Optional[Escalation] = None
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_models.py -v`
Expected: PASS (2 tests)

- [ ] **Step 8: Commit**

```bash
git add backend/pyproject.toml backend/mars/__init__.py backend/mars/models.py backend/tests/__init__.py backend/tests/test_models.py
git commit -m "feat(backend): scaffold package and core Pydantic models"
```

---

## Task 2: Synthetic volume generator

**Files:**
- Create: `backend/mars/data/__init__.py`
- Create: `backend/mars/data/synthetic.py`
- Create: `backend/tests/test_synthetic.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `generate_volume(shape, center, radius, intensity, background, noise_std, seed) -> np.ndarray`
  and `voxel_centroid(mask: np.ndarray) -> tuple[float, float, float]` in
  `backend/mars/data/synthetic.py`, used by Task 3 (fixtures) and agent tests.

- [ ] **Step 1: Create `backend/mars/data/__init__.py`**

```python
"""Synthetic CT volume generation and mock case fixtures for MARS."""
```

- [ ] **Step 2: Write the failing test `backend/tests/test_synthetic.py`**

```python
import numpy as np
from mars.data.synthetic import generate_volume, voxel_centroid


def test_generate_volume_shape_and_determinism():
    v1 = generate_volume(shape=(16, 32, 32), center=(8, 16, 16), radius=4.0, seed=42)
    v2 = generate_volume(shape=(16, 32, 32), center=(8, 16, 16), radius=4.0, seed=42)
    assert v1.shape == (16, 32, 32)
    assert np.array_equal(v1, v2)


def test_generate_volume_lesion_brighter_than_background():
    v = generate_volume(
        shape=(16, 32, 32), center=(8, 16, 16), radius=4.0,
        intensity=200.0, background=50.0, noise_std=0.0, seed=1,
    )
    assert v[8, 16, 16] > v[0, 0, 0]


def test_voxel_centroid_of_simple_mask():
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    centroid = voxel_centroid(mask)
    assert centroid == (1.5, 1.5, 1.5)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_synthetic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mars.data.synthetic'`

- [ ] **Step 4: Write `backend/mars/data/synthetic.py`**

```python
"""Generates synthetic 3D CT-like volumes with an embedded Gaussian-blob lesion, for mock pipeline testing."""

import numpy as np


def generate_volume(
    shape: tuple[int, int, int] = (32, 64, 64),
    center: tuple[float, float, float] | None = None,
    radius: float = 6.0,
    intensity: float = 200.0,
    background: float = 50.0,
    noise_std: float = 10.0,
    seed: int = 0,
) -> np.ndarray:
    if center is None:
        center = tuple(s / 2 for s in shape)

    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    dist = np.sqrt(
        (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    )
    blob = intensity * np.exp(-(dist**2) / (2 * radius**2))
    volume = background + blob
    if noise_std > 0:
        volume = volume + rng.normal(0.0, noise_std, size=shape)
    return volume.astype(np.float32)


def voxel_centroid(mask: np.ndarray) -> tuple[float, float, float]:
    coords = np.argwhere(mask > 0)
    mean = coords.mean(axis=0)
    return tuple(float(v) for v in mean)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_synthetic.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add backend/mars/data/__init__.py backend/mars/data/synthetic.py backend/tests/test_synthetic.py
git commit -m "feat(backend): synthetic volume generator with deterministic seeding"
```

---

## Task 3: Segmentation Agent

**Files:**
- Create: `backend/mars/agents/__init__.py`
- Create: `backend/mars/agents/segmentation.py`
- Create: `backend/tests/test_segmentation.py`

**Interfaces:**
- Consumes: `generate_volume` from Task 2 (test fixtures only);
  `SegmentationResult` from Task 1.
- Produces: `segment(volume: np.ndarray, threshold: float = 120.0) -> SegmentationResult`
  in `backend/mars/agents/segmentation.py`, used by Task 6 (pipeline).

- [ ] **Step 1: Create `backend/mars/agents/__init__.py`**

```python
"""The four MARS agents: segmentation, validation, uncertainty, reporting."""
```

- [ ] **Step 2: Write the failing test `backend/tests/test_segmentation.py`**

```python
import numpy as np
from mars.agents.segmentation import segment
from mars.data.synthetic import generate_volume


def test_segment_detects_lesion_blob():
    volume = generate_volume(
        shape=(16, 32, 32), center=(8, 16, 16), radius=4.0,
        intensity=200.0, background=50.0, noise_std=0.0, seed=1,
    )
    result = segment(volume, threshold=120.0)
    mask = np.array(result.mask)
    assert mask.shape == volume.shape
    assert mask[8, 16, 16] == 1
    assert mask[0, 0, 0] == 0
    assert 0.0 <= result.aggregate_confidence <= 1.0


def test_segment_low_contrast_blob_has_lower_confidence_than_high_contrast():
    high_contrast = generate_volume(
        shape=(16, 32, 32), center=(8, 16, 16), radius=5.0,
        intensity=200.0, background=50.0, noise_std=0.0, seed=1,
    )
    low_contrast = generate_volume(
        shape=(16, 32, 32), center=(8, 16, 16), radius=5.0,
        intensity=130.0, background=50.0, noise_std=0.0, seed=1,
    )
    high_result = segment(high_contrast, threshold=120.0)
    low_result = segment(low_contrast, threshold=120.0)
    assert low_result.aggregate_confidence < high_result.aggregate_confidence


def test_segment_no_lesion_returns_empty_mask():
    flat = np.full((8, 8, 8), 50.0, dtype=np.float32)
    result = segment(flat, threshold=120.0)
    mask = np.array(result.mask)
    assert mask.sum() == 0
    assert result.aggregate_confidence == 0.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_segmentation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mars.agents.segmentation'`

- [ ] **Step 4: Write `backend/mars/agents/segmentation.py`**

```python
"""Agent 1: thresholding + connected-component segmentation with per-voxel confidence, mocking a learned segmentation model."""

import numpy as np
from scipy import ndimage

from mars.models import SegmentationResult


def segment(volume: np.ndarray, threshold: float = 120.0) -> SegmentationResult:
    above = volume > threshold
    labeled, num_components = ndimage.label(above)

    mask = np.zeros_like(volume, dtype=np.uint8)
    confidence = np.zeros_like(volume, dtype=np.float32)

    if num_components > 0:
        sizes = ndimage.sum(above, labeled, index=range(1, num_components + 1))
        largest_label = int(np.argmax(sizes)) + 1
        mask = (labeled == largest_label).astype(np.uint8)

        vmax = float(volume.max())
        span = max(vmax - threshold, 1e-6)
        voxel_confidence = np.clip((volume - threshold) / span, 0.0, 1.0)
        confidence = np.where(mask == 1, voxel_confidence, 0.0).astype(np.float32)

    aggregate = float(confidence[mask == 1].mean()) if mask.sum() > 0 else 0.0

    return SegmentationResult(
        mask=mask.tolist(),
        per_voxel_confidence=confidence.tolist(),
        aggregate_confidence=aggregate,
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_segmentation.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add backend/mars/agents/__init__.py backend/mars/agents/segmentation.py backend/tests/test_segmentation.py
git commit -m "feat(backend): segmentation agent with per-voxel confidence"
```

---

## Task 4: Validation Agent

**Files:**
- Create: `backend/mars/agents/validation.py`
- Create: `backend/tests/test_validation.py`

**Interfaces:**
- Consumes: `ClinicalMetadata`, `ValidationResult` from Task 1;
  `voxel_centroid` from Task 2.
- Produces: `validate(mask: np.ndarray, metadata: ClinicalMetadata) -> ValidationResult`
  in `backend/mars/agents/validation.py`, used by Task 6 (pipeline). Also
  exposes `SITES: dict[str, tuple[float, float]]` mapping a site name to a
  `(z_lo_fraction, z_hi_fraction)` range of the volume's z-axis (as a
  fraction of total depth, e.g. `"anterior": (0.0, 0.5)`), used by Task 5
  (fixtures) to build cases whose mask lands in or out of the expected site.

- [ ] **Step 1: Write the failing test `backend/tests/test_validation.py`**

```python
import numpy as np
from mars.agents.validation import validate, SITES
from mars.models import ClinicalMetadata


def _mask_at_z(shape, z_index):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[z_index, shape[1] // 2, shape[2] // 2] = 1
    mask[z_index, shape[1] // 2 - 1 : shape[1] // 2 + 1, shape[2] // 2 - 1 : shape[2] // 2 + 1] = 1
    return mask


def test_validate_consistent_site_and_size_has_high_score_no_conflicts():
    shape = (16, 32, 32)
    mask = _mask_at_z(shape, z_index=2)  # within "anterior" (0.0-0.5 of depth)
    meta = ClinicalMetadata(
        case_id="c1", expected_site="anterior",
        expected_size_range_mm3=(1.0, 100.0), prior_history="none",
    )
    result = validate(mask, meta)
    assert result.consistency_score == 1.0
    assert result.conflicts == []


def test_validate_wrong_site_flags_conflict_and_lowers_score():
    shape = (16, 32, 32)
    mask = _mask_at_z(shape, z_index=14)  # within "posterior" (0.5-1.0 of depth)
    meta = ClinicalMetadata(
        case_id="c2", expected_site="anterior",
        expected_size_range_mm3=(1.0, 100.0), prior_history="none",
    )
    result = validate(mask, meta)
    assert result.consistency_score < 1.0
    assert any("site" in c.lower() for c in result.conflicts)


def test_validate_size_out_of_range_flags_conflict():
    shape = (16, 32, 32)
    mask = _mask_at_z(shape, z_index=2)
    meta = ClinicalMetadata(
        case_id="c3", expected_site="anterior",
        expected_size_range_mm3=(1000.0, 5000.0), prior_history="none",
    )
    result = validate(mask, meta)
    assert result.consistency_score < 1.0
    assert any("size" in c.lower() for c in result.conflicts)


def test_sites_covers_anterior_and_posterior():
    assert "anterior" in SITES
    assert "posterior" in SITES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_validation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mars.agents.validation'`

- [ ] **Step 3: Write `backend/mars/agents/validation.py`**

```python
"""Agent 2: checks a predicted mask against clinical metadata for anatomical and size plausibility."""

import numpy as np

from mars.data.synthetic import voxel_centroid
from mars.models import ClinicalMetadata, ValidationResult

SITES: dict[str, tuple[float, float]] = {
    "anterior": (0.0, 0.5),
    "posterior": (0.5, 1.0),
}


def validate(mask: np.ndarray, metadata: ClinicalMetadata) -> ValidationResult:
    conflicts: list[str] = []
    voxel_count = int(mask.sum())

    if voxel_count == 0:
        return ValidationResult(consistency_score=0.0, conflicts=["no lesion detected in mask"])

    z_centroid, _, _ = voxel_centroid(mask)
    z_depth = mask.shape[0]
    z_fraction = z_centroid / max(z_depth - 1, 1)

    site_range = SITES.get(metadata.expected_site)
    site_ok = site_range is not None and site_range[0] <= z_fraction <= site_range[1]
    if not site_ok:
        conflicts.append(
            f"mask centroid at z-fraction {z_fraction:.2f} is outside expected site '{metadata.expected_site}'"
        )

    lo, hi = metadata.expected_size_range_mm3
    size_ok = lo <= voxel_count <= hi
    if not size_ok:
        conflicts.append(
            f"mask size {voxel_count} voxels is outside expected range ({lo}, {hi})"
        )

    checks_passed = sum([site_ok, size_ok])
    score = checks_passed / 2.0

    return ValidationResult(consistency_score=score, conflicts=conflicts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_validation.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/mars/agents/validation.py backend/tests/test_validation.py
git commit -m "feat(backend): validation agent with site/size plausibility checks"
```

---

## Task 5: Uncertainty Agent

**Files:**
- Create: `backend/mars/agents/uncertainty.py`
- Create: `backend/tests/test_uncertainty.py`

**Interfaces:**
- Consumes: `UncertaintyResult` from Task 1.
- Produces: `compute_uncertainty(confidence: float, consistency: float, w_confidence: float = 0.5, w_consistency: float = 0.5) -> UncertaintyResult`
  in `backend/mars/agents/uncertainty.py`, used by Task 6 (pipeline).

- [ ] **Step 1: Write the failing test `backend/tests/test_uncertainty.py`**

```python
from mars.agents.uncertainty import compute_uncertainty


def test_high_confidence_and_consistency_gives_low_uncertainty():
    result = compute_uncertainty(confidence=0.95, consistency=0.95)
    assert result.value < 0.1


def test_low_confidence_and_consistency_gives_high_uncertainty():
    result = compute_uncertainty(confidence=0.1, consistency=0.1)
    assert result.value > 0.8


def test_dominant_factor_attribution_picks_larger_term():
    result = compute_uncertainty(confidence=0.2, consistency=0.9)
    assert result.dominant_factor == "confidence"
    assert result.breakdown["confidence_term"] > result.breakdown["consistency_term"]


def test_uncertainty_value_bounded_zero_to_one():
    result = compute_uncertainty(confidence=0.0, consistency=0.0)
    assert 0.0 <= result.value <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_uncertainty.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mars.agents.uncertainty'`

- [ ] **Step 3: Write `backend/mars/agents/uncertainty.py`**

```python
"""Agent 3: aggregates segmentation confidence and validation consistency into a single calibrated uncertainty score."""

from mars.models import UncertaintyResult


def compute_uncertainty(
    confidence: float,
    consistency: float,
    w_confidence: float = 0.5,
    w_consistency: float = 0.5,
) -> UncertaintyResult:
    confidence_term = w_confidence * (1.0 - confidence)
    consistency_term = w_consistency * (1.0 - consistency)
    value = confidence_term + consistency_term

    dominant_factor = "confidence" if confidence_term >= consistency_term else "consistency"

    return UncertaintyResult(
        value=round(value, 6),
        dominant_factor=dominant_factor,
        breakdown={
            "confidence_term": round(confidence_term, 6),
            "consistency_term": round(consistency_term, 6),
        },
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_uncertainty.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/mars/agents/uncertainty.py backend/tests/test_uncertainty.py
git commit -m "feat(backend): uncertainty agent with weighted aggregation and attribution"
```

---

## Task 6: Reporting Agent + Pipeline orchestrator

**Files:**
- Create: `backend/mars/agents/reporting.py`
- Create: `backend/mars/pipeline.py`
- Create: `backend/tests/test_reporting.py`
- Create: `backend/tests/test_pipeline.py`

**Interfaces:**
- Consumes: `segment` (Task 3), `validate` (Task 4), `compute_uncertainty`
  (Task 5), all models from Task 1, `generate_volume` (Task 2).
- Produces:
  - `build_case_result(case_id: str, tau: float, segmentation: SegmentationResult, validation: ValidationResult, uncertainty: UncertaintyResult) -> CaseResult`
    in `backend/mars/agents/reporting.py`.
  - `run_case(volume: np.ndarray, metadata: ClinicalMetadata, tau: float = 0.5, threshold: float = 120.0) -> CaseResult`
    in `backend/mars/pipeline.py`, used by Task 7 (fixtures/API integration
    tests) and Task 8 (API).

- [ ] **Step 1: Write the failing test `backend/tests/test_reporting.py`**

```python
from mars.agents.reporting import build_case_result
from mars.models import SegmentationResult, ValidationResult, UncertaintyResult


def _seg(agg_conf=0.9):
    return SegmentationResult(mask=[[[1]]], per_voxel_confidence=[[[agg_conf]]], aggregate_confidence=agg_conf)


def _val(score=0.9, conflicts=None):
    return ValidationResult(consistency_score=score, conflicts=conflicts or [])


def _unc(value, dominant="confidence"):
    return UncertaintyResult(value=value, dominant_factor=dominant, breakdown={"confidence_term": value / 2, "consistency_term": value / 2})


def test_low_uncertainty_produces_report_not_escalation():
    result = build_case_result("case_a", tau=0.5, segmentation=_seg(), validation=_val(), uncertainty=_unc(0.1))
    assert result.decision == "auto"
    assert result.report is not None
    assert result.escalation is None
    assert result.report.case_id == "case_a"


def test_high_uncertainty_produces_escalation_not_report():
    result = build_case_result(
        "case_b", tau=0.5, segmentation=_seg(agg_conf=0.2),
        validation=_val(score=0.3, conflicts=["mask outside expected site"]), uncertainty=_unc(0.7),
    )
    assert result.decision == "escalate"
    assert result.escalation is not None
    assert result.report is None
    assert "mask outside expected site" in result.escalation.reasons


def test_uncertainty_exactly_at_threshold_escalates():
    result = build_case_result("case_c", tau=0.5, segmentation=_seg(), validation=_val(), uncertainty=_unc(0.5))
    assert result.decision == "escalate"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_reporting.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mars.agents.reporting'`

- [ ] **Step 3: Write `backend/mars/agents/reporting.py`**

```python
"""Agent 4: routes a case to an automated report or a radiologist escalation based on the uncertainty threshold."""

from mars.models import CaseResult, Escalation, Report, SegmentationResult, UncertaintyResult, ValidationResult


def build_case_result(
    case_id: str,
    tau: float,
    segmentation: SegmentationResult,
    validation: ValidationResult,
    uncertainty: UncertaintyResult,
) -> CaseResult:
    if uncertainty.value < tau:
        decision = "auto"
        voxel_count = sum(
            1
            for plane in segmentation.mask
            for row in plane
            for v in row
            if v == 1
        )
        report = Report(
            case_id=case_id,
            findings="Lesion detected and cleared automatically." if voxel_count > 0 else "No lesion detected.",
            measurements={"voxel_count": voxel_count},
            confidence=segmentation.aggregate_confidence,
            narrative=(
                f"Automated analysis identified a lesion of {voxel_count} voxels "
                f"with {segmentation.aggregate_confidence:.0%} confidence. "
                f"Clinical consistency score: {validation.consistency_score:.0%}. "
                f"Overall uncertainty {uncertainty.value:.2f} is below the safety threshold {tau:.2f}."
            ),
        )
        return CaseResult(
            case_id=case_id, tau=tau, segmentation=segmentation, validation=validation,
            uncertainty=uncertainty, decision=decision, report=report, escalation=None,
        )

    decision = "escalate"
    reasons = list(validation.conflicts)
    reasons.append(
        f"uncertainty {uncertainty.value:.2f} >= threshold {tau:.2f} "
        f"(dominant factor: {uncertainty.dominant_factor})"
    )
    escalation = Escalation(case_id=case_id, uncertainty=uncertainty.value, reasons=reasons)
    return CaseResult(
        case_id=case_id, tau=tau, segmentation=segmentation, validation=validation,
        uncertainty=uncertainty, decision=decision, report=None, escalation=escalation,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_reporting.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Write the failing test `backend/tests/test_pipeline.py`**

```python
import numpy as np
from mars.pipeline import run_case
from mars.data.synthetic import generate_volume
from mars.models import ClinicalMetadata


def test_run_case_clean_lesion_matching_metadata_auto_clears():
    volume = generate_volume(
        shape=(16, 32, 32), center=(4, 16, 16), radius=5.0,
        intensity=200.0, background=50.0, noise_std=0.0, seed=1,
    )
    meta = ClinicalMetadata(
        case_id="clean_case", expected_site="anterior",
        expected_size_range_mm3=(50.0, 5000.0), prior_history="none",
    )
    result = run_case(volume, meta, tau=0.5)
    assert result.case_id == "clean_case"
    assert result.decision == "auto"
    assert result.report is not None


def test_run_case_mismatched_site_escalates():
    volume = generate_volume(
        shape=(16, 32, 32), center=(14, 16, 16), radius=5.0,
        intensity=200.0, background=50.0, noise_std=0.0, seed=1,
    )
    meta = ClinicalMetadata(
        case_id="mismatch_case", expected_site="anterior",
        expected_size_range_mm3=(50.0, 5000.0), prior_history="none",
    )
    result = run_case(volume, meta, tau=0.3)
    assert result.decision == "escalate"
    assert result.escalation is not None
    assert len(result.escalation.reasons) > 0
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mars.pipeline'`

- [ ] **Step 7: Write `backend/mars/pipeline.py`**

```python
"""Orchestrates the four MARS agents in sequence for a single case: segment, validate, aggregate uncertainty, route."""

import numpy as np

from mars.agents.reporting import build_case_result
from mars.agents.segmentation import segment
from mars.agents.uncertainty import compute_uncertainty
from mars.agents.validation import validate
from mars.models import CaseResult, ClinicalMetadata


def run_case(
    volume: np.ndarray,
    metadata: ClinicalMetadata,
    tau: float = 0.5,
    threshold: float = 120.0,
) -> CaseResult:
    segmentation = segment(volume, threshold=threshold)
    mask = np.array(segmentation.mask, dtype=np.uint8)
    validation = validate(mask, metadata)
    uncertainty = compute_uncertainty(
        confidence=segmentation.aggregate_confidence,
        consistency=validation.consistency_score,
    )
    return build_case_result(
        case_id=metadata.case_id,
        tau=tau,
        segmentation=segmentation,
        validation=validation,
        uncertainty=uncertainty,
    )
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_pipeline.py -v`
Expected: PASS (2 tests)

- [ ] **Step 9: Commit**

```bash
git add backend/mars/agents/reporting.py backend/mars/pipeline.py backend/tests/test_reporting.py backend/tests/test_pipeline.py
git commit -m "feat(backend): reporting agent and full pipeline orchestrator"
```

---

## Task 7: Mock case fixtures

**Files:**
- Create: `backend/mars/data/generate_fixtures.py`
- Create: `backend/mars/data/cases/` (generated `.npy` + `.json` files, committed)
- Create: `backend/tests/test_fixtures.py`

**Interfaces:**
- Consumes: `generate_volume` (Task 2), `ClinicalMetadata` (Task 1),
  `run_case` (Task 6, used only by the test to sanity-check fixtures).
- Produces: `backend/mars/data/cases/<case_id>.npy` and
  `backend/mars/data/cases/<case_id>.json` fixture files, plus
  `CASES_DIR: pathlib.Path` constant in `backend/mars/data/generate_fixtures.py`,
  used by Task 8's `load_case`/`list_cases`.

- [ ] **Step 1: Write `backend/mars/data/generate_fixtures.py`**

```python
"""Generates and writes the ~15 hand-crafted mock CT cases (volume + metadata) used by the API and tests."""

import json
from pathlib import Path

from mars.data.synthetic import generate_volume
from mars.models import ClinicalMetadata

CASES_DIR = Path(__file__).parent / "cases"

SHAPE = (16, 32, 32)

# Each entry: (case_id, label, center, radius, intensity, background, noise_std, seed,
#              expected_site, expected_size_range_mm3, prior_history)
CASE_SPECS = [
    ("case_001", "Clean anterior lesion, high confidence", (4, 16, 16), 5.0, 200.0, 50.0, 2.0, 1, "anterior", (50.0, 5000.0), "none"),
    ("case_002", "Clean anterior lesion, matches history", (3, 12, 20), 4.5, 190.0, 50.0, 2.0, 2, "anterior", (30.0, 5000.0), "prior lung nodule"),
    ("case_003", "Clean posterior lesion, high confidence", (12, 16, 16), 5.0, 200.0, 50.0, 2.0, 3, "posterior", (50.0, 5000.0), "none"),
    ("case_004", "Clean posterior lesion, matches history", (13, 20, 10), 4.5, 195.0, 50.0, 2.0, 4, "posterior", (30.0, 5000.0), "none"),
    ("case_005", "Low-contrast anterior lesion", (4, 16, 16), 5.0, 128.0, 50.0, 2.0, 5, "anterior", (50.0, 5000.0), "none"),
    ("case_006", "Low-contrast posterior lesion", (12, 16, 16), 5.0, 126.0, 50.0, 2.0, 6, "posterior", (50.0, 5000.0), "none"),
    ("case_007", "Small low-contrast lesion", (4, 16, 16), 2.0, 130.0, 50.0, 2.0, 7, "anterior", (50.0, 5000.0), "none"),
    ("case_008", "Noisy borderline lesion", (4, 16, 16), 4.0, 150.0, 50.0, 20.0, 8, "anterior", (50.0, 5000.0), "none"),
    ("case_009", "Site mismatch: lesion posterior, expected anterior", (12, 16, 16), 5.0, 200.0, 50.0, 2.0, 9, "anterior", (50.0, 5000.0), "none"),
    ("case_010", "Site mismatch: lesion anterior, expected posterior", (4, 16, 16), 5.0, 200.0, 50.0, 2.0, 10, "posterior", (50.0, 5000.0), "none"),
    ("case_011", "Size mismatch: lesion too small for expected range", (4, 16, 16), 2.0, 200.0, 50.0, 2.0, 11, "anterior", (2000.0, 5000.0), "none"),
    ("case_012", "Size mismatch: lesion too large for expected range", (4, 16, 16), 7.0, 200.0, 50.0, 2.0, 12, "anterior", (1.0, 20.0), "none"),
    ("case_013", "Combined site and size mismatch", (12, 16, 16), 7.0, 200.0, 50.0, 2.0, 13, "anterior", (1.0, 20.0), "none"),
    ("case_014", "No detectable lesion (background only)", (4, 16, 16), 5.0, 55.0, 50.0, 2.0, 14, "anterior", (50.0, 5000.0), "none"),
    ("case_015", "Clean lesion, mid-size, borderline uncertainty", (4, 16, 16), 3.5, 145.0, 50.0, 5.0, 15, "anterior", (50.0, 5000.0), "none"),
]


def generate_all_fixtures() -> None:
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    for (case_id, label, center, radius, intensity, background, noise_std, seed,
         expected_site, expected_size_range_mm3, prior_history) in CASE_SPECS:
        volume = generate_volume(
            shape=SHAPE, center=center, radius=radius, intensity=intensity,
            background=background, noise_std=noise_std, seed=seed,
        )
        import numpy as np
        np.save(CASES_DIR / f"{case_id}.npy", volume)

        metadata = ClinicalMetadata(
            case_id=case_id, expected_site=expected_site,
            expected_size_range_mm3=expected_size_range_mm3, prior_history=prior_history,
        )
        meta_dict = metadata.model_dump()
        meta_dict["label"] = label
        with open(CASES_DIR / f"{case_id}.json", "w") as f:
            json.dump(meta_dict, f, indent=2)


if __name__ == "__main__":
    generate_all_fixtures()
    print(f"Generated {len(CASE_SPECS)} fixture cases in {CASES_DIR}")
```

- [ ] **Step 2: Run the generator to produce fixture files**

Run: `cd backend && python -m mars.data.generate_fixtures`
Expected: prints `Generated 15 fixture cases in .../mars/data/cases`, and
`backend/mars/data/cases/` now contains `case_001.npy` .. `case_015.json`
(30 files total).

- [ ] **Step 3: Write the failing test `backend/tests/test_fixtures.py`**

```python
import json
import numpy as np
import pytest

from mars.data.generate_fixtures import CASES_DIR, CASE_SPECS
from mars.models import ClinicalMetadata
from mars.pipeline import run_case


def test_all_fixture_files_exist():
    for spec in CASE_SPECS:
        case_id = spec[0]
        assert (CASES_DIR / f"{case_id}.npy").exists()
        assert (CASES_DIR / f"{case_id}.json").exists()


def test_fixture_json_loads_into_clinical_metadata():
    with open(CASES_DIR / "case_001.json") as f:
        raw = json.load(f)
    raw.pop("label")
    meta = ClinicalMetadata(**raw)
    assert meta.case_id == "case_001"


@pytest.mark.parametrize("case_id,expected_decision", [
    ("case_001", "auto"),
    ("case_009", "escalate"),
    ("case_010", "escalate"),
    ("case_014", "escalate"),
])
def test_designed_fixtures_route_as_intended(case_id, expected_decision):
    volume = np.load(CASES_DIR / f"{case_id}.npy")
    with open(CASES_DIR / f"{case_id}.json") as f:
        raw = json.load(f)
    raw.pop("label")
    meta = ClinicalMetadata(**raw)
    result = run_case(volume, meta, tau=0.5)
    assert result.decision == expected_decision


def test_mismatched_fixture_has_higher_uncertainty_than_clean_fixture():
    def _run(case_id):
        volume = np.load(CASES_DIR / f"{case_id}.npy")
        with open(CASES_DIR / f"{case_id}.json") as f:
            raw = json.load(f)
        raw.pop("label")
        meta = ClinicalMetadata(**raw)
        return run_case(volume, meta, tau=0.5)

    clean = _run("case_001")
    mismatched = _run("case_013")
    assert mismatched.uncertainty.value > clean.uncertainty.value
```

- [ ] **Step 4: Run test to verify it fails, then passes**

Run: `cd backend && python -m pytest tests/test_fixtures.py -v`
Expected: If fixtures were not yet generated, this fails on
`test_all_fixture_files_exist`. After Step 2 has been run, all tests PASS
(7 tests). If any `test_designed_fixtures_route_as_intended` case fails,
adjust that case's `radius`/`intensity`/`center` in `CASE_SPECS` (Step 1)
until it routes as intended, then re-run Step 2 and this test.

- [ ] **Step 5: Commit**

```bash
git add backend/mars/data/generate_fixtures.py backend/mars/data/cases backend/tests/test_fixtures.py
git commit -m "feat(backend): generate and commit 15 mock case fixtures"
```

---

## Task 8: FastAPI layer

**Files:**
- Create: `backend/mars/api.py`
- Create: `backend/tests/test_api.py`

**Interfaces:**
- Consumes: `run_case` (Task 6), `CASES_DIR`/`CASE_SPECS` (Task 7),
  `ClinicalMetadata` (Task 1).
- Produces: `app: FastAPI` instance in `backend/mars/api.py`, plus
  `load_case(case_id: str) -> tuple[np.ndarray, ClinicalMetadata, str]`
  (volume, metadata, label) and `list_case_ids() -> list[str]` helper
  functions in the same module — used by the frontend via HTTP only, not
  imported elsewhere.

- [ ] **Step 1: Write the failing test `backend/tests/test_api.py`**

```python
from fastapi.testclient import TestClient
from mars.api import app

client = TestClient(app)


def test_list_cases_returns_all_fixtures():
    response = client.get("/cases")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 15
    assert {"id", "label"} <= set(body[0].keys())


def test_get_volume_returns_slices_matching_shape():
    response = client.get("/cases/case_001/volume")
    assert response.status_code == 200
    body = response.json()
    assert body["shape"] == [16, 32, 32]
    assert len(body["slices"]) == 16


def test_get_volume_unknown_case_returns_404():
    response = client.get("/cases/not_a_real_case/volume")
    assert response.status_code == 404


def test_run_case_default_tau_returns_case_result():
    response = client.get("/cases/case_001/run")
    assert response.status_code == 200
    body = response.json()
    assert body["case_id"] == "case_001"
    assert body["tau"] == 0.5
    assert body["decision"] in ("auto", "escalate")


def test_run_case_custom_tau_changes_decision_for_borderline_case():
    low_tau = client.get("/cases/case_009/run", params={"tau": 0.01}).json()
    high_tau = client.get("/cases/case_009/run", params={"tau": 0.99}).json()
    assert low_tau["decision"] == "escalate"
    assert high_tau["decision"] == "auto"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mars.api'`

- [ ] **Step 3: Write `backend/mars/api.py`**

```python
"""FastAPI layer exposing the MARS pipeline: list mock cases, fetch a case's volume, and run the pipeline with a given tau."""

import json

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from mars.data.generate_fixtures import CASES_DIR, CASE_SPECS
from mars.models import CaseResult, ClinicalMetadata
from mars.pipeline import run_case

app = FastAPI(title="MARS Mock Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_LABELS = {spec[0]: spec[1] for spec in CASE_SPECS}


def list_case_ids() -> list[str]:
    return sorted(p.stem for p in CASES_DIR.glob("*.npy"))


def load_case(case_id: str) -> tuple[np.ndarray, ClinicalMetadata, str]:
    volume_path = CASES_DIR / f"{case_id}.npy"
    meta_path = CASES_DIR / f"{case_id}.json"
    if not volume_path.exists() or not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"case '{case_id}' not found")

    volume = np.load(volume_path)
    with open(meta_path) as f:
        raw = json.load(f)
    label = raw.pop("label", case_id)
    metadata = ClinicalMetadata(**raw)
    return volume, metadata, label


@app.get("/cases")
def list_cases() -> list[dict]:
    return [{"id": case_id, "label": _LABELS.get(case_id, case_id)} for case_id in list_case_ids()]


@app.get("/cases/{case_id}/volume")
def get_volume(case_id: str) -> dict:
    volume, _, _ = load_case(case_id)
    return {"shape": list(volume.shape), "slices": volume.tolist()}


@app.get("/cases/{case_id}/run", response_model=CaseResult)
def run(case_id: str, tau: float = 0.5) -> CaseResult:
    volume, metadata, _ = load_case(case_id)
    return run_case(volume, metadata, tau=tau)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_api.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the full backend test suite**

Run: `cd backend && python -m pytest -v`
Expected: All tests across every module PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/mars/api.py backend/tests/test_api.py
git commit -m "feat(backend): FastAPI layer for cases, volumes, and pipeline runs"
```

---

## Task 9: Frontend scaffold + API client

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/jest.config.ts`
- Create: `frontend/babel.config.cjs`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/api.ts`
- Create: `frontend/src/__tests__/api.test.ts`

**Interfaces:**
- Consumes: the backend API contract from Task 8 (`GET /cases`,
  `GET /cases/{id}/volume`, `GET /cases/{id}/run?tau=`).
- Produces: TypeScript types `CaseSummary`, `VolumePayload`, `CaseResult`
  (mirroring the backend Pydantic models) and functions `fetchCases()`,
  `fetchVolume(caseId)`, `runCase(caseId, tau)` in `frontend/src/api.ts`,
  used by every component task (10-12).

- [ ] **Step 1: Create `frontend/package.json`**

```json
{
  "name": "mars-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "test": "jest"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@babel/preset-env": "^7.24.0",
    "@babel/preset-react": "^7.23.3",
    "@babel/preset-typescript": "^7.23.3",
    "@testing-library/jest-dom": "^6.4.0",
    "@testing-library/react": "^14.2.0",
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@vitejs/plugin-react": "^4.2.0",
    "jest": "^29.7.0",
    "jest-environment-jsdom": "^29.7.0",
    "typescript": "^5.3.0",
    "vite": "^5.1.0"
  }
}
```

- [ ] **Step 2: Create `frontend/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["ES2020", "DOM"],
    "jsx": "react-jsx",
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src"]
}
```

- [ ] **Step 3: Create `frontend/vite.config.ts`**

```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
});
```

- [ ] **Step 4: Create `frontend/babel.config.cjs`**

```javascript
module.exports = {
  presets: [
    ["@babel/preset-env", { targets: { node: "current" } }],
    ["@babel/preset-react", { runtime: "automatic" }],
    "@babel/preset-typescript",
  ],
};
```

- [ ] **Step 5: Create `frontend/jest.config.ts`**

```typescript
import type { Config } from "jest";

const config: Config = {
  testEnvironment: "jsdom",
  setupFilesAfterEnv: ["<rootDir>/src/setupTests.ts"],
  moduleFileExtensions: ["ts", "tsx", "js", "jsx"],
  testMatch: ["<rootDir>/src/__tests__/**/*.test.{ts,tsx}"],
};

export default config;
```

- [ ] **Step 6: Create `frontend/src/setupTests.ts`**

```typescript
import "@testing-library/jest-dom";
```

- [ ] **Step 7: Create `frontend/index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>MARS Mock Pipeline</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 8: Create `frontend/src/main.tsx`**

```tsx
import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

- [ ] **Step 9: Write the failing test `frontend/src/__tests__/api.test.ts`**

```typescript
import { fetchCases, fetchVolume, runCase } from "../api";

describe("api client", () => {
  beforeEach(() => {
    global.fetch = jest.fn();
  });

  it("fetchCases calls /cases and returns parsed JSON", async () => {
    const mockCases = [{ id: "case_001", label: "Clean anterior lesion" }];
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockCases,
    });

    const result = await fetchCases();

    expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining("/cases"));
    expect(result).toEqual(mockCases);
  });

  it("fetchVolume calls /cases/{id}/volume", async () => {
    const mockVolume = { shape: [2, 2, 2], slices: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] };
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockVolume,
    });

    const result = await fetchVolume("case_001");

    expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining("/cases/case_001/volume"));
    expect(result).toEqual(mockVolume);
  });

  it("runCase calls /cases/{id}/run with tau query param", async () => {
    const mockResult = { case_id: "case_001", tau: 0.7, decision: "auto" };
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockResult,
    });

    const result = await runCase("case_001", 0.7);

    expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining("/cases/case_001/run?tau=0.7"));
    expect(result).toEqual(mockResult);
  });
});
```

- [ ] **Step 10: Run test to verify it fails**

Run: `cd frontend && npm install && npm test`
Expected: FAIL with `Cannot find module '../api'`

- [ ] **Step 11: Write `frontend/src/api.ts`**

```typescript
export const API_BASE = "http://localhost:8000";

export interface CaseSummary {
  id: string;
  label: string;
}

export interface VolumePayload {
  shape: [number, number, number];
  slices: number[][][];
}

export interface SegmentationResult {
  mask: number[][][];
  per_voxel_confidence: number[][][];
  aggregate_confidence: number;
}

export interface ValidationResult {
  consistency_score: number;
  conflicts: string[];
}

export interface UncertaintyResult {
  value: number;
  dominant_factor: string;
  breakdown: { confidence_term: number; consistency_term: number };
}

export interface Report {
  case_id: string;
  findings: string;
  measurements: Record<string, unknown>;
  confidence: number;
  narrative: string;
}

export interface Escalation {
  case_id: string;
  uncertainty: number;
  reasons: string[];
}

export interface CaseResult {
  case_id: string;
  tau: number;
  segmentation: SegmentationResult;
  validation: ValidationResult;
  uncertainty: UncertaintyResult;
  decision: "auto" | "escalate";
  report: Report | null;
  escalation: Escalation | null;
}

export async function fetchCases(): Promise<CaseSummary[]> {
  const response = await fetch(`${API_BASE}/cases`);
  return response.json();
}

export async function fetchVolume(caseId: string): Promise<VolumePayload> {
  const response = await fetch(`${API_BASE}/cases/${caseId}/volume`);
  return response.json();
}

export async function runCase(caseId: string, tau: number): Promise<CaseResult> {
  const response = await fetch(`${API_BASE}/cases/${caseId}/run?tau=${tau}`);
  return response.json();
}
```

- [ ] **Step 12: Run test to verify it passes**

Run: `cd frontend && npm test`
Expected: PASS (3 tests). Note: `App.tsx` does not exist yet, so `npm run dev`/`build` will fail until Task 12 — that's expected at this point.

- [ ] **Step 13: Commit**

```bash
git add frontend/package.json frontend/tsconfig.json frontend/vite.config.ts frontend/jest.config.ts frontend/babel.config.cjs frontend/index.html frontend/src/main.tsx frontend/src/api.ts frontend/src/setupTests.ts frontend/src/__tests__/api.test.ts
git commit -m "feat(frontend): scaffold Vite/React/Jest project and typed API client"
```

---

## Task 10: SliceViewer and ConfidenceBadge components

**Files:**
- Create: `frontend/src/components/SliceViewer.tsx`
- Create: `frontend/src/components/ConfidenceBadge.tsx`
- Create: `frontend/src/__tests__/SliceViewer.test.tsx`
- Create: `frontend/src/__tests__/ConfidenceBadge.test.tsx`

**Interfaces:**
- Consumes: `VolumePayload` type, `SegmentationResult` type (Task 9).
- Produces: `SliceViewer({ volume, mask }: { volume: VolumePayload; mask: number[][][] })`
  and `ConfidenceBadge({ label, value }: { label: string; value: number })`
  React components, used by Task 12 (`App.tsx`).

- [ ] **Step 1: Write the failing test `frontend/src/__tests__/ConfidenceBadge.test.tsx`**

```tsx
import { render, screen } from "@testing-library/react";
import { ConfidenceBadge } from "../components/ConfidenceBadge";

describe("ConfidenceBadge", () => {
  it("renders the label and percentage", () => {
    render(<ConfidenceBadge label="Confidence" value={0.87} />);
    expect(screen.getByText(/Confidence/)).toBeInTheDocument();
    expect(screen.getByText(/87%/)).toBeInTheDocument();
  });

  it("applies a low-value style hint when value is below 0.5", () => {
    render(<ConfidenceBadge label="Confidence" value={0.2} />);
    expect(screen.getByTestId("confidence-badge")).toHaveAttribute("data-level", "low");
  });

  it("applies a high-value style hint when value is at or above 0.5", () => {
    render(<ConfidenceBadge label="Confidence" value={0.6} />);
    expect(screen.getByTestId("confidence-badge")).toHaveAttribute("data-level", "high");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- ConfidenceBadge`
Expected: FAIL with `Cannot find module '../components/ConfidenceBadge'`

- [ ] **Step 3: Write `frontend/src/components/ConfidenceBadge.tsx`**

```tsx
export function ConfidenceBadge({ label, value }: { label: string; value: number }) {
  const level = value >= 0.5 ? "high" : "low";
  return (
    <span data-testid="confidence-badge" data-level={level}>
      {label}: {(value * 100).toFixed(0)}%
    </span>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test -- ConfidenceBadge`
Expected: PASS (3 tests)

- [ ] **Step 5: Write the failing test `frontend/src/__tests__/SliceViewer.test.tsx`**

```tsx
import { render, screen, fireEvent } from "@testing-library/react";
import { SliceViewer } from "../components/SliceViewer";
import type { VolumePayload } from "../api";

const volume: VolumePayload = {
  shape: [3, 2, 2],
  slices: [
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]],
    [[9, 10], [11, 12]],
  ],
};

const mask = [
  [[0, 0], [0, 0]],
  [[0, 1], [0, 0]],
  [[0, 0], [0, 0]],
];

describe("SliceViewer", () => {
  it("renders a depth slider ranging over the number of slices", () => {
    render(<SliceViewer volume={volume} mask={mask} />);
    const slider = screen.getByLabelText(/slice/i) as HTMLInputElement;
    expect(slider.min).toBe("0");
    expect(slider.max).toBe("2");
  });

  it("shows the current slice index", () => {
    render(<SliceViewer volume={volume} mask={mask} />);
    expect(screen.getByText(/slice 0 \/ 2/i)).toBeInTheDocument();
  });

  it("updates the displayed slice index when the slider changes", () => {
    render(<SliceViewer volume={volume} mask={mask} />);
    const slider = screen.getByLabelText(/slice/i);
    fireEvent.change(slider, { target: { value: "1" } });
    expect(screen.getByText(/slice 1 \/ 2/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd frontend && npm test -- SliceViewer`
Expected: FAIL with `Cannot find module '../components/SliceViewer'`

- [ ] **Step 7: Write `frontend/src/components/SliceViewer.tsx`**

```tsx
import { useState } from "react";
import type { VolumePayload } from "../api";

export function SliceViewer({ volume, mask }: { volume: VolumePayload; mask: number[][][] }) {
  const [sliceIndex, setSliceIndex] = useState(0);
  const depth = volume.shape[0];
  const slice = volume.slices[sliceIndex];
  const maskSlice = mask[sliceIndex];

  return (
    <div>
      <label htmlFor="slice-slider">
        Slice
        <input
          id="slice-slider"
          aria-label="slice"
          type="range"
          min={0}
          max={depth - 1}
          value={sliceIndex}
          onChange={(e) => setSliceIndex(Number(e.target.value))}
        />
      </label>
      <p>
        Slice {sliceIndex} / {depth - 1}
      </p>
      <svg width={slice[0].length * 10} height={slice.length * 10} data-testid="slice-svg">
        {slice.map((row, y) =>
          row.map((value, x) => {
            const isMasked = maskSlice?.[y]?.[x] === 1;
            const gray = Math.min(255, Math.max(0, Math.round(value)));
            return (
              <rect
                key={`${y}-${x}`}
                x={x * 10}
                y={y * 10}
                width={10}
                height={10}
                fill={isMasked ? "rgba(255,0,0,0.5)" : `rgb(${gray},${gray},${gray})`}
              />
            );
          })
        )}
      </svg>
    </div>
  );
}
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cd frontend && npm test -- SliceViewer`
Expected: PASS (3 tests)

- [ ] **Step 9: Commit**

```bash
git add frontend/src/components/SliceViewer.tsx frontend/src/components/ConfidenceBadge.tsx frontend/src/__tests__/SliceViewer.test.tsx frontend/src/__tests__/ConfidenceBadge.test.tsx
git commit -m "feat(frontend): slice viewer with mask overlay and confidence badge"
```

---

## Task 11: EscalationPanel, ReportView, TauSlider components

**Files:**
- Create: `frontend/src/components/EscalationPanel.tsx`
- Create: `frontend/src/components/ReportView.tsx`
- Create: `frontend/src/components/TauSlider.tsx`
- Create: `frontend/src/__tests__/EscalationPanel.test.tsx`
- Create: `frontend/src/__tests__/ReportView.test.tsx`
- Create: `frontend/src/__tests__/TauSlider.test.tsx`

**Interfaces:**
- Consumes: `Report`, `Escalation` types (Task 9).
- Produces: `EscalationPanel({ escalation }: { escalation: Escalation })`,
  `ReportView({ report }: { report: Report })`,
  `TauSlider({ tau, onChange }: { tau: number; onChange: (tau: number) => void })`
  components, used by Task 12 (`App.tsx`).

- [ ] **Step 1: Write the failing test `frontend/src/__tests__/EscalationPanel.test.tsx`**

```tsx
import { render, screen } from "@testing-library/react";
import { EscalationPanel } from "../components/EscalationPanel";
import type { Escalation } from "../api";

const escalation: Escalation = {
  case_id: "case_009",
  uncertainty: 0.72,
  reasons: ["mask centroid outside expected site 'anterior'", "uncertainty 0.72 >= threshold 0.50"],
};

describe("EscalationPanel", () => {
  it("shows the uncertainty value and every reason", () => {
    render(<EscalationPanel escalation={escalation} />);
    expect(screen.getByText(/0.72/)).toBeInTheDocument();
    expect(screen.getByText(/outside expected site/)).toBeInTheDocument();
    expect(screen.getByText(/uncertainty 0.72 >= threshold 0.50/)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- EscalationPanel`
Expected: FAIL with `Cannot find module '../components/EscalationPanel'`

- [ ] **Step 3: Write `frontend/src/components/EscalationPanel.tsx`**

```tsx
import type { Escalation } from "../api";

export function EscalationPanel({ escalation }: { escalation: Escalation }) {
  return (
    <div data-testid="escalation-panel">
      <h3>Escalated to radiologist</h3>
      <p>Uncertainty: {escalation.uncertainty.toFixed(2)}</p>
      <ul>
        {escalation.reasons.map((reason, i) => (
          <li key={i}>{reason}</li>
        ))}
      </ul>
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test -- EscalationPanel`
Expected: PASS (1 test)

- [ ] **Step 5: Write the failing test `frontend/src/__tests__/ReportView.test.tsx`**

```tsx
import { render, screen } from "@testing-library/react";
import { ReportView } from "../components/ReportView";
import type { Report } from "../api";

const report: Report = {
  case_id: "case_001",
  findings: "Lesion detected and cleared automatically.",
  measurements: { voxel_count: 42 },
  confidence: 0.91,
  narrative: "Automated analysis identified a lesion with 91% confidence.",
};

describe("ReportView", () => {
  it("shows findings, confidence, and narrative", () => {
    render(<ReportView report={report} />);
    expect(screen.getByText(/Lesion detected and cleared automatically/)).toBeInTheDocument();
    expect(screen.getByText(/91%/)).toBeInTheDocument();
    expect(screen.getByText(/Automated analysis identified/)).toBeInTheDocument();
  });
});
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd frontend && npm test -- ReportView`
Expected: FAIL with `Cannot find module '../components/ReportView'`

- [ ] **Step 7: Write `frontend/src/components/ReportView.tsx`**

```tsx
import type { Report } from "../api";

export function ReportView({ report }: { report: Report }) {
  return (
    <div data-testid="report-view">
      <h3>Automated report</h3>
      <p>{report.findings}</p>
      <p>Confidence: {(report.confidence * 100).toFixed(0)}%</p>
      <p>{report.narrative}</p>
    </div>
  );
}
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cd frontend && npm test -- ReportView`
Expected: PASS (1 test)

- [ ] **Step 9: Write the failing test `frontend/src/__tests__/TauSlider.test.tsx`**

```tsx
import { render, screen, fireEvent } from "@testing-library/react";
import { TauSlider } from "../components/TauSlider";

describe("TauSlider", () => {
  it("displays the current tau value", () => {
    render(<TauSlider tau={0.5} onChange={jest.fn()} />);
    expect(screen.getByText(/0.50/)).toBeInTheDocument();
  });

  it("calls onChange with the new tau when moved", () => {
    const handleChange = jest.fn();
    render(<TauSlider tau={0.5} onChange={handleChange} />);
    const slider = screen.getByLabelText(/threshold/i);
    fireEvent.change(slider, { target: { value: "0.8" } });
    expect(handleChange).toHaveBeenCalledWith(0.8);
  });
});
```

- [ ] **Step 10: Run test to verify it fails**

Run: `cd frontend && npm test -- TauSlider`
Expected: FAIL with `Cannot find module '../components/TauSlider'`

- [ ] **Step 11: Write `frontend/src/components/TauSlider.tsx`**

```tsx
export function TauSlider({ tau, onChange }: { tau: number; onChange: (tau: number) => void }) {
  return (
    <label htmlFor="tau-slider">
      Threshold (τ): {tau.toFixed(2)}
      <input
        id="tau-slider"
        aria-label="threshold"
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={tau}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}
```

- [ ] **Step 12: Run test to verify it passes**

Run: `cd frontend && npm test -- TauSlider`
Expected: PASS (2 tests)

- [ ] **Step 13: Commit**

```bash
git add frontend/src/components/EscalationPanel.tsx frontend/src/components/ReportView.tsx frontend/src/components/TauSlider.tsx frontend/src/__tests__/EscalationPanel.test.tsx frontend/src/__tests__/ReportView.test.tsx frontend/src/__tests__/TauSlider.test.tsx
git commit -m "feat(frontend): escalation panel, report view, and tau slider"
```

---

## Task 12: CaseList component + App wiring

**Files:**
- Create: `frontend/src/components/CaseList.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/__tests__/CaseList.test.tsx`
- Create: `frontend/src/__tests__/App.test.tsx`

**Interfaces:**
- Consumes: `CaseSummary`, `fetchCases`, `fetchVolume`, `runCase` (Task 9);
  `SliceViewer`, `ConfidenceBadge` (Task 10); `EscalationPanel`,
  `ReportView`, `TauSlider` (Task 11).
- Produces: `CaseList({ cases, selectedId, onSelect }: { cases: CaseSummary[]; selectedId: string | null; onSelect: (id: string) => void })`
  and the top-level `App()` component in `frontend/src/App.tsx`. Nothing
  downstream consumes these — this is the final task.

- [ ] **Step 1: Write the failing test `frontend/src/__tests__/CaseList.test.tsx`**

```tsx
import { render, screen, fireEvent } from "@testing-library/react";
import { CaseList } from "../components/CaseList";
import type { CaseSummary } from "../api";

const cases: CaseSummary[] = [
  { id: "case_001", label: "Clean anterior lesion" },
  { id: "case_009", label: "Site mismatch" },
];

describe("CaseList", () => {
  it("renders every case label", () => {
    render(<CaseList cases={cases} selectedId={null} onSelect={jest.fn()} />);
    expect(screen.getByText(/Clean anterior lesion/)).toBeInTheDocument();
    expect(screen.getByText(/Site mismatch/)).toBeInTheDocument();
  });

  it("calls onSelect with the case id when clicked", () => {
    const handleSelect = jest.fn();
    render(<CaseList cases={cases} selectedId={null} onSelect={handleSelect} />);
    fireEvent.click(screen.getByText(/Site mismatch/));
    expect(handleSelect).toHaveBeenCalledWith("case_009");
  });

  it("marks the selected case", () => {
    render(<CaseList cases={cases} selectedId="case_001" onSelect={jest.fn()} />);
    expect(screen.getByTestId("case-case_001")).toHaveAttribute("aria-selected", "true");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- CaseList`
Expected: FAIL with `Cannot find module '../components/CaseList'`

- [ ] **Step 3: Write `frontend/src/components/CaseList.tsx`**

```tsx
import type { CaseSummary } from "../api";

export function CaseList({
  cases,
  selectedId,
  onSelect,
}: {
  cases: CaseSummary[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <ul>
      {cases.map((c) => (
        <li
          key={c.id}
          data-testid={`case-${c.id}`}
          aria-selected={c.id === selectedId}
          onClick={() => onSelect(c.id)}
          style={{ cursor: "pointer", fontWeight: c.id === selectedId ? "bold" : "normal" }}
        >
          {c.label}
        </li>
      ))}
    </ul>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test -- CaseList`
Expected: PASS (3 tests)

- [ ] **Step 5: Write the failing test `frontend/src/__tests__/App.test.tsx`**

```tsx
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { App } from "../App";

const mockCases = [{ id: "case_001", label: "Clean anterior lesion" }];
const mockVolume = { shape: [1, 1, 1], slices: [[[100]]] };
const mockAutoResult = {
  case_id: "case_001",
  tau: 0.5,
  segmentation: { mask: [[[1]]], per_voxel_confidence: [[[0.9]]], aggregate_confidence: 0.9 },
  validation: { consistency_score: 0.95, conflicts: [] },
  uncertainty: { value: 0.1, dominant_factor: "confidence", breakdown: { confidence_term: 0.05, consistency_term: 0.05 } },
  decision: "auto",
  report: {
    case_id: "case_001",
    findings: "Lesion detected and cleared automatically.",
    measurements: { voxel_count: 1 },
    confidence: 0.9,
    narrative: "Automated analysis identified a lesion.",
  },
  escalation: null,
};

describe("App", () => {
  beforeEach(() => {
    global.fetch = jest.fn((url: string) => {
      if (url.includes("/cases") && !url.includes("/run") && !url.includes("/volume")) {
        return Promise.resolve({ ok: true, json: async () => mockCases } as Response);
      }
      if (url.includes("/volume")) {
        return Promise.resolve({ ok: true, json: async () => mockVolume } as Response);
      }
      if (url.includes("/run")) {
        return Promise.resolve({ ok: true, json: async () => mockAutoResult } as Response);
      }
      return Promise.reject(new Error(`unexpected url ${url}`));
    }) as jest.Mock;
  });

  it("loads cases, selects the first case's data, and shows the report for an auto-cleared case", async () => {
    render(<App />);

    await waitFor(() => expect(screen.getByText(/Clean anterior lesion/)).toBeInTheDocument());

    fireEvent.click(screen.getByText(/Clean anterior lesion/));

    await waitFor(() => expect(screen.getByTestId("report-view")).toBeInTheDocument());
    expect(screen.queryByTestId("escalation-panel")).not.toBeInTheDocument();
  });
});
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd frontend && npm test -- App`
Expected: FAIL with `Cannot find module '../App'`

- [ ] **Step 7: Write `frontend/src/App.tsx`**

```tsx
import { useEffect, useState } from "react";
import { fetchCases, fetchVolume, runCase } from "./api";
import type { CaseResult, CaseSummary, VolumePayload } from "./api";
import { CaseList } from "./components/CaseList";
import { SliceViewer } from "./components/SliceViewer";
import { ConfidenceBadge } from "./components/ConfidenceBadge";
import { TauSlider } from "./components/TauSlider";
import { ReportView } from "./components/ReportView";
import { EscalationPanel } from "./components/EscalationPanel";

export function App() {
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [volume, setVolume] = useState<VolumePayload | null>(null);
  const [result, setResult] = useState<CaseResult | null>(null);
  const [tau, setTau] = useState(0.5);

  useEffect(() => {
    fetchCases().then(setCases);
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    fetchVolume(selectedId).then(setVolume);
    runCase(selectedId, tau).then(setResult);
  }, [selectedId, tau]);

  return (
    <div>
      <h1>MARS Mock Pipeline</h1>
      <CaseList cases={cases} selectedId={selectedId} onSelect={setSelectedId} />
      {volume && result && (
        <div>
          <SliceViewer volume={volume} mask={result.segmentation.mask} />
          <ConfidenceBadge label="Segmentation confidence" value={result.segmentation.aggregate_confidence} />
          <ConfidenceBadge label="Clinical consistency" value={result.validation.consistency_score} />
          <TauSlider tau={tau} onChange={setTau} />
          {result.decision === "auto" && result.report && <ReportView report={result.report} />}
          {result.decision === "escalate" && result.escalation && <EscalationPanel escalation={result.escalation} />}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cd frontend && npm test -- App`
Expected: PASS (1 test)

- [ ] **Step 9: Run the full frontend test suite**

Run: `cd frontend && npm test`
Expected: All tests across every component PASS.

- [ ] **Step 10: Commit**

```bash
git add frontend/src/components/CaseList.tsx frontend/src/App.tsx frontend/src/__tests__/CaseList.test.tsx frontend/src/__tests__/App.test.tsx
git commit -m "feat(frontend): case list and top-level App wiring the full flow"
```

---

## Task 13: AGENT.md and README wiring

**Files:**
- Create: `AGENT.md`
- Create: `README.md`

**Interfaces:**
- Consumes: nothing (documentation only, written after the system exists so
  commands and paths are verified against the real repo).
- Produces: onboarding docs at the repo root.

- [ ] **Step 1: Verify the backend runs end-to-end**

Run: `cd backend && python -m uvicorn mars.api:app --reload --port 8000 &` then
`curl http://localhost:8000/cases | head -c 200` then stop the server.
Expected: JSON array of 15 cases is returned.

- [ ] **Step 2: Verify the frontend runs end-to-end**

Run: `cd frontend && npm run dev &`, open `http://localhost:5173` in a
browser (or `curl` it), then stop the server.
Expected: page loads without a build error.

- [ ] **Step 3: Write `AGENT.md`**

```markdown
# AGENT.md — MARS Mock Pipeline

This repo contains a mock, fully-working implementation of the MARS
4-agent pipeline described in `requirements.md`. It uses synthetic data
and simple algorithms (not trained ML models) so the *architecture,
interfaces, and routing logic* are real and testable end-to-end, per
`docs/superpowers/specs/2026-09-03-mars-mock-pipeline-design.md`.

## Architecture

```
CT volume + metadata
   -> Segmentation Agent   (backend/mars/agents/segmentation.py)
   -> Validation Agent     (backend/mars/agents/validation.py)
   -> Uncertainty Agent    (backend/mars/agents/uncertainty.py)
   -> Reporting Agent      (backend/mars/agents/reporting.py)
   -> CaseResult (report or escalation)
```

`backend/mars/pipeline.py` orchestrates the four agents in sequence.
`backend/mars/api.py` exposes them over HTTP. `frontend/` is a React app
that lists cases, renders a slice viewer with mask overlay, and shows the
report or escalation for the selected case at the current τ.

## Requirements-to-code map

| requirements.md section | Code |
|---|---|
| §3.2 Segmentation Agent | `backend/mars/agents/segmentation.py` |
| §3.3 Validation Agent | `backend/mars/agents/validation.py` |
| §3.4 Uncertainty Agent | `backend/mars/agents/uncertainty.py` |
| §3.5 Reporting Agent | `backend/mars/agents/reporting.py` |
| FR-6 (threshold routing) | `backend/mars/agents/reporting.py::build_case_result`, `tau` query param in `backend/mars/api.py` |
| FR-9 (mask overlay UI) | `frontend/src/components/SliceViewer.tsx` |
| NFR-6 (configurable τ) | `frontend/src/components/TauSlider.tsx` + `GET /cases/{id}/run?tau=` |

## Mock data

15 synthetic cases live in `backend/mars/data/cases/` (`.npy` volume +
`.json` metadata each), generated by `backend/mars/data/generate_fixtures.py`
from the `CASE_SPECS` list in that file. They're deliberately designed to
cover: clean auto-clearing cases, low-confidence cases, and cases with a
site/size mismatch against their metadata (these should escalate). To
regenerate: `cd backend && python -m mars.data.generate_fixtures`.

## Running it

Backend:
```bash
cd backend
pip install -e ".[dev]"
python -m uvicorn mars.api:app --reload --port 8000
```

Frontend (in a second terminal):
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Running tests

```bash
cd backend && python -m pytest -v
cd frontend && npm test
```

## Extending this toward the real system

- Replace `agents/segmentation.py`'s thresholding with a real SAM/nnU-Net
  inference call, keeping the same `SegmentationResult` return type.
- Replace `agents/validation.py`'s two-region `SITES` map with real
  anatomical-region lookup against DICOM/NIfTI spatial metadata.
- Replace `agents/uncertainty.py`'s fixed-weight sum with a learned/
  calibrated aggregator — evaluate with Expected Calibration Error and
  risk-coverage curves per requirements.md §6.
- Each agent is independently replaceable and independently testable
  (NFR-4) as long as it keeps producing the same Pydantic model shape
  from `backend/mars/models.py`.
```

- [ ] **Step 4: Write `README.md`**

```markdown
# MARS — Mock Pipeline

See `AGENT.md` for architecture, running instructions, and the
requirements-to-code map. See `requirements.md` for the full project
requirements and `docs/superpowers/specs/` for the design spec this
implementation follows.
```

- [ ] **Step 5: Commit**

```bash
git add AGENT.md README.md
git commit -m "docs: add AGENT.md architecture guide and README"
```
