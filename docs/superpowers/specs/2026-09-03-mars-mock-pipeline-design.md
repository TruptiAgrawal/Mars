# MARS Mock Pipeline — Design Spec

**Status:** Approved
**Date:** 2026-09-03
**Scope:** A simple, fully working (non-ML, mock-data) version of the MARS 4-agent
pipeline described in `requirements.md`, built test-first (TDD), with a FastAPI
backend and a React frontend.

## Goal

Demonstrate the MARS architecture end-to-end — segmentation → validation →
uncertainty → routing/reporting — on synthetic data, so the pipeline's shape,
interfaces, and routing logic are real and testable, even though no trained
ML model is involved yet.

## Non-goals (explicitly out of scope for this version)

- Real SAM / nnU-Net / PyTorch models
- Real DICOM/NIfTI I/O
- Auth, persistence/database (cases served from static fixture files)
- Deployment, PACS integration, regulatory concerns
- Calibration study with real statistical rigor (NFR-1 is illustrated, not proven)

## Repository layout

```
mars_agent/
├── backend/
│   ├── mars/
│   │   ├── agents/
│   │   │   ├── segmentation.py   # Agent 1
│   │   │   ├── validation.py     # Agent 2
│   │   │   ├── uncertainty.py    # Agent 3
│   │   │   └── reporting.py      # Agent 4
│   │   ├── pipeline.py           # orchestrates all 4 agents + routing
│   │   ├── data/
│   │   │   ├── synthetic.py      # generates mock 3D volumes w/ embedded "lesion" blobs
│   │   │   └── cases/            # ~15 hand-crafted case fixtures (.npy + .json)
│   │   ├── models.py             # dataclasses/pydantic schemas
│   │   └── api.py                # FastAPI app exposing the pipeline
│   ├── tests/                    # pytest, one test module per agent + pipeline + api
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── components/ (SliceViewer, ConfidenceBadge, EscalationPanel, ReportView, CaseList, TauSlider)
│   │   ├── api.ts
│   │   └── App.tsx
│   ├── src/__tests__/            # Jest + React Testing Library
│   └── package.json
├── AGENT.md
├── docs/superpowers/specs/       # this file
└── requirements.md
```

## Agent logic

- **Segmentation Agent** (`agents/segmentation.py`): input is a synthetic 3D
  numpy volume (Gaussian-blob "lesion" + noise). Segmentation = intensity
  thresholding + connected-component labeling (`scipy.ndimage`). Per-voxel
  confidence = normalized distance from the threshold boundary; aggregate
  confidence = mean confidence over masked voxels.
- **Validation Agent** (`agents/validation.py`): rule-based checks of the
  predicted mask (centroid location, voxel volume) against JSON metadata
  (`expected_site`, `expected_size_range_mm3`, `prior_history`) → consistency
  score in `[0,1]` + list of specific conflict strings.
- **Uncertainty Agent** (`agents/uncertainty.py`): `U = w1*(1-confidence) +
  w2*(1-consistency)` with fixed weights (`w1=0.5, w2=0.5` by default,
  configurable), plus an attribution of which term dominated.
- **Reporting Agent** (`agents/reporting.py`): if `U < τ` → structured report
  dict (findings, measurements, confidence, narrative string). If `U ≥ τ` →
  escalation dict (mask ref, `U`, attributed reasons, consistency conflicts).

## Data flow

`SyntheticCase (.npy + .json)` → `pipeline.run_case(case, tau)` → calls the 4
agents in sequence → returns a single `CaseResult` → API serializes it →
React fetches and renders.

## API (FastAPI)

- `GET /cases` — list available mock cases (id, label)
- `GET /cases/{id}/volume` — raw slices (list of 2D arrays) for the viewer
- `GET /cases/{id}/run?tau=0.5` — runs the full pipeline for the case,
  returns mask, confidence, consistency, `U`, routing decision, and either
  the report or the escalation payload

`τ` defaults to `0.5` and is overridable per-request via query param, per
NFR-6 (configurable without retraining/restart).

## Frontend

Case list → select case → 2D slice viewer (canvas, depth slider) with mask
overlay → confidence/uncertainty display → τ slider (re-runs `/run` on
change) → conditionally renders `ReportView` (auto-cleared) or
`EscalationPanel` (escalated, showing the reason).

## Testing strategy (TDD on both sides)

**Backend** — pytest, test-first per agent:
- Deterministic synthetic fixtures with a fixed random seed
- Per-agent unit tests (segmentation on a known blob, validation against
  matching/mismatching metadata, uncertainty weighting/attribution,
  reporting routing at the threshold boundary)
- Pipeline integration tests (full case → `CaseResult`)
- API tests via FastAPI `TestClient`
- One calibration-flavored test: a fixture deliberately constructed to be
  anatomically implausible should produce higher `U` than a clean fixture —
  illustrates NFR-1 without a full calibration study

**Frontend** — Jest + React Testing Library, test-first per component:
- `SliceViewer` renders slices and mask overlay
- `EscalationPanel` shows only when `U ≥ τ`; `ReportView` shows only when
  `U < τ`
- `TauSlider` triggers a refetch with the new `τ`
- `CaseList` renders cases and handles selection

## Documentation

- Every backend module gets a one-line module docstring stating its single
  responsibility (NFR-2/NFR-4 in miniature)
- `AGENT.md` at repo root: architecture overview, how to run
  pipeline/tests/dev servers, and a mapping from each agent back to the
  relevant `requirements.md` section — written as onboarding for a human or
  another Claude session working in this repo

## Mock dataset

~15 hand-crafted synthetic cases in `backend/mars/data/cases/`, each a
`.npy` volume (small, e.g. 32×64×64) + a `.json` metadata sidecar. Mix of:
clean/consistent cases (should auto-clear), low-confidence cases (small/
low-contrast blob), and clinically-inconsistent cases (mask generated in a
location that conflicts with metadata) — the U≥τ paths need actual designed
fixtures, not just random noise, so tests are meaningful.
