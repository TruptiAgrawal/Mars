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
