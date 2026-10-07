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
