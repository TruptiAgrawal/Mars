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
