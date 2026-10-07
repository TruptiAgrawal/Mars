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
