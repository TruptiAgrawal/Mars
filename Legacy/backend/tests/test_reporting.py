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
