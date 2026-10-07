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
