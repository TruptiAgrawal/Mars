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
