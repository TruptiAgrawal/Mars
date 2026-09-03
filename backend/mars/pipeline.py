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
