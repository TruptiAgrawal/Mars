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
