"""Pydantic schemas shared across all MARS agents and the API layer."""

from typing import Optional
from pydantic import BaseModel


class ClinicalMetadata(BaseModel):
    case_id: str
    expected_site: str
    expected_size_range_mm3: tuple[float, float]
    prior_history: str


class SegmentationResult(BaseModel):
    mask: list
    per_voxel_confidence: list
    aggregate_confidence: float


class ValidationResult(BaseModel):
    consistency_score: float
    conflicts: list[str]


class UncertaintyResult(BaseModel):
    value: float
    dominant_factor: str
    breakdown: dict[str, float]


class Report(BaseModel):
    case_id: str
    findings: str
    measurements: dict
    confidence: float
    narrative: str


class Escalation(BaseModel):
    case_id: str
    uncertainty: float
    reasons: list[str]


class CaseResult(BaseModel):
    case_id: str
    tau: float
    segmentation: SegmentationResult
    validation: ValidationResult
    uncertainty: UncertaintyResult
    decision: str
    report: Optional[Report] = None
    escalation: Optional[Escalation] = None
