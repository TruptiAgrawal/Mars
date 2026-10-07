export const API_BASE = "http://localhost:8000";

export interface CaseSummary {
  id: string;
  label: string;
}

export interface VolumePayload {
  shape: [number, number, number];
  slices: number[][][];
}

export interface SegmentationResult {
  mask: number[][][];
  per_voxel_confidence: number[][][];
  aggregate_confidence: number;
}

export interface ValidationResult {
  consistency_score: number;
  conflicts: string[];
}

export interface UncertaintyResult {
  value: number;
  dominant_factor: string;
  breakdown: { confidence_term: number; consistency_term: number };
}

export interface Report {
  case_id: string;
  findings: string;
  measurements: Record<string, unknown>;
  confidence: number;
  narrative: string;
}

export interface Escalation {
  case_id: string;
  uncertainty: number;
  reasons: string[];
}

export interface CaseResult {
  case_id: string;
  tau: number;
  segmentation: SegmentationResult;
  validation: ValidationResult;
  uncertainty: UncertaintyResult;
  decision: "auto" | "escalate";
  report: Report | null;
  escalation: Escalation | null;
}

export async function fetchCases(): Promise<CaseSummary[]> {
  const response = await fetch(`${API_BASE}/cases`);
  return response.json();
}

export async function fetchVolume(caseId: string): Promise<VolumePayload> {
  const response = await fetch(`${API_BASE}/cases/${caseId}/volume`);
  return response.json();
}

export async function runCase(caseId: string, tau: number): Promise<CaseResult> {
  const response = await fetch(`${API_BASE}/cases/${caseId}/run?tau=${tau}`);
  return response.json();
}
