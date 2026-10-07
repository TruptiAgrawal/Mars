import { render, screen } from "@testing-library/react";
import { AgentPipeline } from "../components/AgentPipeline";
import type { CaseResult } from "../api";

const autoResult: CaseResult = {
  case_id: "case_001",
  tau: 0.5,
  segmentation: { mask: [[[1]]], per_voxel_confidence: [[[0.9]]], aggregate_confidence: 0.9 },
  validation: { consistency_score: 0.95, conflicts: [] },
  uncertainty: { value: 0.1, dominant_factor: "confidence", breakdown: { confidence_term: 0.05, consistency_term: 0.05 } },
  decision: "auto",
  report: null,
  escalation: null,
};

const escalateResult: CaseResult = {
  ...autoResult,
  validation: { consistency_score: 0.5, conflicts: ["site mismatch"] },
  decision: "escalate",
};

describe("AgentPipeline", () => {
  it("shows all four agent stages with their key metric for an auto-cleared case", () => {
    render(<AgentPipeline result={autoResult} />);
    expect(screen.getByText("Segmentation")).toBeInTheDocument();
    expect(screen.getByText("Validation")).toBeInTheDocument();
    expect(screen.getByText("Uncertainty")).toBeInTheDocument();
    expect(screen.getByText("Reporting")).toBeInTheDocument();
    expect(screen.getByText("90%")).toBeInTheDocument();
    expect(screen.getByText("95%")).toBeInTheDocument();
    expect(screen.getByText("0.10")).toBeInTheDocument();
    expect(screen.getByText("no conflicts")).toBeInTheDocument();
    expect(screen.getByText("auto")).toBeInTheDocument();
  });

  it("shows the conflict count and escalate route for an escalated case", () => {
    render(<AgentPipeline result={escalateResult} />);
    expect(screen.getByText("1 conflict")).toBeInTheDocument();
    expect(screen.getByText("escalate")).toBeInTheDocument();
    expect(screen.getByText("radiologist review")).toBeInTheDocument();
  });
});
