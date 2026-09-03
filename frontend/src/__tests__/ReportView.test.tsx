import { render, screen } from "@testing-library/react";
import { ReportView } from "../components/ReportView";
import type { Report } from "../api";

const report: Report = {
  case_id: "case_001",
  findings: "Lesion detected and cleared automatically.",
  measurements: { voxel_count: 42 },
  confidence: 0.91,
  narrative: "Automated analysis identified a lesion with 91% confidence.",
};

describe("ReportView", () => {
  it("shows findings, confidence, and narrative", () => {
    render(<ReportView report={report} />);
    expect(screen.getByText(/Lesion detected and cleared automatically/)).toBeInTheDocument();
    expect(screen.getByText(/Confidence:.*91%/)).toBeInTheDocument();
    expect(screen.getByText(/Automated analysis identified/)).toBeInTheDocument();
  });
});
