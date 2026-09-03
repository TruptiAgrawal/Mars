import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { App } from "../App";

const mockCases = [{ id: "case_001", label: "Clean anterior lesion" }];
const mockVolume = { shape: [1, 1, 1], slices: [[[100]]] };
const mockAutoResult = {
  case_id: "case_001",
  tau: 0.5,
  segmentation: { mask: [[[1]]], per_voxel_confidence: [[[0.9]]], aggregate_confidence: 0.9 },
  validation: { consistency_score: 0.95, conflicts: [] },
  uncertainty: { value: 0.1, dominant_factor: "confidence", breakdown: { confidence_term: 0.05, consistency_term: 0.05 } },
  decision: "auto",
  report: {
    case_id: "case_001",
    findings: "Lesion detected and cleared automatically.",
    measurements: { voxel_count: 1 },
    confidence: 0.9,
    narrative: "Automated analysis identified a lesion.",
  },
  escalation: null,
};

describe("App", () => {
  beforeEach(() => {
    global.fetch = jest.fn((url: string) => {
      if (url.includes("/cases") && !url.includes("/run") && !url.includes("/volume")) {
        return Promise.resolve({ ok: true, json: async () => mockCases } as Response);
      }
      if (url.includes("/volume")) {
        return Promise.resolve({ ok: true, json: async () => mockVolume } as Response);
      }
      if (url.includes("/run")) {
        return Promise.resolve({ ok: true, json: async () => mockAutoResult } as Response);
      }
      return Promise.reject(new Error(`unexpected url ${url}`));
    }) as jest.Mock;
  });

  it("loads cases, selects the first case's data, and shows the report for an auto-cleared case", async () => {
    render(<App />);

    await waitFor(() => expect(screen.getByText(/Clean anterior lesion/)).toBeInTheDocument());

    fireEvent.click(screen.getByText(/Clean anterior lesion/));

    await waitFor(() => expect(screen.getByTestId("report-view")).toBeInTheDocument());
    expect(screen.queryByTestId("escalation-panel")).not.toBeInTheDocument();
  });
});
