import { render, screen } from "@testing-library/react";
import { ConfidenceBadge } from "../components/ConfidenceBadge";

describe("ConfidenceBadge", () => {
  it("renders the label and percentage", () => {
    render(<ConfidenceBadge label="Confidence" value={0.87} />);
    expect(screen.getByText(/Confidence/)).toBeInTheDocument();
    expect(screen.getByText(/87%/)).toBeInTheDocument();
  });

  it("applies a low-value style hint when value is below 0.5", () => {
    render(<ConfidenceBadge label="Confidence" value={0.2} />);
    expect(screen.getByTestId("confidence-badge")).toHaveAttribute("data-level", "low");
  });

  it("applies a high-value style hint when value is at or above 0.5", () => {
    render(<ConfidenceBadge label="Confidence" value={0.6} />);
    expect(screen.getByTestId("confidence-badge")).toHaveAttribute("data-level", "high");
  });
});
