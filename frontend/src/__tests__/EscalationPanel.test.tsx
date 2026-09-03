import { render, screen } from "@testing-library/react";
import { EscalationPanel } from "../components/EscalationPanel";
import type { Escalation } from "../api";

const escalation: Escalation = {
  case_id: "case_009",
  uncertainty: 0.72,
  reasons: ["mask centroid outside expected site 'anterior'", "uncertainty 0.72 >= threshold 0.50"],
};

describe("EscalationPanel", () => {
  it("shows the uncertainty value and every reason", () => {
    render(<EscalationPanel escalation={escalation} />);
    expect(screen.getByText(/Uncertainty.*0.72/)).toBeInTheDocument();
    expect(screen.getByText(/outside expected site/)).toBeInTheDocument();
    expect(screen.getByText(/uncertainty 0.72 >= threshold 0.50/)).toBeInTheDocument();
  });
});
