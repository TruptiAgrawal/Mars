import { render, screen, fireEvent } from "@testing-library/react";
import { CaseList } from "../components/CaseList";
import type { CaseSummary } from "../api";

const cases: CaseSummary[] = [
  { id: "case_001", label: "Clean anterior lesion" },
  { id: "case_009", label: "Site mismatch" },
];

describe("CaseList", () => {
  it("renders every case label", () => {
    render(<CaseList cases={cases} selectedId={null} onSelect={jest.fn()} />);
    expect(screen.getByText(/Clean anterior lesion/)).toBeInTheDocument();
    expect(screen.getByText(/Site mismatch/)).toBeInTheDocument();
  });

  it("calls onSelect with the case id when clicked", () => {
    const handleSelect = jest.fn();
    render(<CaseList cases={cases} selectedId={null} onSelect={handleSelect} />);
    fireEvent.click(screen.getByText(/Site mismatch/));
    expect(handleSelect).toHaveBeenCalledWith("case_009");
  });

  it("marks the selected case", () => {
    render(<CaseList cases={cases} selectedId="case_001" onSelect={jest.fn()} />);
    expect(screen.getByTestId("case-case_001")).toHaveAttribute("aria-selected", "true");
  });
});
