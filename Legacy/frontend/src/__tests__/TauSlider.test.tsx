import { render, screen, fireEvent } from "@testing-library/react";
import { TauSlider } from "../components/TauSlider";

describe("TauSlider", () => {
  it("displays the current tau value", () => {
    render(<TauSlider tau={0.5} onChange={jest.fn()} />);
    expect(screen.getByText(/0.50/)).toBeInTheDocument();
  });

  it("calls onChange with the new tau when moved", () => {
    const handleChange = jest.fn();
    render(<TauSlider tau={0.5} onChange={handleChange} />);
    const slider = screen.getByLabelText(/threshold/i);
    fireEvent.change(slider, { target: { value: "0.8" } });
    expect(handleChange).toHaveBeenCalledWith(0.8);
  });
});
