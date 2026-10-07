import { render, screen, fireEvent } from "@testing-library/react";
import { SliceViewer } from "../components/SliceViewer";
import type { VolumePayload } from "../api";

const volume: VolumePayload = {
  shape: [3, 2, 2],
  slices: [
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]],
    [[9, 10], [11, 12]],
  ],
};

const mask = [
  [[0, 0], [0, 0]],
  [[0, 1], [0, 0]],
  [[0, 0], [0, 0]],
];

describe("SliceViewer", () => {
  it("renders a depth slider ranging over the number of slices", () => {
    render(<SliceViewer volume={volume} mask={mask} />);
    const slider = screen.getByLabelText(/slice/i) as HTMLInputElement;
    expect(slider.min).toBe("0");
    expect(slider.max).toBe("2");
  });

  it("shows the current slice index", () => {
    render(<SliceViewer volume={volume} mask={mask} />);
    expect(screen.getByText(/slice 0 \/ 2/i)).toBeInTheDocument();
  });

  it("updates the displayed slice index when the slider changes", () => {
    render(<SliceViewer volume={volume} mask={mask} />);
    const slider = screen.getByLabelText(/slice/i);
    fireEvent.change(slider, { target: { value: "1" } });
    expect(screen.getByText(/slice 1 \/ 2/i)).toBeInTheDocument();
  });
});
