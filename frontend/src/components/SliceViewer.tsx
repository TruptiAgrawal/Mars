import { useState } from "react";
import type { VolumePayload } from "../api";

export function SliceViewer({ volume, mask }: { volume: VolumePayload; mask: number[][][] }) {
  const [sliceIndex, setSliceIndex] = useState(0);
  const depth = volume.shape[0];
  const slice = volume.slices[sliceIndex];
  const maskSlice = mask[sliceIndex];

  return (
    <div>
      <label htmlFor="slice-slider">
        Slice
        <input
          id="slice-slider"
          aria-label="slice"
          type="range"
          min={0}
          max={depth - 1}
          value={sliceIndex}
          onChange={(e) => setSliceIndex(Number(e.target.value))}
        />
      </label>
      <p>
        Slice {sliceIndex} / {depth - 1}
      </p>
      <svg width={slice[0].length * 10} height={slice.length * 10} data-testid="slice-svg">
        {slice.map((row, y) =>
          row.map((value, x) => {
            const isMasked = maskSlice?.[y]?.[x] === 1;
            const gray = Math.min(255, Math.max(0, Math.round(value)));
            return (
              <rect
                key={`${y}-${x}`}
                x={x * 10}
                y={y * 10}
                width={10}
                height={10}
                fill={isMasked ? "rgba(255,0,0,0.5)" : `rgb(${gray},${gray},${gray})`}
              />
            );
          })
        )}
      </svg>
    </div>
  );
}
