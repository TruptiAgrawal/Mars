export function TauSlider({ tau, onChange }: { tau: number; onChange: (tau: number) => void }) {
  return (
    <label htmlFor="tau-slider">
      Threshold (τ): {tau.toFixed(2)}
      <input
        id="tau-slider"
        aria-label="threshold"
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={tau}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}
