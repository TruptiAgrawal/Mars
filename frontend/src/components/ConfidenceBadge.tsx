export function ConfidenceBadge({ label, value }: { label: string; value: number }) {
  // 0.3 split matches the achievable confidence range of this mock's segmentation formula.
  const level = value >= 0.3 ? "high" : "low";
  return (
    <div className="confidence-badge" data-testid="confidence-badge" data-level={level}>
      <span className="label">{label}</span>
      <span className="value">{(value * 100).toFixed(0)}%</span>
    </div>
  );
}
