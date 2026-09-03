export function ConfidenceBadge({ label, value }: { label: string; value: number }) {
  const level = value >= 0.5 ? "high" : "low";
  return (
    <span data-testid="confidence-badge" data-level={level}>
      {label}: {(value * 100).toFixed(0)}%
    </span>
  );
}
