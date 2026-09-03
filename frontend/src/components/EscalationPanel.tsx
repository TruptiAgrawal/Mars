import type { Escalation } from "../api";

export function EscalationPanel({ escalation }: { escalation: Escalation }) {
  return (
    <div className="escalation-panel" data-testid="escalation-panel">
      <h3>Escalated to radiologist</h3>
      <p className="uncertainty-line">Uncertainty: {escalation.uncertainty.toFixed(2)}</p>
      <ul>
        {escalation.reasons.map((reason, i) => (
          <li key={i}>{reason}</li>
        ))}
      </ul>
    </div>
  );
}
