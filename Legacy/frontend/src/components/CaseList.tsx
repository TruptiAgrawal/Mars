import type { CaseSummary } from "../api";

export function CaseList({
  cases,
  selectedId,
  onSelect,
}: {
  cases: CaseSummary[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="case-list-pane">
      <p className="pane-label">Cases</p>
      <ul>
        {cases.map((c) => (
          <li
            key={c.id}
            className="case-item"
            data-testid={`case-${c.id}`}
            aria-selected={c.id === selectedId}
            onClick={() => onSelect(c.id)}
          >
            <span className="dot" />
            {c.label}
          </li>
        ))}
      </ul>
    </div>
  );
}
