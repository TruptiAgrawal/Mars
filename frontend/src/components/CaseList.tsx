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
    <ul>
      {cases.map((c) => (
        <li
          key={c.id}
          data-testid={`case-${c.id}`}
          aria-selected={c.id === selectedId}
          onClick={() => onSelect(c.id)}
          style={{ cursor: "pointer", fontWeight: c.id === selectedId ? "bold" : "normal" }}
        >
          {c.label}
        </li>
      ))}
    </ul>
  );
}
