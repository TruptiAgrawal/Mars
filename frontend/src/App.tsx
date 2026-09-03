import { useEffect, useState } from "react";
import { fetchCases, fetchVolume, runCase } from "./api";
import type { CaseResult, CaseSummary, VolumePayload } from "./api";
import { CaseList } from "./components/CaseList";
import { SliceViewer } from "./components/SliceViewer";
import { ConfidenceBadge } from "./components/ConfidenceBadge";
import { TauSlider } from "./components/TauSlider";
import { ReportView } from "./components/ReportView";
import { EscalationPanel } from "./components/EscalationPanel";

export function App() {
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [volume, setVolume] = useState<VolumePayload | null>(null);
  const [result, setResult] = useState<CaseResult | null>(null);
  const [tau, setTau] = useState(0.5);

  useEffect(() => {
    fetchCases().then(setCases);
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    fetchVolume(selectedId).then(setVolume);
    runCase(selectedId, tau).then(setResult);
  }, [selectedId, tau]);

  return (
    <div>
      <h1>MARS Mock Pipeline</h1>
      <CaseList cases={cases} selectedId={selectedId} onSelect={setSelectedId} />
      {volume && result && (
        <div>
          <SliceViewer volume={volume} mask={result.segmentation.mask} />
          <ConfidenceBadge label="Segmentation confidence" value={result.segmentation.aggregate_confidence} />
          <ConfidenceBadge label="Clinical consistency" value={result.validation.consistency_score} />
          <TauSlider tau={tau} onChange={setTau} />
          {result.decision === "auto" && result.report && <ReportView report={result.report} />}
          {result.decision === "escalate" && result.escalation && <EscalationPanel escalation={result.escalation} />}
        </div>
      )}
    </div>
  );
}
