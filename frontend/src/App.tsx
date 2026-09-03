import { useEffect, useState } from "react";
import { fetchCases, fetchVolume, runCase } from "./api";
import type { CaseResult, CaseSummary, VolumePayload } from "./api";
import { CaseList } from "./components/CaseList";
import { SliceViewer } from "./components/SliceViewer";
import { ConfidenceBadge } from "./components/ConfidenceBadge";
import { TauSlider } from "./components/TauSlider";
import { ReportView } from "./components/ReportView";
import { EscalationPanel } from "./components/EscalationPanel";
import { AgentPipeline } from "./components/AgentPipeline";

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
    <div className="app">
      <header className="app-header">
        <h1>MARS</h1>
        <span className="subtitle">Segmentation review pipeline</span>
      </header>
      <div className="app-body">
        <CaseList cases={cases} selectedId={selectedId} onSelect={setSelectedId} />
        {volume && result ? (
          <>
            <div className="main-pane">
              <AgentPipeline result={result} />
              <div className="viewer-pane">
                <SliceViewer volume={volume} mask={result.segmentation.mask} />
              </div>
            </div>
            <div className="metrics-pane">
              <div className="metrics-section">
                <p className="pane-label">Confidence</p>
                <ConfidenceBadge label="Segmentation confidence" value={result.segmentation.aggregate_confidence} />
                <ConfidenceBadge label="Clinical consistency" value={result.validation.consistency_score} />
              </div>
              <div className="metrics-section">
                <TauSlider tau={tau} onChange={setTau} />
              </div>
              <div className="metrics-section">
                {result.decision === "auto" && result.report && <ReportView report={result.report} />}
                {result.decision === "escalate" && result.escalation && (
                  <EscalationPanel escalation={result.escalation} />
                )}
              </div>
            </div>
          </>
        ) : (
          <div className="app-empty">Select a case to begin review.</div>
        )}
      </div>
    </div>
  );
}
