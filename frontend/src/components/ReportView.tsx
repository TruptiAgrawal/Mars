import type { Report } from "../api";

export function ReportView({ report }: { report: Report }) {
  return (
    <div className="report-view" data-testid="report-view">
      <h3>Automated report</h3>
      <p>{report.findings}</p>
      <p className="confidence-line">Confidence: {(report.confidence * 100).toFixed(0)}%</p>
      <p>{report.narrative}</p>
    </div>
  );
}
