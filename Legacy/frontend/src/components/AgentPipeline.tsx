import type { CaseResult } from "../api";

export function AgentPipeline({ result }: { result: CaseResult }) {
  const { segmentation, validation, uncertainty, decision } = result;
  const conflictCount = validation.conflicts.length;

  return (
    <div className="pipeline" data-testid="agent-pipeline">
      <PipelineStage
        index={1}
        name="Segmentation"
        metric={`${(segmentation.aggregate_confidence * 100).toFixed(0)}%`}
        detail="confidence"
      />
      <PipelineConnector />
      <PipelineStage
        index={2}
        name="Validation"
        metric={`${(validation.consistency_score * 100).toFixed(0)}%`}
        detail={conflictCount === 0 ? "no conflicts" : `${conflictCount} conflict${conflictCount === 1 ? "" : "s"}`}
      />
      <PipelineConnector />
      <PipelineStage
        index={3}
        name="Uncertainty"
        metric={uncertainty.value.toFixed(2)}
        detail={uncertainty.dominant_factor}
      />
      <PipelineConnector route={decision} />
      <PipelineStage
        index={4}
        name="Reporting"
        metric={decision}
        detail={decision === "auto" ? "cleared" : "radiologist review"}
        route={decision}
      />
    </div>
  );
}

function PipelineStage({
  index,
  name,
  metric,
  detail,
  route,
}: {
  index: number;
  name: string;
  metric: string;
  detail: string;
  route?: string;
}) {
  return (
    <div className="pipeline-stage" data-route={route}>
      <span className="pipeline-stage-index mono">{index}</span>
      <p className="pipeline-stage-name">{name}</p>
      <p className="pipeline-stage-metric mono">{metric}</p>
      <p className="pipeline-stage-detail">{detail}</p>
    </div>
  );
}

function PipelineConnector({ route }: { route?: string }) {
  return <div className="pipeline-connector" data-route={route} />;
}
