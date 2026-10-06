# Completed checkpoint audits

Archived with user approval on 1 October 2026. The three former top-level diagnostic folders were removed after preserving and verifying their results. Training checkpoints, training code, caches, and the box-padding experiment were not altered.

| Experiment | Report | Console output | Raw records |
| --- | --- | --- | --- |
| BS=2 versus BS=4 | [Report](matched_bs_comparison.report.md) | [Log](matched_bs_comparison.run.log) | [Evidence](matched_bs_comparison.evidence.log) |
| Official checkpoint | [Report](official_checkpoint_comparison.report.md) | [Log](official_checkpoint_comparison.run.log) | [Evidence](official_checkpoint_comparison.evidence.log) |
| Fold-5 trajectory | [Report](checkpoint_trajectory.report.md) | [Log](checkpoint_trajectory.run.log) | [Evidence](checkpoint_trajectory.evidence.log) |

Each evidence log is a JSON object with a `files` mapping. Each entry contains the original file text and its SHA-256 hash. It preserves final results, per-patient CSV, manifests, source hashes and available verification records. Partial result files were redundant and removed. Historical logs and the old comparison README are retained. Source snapshots, runners, shell scripts and summarizers were removed; their old commands no longer run. Source hashes preserve provenance but cannot reconstruct deleted source snapshots.

To inspect results without recreating files, from the project root:

```python
import json
from pathlib import Path
archive = Path('btcv_cross_validation/logs/checkpoint_audits')
records = json.loads((archive / 'official_checkpoint_comparison.evidence.log').read_text())
results = json.loads(records['files']['results.json']['text'])
print(results['models']['official_slidesam_b']['macro_mean_dice'])
```

Continue using the [current roadmap](../../../KT/Next_Roadmap_and_Agent_Handoff_2026-10-01.md).
