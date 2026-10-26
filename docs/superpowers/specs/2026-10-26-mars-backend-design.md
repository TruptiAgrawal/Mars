# MARS Backend — How It Works

Date: 2026-10-26
Parent spec: `2026-10-26-mars-uncertainty-design.md` (the science and evaluation protocol live there; this document is the engineering view).

## 1. One-paragraph summary

The backend is a Python package, `mars`, with four agents that are plain classes with one `run(...)` method each. A `pipeline.run_case` function calls them in order and always returns a `CaseResult`, even if an agent crashes. A thin FastAPI app starts a case in a background thread, stores the result as JSON on disk, and serves it (plus rendered CT slices) to the React frontend. Everything that needs a GPU is behind two small interfaces (`PromptSource`, `Refiner`), so all other logic is testable on a laptop with synthetic arrays.

## 2. Two modes of use

| Mode | When | What runs |
|---|---|---|
| **Offline (research)** | Building and evaluating the system on BTCV / TotalSegmentator | `scripts/*` build tables, pick `deploy.json`, `targets.json`, fit `calibration.json`, run nested CV. Needs GPU for Slide-SAM/TotalSegmentator. |
| **Online (demo)** | Frontend asks for a report on one scan | `POST /cases/{id}/run` → `run_case` → `result.json`. Uses the frozen config files from offline mode. |

The offline mode produces three frozen config files that the online mode only reads:

| File | Produced by | Meaning |
|---|---|---|
| `configs/deploy.json` | Segmentation plan, Task 5 | Per organ: use `"refined"` (Slide-SAM) or `"auto"` (prompt-source mask) |
| `configs/targets.json` | Uncertainty plan, Task 6 (by hand) | Per organ Dice target `d*ₒ` |
| `configs/calibration.json` | Uncertainty plan, Task 6 | Logistic weights, feature stats, per-organ `τₒ`, reason bounds, `targets_hash` |
| `configs/volume_norms.csv` | Team (cited) | Age/sex organ volume reference for the clinical agent |

## 3. Components

```
mars/
  contracts.py        # pydantic models = the JSON contract (single source of truth)
  config.py           # load targets.json, hash it
  volumes.py          # load_ct / load_mask / save_mask / atomic JSON write
  metrics.py          # dice, cluster_bootstrap_ci
  maskstats.py        # volume, #components, touches-edge for a mask
  fallbacks.py        # reason(), empty/flagged outputs when an agent fails
  pipeline.py         # run_case(...) -> CaseResult
  store.py            # save_result / load_result (atomic)
  render.py           # CT slice + mask overlay -> PNG bytes
  api.py              # FastAPI app factory
  wiring.py           # builds the real agents from env settings
  run.py              # CLI: python -m mars.run ...
  agents/
    segmentation/     # prompt_source, refiner, boxes, agent, deploy
    clinical/         # meta, vocab, norms, rules, agent
    uncertainty/      # features, calibrator, agent
    report/           # templates, validator, llm, agent
  eval/               # dev_config, tables, nested CV (offline only)
scripts/              # offline entry points
configs/              # frozen config (committed)
outputs/              # per-case results (git-ignored)
```

Rules that keep it simple:
- Agents import **only** `mars.contracts`, `mars.volumes`, `mars.metrics`, `mars.maskstats`, `mars.fallbacks`, and their own package. No agent imports another agent.
- Agents pass **file paths**, not big arrays, in the JSON contract (`mask_path`, `ct_path`). Arrays are `.npy` files in `outputs/{case_id}/masks/`.
- Axis order everywhere is `(z, y, x)`, spacing is mm. `load_ct` does the NIfTI transpose once.

## 4. Online data flow

```
POST /cases/{id}/run {ct_path, image_id?, synthetic_meta?}
   │  (ct_path must resolve under MARS_DATA_ROOT)
   ▼
background thread → run_case
   1. SegmentationAgent.run(case_id, ct_path, out_dir) → SegmentationOutput
        TotalSegmentator → boxes → Slide-SAM refine → apply deploy.json → save masks
   2. ClinicalAgent.run(seg, meta, provenance)         → ClinicalCheck
   3. UncertaintyAgent.run(seg)                        → UncertaintyReport
        jitter prompts → features → logistic p_ok → compare with τ → reasons
   4. ReportAgent.run(seg, clin, unc)                  → Report
        templates (+ optional LLM rewording gated by validator)
   ▼
store.save_result(out/{id}/result.json)   status → done
GET /cases/{id}  →  {status, result}
GET /cases/{id}/slice?organ=&z=  →  PNG
```

Agents 2 and 3 do not depend on each other; they run sequentially because it is simpler and fast enough.

## 5. Failure model (from the parent spec)

Any exception inside an agent is caught in `run_case`, logged with traceback, and converted to a structured result with `AGENT_ERROR` reasons. The affected organs become FLAG. The case still produces a report. Only a failure of the *fallback* itself raises.

| Failure | Result |
|---|---|
| Segmentation crashes / CT unreadable | All organs FLAG, `AGENT_ERROR`, minimal report |
| Prompt source finds no organ | That organ: `NO_PROMPT`, FLAG, no mask |
| Clinical agent crashes | `relevance = null`, `AGENT_ERROR` finding; uncertainty unaffected |
| Uncertainty agent crashes | All organs FLAG, `AGENT_ERROR` |
| LLM rewording fails validation | Template text used |
| Report agent crashes | `minimal_report` (statuses only) |

## 6. API

| Method + path | Purpose | Notes |
|---|---|---|
| `GET /health` | Liveness | |
| `GET /cases` | List case ids and status | Reads `outputs/` |
| `POST /cases/{id}/run` | Start a case | Body `{ct_path, image_id?, synthetic_meta?}`. Returns `{"status":"running"}`; if already running returns the same, no second job |
| `GET /cases/{id}` | Status + `CaseResult` when done | `status ∈ running, done, error` |
| `GET /cases/{id}/slice?organ=&z=` | PNG of CT slice with organ overlay (green accepted / orange flagged) | If the organ has no mask, plain CT slice |

Validation: `case_id` matches `^[A-Za-z0-9_-]{1,64}$`; `ct_path` is resolved and must be an existing file under `MARS_DATA_ROOT`; `organ` must be one of the four.

## 7. Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MARS_DATA_ROOT` | `datasets` | Only place CT files may be read from |
| `MARS_OUT_ROOT` | `outputs/cases` | Where results are written |
| `MARS_META_CSV` | `datasets/meta.csv` | TotalSegmentator metadata |
| `MARS_SLIDESAM_CKPT` | (required for real runs) | Checkpoint for the demo model |
| `MARS_LLM` | `none` | `none` or `anthropic` (report rewording + pathology tagging) |

## 8. Reproducibility and provenance

Every `UncertaintyReport` carries `calibration_version` and `targets_hash`. Jitter uses a fixed seed (`seed=0` default), so the same scan gives the same report. Offline tables are written to `outputs/dev/` together with the git commit hash that produced them.

## 9. Security and privacy

- `ct_path` and `case_id` are validated (no path traversal).
- When the LLM is enabled, only structured numbers and reason sentences are sent. No patient identifiers, no images.
- Synthetic-metadata cases are labelled in the contract (`metadata_provenance`) and in the report banner.

## 10. Testing strategy

- Unit tests per module on synthetic arrays (CPU only).
- Contract round-trip tests.
- Pipeline tests with stub agents (crash injection).
- API tests with FastAPI `TestClient` and a stub runner.
- Golden end-to-end cases (clean, flagged, missing metadata) with real Report agent and stubbed ML agents.
- GPU smoke tests marked `@pytest.mark.gpu`, run manually on the remote lab.
- Leakage test in the offline evaluation (no patient in both fit and evaluation sets).
