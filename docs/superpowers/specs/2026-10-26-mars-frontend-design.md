# MARS Frontend — How It Works

Date: 2026-10-26
Parent specs: `2026-10-26-mars-uncertainty-design.md`, `2026-10-26-mars-backend-design.md`

## 1. One-paragraph summary

A single-page React + TypeScript app (Vite) for a radiologist-style review. The user picks or enters a case, the app asks the backend to run it, polls until done, and then shows (a) the draft report organised by organ, with each organ clearly marked **Reliable** or **Needs review** and the reasons spelled out, and (b) a CT slice viewer with the organ outline overlaid. The frontend contains no medical logic: it only displays what the backend's JSON says.

## 2. Principles

1. **Never imply "normal".** "Reliable" means the outline is reliable, not that the organ is healthy. The status chip text is *Outline reliable* / *Needs review*.
2. **Two axes, shown separately.** Outline reliability (from `p_ok`) and clinical consistency (from `relevance`) are different panels in each organ card.
3. **Colour is never the only signal.** Every status has an icon and text.
4. **Always show provenance.** A persistent banner states the disclaimer ("Draft for radiologist review; not a diagnosis") and, when applicable, "Clinical metadata is synthetic (demo only)" or "No clinical metadata".
5. **Show your working.** Each flagged organ lists its reasons with the numbers from `evidence`.

## 3. Stack

- Vite + React 18 + TypeScript. No state library: `useState` + one `useCase` hook.
- Plain `fetch`; polling every 1.5 s while `status === "running"`.
- Types generated from the backend contract: `python scripts/export_ts_types.py` writes `frontend/src/types/contract.ts` from `CaseResult.model_json_schema()` via `npx json-schema-to-typescript`. Re-run when `contracts.py` changes.
- Dev: Vite proxy `/api → http://localhost:8000`. Prod: `vite build`, served by the same host as the API (static folder).

## 4. Screen layout (one page)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ MARS — draft report for radiologist review     [banner: provenance/disclaimer] │
├───────────────┬─────────────────────────────────┬────────────────────────┤
│ Case panel    │ Report                          │ Slice viewer           │
│               │ Summary: 3/4 accepted; stomach  │ ┌────────────────────┐ │
│ [case id    ] │ flagged.                        │ │   CT slice + mask  │ │
│ [ct path    ] │                                 │ └────────────────────┘ │
│ [meta id    ] │ ┌ Liver  ✔ Outline reliable ──┐ │ organ: ( liver ▾ )     │
│ [Run]         │ │ p_ok 0.96  ▓▓▓▓▓▓▓▓▓░ τ 0.90│ │ z: ───●──────  42/118  │
│               │ │ Clinical consistency: 1.00  │ │                        │
│ Previous      │ └─────────────────────────────┘ │ ● accepted ● flagged   │
│ cases ▸       │ ┌ Stomach ⚠ Needs review ─────┐ │                        │
│  case_01 done │ │ p_ok 0.42  ▓▓▓▓░░░░░░ τ 0.90│ │                        │
│  case_02 err  │ │ • Outline changed when ...  │ │                        │
│               │ │ [Show in viewer]            │ │                        │
│               │ └─────────────────────────────┘ │                        │
└───────────────┴─────────────────────────────────┴────────────────────────┘
```

On narrow screens the three columns stack: case panel, report, viewer.

## 5. Components

| Component | Props | Responsibility |
|---|---|---|
| `App` | — | Layout; owns `caseId`, `selectedOrgan`, `z` |
| `CasePanel` | `onRun(req)`, `cases[]`, `onSelect(id)` | Form to start a run; list of previous cases with status |
| `Banner` | `report` | Disclaimer + provenance banner |
| `ReportView` | `result` | Summary line + one `OrganCard` per organ |
| `OrganCard` | `section`, `unc`, `clin`, `onShow` | Status chip, `p_ok` vs `τ` bar, sentences, clinical-consistency row, "Show in viewer" |
| `ReliabilityBar` | `pOk`, `tau` | Horizontal bar with a tick at `τ`; renders "not calibrated" when `pOk` is null |
| `SliceViewer` | `caseId`, `organ`, `z`, `zMax`, `onZ` | `<img src="/api/cases/{id}/slice?organ=&z=">`, slider, organ select, legend |
| `useCase` | `caseId` | Starts/polls, returns `{status, result, error}` |

## 6. Data flow

```
CasePanel.onRun ──POST /api/cases/{id}/run──► backend
        │
   useCase(caseId) polls GET /api/cases/{id}  (stops on done | error)
        │
   result: CaseResult ─► ReportView ─► OrganCard × 4
        │
   "Show in viewer" sets selectedOrgan and z = centre of that organ's box_prompt
        ▼
   SliceViewer requests PNGs on demand (server renders overlay)
```

The viewer's slice count comes from `result.segmentation.shape[0]`. Starting `z` is `(box_prompt[0] + box_prompt[3]) / 2` for the organ, or the middle slice if the organ has no box.

## 7. States to handle

| State | UI |
|---|---|
| Idle | Empty report area with instructions |
| `running` | Spinner + "Running segmentation, uncertainty and report..." |
| `done` | Report + viewer |
| `error` / HTTP error | Message with the backend's detail; retry button |
| Organ with no mask (`deployed = "none"`) | Card says "No outline was produced"; viewer shows plain CT |
| `p_ok = null` | `ReliabilityBar` shows "not calibrated"; card still lists reasons |
| `relevance = null` | Clinical row says "No clinical context" |

## 8. Wording rules (copied from the report spec)

- Status chips: **Outline reliable** and **Needs review**. Never "Normal", "Healthy", "OK".
- The ACCEPT sentence is rendered exactly as the backend sends it (it already says "not an assessment of the organ's health").
- The frontend never rewrites report sentences, and never computes statuses itself.

## 9. Accessibility and quality

- Status conveyed by icon + text + colour; contrast ≥ 4.5:1 for text.
- The slider is a native `<input type="range">` with a label and keyboard support; `<img>` has alt text such as "Liver, slice 42, outline accepted".
- Layout works at 360 px width (stacked columns).

## 10. Testing

- Vitest + React Testing Library with a mocked `fetch` and three fixture JSON files exported from the backend golden tests (clean, flagged, missing metadata).
- Tests assert: chip wording, banner text per provenance, `ReliabilityBar` null handling, viewer start slice, and that no component contains the strings "normal" or "healthy" (a grep-style test over rendered fixture output).
- Manual check: run backend + frontend on the remote lab with one BTCV case and one TotalSegmentator case.

## 11. Out of scope

Authentication, multi-user storage, DICOM import, editing masks, saving radiologist feedback.
