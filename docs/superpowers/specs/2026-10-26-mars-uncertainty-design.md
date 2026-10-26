# MARS: Per-Organ Uncertainty-Aware CT Report Generation — Design

Date: 2026-10-26
Status: Draft for review
Context: Final-year research prototype (BVRIT). Must withstand academic review; not clinical-grade.

## 1. Purpose and scope

Given a CT scan and clinical metadata, produce a draft radiology report in which every organ is either **accepted** (segmentation is calibrated-reliable) or **flagged** with a structured, evidence-backed explanation of where and why the system may be wrong.

**In scope**
- Organ-level segmentation on BTCV: liver, spleen, left kidney, stomach. Other datasets and lesions are future work.
- Per-organ calibrated reliability score `p_ok` and an accept/flag decision.
- Clinical cross-check using `datasets/meta.csv` (TotalSegmentator metadata: age, gender, study type, pathology, pathology location) as a stand-in for clinical history.
- Draft report whose every claim is grounded in structured agent outputs.
- React frontend consuming the JSON contract (frontend design is out of scope for this spec).

**Out of scope**
- Final diagnosis; the output is a draft for radiologist review.
- Lesion segmentation.
- Real patient reports.

**Requirements change:** the requirements doc currently specifies one scalar uncertainty `U` per case. This design replaces it with a per-organ `UncertaintyReport`.

## 2. Existing assets

- Slide-SAM, LoRA fine-tuned on BTCV (4 organs), 5-fold CV. Fold 1 mean Dice 0.698 **with oracle (ground-truth) box prompts**.
- Stack: Python, PyTorch, `nnunetv2`, `segment-anything`, remote JupyterLab.
- ~30 labelled BTCV cases (~120 organ-cases).

## 3. Key design decisions

1. **Calibrate on the deployed pipeline.** Oracle-prompt Dice is not the deployed behaviour. The calibrated pipeline is: *automatic prompt source → Slide-SAM*. Oracle-prompt results are reported only as an ablation.
2. **Per-organ decisions**, not per-scan.
3. **Per-organ quality target `d*ₒ`**, with 0.90 as the headline where reachable (section 6.1).
4. **Reasons are structured codes with evidence; the LLM only verbalises them.**
5. **Relevance (cross-check) and `p_ok` stay separate axes.** They are never multiplied.
6. **Failures degrade to FLAG, never to a silent pass.**

## 4. Architecture

```
CT + case_id ──► [1 Segmentation] ──► SegmentationOutput ─┬─► [2 Clinical cross-check] ─► ClinicalCheck ─┐
                 prompt source (auto)                      │                                               ├─► [4 Report] ─► Report
                 + Slide-SAM refiner                       └─► [3 Uncertainty] ───────► UncertaintyReport ─┘
```

Agents 2 and 3 are independent of each other and consume Agent 1's output. Agent 4 consumes all three.

### Agent 1: Segmentation (ML)
- Automatic prompt source (TotalSegmentator or per-fold nnU-Net) produces a coarse mask and box per organ.
- Slide-SAM (LoRA) refines using that box.
- Open item: choose the prompt source (see section 9).

### Agent 2: Clinical cross-check (rules + LLM)
- Inputs: masks (volumes, locations) and the `meta.csv` row.
- Produces per-organ `relevance` in [0, 1] and `findings` (`Reason` list), e.g. pathology location is "liver" but liver volume is far below the age/sex norm.
- Missing metadata yields `relevance = null` with reason `NO_CLINICAL_CONTEXT` (not 0).

### Agent 3: Uncertainty (ML)
Per-organ features:
1. **Jitter disagreement:** perturb the box prompt (shift/scale) N times; mean pairwise Dice/IoU between resulting masks.
2. **Auto-vs-refined agreement:** Dice between the auto-model mask and the Slide-SAM mask (`prompt_mask_dice`).
3. **Anatomical sanity:** volume, number of connected components, HU statistics inside the mask, left/right position.
4. **Boundary entropy** from SAM logits.

A calibrator maps features to `p_ok = P(Dice ≥ d*ₒ)`. Decision: `FLAG` if `p_ok < τₒ`, else `ACCEPT`. Reason codes are derived from which features are outside their expected range.

### Agent 4: Report generation (LLM, constrained)
- Per organ: a clean statement for ACCEPT; a paragraph rendered from the organ's `Reason`s for FLAG.
- Ends with a one-line summary (for example "3/4 organs auto-accepted; stomach flagged").
- A validator checks every number and organ name in the prose against the structured inputs. On failure, fall back to a deterministic template report.

## 5. Contract

```
SegmentationOutput
  case_id, ct_spacing, ct_shape
  organs: { organ -> OrganMask }
OrganMask
  mask (RLE or ndarray), box_prompt, prompt_source ("totalseg" | "nnunet")
  prompt_mask_dice, logits_summary

ClinicalCheck
  organs: { organ -> { relevance: float | null, findings: [Reason] } }

UncertaintyReport
  calibration_version, target_config_hash, checkpoint_ids
  organs: { organ -> { p_ok, status: ACCEPT | FLAG,
                       signals: {...}, reasons: [Reason] } }

Reason
  code      # JITTER_HIGH, DISAGREE_AUTOMODEL, VOLUME_IMPLAUSIBLE,
            # TOUCHING_ORGAN_LEAK, PATHOLOGY_LOC_MISMATCH,
            # NO_PROMPT, NO_CLINICAL_CONTEXT
  organ, severity, evidence {numbers}, slices (optional)
```

All objects serialise to JSON (consumed by the React frontend).

## 6. Calibration and evaluation protocol

### 6.1 Per-organ target `d*ₒ`
```
d*ₒ = min(0.90, published inter-observer Dice for organ o)
```
- One citation per organ is recorded in this spec before test-fold results are examined. (Citations to be added by the team; values are not asserted here.)
- Fallback when no citation exists: median Dice of the auto-prompt model alone on the **development** folds.
- Values are committed in a config file before any test-fold evaluation. Git history is the pre-registration.

### 6.2 Data splits and leakage control
- Unit of analysis and of bootstrapping is the **patient** (organs within a CT are correlated).
- Level 1: Slide-SAM fold k predicts only its held-out cases. Only these out-of-fold predictions feed Agent 3 training/evaluation.
- Level 2: the calibrator and `τₒ` are fit on out-of-fold predictions of other patients and evaluated on the held-out patient group (leave-patients-out, repeated).
- The auto-prompt model must use the same folds as Slide-SAM. If TotalSegmentator is used, check for BTCV overlap with its training data and report it.

### 6.3 Calibrator
- Logistic regression, L2-regularised, on 4–6 features, target `1[Dice ≥ d*ₒ]`.
- **Shared slopes with a per-organ intercept.** Per-organ slopes are an ablation only.
- Features z-scored with training-fold statistics only.
- Degenerate organs (target almost never reached) are reported as "always FLAG"; they are not tuned away.

### 6.4 Threshold `τₒ`
Smallest `τₒ` such that accepted calibration-fold cases have empirical `P(Dice ≥ d*ₒ) ≥ 0.90`. Held-out coverage and accepted-precision are the headline results.

### 6.5 Metrics (patient-clustered bootstrap CIs)
- ECE (5 equal-mass bins), Brier score, reliability diagrams.
- AUROC of `p_ok` for `Dice ≥ d*ₒ`.
- Risk-coverage curve and AURC per organ.
- Headline claim: "When `p_ok ≥ τ`, Dice ≥ d* in X% [CI] of accepted cases at Y% coverage."
- State explicitly that n≈30 patients gives wide CIs.

### 6.6 Ablations
- Each signal alone; leave-one-signal-out.
- Oracle prompts vs auto-prompts.
- Uncalibrated raw score vs calibrated.
- Shared-slope vs per-organ-slope calibrator.

## 7. Failure handling

| Failure | Behaviour |
|---|---|
| Prompt source finds no organ | `NO_PROMPT`, FLAG, no mask |
| `meta.csv` row missing/empty | `relevance = null`, `NO_CLINICAL_CONTEXT` |
| Report prose fails validator | Deterministic template report from the same structured data |
| Any agent exception | Affected organs FLAG; case still produces a report |

Every output records `calibration_version`, `d*ₒ` config hash, and model checkpoint IDs.

## 8. Testing

- Schema/round-trip JSON tests for every contract object.
- Calibrator: monotonicity in main features; ECE on synthetic data with known true probabilities.
- Leakage test: no patient ID in both fit and evaluation sets at either cross-fitting level.
- Golden end-to-end cases with the LLM stubbed: clean, flagged, missing metadata.
- Report-validator tests with deliberately hallucinated sentences.

## 9. Open items

1. Prompt source: TotalSegmentator vs per-fold nnU-Net (leakage vs effort trade-off).
2. Inter-observer Dice citations per organ.
3. Jitter parameters (N, shift/scale ranges); tune on development folds only.
4. Which LLM backs Agents 2 and 4.
5. Update the requirements doc (scalar `U` to per-organ `UncertaintyReport`).
