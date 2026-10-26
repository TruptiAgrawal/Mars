# MARS: Per-Organ Uncertainty-Aware CT Report Generation — Design

Date: 2026-10-26 (rev. 2)
Status: Draft for review
Context: Final-year research prototype (BVRIT). Must withstand academic review; not clinical-grade.

## 1. Purpose and scope

Given a CT scan and clinical metadata, produce a draft radiology report in which every organ is either **accepted** (the segmentation is calibrated-reliable) or **flagged** with a structured, evidence-backed explanation of where and why the system may be wrong.

**In scope**
- Organ-level segmentation: liver, spleen, left kidney, stomach.
- Per-organ calibrated reliability score `p_ok` and an accept/flag decision.
- Clinical cross-check against metadata (section 4, Agent 2; data in section 3).
- Draft report whose every sentence is grounded in structured agent outputs.
- JSON contract consumed by the planned React frontend (frontend design is a separate spec).

**Out of scope**
- Final diagnosis; the output is a draft for radiologist review.
- Lesion segmentation and datasets beyond BTCV/TotalSegmentator (future work).
- Clinical validation.

**Requirements change:** the requirements doc specifies one scalar uncertainty `U` per case. This design replaces it with a per-organ `UncertaintyReport`.

**Left kidney only:** the rationale must be stated in the final report. BTCV also labels the right kidney (to be confirmed by the team). Open item 6.

## 2. Existing assets

- Slide-SAM, LoRA fine-tuned on BTCV (4 organs), 5-fold CV. Fold 1 mean Dice 0.698 **with oracle (ground-truth) box prompts**.
- Stack: Python, PyTorch, `nnunetv2`, `segment-anything`, remote JupyterLab.
- ~30 labelled BTCV cases (~120 organ-cases).
- `datasets/meta.csv`: TotalSegmentator metadata (age, gender, study type, pathology, pathology location).

## 3. Data roles (resolves the metadata mismatch)

`meta.csv` describes TotalSegmentator scans, not BTCV. So BTCV cases have no clinical metadata, and Agent 2 would return `NO_CLINICAL_CONTEXT` for all of them. Approaches considered:

| | Approach | Verdict |
|---|---|---|
| A | Synthetic metadata attached to BTCV cases | Controlled and ground truth is known, but any claim about real clinical consistency would be circular |
| B | Use TotalSegmentator test-split scans, which have real metadata and organ labels | Real metadata and independent scans; adds domain shift and its ground truth is partly model-assisted |
| **C (recommended)** | **B as the external evaluation set; A only for fault-injection unit tests and BTCV demo cases, always labelled "synthetic"** | Real metadata for Agent 2 evaluation; the external set doubles as an external validity test of the calibration |

Roles under C:

- **Development set = the ~30 BTCV patients.** Every design choice is made here: `d*ₒ`, features, jitter parameters, prompt-source choice, calibrator, `τ`. Evaluation inside it uses nested repeated leave-patients-out cross-validation (section 7).
- **External test set = TotalSegmentator test-split scans with the four organ labels and a non-empty metadata row.** Touched once, after the config is frozen in a dated commit. (Verify that `meta.csv` has a split column and how many test-split scans carry pathology text; the count sets how strong any Agent 2 claim can be.)
- On the external set, run each of the 5 fold models separately and report mean and spread. This matches the distribution the calibrator was fit on. Do not use an all-data final model for external evaluation.
- Known limits, to be reported: domain shift (scanner, protocol), TotalSegmentator ground truth being partly model-assisted, and, if TotalSegmentator is also the prompt source, partial circularity on this set. Prefer nnU-Net as the prompt source for the external evaluation if effort allows (open item 1).
- "Development folds" everywhere in this spec means BTCV patients only.

## 4. Architecture

```
CT + case_id ──► [1 Segmentation] ──► SegmentationOutput ─┬─► [2 Clinical cross-check] ─► ClinicalCheck ─┐
                 prompt source (auto)                      │                                               ├─► [4 Report] ─► Report
                 + Slide-SAM refiner                       └─► [3 Uncertainty] ───────► UncertaintyReport ─┘
```

Agents 2 and 3 are independent and both consume Agent 1's output. Agent 4 consumes all three.

### Agent 1: Segmentation (ML)
- The automatic prompt source (TotalSegmentator or per-fold nnU-Net) produces a coarse mask and box per organ.
- Slide-SAM (LoRA) refines using that box.
- **Refinement must earn its place.** The prompt source's own mask is the baseline (`mask_auto`). Per organ, on the development set, compare Dice of Slide-SAM-refined vs `mask_auto` with a patient-paired bootstrap. Pre-registered rule: if refinement is not better with the CI excluding zero, then for that organ the **deployed mask is `mask_auto`**, and Slide-SAM remains only as a source of uncertainty signals (jitter, agreement). The report states which organs fell back and why. This is a finding, not a failure.

### Agent 2: Clinical cross-check (deterministic rules; LLM only for parsing)
Relevance is a **rule-based score**, not an LLM judgement. For organ `o`, rules each return a severity in [0, 1] plus a `Reason`:

1. **Volume plausibility.** z-score of log organ volume against a sex/age-adjusted reference table. The table and its citations are supplied by the team and committed before the external set is touched. Severity increases with |z|. No norms are asserted in this spec.
2. **Pathology-location consistency.** If the metadata's pathology location names organ `o` (after normalising free text to a fixed organ vocabulary), then a missing, fragmented, or implausibly small mask for `o` produces `PATHOLOGY_LOC_MISMATCH`.
3. **Field-of-view truncation.** The mask touches the scan boundary on the craniocaudal axis (`FOV_TRUNCATED`).

`relevance = 1 - max(severity over rules)`. Missing metadata gives `relevance = null` with `NO_CLINICAL_CONTEXT`, never 0.

An LLM is used only to map free-text pathology fields to the fixed organ vocabulary. That mapping is checked against a hand-labelled sample from the team.

**Evaluation of Agent 2**
- Fault injection (synthetic): deliberately shrink, remove, or shift a mask, or edit metadata, and measure rule recall and false-alarm rate against the known injection.
- External set: team hand-annotates a sample of cases (target 30–50) for whether the metadata and mask are consistent, and reports agreement with the rules (precision, recall).
- State plainly that this is a rule-based plausibility check, not clinically validated.

### Agent 3: Uncertainty (ML)
Per-organ features:
1. **Jitter disagreement:** perturb the box prompt (shift/scale) N times; mean pairwise Dice between resulting masks.
2. **Auto-vs-refined agreement:** Dice between `mask_auto` and the Slide-SAM mask (`prompt_mask_dice`).
3. **Anatomical sanity:** volume, number of connected components, HU statistics inside the mask, left/right position.
4. **Boundary entropy** from SAM logits.

A calibrator maps features to `p_ok = P(Dice ≥ d*ₒ)`. Decision: `FLAG` if `p_ok < τₒ`, else `ACCEPT`. Reason codes derive from which features are outside their expected range.

**Compute cost of jitter.** Image embeddings are computed once per slice and cached, so jitter cost is N × (organs) × (slices) mask-decoder passes, not encoder passes. The wall-clock figure is to be measured on the remote lab and recorded in the spec. If it is too high, reduce N or subsample slices (open item 3).

### Agent 4: Report generation (templates + constrained LLM)
Every sentence in the report is traceable to a template ID or a `Reason`.

**ACCEPT wording is a fixed template with no LLM freedom:**
> "{Organ} segmentation met the reliability criterion (p_ok = {p_ok:.2f}, threshold {tau:.2f}); the automated outline is likely accurate. This is not an assessment of the {organ}'s health or appearance. Measured volume: {vol} mL."

The report may never say that an organ is normal, healthy, or unremarkable. ACCEPT means only that the segmentation is reliable.

**FLAG paragraphs:** the LLM may rephrase the sentences generated from `Reason` codes. It emits `{sentence, cites:[reason_id | field_path]}` objects.

**Validator (rejects the whole organ paragraph on any failure, then falls back to the deterministic template):**
1. Every sentence has at least one cite and every cite exists in the structured inputs.
2. Every number in a sentence matches the cited value (within rounding).
3. Every sentence maps to a permitted claim type for its cited `Reason` code (a fixed table), so a number-free claim like "this suggests a lesion" is rejected.
4. Deny-list for diagnostic vocabulary (for example normal, abnormal, lesion, tumour, healthy, disease, diagnosis, malignant, benign).

Summary line: "{k}/4 organs auto-accepted; {names} flagged."

## 5. Contract

```
SegmentationOutput
  case_id, ct_spacing, ct_shape
  organs: { organ -> OrganMask }
OrganMask
  mask (RLE or ndarray), mask_auto, box_prompt,
  prompt_source ("totalseg" | "nnunet"), deployed ("refined" | "auto"),
  prompt_mask_dice, logits_summary

ClinicalCheck
  metadata_provenance ("real" | "synthetic" | "none")
  organs: { organ -> { relevance: float | null, findings: [Reason] } }

UncertaintyReport
  calibration_version, target_config_hash, checkpoint_ids
  organs: { organ -> { p_ok, status: ACCEPT | FLAG,
                       signals: {...}, reasons: [Reason] } }

Reason
  id, code, organ, severity, evidence {numbers}, slices (optional)
  # codes: JITTER_HIGH, DISAGREE_AUTOMODEL, VOLUME_IMPLAUSIBLE,
  #        TOUCHING_ORGAN_LEAK, PATHOLOGY_LOC_MISMATCH, FOV_TRUNCATED,
  #        NO_PROMPT, NO_CLINICAL_CONTEXT, NOT_CALIBRATABLE
```

All objects serialise to JSON. Reports on synthetic metadata carry a visible "synthetic metadata" banner.

## 6. Quality target `d*ₒ` and feasibility

### 6.1 Rule
```
d*ₒ = min(0.90, published inter-observer Dice for organ o)
```
One citation per organ is recorded before test-set results are examined (the team supplies citations; no values are asserted here).

### 6.2 Feasibility check (Step 0, before `d*ₒ` is committed)
On the development set, compute the per-organ distribution of auto-prompt Dice (both `mask_auto` and refined). Define positives as cases with Dice ≥ `d*ₒ`, counted per organ in patient clusters.

An organ is **calibratable** if it has at least 8 positives and at least 8 negatives. Otherwise:
- **Too few positives:** report the Dice distribution and the risk–coverage curve on continuous Dice. The organ's status is always FLAG with reason `NOT_CALIBRATABLE`. We report "the pipeline does not reach `d*ₒ` for this organ" as the result.
- **Too few negatives:** report the same; the calibrator is not fit and coverage is stated with the caveat.
- **One permitted adjustment:** before any external-set result, `d*ₒ` may be lowered once, per organ, to the development-set median Dice of the pipeline, recorded in a dated commit with the reason. The headline claim for that organ is then stated at the lowered target, and 0.90 results are still shown.

## 7. Calibration and evaluation protocol

### 7.1 Splits and leakage control
- The unit of analysis and bootstrapping is the **patient**. Organs within a CT are correlated.
- Level 1: Slide-SAM fold k predicts only its held-out cases. Only these out-of-fold predictions feed Agent 3.
- Level 2: repeated leave-patients-out (R repeats). In each repeat, the calibrator and `τₒ` are fit on out-of-fold predictions of the training patients and scored on the held-out patients.
- The auto-prompt model uses the same folds as Slide-SAM. For TotalSegmentator as prompt source, check BTCV overlap with its training data and report it.

### 7.2 Calibrator
- Logistic regression, L2-regularised, 4–6 features, target `1[Dice ≥ d*ₒ]`.
- Shared slopes with a per-organ intercept. Per-organ slopes are an ablation only.
- Features z-scored with training-fold statistics only.

### 7.3 Threshold `τₒ` and precision target `α`
- `α` is the **accepted-precision target**, fixed at 0.90. It is distinct from the Dice target `d*ₒ`.
- `τₒ` is the smallest threshold such that accepted training-patient cases have empirical `P(Dice ≥ d*ₒ) ≥ α`.

### 7.4 Pooling across repeats and variability of `τ`
- Within each repeat, `τₒ` is fit only on that repeat's training patients and applied to its held-out patients.
- For metrics, each patient's held-out `p_ok` and accept/flag decision are pooled across repeats (average `p_ok`; the decision uses the repeat-wise `τₒ`, and the per-patient accept rate across repeats is reported). Metrics are then computed once on the pooled predictions, with the patient-clustered bootstrap.
- `τₒ` variability: report the distribution of `τₒ` across repeats and across bootstrap resamples (median and IQR). The deployed `τₒ` is the value refit on all development patients. If its IQR is wide, state that the threshold is unstable at this sample size.

### 7.5 Metrics (patient-clustered bootstrap CIs)
- ECE (5 equal-mass bins) with bootstrap CIs, Brier score, reliability diagrams.
- AUROC of `p_ok` for `Dice ≥ d*ₒ`.
- Risk-coverage curve and AURC per organ.
- Secondary segmentation metric: Normalised Surface Dice with the tolerance fixed at 2 mm for all organs (chosen before test, stated in the report), alongside Dice.
- Headline claim: "When `p_ok ≥ τ`, Dice ≥ d* in X% [CI] of accepted cases at Y% coverage," on the development set (nested CV) and, separately, on the external set.
- State explicitly that n≈30 patients gives wide CIs.

### 7.6 Ablations
- **Auto-model-only baseline** (prompt source's own mask `mask_auto`): Dice, and the same uncertainty and calibration pipeline run on it.
- Slide-SAM refined vs `mask_auto`: the fallback rule from Agent 1.
- Each signal alone; leave-one-signal-out.
- Oracle prompts vs auto-prompts.
- Uncalibrated raw score vs calibrated.
- Shared-slope vs per-organ-slope calibrator.

### 7.7 Report faithfulness
- Automatic: validator pass rate and fallback-to-template rate on all reports.
- Manual: the team audits a sample of reports (target 20) against a checklist — every sentence supported by a cited input, no diagnostic claim, numbers correct — and reports the unsupported-sentence rate.

## 8. Failure handling and testing

| Failure | Behaviour |
|---|---|
| Prompt source finds no organ | `NO_PROMPT`, FLAG, no mask |
| `meta.csv` row missing/empty | `relevance = null`, `NO_CLINICAL_CONTEXT` |
| Organ not calibratable | FLAG, `NOT_CALIBRATABLE` |
| Report prose fails validator | Deterministic template report from the same data |
| Any agent exception | Affected organs FLAG; case still produces a report |

Every output records `calibration_version`, the `d*ₒ` config hash, and checkpoint IDs.

**Tests**
- Schema/round-trip JSON tests for each contract object.
- Calibrator: monotonic in main features; ECE on synthetic data with known probabilities.
- Leakage test: no patient ID in both fit and evaluation sets at either level; no external-set patient in development.
- Golden end-to-end cases with the LLM stubbed: clean, flagged, missing metadata.
- Validator tests with deliberately hallucinated sentences, including number-free claims and deny-list words.
- Agent 2 fault-injection tests (section 4).

## 9. Open items

1. Prompt source: TotalSegmentator vs per-fold nnU-Net (leakage and circularity vs effort).
2. Inter-observer Dice citations per organ; volume reference-table citations for Agent 2.
3. Jitter parameters (N, shift/scale ranges), tuned on development data only; measure jitter wall-clock on the remote lab.
4. Which LLM backs the pathology-text parser and Agent 4.
5. Update the requirements doc (scalar `U` to per-organ `UncertaintyReport`).
6. Why left kidney only; confirm whether BTCV right-kidney labels are used or excluded.
7. Number of TotalSegmentator test-split scans with non-empty pathology metadata (sets the strength of the Agent 2 evaluation).
