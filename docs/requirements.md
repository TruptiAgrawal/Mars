# MARS — Requirements

**Multi-Agent Framework for Reliable Radiological Segmentation and Report Generation**

| | |
|---|---|
| **Team** | Team 1 |
| **Members** | Trupthi Mantri (23WH1A1271), Trupti Agrawal (23WH1A1277), Navya Sree Balu (23WH1A1286), Ankam Hitha (23WH1A1289) |
| **Guide** | Dr. K. Srikar Goud, Assistant Professor, Dept. of IT |
| **Institution** | BVRIT Hyderabad College of Engineering for Women |
| **Version** | 1.0 |
| **Date** | September 3, 2026 |

---

## 1. Problem Statement

### 1.1 Background

Deep learning models can already segment lesions in CT scans with accuracy approaching that of trained radiologists. Despite this, such models are rarely part of routine hospital workflow. The bottleneck is not accuracy — it is trust.

A segmentation model outputs a mask. It does not output an honest measure of how much that mask should be believed. This leaves a radiologist with two options, both bad:

- **Verify every prediction manually** — the AI saves no time, so there is no reason to deploy it.
- **Verify none of them** — clinically unsafe, since a silent failure on one scan can lead to a missed or mis-staged lesion.

A second gap sits alongside the first: a mask is not a diagnosis. Segmentation in isolation ignores the patient's clinical history and produces nothing a clinician can put in a file. The reporting step, which is where the clinical value actually lands, is missing.

### 1.2 The Need

An AI system that **knows when it does not know** — one that quantifies its own reliability, checks its output against clinical context, and routes cases accordingly instead of presenting every prediction with the same unearned confidence.

### 1.3 Formal Formulation

Given a 3D CT volume `X` and contextual clinical metadata `C`:

```
M = f_seg(X)                      # volumetric segmentation mask
U = f_uncertainty(X, M, C)        # scalar uncertainty estimate

        ⎧ Automatic processing,   U <  τ
D =     ⎨
        ⎩ Expert review,          U ≥ τ
```

| Symbol | Meaning |
|---|---|
| `X` | Input 3D CT volume |
| `C` | Clinical metadata / patient history |
| `M` | Predicted segmentation mask |
| `U` | Quantified prediction uncertainty |
| `τ` | Safety threshold for routing |
| `D` | Routing decision |

The objective is to generate precise volumetric segmentation masks while **concurrently** quantifying prediction uncertainty, then evaluate that confidence against a safety threshold to dynamically route each case toward automated processing or expert review.

### 1.4 Domain

- Machine Learning / Deep Learning
- Medical Imaging
- CT-based Image Segmentation
- Clinical Decision Support

### 1.5 Objectives

1. Perform accurate pixel-level lesion segmentation on CT scans, along with per-prediction confidence scores.
2. Validate every generated mask against the patient's clinical history for consistency.
3. Estimate the system's own uncertainty so that reliable and unreliable cases can be separated.
4. Route confident cases through automatically and refer uncertain cases to a radiologist.
5. Generate a structured, explainable radiological report for each case.

### 1.6 Scope

**In scope**
- Lesion segmentation on 3D CT volumes
- Per-voxel and per-case confidence estimation
- Consistency checking against structured clinical metadata
- Threshold-based routing between automated and expert paths
- Structured report generation with traceable reasoning

**Out of scope (v1)**
- Modalities other than CT (MRI, X-ray, ultrasound)
- Real-time intra-operative use
- Final diagnostic authority — the system supports a radiologist, it does not replace one
- Deployment on hospital PACS infrastructure or regulatory clearance

---

## 2. Tech Stack

### 2.1 Overview

| Category | Technologies |
|---|---|
| Frontend | React.js |
| Backend / Implementation | Python, JupyterLab |
| AI / Medical Image Processing | `segment-anything`, `nnunetv2`, PyTorch |
| Scientific Libraries | NumPy, Pandas, SciPy |

### 2.2 Detail

**Frontend — React.js**
Radiologist-facing interface. Displays the CT volume with the predicted mask overlaid, shows the confidence score and uncertainty band for the case, flags cases escalated for review, and renders the final structured report.

**Backend — Python**
Orchestrates the four agents, manages the routing logic against threshold `τ`, and serves results to the frontend. JupyterLab is used for experimentation, model evaluation, and ablation studies during development.

**Segmentation — `segment-anything` (SAM) + `nnunetv2`**
SAM provides strong general-purpose segmentation with prompt-based control; nnU-Net provides a self-configuring pipeline tuned specifically for 3D medical volumes. Using both allows comparison and potential ensembling — and disagreement between them is itself a usable uncertainty signal.

**Deep Learning — PyTorch**
Model definition, training, inference, and the uncertainty machinery (e.g. Monte Carlo dropout passes, ensemble variance).

**Scientific Stack — NumPy / Pandas / SciPy**
Volume array manipulation, clinical metadata handling, and statistical computation for the uncertainty and validation agents.

### 2.3 Data Requirements

- Publicly available annotated CT datasets with lesion-level ground truth
- Accompanying clinical metadata where available (lesion site, prior findings, patient history fields)
- Standard medical imaging I/O support (DICOM / NIfTI)

---

## 3. Agent Workflow

MARS divides the pipeline across four specialized agents. Each agent has one responsibility and passes an explicit, inspectable output forward — so that when the system is unsure, it is possible to say *where* the uncertainty came from.

### 3.1 Pipeline

```
   CT Volume (X)  +  Clinical Metadata (C)
                │
                ▼
   ┌────────────────────────────┐
   │  1. SEGMENTATION AGENT     │
   │  → mask M                  │
   │  → per-voxel confidence    │
   └────────────┬───────────────┘
                ▼
   ┌────────────────────────────┐
   │  2. VALIDATION AGENT       │
   │  M checked against C       │
   │  → consistency score       │
   └────────────┬───────────────┘
                ▼
   ┌────────────────────────────┐
   │  3. UNCERTAINTY AGENT      │
   │  → aggregate uncertainty U │
   └────────────┬───────────────┘
                ▼
          ┌─────┴─────┐
      U < τ           U ≥ τ
          │               │
          ▼               ▼
   ┌──────────────┐  ┌──────────────────┐
   │ 4. REPORTING │  │ ESCALATE TO      │
   │    AGENT     │  │ RADIOLOGIST      │
   │ → structured │  │ (mask + reason   │
   │    report    │  │  for the doubt)  │
   └──────────────┘  └──────────────────┘
```

### 3.2 Agent 1 — Segmentation Agent

**Input:** 3D CT volume `X`

**Task:** Produce a pixel-level lesion mask.

**Output:** Segmentation mask `M`, plus per-voxel and aggregate confidence scores.

**Notes:** Confidence is produced *alongside* the mask, not inferred afterwards. Boundary voxels and small or low-contrast lesions are expected to carry lower confidence, and that signal is preserved rather than thresholded away.

---

### 3.3 Agent 2 — Validation Agent

**Input:** Mask `M`, clinical metadata `C`

**Task:** Check whether the predicted mask is *clinically plausible* given what is known about the patient — anatomical location, lesion size relative to reported findings, and consistency with prior history.

**Output:** Consistency score and a list of any specific conflicts detected.

**Notes:** This is the agent that catches a confidently-wrong prediction. A model can be certain about a mask in an anatomically implausible location; segmentation confidence alone will not flag that, but a clinical cross-check will.

---

### 3.4 Agent 3 — Uncertainty Agent

**Input:** Confidence scores from Agent 1, consistency score from Agent 2

**Task:** Aggregate the available evidence into a single, calibrated estimate of how reliable this prediction is overall.

**Output:** Scalar uncertainty value `U`, with an attribution of which factors drove it.

**Notes:** This is the core of the contribution. The value of `U` is only meaningful if it is *calibrated* — high uncertainty must actually correlate with high error rate. Evaluation should therefore measure not just segmentation accuracy but the quality of the uncertainty estimate itself.

---

### 3.5 Agent 4 — Reporting Agent

**Input:** Mask `M`, uncertainty `U`, validation findings, threshold `τ`

**Task:** Route the case and produce the output document.

**Output:**
- If `U < τ` — a structured, explainable radiological report generated automatically.
- If `U ≥ τ` — the case is escalated to a radiologist, accompanied by the mask, the uncertainty value, and the specific reason the system was not confident.

**Notes:** An escalation should never be a bare "unsure." Telling the radiologist *why* the system hesitated is what makes the review fast enough to be worth doing.

---

## 4. Functional Requirements

| ID | Requirement |
|---|---|
| FR-1 | The system shall accept a 3D CT volume and optional structured clinical metadata as input. |
| FR-2 | The system shall generate a voxel-level lesion segmentation mask. |
| FR-3 | The system shall produce per-prediction confidence scores alongside the mask. |
| FR-4 | The system shall validate each mask against available clinical metadata and report detected inconsistencies. |
| FR-5 | The system shall compute a single aggregate uncertainty value per case. |
| FR-6 | The system shall route each case to the automated or expert path based on a configurable threshold `τ`. |
| FR-7 | The system shall generate a structured radiological report for automatically-cleared cases. |
| FR-8 | The system shall present escalated cases with the mask, the uncertainty value, and the reason for escalation. |
| FR-9 | The interface shall allow a radiologist to view the CT volume with the mask overlaid. |

## 5. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-1 | **Calibration** — uncertainty estimates must correlate measurably with actual error; this is evaluated explicitly, not assumed. |
| NFR-2 | **Explainability** — every routing decision must be traceable to a specific agent output. |
| NFR-3 | **Safety bias** — where the system is in doubt, it escalates. False escalations are acceptable; silent false negatives are not. |
| NFR-4 | **Modularity** — each agent must be independently replaceable and independently evaluable. |
| NFR-5 | **Reproducibility** — fixed random seeds and versioned dependencies for all reported results. |
| NFR-6 | **Configurability** — the threshold `τ` must be adjustable without retraining. |

## 6. Evaluation Plan

**Segmentation quality** — Dice coefficient, IoU, Hausdorff distance against ground-truth annotations.

**Uncertainty quality** — Expected Calibration Error; risk–coverage curves showing accuracy on the auto-cleared subset as a function of how much is escalated.

**System-level** — the key result: *at what escalation rate does the auto-cleared subset reach clinically acceptable accuracy?* A system that escalates 30% of cases and is near-perfect on the remaining 70% is a success. One that escalates 5% but still errs on the rest is not.

**Validation agent** — measured by how many confidently-wrong predictions it catches that segmentation confidence alone misses.

---

## 7. References

1. *Training-free Prompt Placement by Propagation for SAM Predictions in 3D Bone CT Scans* — no retraining required; enables efficient 3D segmentation through automatic prompt propagation across CT slices. Limitation: addresses segmentation only, with no clinical validation, uncertainty estimation, or report generation.

2. *Slide-SAM: Medical SAM Meets Sliding Window* — extends SAM to 3D medical image segmentation using a three-slice sliding window and minimal box prompts, preserving pretrained SAM knowledge through parameter-efficient adaptation. Limitation: requires task-specific training and is sensitive to large inter-slice spacing and prompt propagation errors.

3. *Attention-augmented U-Net (AA-U-Net) for Semantic Segmentation* — enhances U-Net with an attention mechanism to capture both local and global contextual information. Limitation: higher computational cost and limited generalization to new imaging tasks without retraining.

4. *Video-CT MAE: Self-supervised Video-CT Domain Adaptation for Vertebral Fracture Diagnosis* — improves generalization through self-supervised learning and domain adaptation. Limitation: targets fracture diagnosis only; lacks multi-agent clinical decision-making and explainability.