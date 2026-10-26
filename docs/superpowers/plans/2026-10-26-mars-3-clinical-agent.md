# MARS Plan 3 — Clinical Cross-Check Agent (Agent 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Given the organ masks and one metadata row, return a per-organ relevance score and findings from three explicit rules. No LLM judgement: the score is deterministic.

**Architecture:** `ClinicalAgent` loops over the four organs. For each, `mask_stats` gives volume/pieces/edge contact, `VolumeNorms` gives a z-score against a cited reference table, and three small rule functions each return a `Reason` (or nothing). `relevance = 1 − max(severity)`. An optional LLM only maps free-text pathology to organ names.

**Tech Stack:** pandas, numpy, pytest. Optional LLM function injected as `llm_fn(prompt: str) -> str`.

**Spec:** `docs/superpowers/specs/2026-10-26-mars-uncertainty-design.md` (§3 data roles, §4 Agent 2).

**Prerequisite:** Plan 1 Tasks 1–3 done (`contracts`, `maskstats`, `volumes`, `fallbacks.reason`).

## Global Constraints

- Missing or empty metadata → `relevance = None` with a `NO_CLINICAL_CONTEXT` finding, **never** relevance 0.
- Relevance is a rule-based plausibility score, not clinically validated; the final report must say so.
- Severity from a z-score: `clip((|z| − 2) / 2, 0, 1)` (nothing below |z| = 2, maximum at |z| = 4).
- Evidence keys per code come from `contracts.REQUIRED_EVIDENCE` (`VOLUME_IMPLAUSIBLE`: `volume_ml`, `z`; `PATHOLOGY_LOC_MISMATCH`: `volume_ml`, `n_components`).
- The reference volume table `configs/volume_norms.csv` and its citations are supplied by the team and committed before the external set is touched; this plan never invents norms.
- `meta.csv` columns used: `image_id`, `age`, `gender`, `study_type`, `pathology`, `pathology_location` (verify with `df.columns` on the lab).

## Review Focus

1. Metadata row exists but every useful field is empty → treated as no context, not as "normal" (Task 5 test).
2. Age or sex missing but pathology text present → pathology rule still runs; volume rule is skipped (Task 5 test).
3. Pathology text says "right kidney" only → `kidney_left` is **not** tagged (Task 2 test).
4. Organ absent (`deployed="none"`) with the organ named in the pathology text → strong mismatch, relevance 0 (Task 5 test).
5. Age outside the norm table → no z-score, no volume finding, no crash (Task 3 test).

---

### Task 1: Metadata loader

**Files:**
- Create: `mars/agents/clinical/__init__.py`, `mars/agents/clinical/meta.py`
- Test: `tests/agents/clinical/test_meta.py` (create `tests/agents/clinical/__init__.py`)

**Interfaces:**
- Produces: `meta.load_meta_row(csv_path, image_id: str) -> dict | None` returning keys `age: float | None`, `sex: "m" | "f" | None`, `study_type: str | None`, `pathology: str | None`, `pathology_location: str | None`; `None` if the image id is not in the file.

- [ ] **Step 1: Write the failing test**

```python
from mars.agents.clinical.meta import load_meta_row

CSV = (
    "image_id;age;gender;study_type;split;pathology;pathology_location\n"
    "s0001;63;m;ct abdomen;test;hepatocellular carcinoma;liver\n"
    "s0002;;f;ct chest;test;;\n"
    "s0003;x;F;ct abdomen;train;none;\n"
)


def test_known_row(tmp_path):
    (tmp_path / "meta.csv").write_text(CSV)
    row = load_meta_row(tmp_path / "meta.csv", "s0001")
    assert row == {"age": 63.0, "sex": "m", "study_type": "ct abdomen",
                   "pathology": "hepatocellular carcinoma", "pathology_location": "liver"}


def test_empty_fields_become_none_and_bad_age_is_none(tmp_path):
    (tmp_path / "meta.csv").write_text(CSV)
    r2 = load_meta_row(tmp_path / "meta.csv", "s0002")
    assert r2["age"] is None and r2["pathology"] is None and r2["sex"] == "f"
    assert load_meta_row(tmp_path / "meta.csv", "s0003")["age"] is None


def test_unknown_image_id(tmp_path):
    (tmp_path / "meta.csv").write_text(CSV)
    assert load_meta_row(tmp_path / "meta.csv", "nope") is None
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/agents/clinical/test_meta.py -v` → FAIL

- [ ] **Step 3: Implement** `mars/agents/clinical/meta.py`

```python
import pandas as pd


def _clean(value):
    if value is None or pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def load_meta_row(csv_path, image_id: str) -> dict | None:
    df = pd.read_csv(csv_path, sep=None, engine="python", dtype=str)
    rows = df[df["image_id"].astype(str) == str(image_id)]
    if rows.empty:
        return None
    r = rows.iloc[0]
    age = _clean(r.get("age"))
    try:
        age_val = float(age) if age is not None else None
    except ValueError:
        age_val = None
    gender = (_clean(r.get("gender")) or "").lower()
    sex = "m" if gender.startswith("m") else "f" if gender.startswith("f") else None
    return {"age": age_val, "sex": sex, "study_type": _clean(r.get("study_type")),
            "pathology": _clean(r.get("pathology")),
            "pathology_location": _clean(r.get("pathology_location"))}
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/agents/clinical/test_meta.py -v` → 3 PASS

- [ ] **Step 5: Commit**

```bash
git add mars tests
git commit -m "feat(clinical): metadata row loader"
```

---

### Task 2: Organ tagging from pathology text

**Files:**
- Create: `mars/agents/clinical/vocab.py`
- Test: `tests/agents/clinical/test_vocab.py`

**Interfaces:**
- Produces: `vocab.tag_organs(text: str) -> set[str]` (deterministic keywords), `vocab.tag_organs_with_fallback(text: str, llm_fn=None) -> set[str]` (keywords first; if nothing found and `llm_fn` given, ask the LLM and keep only valid organ names).

- [ ] **Step 1: Write the failing test**

```python
from mars.agents.clinical.vocab import tag_organs, tag_organs_with_fallback


def test_keywords():
    assert tag_organs("hepatocellular carcinoma") == {"liver"}
    assert tag_organs("Splenic infarct and gastric wall thickening") == {"spleen", "stomach"}
    assert tag_organs("") == set()


def test_kidney_side_handling():
    assert tag_organs("left kidney cyst") == {"kidney_left"}
    assert tag_organs("renal cyst") == {"kidney_left"}            # side unspecified: tag (conservative)
    assert tag_organs("right kidney cyst") == set()               # right only: not our organ
    assert tag_organs("bilateral kidney stones, right and left") == {"kidney_left"}


def test_llm_only_used_when_keywords_find_nothing_and_output_is_filtered():
    calls = []

    def llm(prompt):
        calls.append(prompt)
        return "liver, pancreas, STOMACH"

    assert tag_organs_with_fallback("hepatic mass", llm) == {"liver"} and calls == []
    assert tag_organs_with_fallback("upper abdominal pain", llm) == {"liver", "stomach"}
    assert len(calls) == 1
    assert tag_organs_with_fallback("upper abdominal pain", None) == set()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/agents/clinical/test_vocab.py -v` → FAIL

- [ ] **Step 3: Implement** `mars/agents/clinical/vocab.py`

```python
from mars.contracts import ORGANS

KEYWORDS = {
    "liver": ("liver", "hepat"),
    "spleen": ("spleen", "splen"),
    "stomach": ("stomach", "gastr"),
}
KIDNEY_WORDS = ("kidney", "renal", "nephr")


def tag_organs(text: str) -> set[str]:
    t = (text or "").lower()
    found = {organ for organ, words in KEYWORDS.items() if any(w in t for w in words)}
    if any(w in t for w in KIDNEY_WORDS):
        right_only = "right" in t and "left" not in t
        if not right_only:
            found.add("kidney_left")
    return found


def tag_organs_with_fallback(text: str, llm_fn=None) -> set[str]:
    found = tag_organs(text)
    if found or not llm_fn or not (text or "").strip():
        return found
    prompt = (
        "Which of these organs does the clinical text refer to: "
        f"{', '.join(ORGANS)}? Answer only with a comma-separated list, or 'none'.\n"
        f"Text: {text}"
    )
    answer = llm_fn(prompt).lower().replace(";", ",")
    return {tok.strip() for tok in answer.split(",") if tok.strip() in ORGANS}
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/agents/clinical/test_vocab.py -v` → 3 PASS

- [ ] **Step 5: Commit**

```bash
git add mars tests
git commit -m "feat(clinical): organ tagging with optional LLM fallback"
```

---

### Task 3: Volume norms

**Files:**
- Create: `mars/agents/clinical/norms.py`
- Test: `tests/agents/clinical/test_norms.py`

**Interfaces:**
- Produces: `norms.VolumeNorms.from_csv(path)`, `VolumeNorms(df)`, `VolumeNorms.zscore(organ: str, sex: str | None, age: float | None, volume_ml: float | None) -> float | None`. CSV columns: `organ,sex,age_min,age_max,mean_log_ml,sd_log_ml` with `sex ∈ {m, f, any}`. Returns `None` when there is no matching row, age missing, or volume ≤ 0.

- [ ] **Step 1: Write the failing test**

```python
import math

import pandas as pd
import pytest

from mars.agents.clinical.norms import VolumeNorms

DF = pd.DataFrame([
    {"organ": "liver", "sex": "any", "age_min": 18, "age_max": 120, "mean_log_ml": math.log(1500), "sd_log_ml": 0.2},
    {"organ": "liver", "sex": "f", "age_min": 18, "age_max": 120, "mean_log_ml": math.log(1300), "sd_log_ml": 0.2},
])


def test_exact_sex_row_preferred_over_any():
    n = VolumeNorms(DF)
    assert n.zscore("liver", "f", 50, 1300) == pytest.approx(0.0)
    assert n.zscore("liver", "m", 50, 1500) == pytest.approx(0.0)   # falls back to "any"
    assert n.zscore("liver", None, 50, 1500) == pytest.approx(0.0)


def test_z_sign_and_scale():
    z = VolumeNorms(DF).zscore("liver", "m", 50, 1500 * math.exp(0.4))
    assert z == pytest.approx(2.0)


@pytest.mark.parametrize("args", [
    ("spleen", "m", 50, 200),     # no row for organ
    ("liver", "m", 10, 1500),     # age outside table
    ("liver", "m", None, 1500),   # age unknown
    ("liver", "m", 50, 0),        # empty organ
])
def test_no_zscore_when_not_computable(args):
    assert VolumeNorms(DF).zscore(*args) is None


def test_from_csv(tmp_path):
    DF.to_csv(tmp_path / "n.csv", index=False)
    assert VolumeNorms.from_csv(tmp_path / "n.csv").zscore("liver", "m", 50, 1500) == pytest.approx(0.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/agents/clinical/test_norms.py -v` → FAIL

- [ ] **Step 3: Implement** `mars/agents/clinical/norms.py`

```python
import math

import pandas as pd


class VolumeNorms:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @classmethod
    def from_csv(cls, path) -> "VolumeNorms":
        return cls(pd.read_csv(path))

    def zscore(self, organ, sex, age, volume_ml):
        if volume_ml is None or volume_ml <= 0 or age is None:
            return None
        d = self.df
        rows = d[(d.organ == organ) & d.sex.isin([sex or "any", "any"])
                 & (d.age_min <= age) & (age <= d.age_max)]
        if rows.empty:
            return None
        r = rows.sort_values("sex", key=lambda s: s == "any").iloc[0]   # exact sex first
        return float((math.log(volume_ml) - r.mean_log_ml) / r.sd_log_ml)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/agents/clinical/test_norms.py -v` → 4+ PASS

- [ ] **Step 5: Commit**

```bash
git add mars tests
git commit -m "feat(clinical): age/sex volume norms with z-score"
```

---

### Task 4: The three rules

**Files:**
- Create: `mars/agents/clinical/rules.py`
- Test: `tests/agents/clinical/test_rules.py`

**Interfaces:**
- Consumes: `maskstats.OrganStats`, `fallbacks.reason`
- Produces (all return `Reason | None`):
  - `rules.severity_from_z(z: float) -> float`
  - `rules.rule_volume(organ, stats: OrganStats, z: float | None)`
  - `rules.rule_pathology_location(organ, tagged: set[str], stats: OrganStats | None, z: float | None)`
  - `rules.rule_fov(organ, stats: OrganStats)`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from mars.agents.clinical.rules import (
    rule_fov, rule_pathology_location, rule_volume, severity_from_z,
)
from mars.contracts import REQUIRED_EVIDENCE
from mars.maskstats import OrganStats

OK = OrganStats(volume_ml=1500.0, n_components=1, touches_z_edge=False)


@pytest.mark.parametrize("z,sev", [(0.0, 0.0), (2.0, 0.0), (3.0, 0.5), (-4.0, 1.0), (9.0, 1.0)])
def test_severity_from_z(z, sev):
    assert severity_from_z(z) == pytest.approx(sev)


def test_volume_rule_fires_only_beyond_two_sd_and_carries_evidence():
    assert rule_volume("liver", OK, 1.9) is None
    assert rule_volume("liver", OK, None) is None
    r = rule_volume("liver", OK, -3.0)
    assert r.code == "VOLUME_IMPLAUSIBLE" and r.severity == pytest.approx(0.5)
    assert set(REQUIRED_EVIDENCE["VOLUME_IMPLAUSIBLE"]) <= set(r.evidence)


def test_pathology_rule_only_for_tagged_organs():
    assert rule_pathology_location("liver", {"spleen"}, None, None) is None


def test_pathology_rule_cases():
    missing = rule_pathology_location("liver", {"liver"}, None, None)
    assert missing.code == "PATHOLOGY_LOC_MISMATCH" and missing.severity == 1.0
    assert missing.evidence["volume_ml"] == 0 and missing.evidence["n_components"] == 0
    empty = rule_pathology_location("liver", {"liver"}, OrganStats(0.0, 0, False), None)
    assert empty.severity == 1.0
    frag = rule_pathology_location("liver", {"liver"}, OrganStats(900.0, 3, False), 0.0)
    assert frag.severity == pytest.approx(0.6)
    small = rule_pathology_location("liver", {"liver"}, OK, -3.0)
    assert small.severity == pytest.approx(0.5)
    assert rule_pathology_location("liver", {"liver"}, OK, 0.5) is None
    assert set(REQUIRED_EVIDENCE["PATHOLOGY_LOC_MISMATCH"]) <= set(frag.evidence)


def test_fov_rule():
    assert rule_fov("liver", OK) is None
    r = rule_fov("liver", OrganStats(1000.0, 1, True))
    assert r.code == "FOV_TRUNCATED" and r.severity == 0.5
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/agents/clinical/test_rules.py -v` → FAIL

- [ ] **Step 3: Implement** `mars/agents/clinical/rules.py`

```python
import numpy as np

from mars.contracts import Reason
from mars.fallbacks import reason
from mars.maskstats import OrganStats


def severity_from_z(z: float) -> float:
    return float(np.clip((abs(z) - 2.0) / 2.0, 0.0, 1.0))


def rule_volume(organ: str, stats: OrganStats, z: float | None) -> Reason | None:
    if z is None:
        return None
    sev = severity_from_z(z)
    if sev <= 0:
        return None
    return reason("clin", organ, "VOLUME_IMPLAUSIBLE", sev,
                  volume_ml=round(stats.volume_ml, 1), z=round(z, 2))


def rule_pathology_location(organ: str, tagged: set[str], stats: OrganStats | None,
                            z: float | None) -> Reason | None:
    """History names this organ but its outline is missing, fragmented, or far too small."""
    if organ not in tagged:
        return None
    if stats is None or stats.volume_ml == 0:
        sev, vol, comps = 1.0, 0.0, 0
    elif stats.n_components > 1:
        sev, vol, comps = 0.6, stats.volume_ml, stats.n_components
    elif z is not None and z < -2.0:
        sev, vol, comps = severity_from_z(z), stats.volume_ml, stats.n_components
    else:
        return None
    return reason("clin", organ, "PATHOLOGY_LOC_MISMATCH", sev,
                  volume_ml=round(vol, 1), n_components=comps)


def rule_fov(organ: str, stats: OrganStats) -> Reason | None:
    if not stats.touches_z_edge:
        return None
    return reason("clin", organ, "FOV_TRUNCATED", 0.5)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/agents/clinical/test_rules.py -v` → all PASS

- [ ] **Step 5: Commit**

```bash
git add mars tests
git commit -m "feat(clinical): volume, pathology-location and FOV rules"
```

---

### Task 5: ClinicalAgent and fault-injection tests

**Files:**
- Create: `mars/agents/clinical/agent.py`
- Test: `tests/agents/clinical/test_agent.py`

**Interfaces:**
- Consumes: everything above, `volumes.load_mask`, `maskstats.mask_stats`
- Produces: `ClinicalAgent(norms: VolumeNorms, llm_fn=None)` with `run(seg: SegmentationOutput, meta: dict | None, provenance: str) -> ClinicalCheck`.

- [ ] **Step 1: Write the failing test**

```python
import math

import numpy as np
import pandas as pd
import pytest

from mars.agents.clinical.agent import ClinicalAgent
from mars.agents.clinical.norms import VolumeNorms
from mars.contracts import ORGANS, REQUIRED_EVIDENCE, OrganMask, SegmentationOutput
from mars.volumes import save_mask

# Spacing 10 mm in every axis makes one voxel exactly 1 mL, so a 10x10x10 cube is 1000 mL.
NORMS = VolumeNorms(pd.DataFrame([
    {"organ": o, "sex": "any", "age_min": 0, "age_max": 120,
     "mean_log_ml": math.log(1000), "sd_log_ml": 0.2} for o in ORGANS]))


def _seg(tmp_path, liver_mask):
    organs = {o: OrganMask() for o in ORGANS}
    if liver_mask is not None:
        save_mask(tmp_path / "liver.npy", liver_mask)
        organs["liver"] = OrganMask(prompt_source="totalseg", deployed="auto",
                                    mask_path=str(tmp_path / "liver.npy"))
    return SegmentationOutput(case_id="c", ct_path="x", spacing=(10.0, 10.0, 10.0),
                              shape=(20, 20, 20), organs=organs)


def _cube(n, start=5):
    m = np.zeros((20, 20, 20), bool); m[start:start + n, start:start + n, start:start + n] = True
    return m


META = {"age": 60.0, "sex": "m", "study_type": "ct abdomen",
        "pathology": "liver lesion", "pathology_location": "liver"}


def _liver(check):
    return check.organs["liver"]


def test_consistent_case_has_relevance_one(tmp_path):
    check = ClinicalAgent(NORMS).run(_seg(tmp_path, _cube(10)), META, "real")
    assert _liver(check).relevance == 1.0 and _liver(check).findings == []
    assert check.metadata_provenance == "real"


def test_fault_injection_shrunk_mask_fires_volume_and_pathology_rules(tmp_path):
    check = ClinicalAgent(NORMS).run(_seg(tmp_path, _cube(4)), META, "real")   # 64 mL vs 1000 mL norm
    codes = {f.code for f in _liver(check).findings}
    assert {"VOLUME_IMPLAUSIBLE", "PATHOLOGY_LOC_MISMATCH"} <= codes
    assert _liver(check).relevance == pytest.approx(0.0)


def test_fault_injection_removed_organ_named_in_history(tmp_path):
    check = ClinicalAgent(NORMS).run(_seg(tmp_path, None), META, "real")
    assert [f.code for f in _liver(check).findings] == ["PATHOLOGY_LOC_MISMATCH"]
    assert _liver(check).relevance == 0.0


def test_absent_organ_not_named_in_history_has_no_opinion(tmp_path):
    meta = {**META, "pathology": None, "pathology_location": None}
    check = ClinicalAgent(NORMS).run(_seg(tmp_path, None), meta, "real")
    assert _liver(check).relevance is None and _liver(check).findings == []


def test_fov_truncation(tmp_path):
    m = np.zeros((20, 20, 20), bool); m[0:10, 5:15, 5:15] = True     # touches z = 0
    check = ClinicalAgent(NORMS).run(_seg(tmp_path, m), {**META, "pathology": None, "pathology_location": None}, "real")
    assert [f.code for f in _liver(check).findings] == ["FOV_TRUNCATED"]
    assert _liver(check).relevance == pytest.approx(0.5)


def test_no_metadata_gives_null_relevance_not_zero(tmp_path):
    check = ClinicalAgent(NORMS).run(_seg(tmp_path, _cube(10)), None, "none")
    assert check.metadata_provenance == "none"
    for o in ORGANS:
        assert check.organs[o].relevance is None
        assert [f.code for f in check.organs[o].findings] == ["NO_CLINICAL_CONTEXT"]


def test_metadata_row_with_no_useful_fields_counts_as_no_context(tmp_path):
    empty = {"age": None, "sex": "m", "study_type": "ct", "pathology": None, "pathology_location": None}
    check = ClinicalAgent(NORMS).run(_seg(tmp_path, _cube(10)), empty, "real")
    assert check.metadata_provenance == "none"
    assert _liver(check).relevance is None


def test_age_missing_skips_volume_rule_but_pathology_rule_still_runs(tmp_path):
    meta = {**META, "age": None}
    check = ClinicalAgent(NORMS).run(_seg(tmp_path, None), meta, "real")
    assert [f.code for f in _liver(check).findings] == ["PATHOLOGY_LOC_MISMATCH"]
    small = ClinicalAgent(NORMS).run(_seg(tmp_path, _cube(4)), meta, "real")
    assert {f.code for f in _liver(small).findings} == set()    # no z-score without age, so no small-organ finding


def test_synthetic_provenance_is_passed_through(tmp_path):
    check = ClinicalAgent(NORMS).run(_seg(tmp_path, _cube(10)), META, "synthetic")
    assert check.metadata_provenance == "synthetic"


def test_every_finding_carries_the_evidence_its_template_needs(tmp_path):
    check = ClinicalAgent(NORMS).run(_seg(tmp_path, _cube(4)), META, "real")
    for f in _liver(check).findings:
        assert set(REQUIRED_EVIDENCE[f.code]) <= set(f.evidence)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/agents/clinical/test_agent.py -v` → FAIL

- [ ] **Step 3: Implement** `mars/agents/clinical/agent.py`

```python
from mars.agents.clinical.rules import rule_fov, rule_pathology_location, rule_volume
from mars.agents.clinical.vocab import tag_organs_with_fallback
from mars.contracts import (
    ClinicalCheck, ORGANS, OrganClinical, SegmentationOutput,
)
from mars.fallbacks import reason
from mars.maskstats import mask_stats
from mars.volumes import load_mask


def _has_context(meta: dict | None) -> bool:
    return bool(meta) and any(meta.get(k) for k in ("age", "pathology", "pathology_location"))


class ClinicalAgent:
    def __init__(self, norms, llm_fn=None):
        self.norms = norms
        self.llm_fn = llm_fn

    def run(self, seg: SegmentationOutput, meta: dict | None, provenance: str) -> ClinicalCheck:
        if not _has_context(meta):
            return ClinicalCheck(
                metadata_provenance="none",
                organs={o: OrganClinical(
                    relevance=None, findings=[reason("clin", o, "NO_CLINICAL_CONTEXT", 0.0)])
                    for o in ORGANS})

        text = " ".join(t for t in (meta.get("pathology_location"), meta.get("pathology")) if t)
        tagged = tag_organs_with_fallback(text, self.llm_fn)
        organs = {}
        for organ in ORGANS:
            om = seg.organs[organ]
            stats = mask_stats(load_mask(om.mask_path), seg.spacing) if om.mask_path else None
            z = (self.norms.zscore(organ, meta.get("sex"), meta.get("age"), stats.volume_ml)
                 if stats else None)
            findings = [r for r in (
                rule_volume(organ, stats, z) if stats else None,
                rule_pathology_location(organ, tagged, stats, z),
                rule_fov(organ, stats) if stats else None,
            ) if r is not None]
            if stats is None and not findings:
                relevance = None                      # nothing to cross-check
            else:
                relevance = 1.0 - max((f.severity for f in findings), default=0.0)
            organs[organ] = OrganClinical(relevance=relevance, findings=findings)
        return ClinicalCheck(metadata_provenance=provenance, organs=organs)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/agents/clinical -v` → all PASS

- [ ] **Step 5: Commit**

```bash
git add mars tests
git commit -m "feat(clinical): ClinicalAgent with fault-injection tests"
```

---

### Task 6: Agreement with hand annotations (external-set evaluation)

**Files:**
- Create: `mars/eval/clinical_agreement.py`, `scripts/eval_clinical_rules.py`
- Test: `tests/eval/test_clinical_agreement.py`

**Interfaces:**
- Consumes: `store.load_result`
- Produces: `clinical_agreement.rule_agreement(results: dict[str, CaseResult], annotations: pd.DataFrame) -> dict` where `annotations` has columns `case_id, organ, consistent` (1 = a human judged the mask consistent with the metadata, 0 = inconsistent). A rule "flags" an organ when it has a `VOLUME_IMPLAUSIBLE` or `PATHOLOGY_LOC_MISMATCH` finding with severity > 0. Returns `{"n": int, "tp": int, "fp": int, "fn": int, "tn": int, "precision": float | None, "recall": float | None}` where positive = inconsistent.

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd

from mars.contracts import CaseResult, OrganClinical, Reason
from mars.eval.clinical_agreement import rule_agreement
from tests.factories import make_clin, make_report, make_seg, make_unc


def _result(flagged_organs):
    clin = make_clin()
    for o in flagged_organs:
        clin.organs[o] = OrganClinical(relevance=0.0, findings=[Reason(
            id=f"clin:{o}:VOLUME_IMPLAUSIBLE", code="VOLUME_IMPLAUSIBLE", organ=o, severity=0.8,
            evidence={"volume_ml": 100.0, "z": -3.0})])
    return CaseResult(segmentation=make_seg(), clinical=clin, uncertainty=make_unc(), report=make_report())


def test_precision_recall():
    results = {"a": _result(["liver"]), "b": _result([])}
    ann = pd.DataFrame([
        {"case_id": "a", "organ": "liver", "consistent": 0},    # flagged, truly inconsistent -> tp
        {"case_id": "a", "organ": "spleen", "consistent": 1},   # not flagged, consistent -> tn
        {"case_id": "b", "organ": "liver", "consistent": 0},    # not flagged, inconsistent -> fn
        {"case_id": "b", "organ": "stomach", "consistent": 1},  # tn
    ])
    out = rule_agreement(results, ann)
    assert (out["tp"], out["fp"], out["fn"], out["tn"]) == (1, 0, 1, 2)
    assert out["precision"] == 1.0 and out["recall"] == 0.5


def test_undefined_when_no_positives():
    out = rule_agreement({"a": _result([])}, pd.DataFrame([{"case_id": "a", "organ": "liver", "consistent": 1}]))
    assert out["precision"] is None and out["recall"] is None
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/eval/test_clinical_agreement.py -v` → FAIL

- [ ] **Step 3: Implement**

`mars/eval/clinical_agreement.py`:
```python
import pandas as pd

RULE_CODES = {"VOLUME_IMPLAUSIBLE", "PATHOLOGY_LOC_MISMATCH"}


def rule_agreement(results: dict, annotations: pd.DataFrame) -> dict:
    tp = fp = fn = tn = 0
    for row in annotations.itertuples():
        findings = results[row.case_id].clinical.organs[row.organ].findings
        flagged = any(f.code in RULE_CODES and f.severity > 0 for f in findings)
        inconsistent = row.consistent == 0
        if flagged and inconsistent: tp += 1
        elif flagged: fp += 1
        elif inconsistent: fn += 1
        else: tn += 1
    return {"n": tp + fp + fn + tn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}
```

`scripts/eval_clinical_rules.py`:
```python
import argparse
from pathlib import Path

import pandas as pd

from mars.eval.clinical_agreement import rule_agreement
from mars.store import load_result


def main():
    p = argparse.ArgumentParser(description="Compare clinical rules with hand annotations")
    p.add_argument("--annotations", required=True, help="CSV: case_id,organ,consistent")
    p.add_argument("--cases-dir", required=True, help="folder with <case_id>/result.json")
    a = p.parse_args()
    ann = pd.read_csv(a.annotations)
    results = {c: load_result(Path(a.cases_dir) / c) for c in ann.case_id.unique()}
    print(rule_agreement(results, ann))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/eval/test_clinical_agreement.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add mars scripts tests
git commit -m "feat(clinical): rule-vs-annotation agreement metric"
```

**Team TODOs (not code):** (1) fill `configs/volume_norms.csv` with cited reference values and commit it before the external set is touched; (2) hand-annotate 30–50 external organ-cases into the annotations CSV; (3) state in the write-up that relevance is a rule-based plausibility check, not clinically validated.
