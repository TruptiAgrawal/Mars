# MARS Plan 5 — Report Agent (Agent 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the three structured outputs into a draft report where every sentence is traceable to a template and a cited reason, the ACCEPT wording never says an organ is normal, and an optional LLM can only reword flagged paragraphs after passing a strict validator.

**Architecture:** The report is built **deterministically from templates first** (this alone is a complete, shippable agent). `validate()` checks that sentences cite real inputs, carry the right claim type, contain only cited numbers, and avoid diagnostic/inference vocabulary. An optional `llm_fn` may reword FLAG paragraphs; its output replaces the template text only if validation and a vocabulary-novelty check both pass; otherwise the template text is kept.

**Tech Stack:** Python stdlib (`re`, `json`), pydantic contracts; optional `anthropic` SDK.

**Spec:** `docs/superpowers/specs/2026-10-26-mars-uncertainty-design.md` (§4 Agent 4, §7.7 faithfulness).

**Prerequisite:** Plan 1 Tasks 1–3 (contracts, factories, fallbacks). Plans 2–4 are not needed to build or test this plan (it consumes their output objects only).

## Global Constraints

- ACCEPT wording is a fixed template: it says the *segmentation* is reliable and that this is "not an assessment of the {organ}'s health or appearance". The report never says an organ is normal, healthy or unremarkable.
- Every sentence has `claim` (a reason code, `ACCEPT`, or `SUMMARY`) and `cites` (`reason:<id>` or `field:<organ>.<name>` with name ∈ `p_ok, tau, d_star, volume_ml`).
- The LLM may change only the `text` of a sentence; `claim` and `cites` are always copied from the template sentence.
- Reliability (`p_ok`) and clinical consistency (`relevance`/findings) are rendered as separate sentences; never merged.
- Summary line format: `"{k}/4 organs auto-accepted"` + `"; {names} flagged."` or `"."` (names lower-case, comma-separated).
- Banner: real metadata → none; synthetic → `"Clinical metadata for this case is synthetic (demo only)."`; none → `"No clinical metadata was available; the clinical cross-check was not performed."`
- Only structured numbers and template sentences go to an LLM; no patient identifiers or images.

## Review Focus

1. LLM returns a sentence with a changed number → rejected, template text kept (Task 4 test).
2. LLM adds an inference with no numbers ("this suggests an infection") → rejected by deny-list/novelty (Tasks 2 and 4 tests).
3. LLM call raises or returns non-JSON → template text, no crash (Task 4 test).
4. Organ flagged with no scores (`NO_PROMPT`, `p_ok = None`) → sentence renders with no numbers and no crash (Task 1 test).
5. ACCEPT organ with a clinical finding → ACCEPT sentence stays and the clinical sentence is added separately (Task 4 test).

---

### Task 1: Templates, summary, banner

**Files:**
- Create: `mars/agents/report/__init__.py`, `mars/agents/report/templates.py`
- Test: `tests/agents/report/test_templates.py` (create `tests/agents/report/__init__.py`)

**Interfaces:**
- Consumes: `contracts.*`, `tests/factories.py`
- Produces:
  - `templates.TEMPLATES: dict[str, str]` (reason code → sentence template; `NO_CLINICAL_CONTEXT` intentionally absent)
  - `templates.reason_sentence(r: Reason) -> ReportSentence` (claim = `r.code`, cites = `["reason:<id>"]`)
  - `templates.accept_sentence(organ: str, seg, unc) -> ReportSentence`
  - `templates.section_sentences(organ: str, seg, clin, unc) -> list[ReportSentence]`
  - `templates.summary_line(unc) -> str`, `templates.banner(provenance: str) -> str | None`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from mars.agents.report.templates import (
    TEMPLATES, accept_sentence, banner, reason_sentence, section_sentences, summary_line,
)
from mars.contracts import ORGANS, REQUIRED_EVIDENCE, OrganClinical, Reason
from mars.fallbacks import reason
from tests.factories import make_clin, make_seg, make_unc


def test_every_reason_code_except_no_context_has_a_template_that_formats_with_required_evidence():
    assert set(TEMPLATES) == set(REQUIRED_EVIDENCE) - {"NO_CLINICAL_CONTEXT"}
    for code in TEMPLATES:
        r = Reason(id=f"x:liver:{code}", code=code, organ="liver", severity=0.5,
                   evidence={k: 0.5 for k in REQUIRED_EVIDENCE[code]})
        s = reason_sentence(r)
        assert "{" not in s.text and s.claim == code and s.cites == [f"reason:{r.id}"]


def test_accept_sentence_is_the_fixed_template():
    s = accept_sentence("liver", make_seg(), make_unc(flag=()))
    assert s.text == ("Liver segmentation met the reliability criterion (p_ok = 0.96, threshold 0.90); "
                      "the automated outline is expected to be accurate. This is not an assessment of the "
                      "liver's health or appearance. Measured volume: 1500 mL.")
    assert s.claim == "ACCEPT"
    assert s.cites == ["field:liver.p_ok", "field:liver.tau", "field:liver.volume_ml"]


def test_flag_organ_lists_reasons_by_severity_and_prints_evidence():
    sents = section_sentences("stomach", make_seg(), make_clin(), make_unc())
    assert [s.claim for s in sents] == ["LOW_RELIABILITY", "JITTER_HIGH"]
    assert "0.71" in sents[1].text and "0.88" in sents[1].text


def test_flag_without_scores_renders_without_numbers():
    unc = make_unc(flag=())
    unc.organs["liver"].status, unc.organs["liver"].p_ok = "FLAG", None
    unc.organs["liver"].reasons = [reason("unc", "liver", "NO_PROMPT")]
    sents = section_sentences("liver", make_seg(), make_clin(), unc)
    assert [s.claim for s in sents] == ["NO_PROMPT"] and not any(c.isdigit() for c in sents[0].text)


def test_clinical_findings_are_separate_sentences_even_on_accepted_organs():
    clin = make_clin()
    clin.organs["liver"] = OrganClinical(relevance=0.5, findings=[
        reason("clin", "liver", "FOV_TRUNCATED", 0.5),
        reason("clin", "liver", "NO_CLINICAL_CONTEXT", 0.0)])
    sents = section_sentences("liver", make_seg(), clin, make_unc(flag=()))
    assert [s.claim for s in sents] == ["ACCEPT", "FOV_TRUNCATED"]      # NO_CLINICAL_CONTEXT is not a sentence


def test_summary_line():
    assert summary_line(make_unc(flag=())) == "4/4 organs auto-accepted."
    assert summary_line(make_unc(flag=("stomach", "kidney_left"))) == \
        "2/4 organs auto-accepted; left kidney, stomach flagged."


@pytest.mark.parametrize("prov,expected", [
    ("real", None),
    ("synthetic", "Clinical metadata for this case is synthetic (demo only)."),
    ("none", "No clinical metadata was available; the clinical cross-check was not performed."),
])
def test_banner(prov, expected):
    assert banner(prov) == expected
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/agents/report/test_templates.py -v` → FAIL

- [ ] **Step 3: Implement** `mars/agents/report/templates.py`

```python
from mars.contracts import (
    ClinicalCheck, ORGANS, ORGAN_LABELS, Reason, ReportSentence, SegmentationOutput,
    UncertaintyReport,
)

ACCEPT_TEMPLATE = (
    "{label} segmentation met the reliability criterion (p_ok = {p_ok:.2f}, threshold {tau:.2f}); "
    "the automated outline is expected to be accurate. This is not an assessment of the "
    "{organ_lower}'s health or appearance. Measured volume: {vol:.0f} mL."
)

TEMPLATES = {
    "JITTER_HIGH": "The outline changed when the starting prompt was nudged slightly (agreement "
                   "{jitter_dice:.2f} versus an expected minimum of {expected_min:.2f}), so the boundary "
                   "is not stable.",
    "DISAGREE_AUTOMODEL": "Two independent segmentation methods disagree on this organ (overlap "
                          "{prompt_mask_dice:.2f} versus an expected minimum of {expected_min:.2f}).",
    "HIGH_BOUNDARY_ENTROPY": "The model was unsure where the organ boundary lies (boundary uncertainty "
                             "{boundary_entropy:.2f} versus an expected maximum of {expected_max:.2f}).",
    "FRAGMENTED": "The outline consists of {n_components:.0f} separate pieces, which is unusual for a "
                  "single organ.",
    "LOW_RELIABILITY": "The combined reliability score ({p_ok:.2f}) is below the acceptance threshold "
                       "({tau:.2f}).",
    "NOT_CALIBRATABLE": "The reliability of this outline could not be established to the required level "
                        "on the development data, so it is always flagged for review.",
    "NO_PROMPT": "The automatic step did not find the {organ_lower}, so no outline was produced.",
    "AGENT_ERROR": "An internal step failed for this organ (details are in the structured output); "
                   "review is required.",
    "VOLUME_IMPLAUSIBLE": "The measured volume ({volume_ml:.0f} mL) is {z_abs:.1f} standard deviations "
                          "from the reference for this age and sex.",
    "PATHOLOGY_LOC_MISMATCH": "The clinical history mentions this organ, but its outline is missing or "
                              "atypical (volume {volume_ml:.0f} mL; {n_components:.0f} piece(s)).",
    "FOV_TRUNCATED": "The outline reaches the edge of the scan, so the organ may be cut off.",
}


def reason_sentence(r: Reason) -> ReportSentence:
    fmt = {**r.evidence, "organ_lower": ORGAN_LABELS[r.organ].lower(),
           "label": ORGAN_LABELS[r.organ]}
    if "z" in r.evidence:
        fmt["z_abs"] = abs(float(r.evidence["z"]))
    try:
        text = TEMPLATES[r.code].format(**fmt)
    except (KeyError, ValueError, TypeError):            # defensive: evidence missing or wrong type
        text = f"A check ({r.code}) was triggered for this organ; see the structured output."
    return ReportSentence(text=text, claim=r.code, cites=[f"reason:{r.id}"])


def accept_sentence(organ: str, seg: SegmentationOutput, unc: UncertaintyReport) -> ReportSentence:
    ou = unc.organs[organ]
    text = ACCEPT_TEMPLATE.format(
        label=ORGAN_LABELS[organ], organ_lower=ORGAN_LABELS[organ].lower(),
        p_ok=ou.p_ok, tau=ou.tau, vol=seg.organs[organ].volume_ml or 0.0)
    return ReportSentence(text=text, claim="ACCEPT",
                          cites=[f"field:{organ}.p_ok", f"field:{organ}.tau", f"field:{organ}.volume_ml"])


def section_sentences(organ: str, seg: SegmentationOutput, clin: ClinicalCheck,
                      unc: UncertaintyReport) -> list[ReportSentence]:
    ou = unc.organs[organ]
    if ou.status == "ACCEPT":
        out = [accept_sentence(organ, seg, unc)]
    else:
        out = [reason_sentence(r) for r in sorted(ou.reasons, key=lambda r: -r.severity)]
        if not out:
            out = [ReportSentence(text=f"{ORGAN_LABELS[organ]} was flagged for review.",
                                  claim="SUMMARY", cites=[])]
    findings = sorted(clin.organs[organ].findings, key=lambda r: -r.severity)
    out += [reason_sentence(r) for r in findings if r.code != "NO_CLINICAL_CONTEXT"]
    return out


def summary_line(unc: UncertaintyReport) -> str:
    flagged = [ORGAN_LABELS[o].lower() for o in ORGANS if unc.organs[o].status == "FLAG"]
    head = f"{len(ORGANS) - len(flagged)}/{len(ORGANS)} organs auto-accepted"
    return head + (f"; {', '.join(flagged)} flagged." if flagged else ".")


def banner(provenance: str) -> str | None:
    if provenance == "synthetic":
        return "Clinical metadata for this case is synthetic (demo only)."
    if provenance == "none":
        return "No clinical metadata was available; the clinical cross-check was not performed."
    return None
```

Note on the summary test: `ORGANS` order is liver, spleen, kidney_left, stomach, so flagged names appear as "left kidney, stomach".

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/agents/report/test_templates.py -v` → all PASS

- [ ] **Step 5: Commit**

```bash
git add mars tests
git commit -m "feat(report): deterministic templates, summary line and banner"
```

---

### Task 2: Validator

**Files:**
- Create: `mars/agents/report/validator.py`
- Test: `tests/agents/report/test_validator.py`

**Interfaces:**
- Consumes: `contracts.ReportSentence`, `templates.section_sentences` (in tests)
- Produces:
  - `validator.validate(sentences: list[ReportSentence], seg, clin, unc) -> list[str]` — empty list means valid; otherwise human-readable errors
  - `validator.novelty_ok(new_text: str, original_text: str, max_ratio: float = 0.25) -> bool`
  - `validator.DENY: re.Pattern`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from mars.agents.report.templates import section_sentences
from mars.agents.report.validator import novelty_ok, validate
from mars.contracts import ORGANS, ReportSentence
from tests.factories import make_clin, make_seg, make_unc

SEG, CLIN, UNC = make_seg(), make_clin(), make_unc()


def _s(text, claim, cites):
    return ReportSentence(text=text, claim=claim, cites=cites)


def test_template_sentences_always_validate():
    for flag in ((), ("stomach",), ORGANS):
        unc = make_unc(flag=flag)
        for organ in ORGANS:
            assert validate(section_sentences(organ, SEG, CLIN, unc), SEG, CLIN, unc) == []


def test_missing_or_unknown_cites_are_rejected():
    assert validate([_s("Something.", "JITTER_HIGH", [])], SEG, CLIN, UNC)
    assert validate([_s("Something.", "JITTER_HIGH", ["reason:unc:stomach:NOPE"])], SEG, CLIN, UNC)
    assert validate([_s("Something.", "ACCEPT", ["field:liver.colour"])], SEG, CLIN, UNC)


def test_claim_must_match_the_cited_reason():
    ok = _s("The outline changed (agreement 0.71).", "JITTER_HIGH", ["reason:unc:stomach:JITTER_HIGH"])
    bad = _s("The outline changed (agreement 0.71).", "FRAGMENTED", ["reason:unc:stomach:JITTER_HIGH"])
    assert validate([ok], SEG, CLIN, UNC) == []
    assert any("claim" in e for e in validate([bad], SEG, CLIN, UNC))


def test_numbers_must_come_from_the_cited_evidence_at_the_precision_shown():
    cite = ["reason:unc:stomach:JITTER_HIGH"]
    assert validate([_s("Agreement 0.71 versus 0.88.", "JITTER_HIGH", cite)], SEG, CLIN, UNC) == []
    assert any("0.99" in e for e in validate([_s("Agreement 0.99.", "JITTER_HIGH", cite)], SEG, CLIN, UNC))
    assert any("7" in e for e in validate([_s("Seen in 7 slices.", "JITTER_HIGH", cite)], SEG, CLIN, UNC))


@pytest.mark.parametrize("text", [
    "The stomach looks normal.", "This is an abnormal finding.", "A lesion is possible.",
    "This suggests an infection.", "The pattern indicates inflammation.", "Probably a tumour.",
    "Consistent with malignancy.", "The liver is healthy.", "This is likely due to disease.",
])
def test_diagnostic_and_inference_vocabulary_is_rejected_even_without_numbers(text):
    errs = validate([_s(text, "JITTER_HIGH", ["reason:unc:stomach:JITTER_HIGH"])], SEG, CLIN, UNC)
    assert any("vocabulary" in e for e in errs)


def test_novelty():
    original = "The outline changed when the starting prompt was nudged slightly, so the boundary is not stable."
    assert novelty_ok("The outline shifted when the starting prompt was nudged slightly, so the boundary is not stable.", original)
    assert not novelty_ok("Radiographic evidence indicates peripheral parenchymal heterogeneity requiring attention.", original)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/agents/report/test_validator.py -v` → FAIL

- [ ] **Step 3: Implement** `mars/agents/report/validator.py`

```python
import re

from mars.contracts import ReportSentence

DENY = re.compile(
    r"\b(normal|abnormal|unremarkable|healthy|diseas\w*|lesion\w*|tumou?r\w*|mass(es)?|cancer\w*|"
    r"malignan\w*|benign|diagnos\w*|patholog\w*|suggest\w*|indicat\w*|likely|probabl\w*|"
    r"consistent with|due to|caus\w*|infect\w*|inflam\w*|abscess\w*|cyst\w*|metasta\w*|"
    r"hemorrhag\w*|infarct\w*|obstruct\w*|enlarg\w*|atroph\w*)\b",
    re.IGNORECASE,
)
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
WORD = re.compile(r"[a-z]+")
STOP = set(
    "a an the of to in on at by for with without and or but so as is are was were be been being it its "
    "this that these those there their than then when while where which who what not no yes can could "
    "may might should would will shall do does did has have had from into over under about between "
    "versus per one two three s".split()          # "s" comes from possessives such as "liver's"
)
FIELDS = ("p_ok", "tau", "d_star", "volume_ml")


def _reason_index(clin, unc) -> dict:
    index = {}
    for organ_unc in unc.organs.values():
        index.update({r.id: r for r in organ_unc.reasons})
    for organ_clin in clin.organs.values():
        index.update({r.id: r for r in organ_clin.findings})
    return index


def _resolve(cite: str, seg, unc, reasons) -> list[float] | None:
    """Numbers a cite licenses, or None if the cite does not exist."""
    if cite.startswith("reason:"):
        r = reasons.get(cite[len("reason:"):])
        if r is None:
            return None
        return [float(v) for v in r.evidence.values() if isinstance(v, (int, float))]
    if cite.startswith("field:"):
        try:
            organ, name = cite[len("field:"):].split(".")
            if name not in FIELDS:
                return None
            value = (seg.organs[organ].volume_ml if name == "volume_ml"
                     else getattr(unc.organs[organ], name))
        except (ValueError, KeyError):
            return None
        return [] if value is None else [float(value)]
    return None


def validate(sentences: list[ReportSentence], seg, clin, unc) -> list[str]:
    reasons, errors = _reason_index(clin, unc), []
    for i, s in enumerate(sentences):
        tag = f"sentence {i}"
        if not s.cites:
            errors.append(f"{tag}: no cites")
            continue
        allowed: list[float] = []
        for c in s.cites:
            nums = _resolve(c, seg, unc, reasons)
            if nums is None:
                errors.append(f"{tag}: unknown cite {c}")
            else:
                allowed += nums + [abs(n) for n in nums]
        cited_codes = {reasons[c[len('reason:'):]].code for c in s.cites
                       if c.startswith("reason:") and c[len("reason:"):] in reasons}
        fields_only = all(c.startswith("field:") for c in s.cites)
        if not (s.claim in cited_codes or (s.claim == "ACCEPT" and fields_only)):
            errors.append(f"{tag}: claim {s.claim} does not match its cites")
        for token in NUMBER.findall(s.text):
            decimals = len(token.split(".")[1]) if "." in token else 0
            if not any(round(v, decimals) == float(token) for v in allowed):
                errors.append(f"{tag}: number {token} is not supported by its cites")
        if DENY.search(s.text):
            errors.append(f"{tag}: forbidden vocabulary '{DENY.search(s.text).group(0)}'")
    return errors


def novelty_ok(new_text: str, original_text: str, max_ratio: float = 0.25) -> bool:
    """Reject rewordings that bring in many words not present in the template sentence."""
    known = set(WORD.findall(original_text.lower())) | STOP
    words = WORD.findall(new_text.lower())
    if not words:
        return False
    return sum(w not in known for w in words) / len(words) <= max_ratio
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/agents/report/test_validator.py -v` → all PASS. If `test_template_sentences_always_validate` fails, a template contains a deny-listed word or a number the cite does not license: fix the template, not the validator.

- [ ] **Step 5: Commit**

```bash
git add mars tests
git commit -m "feat(report): sentence validator (cites, claim, numbers, vocabulary) and novelty check"
```

---

### Task 3: LLM rewording helper

**Files:**
- Create: `mars/agents/report/llm.py`
- Test: `tests/agents/report/test_llm.py`

**Interfaces:**
- Produces:
  - `llm.paraphrase_section(llm_fn, sentences: list[ReportSentence]) -> list[ReportSentence] | None` — `llm_fn(prompt: str) -> str`; returns sentences with only `text` changed, or `None` on any error, wrong length, or non-string items
  - `llm.anthropic_llm_fn(model: str = "claude-sonnet-5-5", max_tokens: int = 1000)` → a function `llm_fn(prompt) -> str` (needs `pip install anthropic` and `ANTHROPIC_API_KEY`)
  - `llm.PROMPT: str` (ends with the JSON list of sentences)

- [ ] **Step 1: Write the failing test**

```python
import json

from mars.agents.report.llm import PROMPT, paraphrase_section
from mars.contracts import ReportSentence

SENTS = [
    ReportSentence(text="The outline changed (agreement 0.71).", claim="JITTER_HIGH", cites=["reason:a"]),
    ReportSentence(text="The score (0.42) is low.", claim="LOW_RELIABILITY", cites=["reason:b"]),
]


def _llm_returning(items):
    return lambda prompt: "Here you go:\n" + json.dumps(items) + "\nDone."


def test_prompt_ends_with_the_sentences_and_forbids_new_facts():
    assert "do not add" in PROMPT.lower() and "{items}" in PROMPT


def test_only_text_changes_claim_and_cites_are_kept():
    out = paraphrase_section(_llm_returning(["The outline shifted (agreement 0.71).", "The score (0.42) is low."]), SENTS)
    assert [s.text for s in out][0].startswith("The outline shifted")
    assert [(s.claim, s.cites) for s in out] == [(s.claim, s.cites) for s in SENTS]


def test_bad_outputs_return_none():
    assert paraphrase_section(_llm_returning(["only one"]), SENTS) is None            # wrong length
    assert paraphrase_section(_llm_returning(["a", 3]), SENTS) is None                # non-string
    assert paraphrase_section(_llm_returning(["a", "  "]), SENTS) is None             # blank
    assert paraphrase_section(lambda p: "no json here", SENTS) is None
    assert paraphrase_section(lambda p: (_ for _ in ()).throw(RuntimeError("api down")), SENTS) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/agents/report/test_llm.py -v` → FAIL

- [ ] **Step 3: Implement** `mars/agents/report/llm.py`

```python
import json

from mars.contracts import ReportSentence

PROMPT = (
    "Reword each sentence below so it reads naturally. Rules: keep every number exactly as written; "
    "do not add new facts, causes, diagnoses or medical terms; do not say whether any organ is normal "
    "or abnormal. Return ONLY a JSON list of strings, same length and same order.\n"
    "{items}"
)


def paraphrase_section(llm_fn, sentences: list[ReportSentence]) -> list[ReportSentence] | None:
    try:
        raw = llm_fn(PROMPT.format(items=json.dumps([s.text for s in sentences])))
        data = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
    except Exception:  # noqa: BLE001 - any failure means "keep the template text"
        return None
    if (not isinstance(data, list) or len(data) != len(sentences)
            or not all(isinstance(t, str) and t.strip() for t in data)):
        return None
    return [s.model_copy(update={"text": t.strip()}) for s, t in zip(sentences, data)]


def anthropic_llm_fn(model: str = "claude-sonnet-5-5", max_tokens: int = 1000):
    """Thin wrapper over the Anthropic SDK. Sends only template sentences (no identifiers, no images)."""
    import anthropic
    client = anthropic.Anthropic()

    def call(prompt: str) -> str:
        msg = client.messages.create(model=model, max_tokens=max_tokens,
                                     messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text

    return call
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/agents/report/test_llm.py -v` → all PASS

- [ ] **Step 5: Commit**

```bash
git add mars tests
git commit -m "feat(report): constrained LLM rewording helper"
```

---

### Task 4: ReportAgent

**Files:**
- Create: `mars/agents/report/agent.py`
- Test: `tests/agents/report/test_agent.py`

**Interfaces:**
- Consumes: `section_sentences`, `summary_line`, `banner`, `paraphrase_section`, `validate`, `novelty_ok`
- Produces: `ReportAgent(llm_fn=None)` with `run(seg, clin, unc) -> Report`. `OrganSection.used_llm` is `True` only for FLAG organs whose reworded text passed all checks.

- [ ] **Step 1: Write the failing test**

```python
import json

from mars.agents.report.agent import ReportAgent
from mars.agents.report.llm import PROMPT
from mars.contracts import ORGANS, OrganClinical
from mars.fallbacks import reason
from tests.factories import make_clin, make_seg, make_unc


def _items(prompt):
    return json.loads(prompt[prompt.index("["):])


def _llm(transform):
    return lambda prompt: json.dumps([transform(t) for t in _items(prompt)])


def _stomach(report):
    return next(s for s in report.sections if s.organ == "stomach")


def _text(section):
    return " ".join(s.text for s in section.sentences)


def test_without_llm_report_is_template_only():
    r = ReportAgent().run(make_seg(), make_clin(), make_unc())
    assert r.summary == "3/4 organs auto-accepted; stomach flagged."
    assert [s.organ for s in r.sections] == list(ORGANS)
    assert all(not s.used_llm for s in r.sections) and r.banner is None
    everything = " ".join(_text(s) for s in r.sections).lower()
    assert "normal" not in everything and "healthy" not in everything
    assert "not an assessment" in _text(r.sections[0])


def test_good_rewording_is_used_only_for_flagged_organs():
    r = ReportAgent(_llm(lambda t: t.replace("The outline changed", "The outline shifted"))).run(
        make_seg(), make_clin(), make_unc())
    assert _stomach(r).used_llm and "shifted" in _text(_stomach(r))
    assert not r.sections[0].used_llm and "met the reliability criterion" in _text(r.sections[0])


def test_changed_number_falls_back_to_template():
    r = ReportAgent(_llm(lambda t: t.replace("0.71", "0.99"))).run(make_seg(), make_clin(), make_unc())
    assert not _stomach(r).used_llm and "0.71" in _text(_stomach(r))


def test_inference_without_numbers_falls_back_to_template():
    r = ReportAgent(_llm(lambda t: t + " This suggests an infection.")).run(make_seg(), make_clin(), make_unc())
    assert not _stomach(r).used_llm and "infection" not in _text(_stomach(r))


def test_heavy_rewording_with_new_vocabulary_falls_back():
    r = ReportAgent(_llm(lambda t: "Peripheral parenchymal heterogeneity requires radiographic attention 0.71 0.88 0.42 0.90")).run(
        make_seg(), make_clin(), make_unc())
    assert not _stomach(r).used_llm


def test_llm_failure_or_garbage_falls_back_without_crashing():
    def boom(prompt):
        raise RuntimeError("api down")

    for llm in (boom, lambda p: "not json"):
        r = ReportAgent(llm).run(make_seg(), make_clin(), make_unc())
        assert not _stomach(r).used_llm and "0.71" in _text(_stomach(r))


def test_clinical_finding_on_accepted_organ_is_added_separately():
    clin = make_clin()
    clin.organs["liver"] = OrganClinical(relevance=0.5, findings=[reason("clin", "liver", "FOV_TRUNCATED", 0.5)])
    liver = ReportAgent().run(make_seg(), clin, make_unc()).sections[0]
    assert liver.status == "ACCEPT" and [s.claim for s in liver.sentences] == ["ACCEPT", "FOV_TRUNCATED"]


def test_banner_follows_provenance_and_reliability_axes_stay_separate():
    r = ReportAgent().run(make_seg(), make_clin("none"), make_unc())
    assert r.banner.startswith("No clinical metadata")
    assert "synthetic" in ReportAgent().run(make_seg(), make_clin("synthetic"), make_unc()).banner


def test_no_prompt_organ_has_no_numbers_and_does_not_crash():
    unc = make_unc(flag=())
    ou = unc.organs["spleen"]
    ou.status, ou.p_ok, ou.tau = "FLAG", None, None
    ou.reasons = [reason("unc", "spleen", "NO_PROMPT")]
    r = ReportAgent().run(make_seg(), make_clin(), unc)
    spleen = next(s for s in r.sections if s.organ == "spleen")
    assert spleen.status == "FLAG" and not any(c.isdigit() for c in _text(spleen))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/agents/report/test_agent.py -v` → FAIL

- [ ] **Step 3: Implement** `mars/agents/report/agent.py`

```python
from mars.agents.report.llm import paraphrase_section
from mars.agents.report.templates import banner, section_sentences, summary_line
from mars.agents.report.validator import novelty_ok, validate
from mars.contracts import ORGANS, OrganSection, Report


class ReportAgent:
    def __init__(self, llm_fn=None):
        self.llm_fn = llm_fn

    def _reworded(self, template_sentences, seg, clin, unc):
        new = paraphrase_section(self.llm_fn, template_sentences)
        if new is None or validate(new, seg, clin, unc):
            return None
        if not all(novelty_ok(n.text, o.text) for n, o in zip(new, template_sentences)):
            return None
        return new

    def run(self, seg, clin, unc) -> Report:
        sections = []
        for organ in ORGANS:
            sentences = section_sentences(organ, seg, clin, unc)
            status, used = unc.organs[organ].status, False
            if status == "FLAG" and self.llm_fn is not None:
                new = self._reworded(sentences, seg, clin, unc)
                if new is not None:
                    sentences, used = new, True
            sections.append(OrganSection(organ=organ, status=status, sentences=sentences, used_llm=used))
        return Report(case_id=seg.case_id, summary=summary_line(unc),
                      banner=banner(clin.metadata_provenance), sections=sections)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/agents/report -v` → all PASS. Then run Plan 1's golden tests: `pytest tests/test_golden.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add mars tests
git commit -m "feat(report): ReportAgent with validated LLM rewording and template fallback"
```

---

### Task 5: Faithfulness audit export

**Files:**
- Create: `mars/eval/report_audit.py`, `scripts/export_report_audit.py`
- Test: `tests/eval/test_report_audit.py`

**Interfaces:**
- Consumes: `store.load_result`
- Produces: `report_audit.audit_rows(results: dict[str, CaseResult], n: int, seed: int = 0) -> pd.DataFrame` — a random sample of `n` sentences across all cases with columns `case_id, organ, sentence, claim, cites, used_llm, supported, diagnostic_claim, numbers_ok` (the last three blank, for the team to fill by hand).

- [ ] **Step 1: Write the failing test**

```python
from mars.agents.report.agent import ReportAgent
from mars.contracts import CaseResult
from mars.eval.report_audit import audit_rows
from tests.factories import make_clin, make_seg, make_unc


def _result(cid):
    seg, clin, unc = make_seg(cid), make_clin(), make_unc()
    return CaseResult(segmentation=seg, clinical=clin, uncertainty=unc, report=ReportAgent().run(seg, clin, unc))


def test_audit_sample_is_reproducible_and_has_blank_columns():
    results = {"a": _result("a"), "b": _result("b")}
    df = audit_rows(results, n=5, seed=1)
    assert len(df) == 5 and df.equals(audit_rows(results, n=5, seed=1))
    assert {"case_id", "organ", "sentence", "claim", "cites", "used_llm"} <= set(df.columns)
    assert (df[["supported", "diagnostic_claim", "numbers_ok"]] == "").all().all()


def test_n_larger_than_available_returns_everything():
    results = {"a": _result("a")}
    total = sum(len(s.sentences) for s in results["a"].report.sections)
    assert len(audit_rows(results, n=999)) == total
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/eval/test_report_audit.py -v` → FAIL

- [ ] **Step 3: Implement**

`mars/eval/report_audit.py`:
```python
import pandas as pd


def audit_rows(results: dict, n: int, seed: int = 0) -> pd.DataFrame:
    rows = []
    for case_id, result in results.items():
        for section in result.report.sections:
            for s in section.sentences:
                rows.append({"case_id": case_id, "organ": section.organ, "sentence": s.text,
                             "claim": s.claim, "cites": ";".join(s.cites), "used_llm": section.used_llm,
                             "supported": "", "diagnostic_claim": "", "numbers_ok": ""})
    df = pd.DataFrame(rows)
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
```

`scripts/export_report_audit.py`:
```python
import argparse
from pathlib import Path

from mars.eval.report_audit import audit_rows
from mars.store import load_result


def main():
    p = argparse.ArgumentParser(description="Export sentences for the manual faithfulness audit")
    p.add_argument("--cases-dir", required=True)
    p.add_argument("--n", type=int, default=60, help="sentences to sample (about 20 reports)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    results = {d.name: load_result(d) for d in sorted(Path(a.cases_dir).iterdir()) if (d / "result.json").exists()}
    audit_rows(results, a.n).to_csv(a.out, index=False)
    print(f"wrote {a.out}. Fill supported / diagnostic_claim / numbers_ok (y/n), then report the unsupported-sentence rate.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/eval/test_report_audit.py -v` → PASS; then the full suite `pytest -v`.

- [ ] **Step 5: Commit**

```bash
git add mars scripts tests
git commit -m "feat(report): manual faithfulness audit export"
```

---

## After this plan

- Report the validator pass rate and fallback-to-template rate over all reports (count `used_llm` per organ section).
- Run the manual audit: sample about 20 reports, fill the three audit columns, and report the unsupported-sentence rate.
- **Residual risk to state in the write-up:** the LLM is limited to rewording template sentences (claim and cites are fixed; numbers, vocabulary and novelty are checked), but a fluent rewording could still shift emphasis. The manual audit is the check for that.
