# MARS Plan 4 — Uncertainty Agent and Calibration (Agent 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For every organ, estimate `p_ok = P(Dice ≥ d*ₒ)` from six cheap signals using a logistic-regression calibrator, decide ACCEPT/FLAG against a per-organ threshold `τₒ`, and explain every FLAG with structured reasons. Includes the offline tooling that builds the feature table, checks feasibility, fits the calibrator and evaluates it with nested patient-level cross-validation.

**Architecture:** `compute_features` perturbs the box prompt and measures mask disagreement, plus agreement with the prompt-source mask, boundary entropy and simple anatomy features. `Calibrator` (shared slopes, per-organ intercept) turns features into `p_ok`; `choose_tau` picks `τₒ` for a precision target `α`. `UncertaintyAgent` applies the frozen calibrator online. Offline scripts produce the table and the frozen `configs/calibration.json`.

**Tech Stack:** numpy, pandas, scikit-learn (LogisticRegression), pytest.

**Spec:** `docs/superpowers/specs/2026-10-26-mars-uncertainty-design.md` (§6 target and feasibility, §7 protocol).

**Prerequisites:** Plan 1 Tasks 1–3 (contracts, helpers, fallbacks); Plan 2 Tasks 3 and 5–6 (`apply_deploy`, `DevConfig`, `load_gt`, `run_dev_seg`, `configs/deploy.json`).

## Global Constraints

- `p_ok` target is `1[Dice ≥ d*ₒ]`; `d*ₒ` comes from `configs/targets.json` (rule: min(0.90, cited inter-observer Dice), fixed before external-set results). The precision target is named **α** (default 0.90) and is not the Dice target.
- Exactly six features, in this order: `jitter_dice, prompt_mask_dice, boundary_entropy, log_volume, n_components, hu_mean`.
- Calibrator: logistic regression, L2 (`C=1.0`), **shared slopes with a per-organ intercept**, features z-scored with training statistics only.
- An organ is calibratable only with ≥ 8 positives and ≥ 8 negatives; otherwise always FLAG (`NOT_CALIBRATABLE`).
- ECE uses 5 equal-mass bins. All CIs resample **patients**, never rows.
- Each patient is held out exactly once per repeat; the calibrator never sees a held-out patient (leakage test).
- Organs without a prompt (`NO_PROMPT`) never enter the calibrator; they always FLAG.
- `UncertaintyReport` carries `calibration_version` and `targets_hash`; the agent refuses a calibrator whose hash does not match `targets.json`.
- Jitter uses a fixed seed (`seed=0`), so the same scan gives the same report.

## Review Focus

1. Segmentation left `prompt_mask_dice`/`boundary_entropy` empty for one organ (refiner crashed) → only that organ gets `AGENT_ERROR`; others unaffected (Task 4 test).
2. Calibration file built from different `targets.json` → agent refuses to start (Task 4 test).
3. An organ almost never reaches `d*ₒ` → `NOT_CALIBRATABLE`, `p_ok = None`, always FLAG, run does not crash (Tasks 3–4 tests).
4. No τ reaches the precision target α → organ always FLAG with a `NOT_CALIBRATABLE` reason, never silently ACCEPT (Task 3 test).
5. Same scan twice → identical report (Task 4 test); every FLAG carries ≥ 1 reason with the evidence keys the report template prints (Task 4 test).

---

### Task 1: Features (jitter, anatomy)

**Files:**
- Create: `mars/agents/uncertainty/__init__.py`, `mars/agents/uncertainty/features.py`
- Test: `tests/agents/uncertainty/test_features.py` (create `tests/agents/uncertainty/__init__.py`)

**Interfaces:**
- Consumes: `OrganMask`, `Refiner.refine`, `metrics.dice`, `maskstats.mask_stats`, `volumes.load_mask`
- Produces:
  - `features.FEATURE_NAMES: list[str]`
  - `features.jitter_boxes(box: list[int], shape, n: int, rng: np.random.Generator, shift_frac=0.05, scale_frac=0.05) -> list[list[int]]`
  - `features.mean_pairwise_dice(masks: list[np.ndarray]) -> float`
  - `features.compute_features(ct, spacing, om: OrganMask, refiner, rng, n_jitter=8) -> dict[str, float]` (raises `ValueError` if `om` lacks `mask_path`, `box_prompt`, `prompt_mask_dice` or `boundary_entropy`)

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest

from mars.agents.uncertainty.features import (
    FEATURE_NAMES, compute_features, jitter_boxes, mean_pairwise_dice,
)
from mars.contracts import OrganMask
from mars.volumes import save_mask


def test_jitter_boxes_valid_deterministic_and_zero_jitter_is_identity():
    box, shape = [2, 2, 2, 12, 12, 12], (16, 16, 16)
    a = jitter_boxes(box, shape, 20, np.random.default_rng(0))
    b = jitter_boxes(box, shape, 20, np.random.default_rng(0))
    assert a == b and len(a) == 20
    for z0, y0, x0, z1, y1, x1 in a:
        assert 0 <= z0 < z1 <= 16 and 0 <= y0 < y1 <= 16 and 0 <= x0 < x1 <= 16
    same = jitter_boxes(box, shape, 3, np.random.default_rng(0), shift_frac=0.0, scale_frac=0.0)
    assert all(s == box for s in same)


def test_box_at_volume_edge_stays_inside():
    boxes = jitter_boxes([0, 0, 0, 16, 16, 16], (16, 16, 16), 10, np.random.default_rng(1), 0.2, 0.2)
    assert all(min(b) >= 0 and max(b[3:]) <= 16 for b in boxes)


def test_mean_pairwise_dice():
    a = np.zeros((4, 4, 4), bool); a[:2] = True
    assert mean_pairwise_dice([a, a, a]) == 1.0
    assert mean_pairwise_dice([a, ~a]) == 0.0
    assert mean_pairwise_dice([a]) == 1.0


class BoxRefiner:
    def refine(self, ct, box):
        z0, y0, x0, z1, y1, x1 = box
        m = np.zeros(ct.shape, bool); m[z0:z1, y0:y1, x0:x1] = True
        return m, m.astype(np.float32)


def _om(tmp_path, **over):
    m = np.zeros((16, 16, 16), bool); m[4:10, 4:10, 4:10] = True       # 216 voxels
    save_mask(tmp_path / "m.npy", m)
    base = dict(prompt_source="totalseg", deployed="auto", mask_path=str(tmp_path / "m.npy"),
                box_prompt=[2, 2, 2, 12, 12, 12], prompt_mask_dice=0.9, boundary_entropy=0.2)
    base.update(over)
    return OrganMask(**base)


def test_compute_features(tmp_path):
    ct = np.full((16, 16, 16), 40.0, np.float32)
    f = compute_features(ct, (1.0, 1.0, 1.0), _om(tmp_path), BoxRefiner(), np.random.default_rng(0), n_jitter=4)
    assert list(f) == FEATURE_NAMES
    assert 0.0 < f["jitter_dice"] <= 1.0
    assert f["prompt_mask_dice"] == 0.9 and f["boundary_entropy"] == 0.2
    assert f["log_volume"] == pytest.approx(np.log(0.216))
    assert f["n_components"] == 1.0 and f["hu_mean"] == 40.0


@pytest.mark.parametrize("missing", ["prompt_mask_dice", "boundary_entropy", "box_prompt", "mask_path"])
def test_missing_inputs_raise_value_error(tmp_path, missing):
    om = _om(tmp_path, **{missing: None})
    with pytest.raises(ValueError, match=missing):
        compute_features(np.zeros((16, 16, 16), np.float32), (1.0, 1.0, 1.0), om, BoxRefiner(),
                         np.random.default_rng(0))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/agents/uncertainty/test_features.py -v` → FAIL

- [ ] **Step 3: Implement** `mars/agents/uncertainty/features.py`

```python
import itertools

import numpy as np

from mars.contracts import OrganMask
from mars.maskstats import mask_stats
from mars.metrics import dice
from mars.volumes import load_mask

FEATURE_NAMES = ["jitter_dice", "prompt_mask_dice", "boundary_entropy",
                 "log_volume", "n_components", "hu_mean"]


def jitter_boxes(box, shape, n, rng, shift_frac=0.05, scale_frac=0.05):
    """n randomly shifted/scaled copies of a [z0,y0,x0,z1,y1,x1] box, clipped to the volume."""
    lo = np.array(box[:3], float)
    hi = np.array(box[3:], float)
    size = hi - lo
    out = []
    for _ in range(n):
        centre = (lo + hi) / 2 + rng.uniform(-shift_frac, shift_frac, 3) * size
        half = size * (1 + rng.uniform(-scale_frac, scale_frac, 3)) / 2
        nlo = np.maximum(np.floor(centre - half), 0)
        nhi = np.minimum(np.ceil(centre + half), shape)
        nhi = np.maximum(nhi, nlo + 1)                      # never an empty box
        out.append([int(v) for v in (*nlo, *nhi)])
    return out


def mean_pairwise_dice(masks) -> float:
    if len(masks) < 2:
        return 1.0
    return float(np.mean([dice(a, b) for a, b in itertools.combinations(masks, 2)]))


def compute_features(ct, spacing, om: OrganMask, refiner, rng, n_jitter: int = 8) -> dict[str, float]:
    for field in ("mask_path", "box_prompt", "prompt_mask_dice", "boundary_entropy"):
        if getattr(om, field) is None:
            raise ValueError(f"organ mask lacks {field}")
    mask = load_mask(om.mask_path)
    stats = mask_stats(mask, spacing)
    boxes = jitter_boxes(om.box_prompt, ct.shape, n_jitter, rng)
    jitter_masks = [refiner.refine(ct, b)[0] for b in boxes]
    return {
        "jitter_dice": mean_pairwise_dice(jitter_masks),
        "prompt_mask_dice": float(om.prompt_mask_dice),
        "boundary_entropy": float(om.boundary_entropy),
        "log_volume": float(np.log(max(stats.volume_ml, 1e-3))),
        "n_components": float(stats.n_components),
        "hu_mean": float(ct[mask].mean()) if mask.any() else 0.0,
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/agents/uncertainty/test_features.py -v` → all PASS

- [ ] **Step 5: Commit**

```bash
git add mars tests
git commit -m "feat(unc): jitter and anatomy features"
```

**Compute note:** `compute_features` runs `refiner.refine` `n_jitter` times per organ. If your Slide-SAM wrapper re-encodes the CT on every call this is slow; cache the image embeddings inside `slide_sam_infer.infer` (keyed by the CT array id) if the run time on the lab is too long. Measure and record the seconds per case in the write-up.

---

### Task 2: Metrics and threshold selection

**Files:**
- Create: `mars/eval/metrics.py`
- Test: `tests/eval/test_calibration_metrics.py`

**Interfaces:**
- Produces:
  - `metrics.ece(p, y, bins=5) -> float` (equal-mass bins)
  - `metrics.brier(p, y) -> float`
  - `metrics.auroc(p, y) -> float` (NaN if only one class)
  - `metrics.aurc(p, dice) -> float` (mean of `1 − mean Dice of the top-k by p`, over k = 1..n)
  - `metrics.choose_tau(p, y, alpha: float, min_accept: int = 3) -> float | None` (smallest τ among the observed `p` values whose accepted set `p ≥ τ` has ≥ `min_accept` cases and precision ≥ α)

- [ ] **Step 1: Write the failing test**

```python
import math

import numpy as np
import pytest

from mars.eval.metrics import aurc, auroc, brier, choose_tau, ece


def test_ece_zero_when_probabilities_are_outcomes_and_large_when_overconfident():
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    assert ece(y.astype(float), y) == pytest.approx(0.0)
    assert ece(np.full(10, 0.9), y) == pytest.approx(0.4)


def test_brier():
    assert brier(np.array([1.0, 0.0]), np.array([1, 0])) == 0.0
    assert brier(np.array([0.5, 0.5]), np.array([1, 0])) == 0.25


def test_auroc_and_degenerate_case():
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert math.isnan(auroc(np.array([0.1, 0.2]), np.array([1, 1])))


def test_aurc_prefers_good_ranking():
    dice = np.array([0.95, 0.9, 0.5, 0.4])
    assert aurc(np.array([0.9, 0.8, 0.3, 0.2]), dice) < aurc(np.array([0.2, 0.3, 0.8, 0.9]), dice)


def test_choose_tau():
    p, y = np.array([0.95, 0.9, 0.8, 0.4]), np.array([1, 1, 0, 0])
    assert choose_tau(p, y, 0.9, min_accept=2) == 0.9
    assert choose_tau(p, y, 0.9, min_accept=3) is None          # needs 3 accepted but only 2 are clean
    assert choose_tau(p, y, 0.99, min_accept=2) == 0.9
    assert choose_tau(np.array([0.9, 0.8]), np.array([0, 0]), 0.9, min_accept=1) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/eval/test_calibration_metrics.py -v` → FAIL

- [ ] **Step 3: Implement** `mars/eval/metrics.py`

```python
import numpy as np
from sklearn.metrics import roc_auc_score


def ece(p, y, bins: int = 5) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    order = np.argsort(p, kind="stable")           # stable: ties keep input order, result is reproducible
    p, y = p[order], y[order]
    total = 0.0
    for idx in np.array_split(np.arange(len(p)), bins):
        if len(idx):
            total += len(idx) / len(p) * abs(p[idx].mean() - y[idx].mean())
    return float(total)


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def auroc(p, y) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def aurc(p, dice) -> float:
    order = np.argsort(-np.asarray(p, float), kind="stable")
    d = np.asarray(dice, float)[order]
    return float(np.mean(1.0 - np.cumsum(d) / np.arange(1, len(d) + 1)))


def choose_tau(p, y, alpha: float, min_accept: int = 3):
    p, y = np.asarray(p, float), np.asarray(y, float)
    for tau in np.unique(p):                       # ascending: smallest valid tau wins
        accepted = p >= tau
        if accepted.sum() >= min_accept and y[accepted].mean() >= alpha:
            return float(tau)
    return None
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/eval/test_calibration_metrics.py -v` → all PASS

- [ ] **Step 5: Commit**

```bash
git add mars tests
git commit -m "feat(unc): ECE, Brier, AUROC, AURC, tau selection"
```

---

### Task 3: Calibrator

**Files:**
- Create: `mars/agents/uncertainty/calibrator.py`, `tests/agents/uncertainty/synth.py`
- Test: `tests/agents/uncertainty/test_calibrator.py`

**Interfaces:**
- Consumes: `FEATURE_NAMES`, `choose_tau`, `config.targets_hash`, `fallbacks.reason`
- Produces:
  - `calibrator.is_calibratable(n_pos: int, n_neg: int, min_each: int = 8) -> bool`
  - `calibrator.Calibrator` with attributes `version, targets_hash, calibratable: dict[str, bool], tau: dict[str, float | None], reason_bounds: dict[str, float]` and methods:
    - `labels(df, d_star) -> pd.Series` (static; `1[dice ≥ d*ₒ]`)
    - `fit(df, d_star: dict[str, float], version: str = "dev") -> Calibrator` (df columns: `organ, dice` + `FEATURE_NAMES`)
    - `predict_df(df) -> np.ndarray` (NaN for non-calibratable organs)
    - `predict_p(organ: str, feats: dict[str, float]) -> float`
    - `fit_tau(df, d_star, alpha: float, min_accept: int = 3) -> None`
    - `reasons(organ: str, feats: dict[str, float], p_ok: float, tau: float | None) -> list[Reason]` (first reason is `LOW_RELIABILITY`, or `NOT_CALIBRATABLE` when `tau is None`; then one reason per out-of-range signal)
    - `save(path)`, `Calibrator.load(path)`
  - Test helper `tests/agents/uncertainty/synth.py: synthetic_table(n=60, seed=0) -> pd.DataFrame`, `D_STAR`

- [ ] **Step 1: Write the test helper** — `tests/agents/uncertainty/synth.py`

```python
import numpy as np
import pandas as pd

from mars.contracts import ORGANS

D_STAR = {o: 0.85 for o in ORGANS}


def synthetic_table(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Dice follows jitter_dice (plus noise), so the calibrator has a real signal to find.
    About 37% of cases reach d*=0.85, so every training fold has well over 8 positives and 8 negatives."""
    rng = np.random.default_rng(seed)
    rows = []
    for organ in ORGANS:
        for i in range(n):
            j = rng.uniform(0.6, 1.0)
            rows.append({
                "patient": f"p{i}", "fold": i % 5, "organ": organ,
                "dice": j + rng.normal(0, 0.02),
                "jitter_dice": j, "prompt_mask_dice": rng.uniform(0.8, 1.0),
                "boundary_entropy": rng.uniform(0, 0.5), "log_volume": rng.normal(7, 0.1),
                "n_components": 1.0, "hu_mean": rng.normal(60, 5),
            })
    return pd.DataFrame(rows)
```

- [ ] **Step 2: Write the failing test**

```python
import numpy as np
import pytest

from mars.agents.uncertainty.calibrator import Calibrator, is_calibratable
from mars.agents.uncertainty.features import FEATURE_NAMES
from mars.contracts import ORGANS, REQUIRED_EVIDENCE
from mars.config import targets_hash
from tests.agents.uncertainty.synth import D_STAR, synthetic_table


def _feats(**over):
    base = {"jitter_dice": 0.95, "prompt_mask_dice": 0.95, "boundary_entropy": 0.1,
            "log_volume": 7.0, "n_components": 1.0, "hu_mean": 60.0}
    base.update(over)
    return base


def test_is_calibratable():
    assert is_calibratable(8, 8) and not is_calibratable(7, 50) and not is_calibratable(50, 7)


def test_fit_predict_is_monotone_in_jitter_and_records_hash():
    cal = Calibrator().fit(synthetic_table(), D_STAR, version="t1")
    assert all(cal.calibratable.values()) and cal.version == "t1"
    assert cal.targets_hash == targets_hash(D_STAR)
    hi = cal.predict_p("liver", _feats(jitter_dice=0.99))
    lo = cal.predict_p("liver", _feats(jitter_dice=0.65))
    assert 0.0 <= lo < hi <= 1.0


def test_organ_that_never_reaches_target_is_not_calibratable():
    df = synthetic_table()
    df.loc[df.organ == "stomach", "dice"] = 0.5
    cal = Calibrator().fit(df, D_STAR)
    assert cal.calibratable["stomach"] is False and cal.calibratable["liver"] is True
    assert np.isnan(cal.predict_p("stomach", _feats()))
    cal.fit_tau(df, D_STAR, alpha=0.9)
    assert cal.tau["stomach"] is None and cal.tau["liver"] is not None


def test_fit_tau_meets_precision_target_on_training_data():
    df = synthetic_table()
    cal = Calibrator().fit(df, D_STAR)
    cal.fit_tau(df, D_STAR, alpha=0.9)
    sub = df[df.organ == "liver"]
    p = cal.predict_df(sub)
    accepted = p >= cal.tau["liver"]
    assert Calibrator.labels(sub, D_STAR).to_numpy()[accepted].mean() >= 0.9


def test_all_organs_degenerate_gives_no_model_and_nan_predictions():
    df = synthetic_table(); df["dice"] = 0.1
    cal = Calibrator().fit(df, D_STAR)
    assert not any(cal.calibratable.values())
    assert np.isnan(cal.predict_p("liver", _feats()))


def test_reasons_include_overall_and_specific_with_required_evidence():
    cal = Calibrator().fit(synthetic_table(), D_STAR)
    rs = cal.reasons("liver", _feats(jitter_dice=0.1, prompt_mask_dice=0.1,
                                     boundary_entropy=5.0, n_components=3.0), 0.3, 0.9)
    codes = [r.code for r in rs]
    assert codes[0] == "LOW_RELIABILITY"
    assert {"JITTER_HIGH", "DISAGREE_AUTOMODEL", "HIGH_BOUNDARY_ENTROPY", "FRAGMENTED"} <= set(codes)
    for r in rs:
        assert set(REQUIRED_EVIDENCE[r.code]) <= set(r.evidence) and 0.0 <= r.severity <= 1.0


def test_reasons_when_no_tau_reaches_alpha():
    cal = Calibrator().fit(synthetic_table(), D_STAR)
    assert cal.reasons("liver", _feats(), 0.7, None)[0].code == "NOT_CALIBRATABLE"


def test_save_load_round_trip(tmp_path):
    df = synthetic_table()
    cal = Calibrator().fit(df, D_STAR, version="v9"); cal.fit_tau(df, D_STAR, 0.9)
    cal.save(tmp_path / "c.json")
    again = Calibrator.load(tmp_path / "c.json")
    assert again.version == "v9" and again.tau == cal.tau and again.calibratable == cal.calibratable
    sample = df.head(20)
    np.testing.assert_allclose(again.predict_df(sample), cal.predict_df(sample))


def test_feature_names_match_table_columns():
    assert set(FEATURE_NAMES) <= set(synthetic_table().columns)
```

- [ ] **Step 3: Run to verify failure**

Run: `pytest tests/agents/uncertainty/test_calibrator.py -v` → FAIL

- [ ] **Step 4: Implement** `mars/agents/uncertainty/calibrator.py`

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mars.agents.uncertainty.features import FEATURE_NAMES
from mars.config import targets_hash
from mars.contracts import ORGANS, Reason
from mars.eval.metrics import choose_tau
from mars.fallbacks import reason


def is_calibratable(n_pos: int, n_neg: int, min_each: int = 8) -> bool:
    return n_pos >= min_each and n_neg >= min_each


def _sev(value: float, bound: float, below: bool) -> float:
    gap = (bound - value) if below else (value - bound)
    return float(np.clip(gap / max(abs(bound), 1e-6), 0.0, 1.0))


class Calibrator:
    def __init__(self):
        self.version = "uncalibrated"
        self.targets_hash = ""
        self.mean = None
        self.std = None
        self.coef = None
        self.intercept = 0.0
        self.calibratable = {o: False for o in ORGANS}
        self.tau: dict[str, float | None] = {o: None for o in ORGANS}
        self.reason_bounds: dict[str, float] = {}

    # ---- fitting -------------------------------------------------------
    @staticmethod
    def labels(df: pd.DataFrame, d_star: dict[str, float]) -> pd.Series:
        return (df["dice"] >= df["organ"].map(d_star)).astype(int)

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        x = (df[FEATURE_NAMES].to_numpy(float) - self.mean) / self.std
        onehot = np.stack([(df["organ"] == o).to_numpy(float) for o in ORGANS], axis=1)
        return np.hstack([x, onehot])

    def fit(self, df: pd.DataFrame, d_star: dict[str, float], version: str = "dev") -> "Calibrator":
        self.version, self.targets_hash = version, targets_hash(d_star)
        y_all = self.labels(df, d_star)
        for organ in ORGANS:
            m = df["organ"] == organ
            self.calibratable[organ] = is_calibratable(int(y_all[m].sum()), int((1 - y_all[m]).sum()))
        train = df[df["organ"].map(self.calibratable).astype(bool)]
        self.coef = None
        if train.empty:
            return self
        feats = train[FEATURE_NAMES].to_numpy(float)
        self.mean, self.std = feats.mean(axis=0), feats.std(axis=0)
        self.std[self.std == 0] = 1.0
        y = self.labels(train, d_star)
        model = LogisticRegression(C=1.0, max_iter=2000).fit(self._design(train), y.to_numpy())
        self.coef, self.intercept = model.coef_[0], float(model.intercept_[0])
        good = train[y.to_numpy() == 1]
        self.reason_bounds = {
            "jitter_dice": float(np.percentile(good["jitter_dice"], 10)),
            "prompt_mask_dice": float(np.percentile(good["prompt_mask_dice"], 10)),
            "boundary_entropy": float(np.percentile(good["boundary_entropy"], 90)),
        }
        return self

    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        out = np.full(len(df), np.nan)
        if self.coef is None or df.empty:
            return out
        ok = df["organ"].map(self.calibratable).to_numpy(bool)
        if ok.any():
            z = self._design(df[ok]) @ self.coef + self.intercept
            out[ok] = 1.0 / (1.0 + np.exp(-z))
        return out

    def predict_p(self, organ: str, feats: dict[str, float]) -> float:
        row = pd.DataFrame([{**feats, "organ": organ}])
        return float(self.predict_df(row)[0])

    def fit_tau(self, df: pd.DataFrame, d_star: dict[str, float], alpha: float,
                min_accept: int = 3) -> None:
        """Smallest threshold per organ whose accepted training cases have precision >= alpha."""
        p, y = self.predict_df(df), self.labels(df, d_star).to_numpy()
        for organ in ORGANS:
            m = (df["organ"] == organ).to_numpy() & ~np.isnan(p)
            self.tau[organ] = choose_tau(p[m], y[m], alpha, min_accept) if m.any() else None

    # ---- explaining ----------------------------------------------------
    def reasons(self, organ: str, feats: dict[str, float], p_ok: float,
                tau: float | None) -> list[Reason]:
        out = []
        if tau is None:
            out.append(reason("unc", organ, "NOT_CALIBRATABLE", 1.0))
        else:
            out.append(reason("unc", organ, "LOW_RELIABILITY", float(np.clip(tau - p_ok, 0, 1)) or 0.1,
                              p_ok=round(p_ok, 3), tau=round(tau, 3)))
        b = self.reason_bounds
        if feats["jitter_dice"] < b["jitter_dice"]:
            out.append(reason("unc", organ, "JITTER_HIGH", _sev(feats["jitter_dice"], b["jitter_dice"], True),
                              jitter_dice=round(feats["jitter_dice"], 3), expected_min=round(b["jitter_dice"], 3)))
        if feats["prompt_mask_dice"] < b["prompt_mask_dice"]:
            out.append(reason("unc", organ, "DISAGREE_AUTOMODEL",
                              _sev(feats["prompt_mask_dice"], b["prompt_mask_dice"], True),
                              prompt_mask_dice=round(feats["prompt_mask_dice"], 3),
                              expected_min=round(b["prompt_mask_dice"], 3)))
        if feats["boundary_entropy"] > b["boundary_entropy"]:
            out.append(reason("unc", organ, "HIGH_BOUNDARY_ENTROPY",
                              _sev(feats["boundary_entropy"], b["boundary_entropy"], False),
                              boundary_entropy=round(feats["boundary_entropy"], 3),
                              expected_max=round(b["boundary_entropy"], 3)))
        if feats["n_components"] > 1:
            out.append(reason("unc", organ, "FRAGMENTED", min(1.0, 0.3 * feats["n_components"]),
                              n_components=feats["n_components"]))
        return out

    # ---- persistence ---------------------------------------------------
    def save(self, path) -> None:
        blob = {
            "version": self.version, "targets_hash": self.targets_hash,
            "mean": None if self.mean is None else self.mean.tolist(),
            "std": None if self.std is None else self.std.tolist(),
            "coef": None if self.coef is None else self.coef.tolist(),
            "intercept": self.intercept, "calibratable": self.calibratable,
            "tau": self.tau, "reason_bounds": self.reason_bounds,
        }
        Path(path).write_text(json.dumps(blob, indent=2))

    @classmethod
    def load(cls, path) -> "Calibrator":
        b = json.loads(Path(path).read_text())
        c = cls()
        c.version, c.targets_hash = b["version"], b["targets_hash"]
        c.mean = None if b["mean"] is None else np.array(b["mean"])
        c.std = None if b["std"] is None else np.array(b["std"])
        c.coef = None if b["coef"] is None else np.array(b["coef"])
        c.intercept, c.calibratable = b["intercept"], b["calibratable"]
        c.tau, c.reason_bounds = b["tau"], b["reason_bounds"]
        return c
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/agents/uncertainty/test_calibrator.py -v` → all PASS

- [ ] **Step 6: Commit**

```bash
git add mars tests
git commit -m "feat(unc): logistic calibrator with shared slopes, tau, reasons, persistence"
```

---

### Task 4: UncertaintyAgent

**Files:**
- Create: `mars/agents/uncertainty/agent.py`
- Test: `tests/agents/uncertainty/test_agent.py`

**Interfaces:**
- Consumes: `compute_features`, `Calibrator`, `config.targets_hash`, `volumes.load_ct`, `fallbacks.reason`
- Produces: `UncertaintyAgent(refiner, calibrator: Calibrator, targets: dict[str, float], n_jitter: int = 8, seed: int = 0)` with `run(seg: SegmentationOutput) -> UncertaintyReport`. Raises `ValueError` in the constructor if `calibrator.targets_hash != targets_hash(targets)`.

- [ ] **Step 1: Write the failing test**

```python
import nibabel as nib
import numpy as np
import pytest

from mars.agents.uncertainty.agent import UncertaintyAgent
from mars.agents.uncertainty.calibrator import Calibrator
from mars.agents.uncertainty.features import FEATURE_NAMES
from mars.contracts import ORGANS, REQUIRED_EVIDENCE, OrganMask, SegmentationOutput
from mars.volumes import save_mask
from tests.agents.uncertainty.synth import D_STAR, synthetic_table


class BoxRefiner:
    def refine(self, ct, box):
        z0, y0, x0, z1, y1, x1 = box
        m = np.zeros(ct.shape, bool); m[z0:z1, y0:y1, x0:x1] = True
        return m, m.astype(np.float32)


@pytest.fixture
def seg(tmp_path):
    nib.save(nib.Nifti1Image(np.full((16, 16, 16), 50.0, np.float32), np.eye(4)), str(tmp_path / "ct.nii.gz"))
    m = np.zeros((16, 16, 16), bool); m[4:10, 4:10, 4:10] = True
    save_mask(tmp_path / "m.npy", m)
    liver = OrganMask(prompt_source="totalseg", deployed="auto", mask_path=str(tmp_path / "m.npy"),
                      auto_mask_path=str(tmp_path / "m.npy"), box_prompt=[2, 2, 2, 12, 12, 12],
                      prompt_mask_dice=0.9, boundary_entropy=0.2, volume_ml=0.216)
    spleen = liver.model_copy(update={"prompt_mask_dice": None})        # refiner crashed upstream
    return SegmentationOutput(case_id="c", ct_path=str(tmp_path / "ct.nii.gz"), spacing=(1.0, 1.0, 1.0),
                              shape=(16, 16, 16),
                              organs={"liver": liver, "spleen": spleen, "kidney_left": OrganMask(),
                                      "stomach": liver.model_copy()})


@pytest.fixture
def cal():
    c = Calibrator().fit(synthetic_table(), D_STAR, version="t1")
    c.fit_tau(synthetic_table(), D_STAR, 0.9)
    df = synthetic_table(); df.loc[df.organ == "stomach", "dice"] = 0.5     # stomach: not calibratable
    c2 = Calibrator().fit(df, D_STAR, version="t1")
    c2.fit_tau(df, D_STAR, 0.9)
    return c2


def _agent(cal, tau=None):
    if tau is not None:
        cal.tau = {o: tau for o in ORGANS}
    return UncertaintyAgent(BoxRefiner(), cal, D_STAR, n_jitter=3)


def test_statuses_and_reasons(seg, cal):
    rep = _agent(cal).run(seg)
    assert rep.calibration_version == "t1" and set(rep.organs) == set(ORGANS)
    assert [r.code for r in rep.organs["kidney_left"].reasons] == ["NO_PROMPT"]
    assert rep.organs["kidney_left"].status == "FLAG" and rep.organs["kidney_left"].p_ok is None
    assert [r.code for r in rep.organs["stomach"].reasons] == ["NOT_CALIBRATABLE"]
    assert [r.code for r in rep.organs["spleen"].reasons] == ["AGENT_ERROR"]   # only this organ failed
    assert rep.organs["liver"].status in ("ACCEPT", "FLAG")


def test_always_accept_and_never_accept_thresholds(seg, cal):
    accept = _agent(cal, tau=0.0).run(seg).organs["liver"]
    assert accept.status == "ACCEPT" and accept.reasons == [] and set(accept.signals) == set(FEATURE_NAMES)
    flag = _agent(cal, tau=1.01).run(seg).organs["liver"]
    assert flag.status == "FLAG" and len(flag.reasons) >= 1
    for r in flag.reasons:
        assert set(REQUIRED_EVIDENCE[r.code]) <= set(r.evidence)


def test_organ_with_no_reachable_tau_flags_with_not_calibratable(seg, cal):
    cal.tau["liver"] = None
    organ = _agent(cal).run(seg).organs["liver"]
    assert organ.status == "FLAG" and organ.reasons[0].code == "NOT_CALIBRATABLE"
    assert organ.p_ok is not None                      # the score is still shown


def test_deterministic(seg, cal):
    assert _agent(cal).run(seg) == _agent(cal).run(seg)


def test_stale_calibration_is_refused(cal):
    with pytest.raises(ValueError, match="targets"):
        UncertaintyAgent(BoxRefiner(), cal, {**D_STAR, "liver": 0.7})
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/agents/uncertainty/test_agent.py -v` → FAIL

- [ ] **Step 3: Implement** `mars/agents/uncertainty/agent.py`

```python
import logging

import numpy as np

from mars.agents.uncertainty.calibrator import Calibrator
from mars.agents.uncertainty.features import compute_features
from mars.config import targets_hash
from mars.contracts import ORGANS, OrganUncertainty, SegmentationOutput, UncertaintyReport
from mars.fallbacks import reason
from mars.volumes import load_ct

log = logging.getLogger("mars.uncertainty")


class UncertaintyAgent:
    def __init__(self, refiner, calibrator: Calibrator, targets: dict[str, float],
                 n_jitter: int = 8, seed: int = 0):
        if calibrator.targets_hash != targets_hash(targets):
            raise ValueError("calibration was fitted for different targets; refit after changing targets.json")
        self.refiner, self.cal, self.targets = refiner, calibrator, targets
        self.n_jitter, self.seed = n_jitter, seed

    def _flag(self, organ, code, **evidence) -> OrganUncertainty:
        return OrganUncertainty(status="FLAG", d_star=self.targets[organ],
                                reasons=[reason("unc", organ, code, 1.0, **evidence)])

    def run(self, seg: SegmentationOutput) -> UncertaintyReport:
        ct, spacing = load_ct(seg.ct_path)
        rng = np.random.default_rng(self.seed)
        organs = {}
        for organ in ORGANS:
            om = seg.organs[organ]
            if om.deployed == "none" or om.mask_path is None or om.box_prompt is None:
                organs[organ] = self._flag(organ, "NO_PROMPT")
            elif not self.cal.calibratable.get(organ):
                organs[organ] = self._flag(organ, "NOT_CALIBRATABLE")
            else:
                try:
                    feats = compute_features(ct, spacing, om, self.refiner, rng, self.n_jitter)
                    p = self.cal.predict_p(organ, feats)
                    tau = self.cal.tau.get(organ)
                    accept = tau is not None and p >= tau
                    organs[organ] = OrganUncertainty(
                        p_ok=p, tau=tau, d_star=self.targets[organ],
                        status="ACCEPT" if accept else "FLAG", signals=feats,
                        reasons=[] if accept else self.cal.reasons(organ, feats, p, tau))
                except Exception as e:  # noqa: BLE001 - isolate the failing organ
                    log.exception("uncertainty failed for %s in %s", organ, seg.case_id)
                    organs[organ] = self._flag(organ, "AGENT_ERROR", detail=str(e))
        return UncertaintyReport(calibration_version=self.cal.version,
                                 targets_hash=targets_hash(self.targets), organs=organs)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/agents/uncertainty -v` → all PASS

- [ ] **Step 5: Commit**

```bash
git add mars tests
git commit -m "feat(unc): UncertaintyAgent with per-organ isolation and stale-calibration guard"
```

---

### Task 5: Offline feature table

**Files:**
- Create: `mars/eval/feature_table.py`, `scripts/make_feature_table.py`
- Test: `tests/eval/test_feature_table.py`

**Interfaces:**
- Consumes: `DevConfig`/`DevCase` (Plan 2; uses `case.key`, see the note below), `apply_deploy`, `load_gt`, `compute_features`, `dice`
- Produces: `feature_table.build_feature_table(cfg: DevConfig, seg_dir, deploy: dict[str, str], make_refiner, n_jitter: int = 8, seed: int = 0) -> pd.DataFrame` where `make_refiner(fold) -> Refiner`. Columns: `patient, case, fold, organ, dice` + `FEATURE_NAMES`. `dice` is the Dice of the **deployed** mask against ground truth. Organs without a prompt or without features are skipped; the count is stored in `df.attrs["skipped"]`.

- [ ] **Step 1: Write the failing test**

```python
import nibabel as nib
import numpy as np

from mars.agents.uncertainty.features import FEATURE_NAMES
from mars.contracts import OrganMask, SegmentationOutput
from mars.eval.dev_config import DevCase, DevConfig
from mars.eval.feature_table import build_feature_table
from mars.volumes import save_mask, write_json_atomic


class BoxRefiner:
    def refine(self, ct, box):
        z0, y0, x0, z1, y1, x1 = box
        m = np.zeros(ct.shape, bool); m[z0:z1, y0:y1, x0:x1] = True
        return m, m.astype(np.float32)


def test_feature_table_rows(tmp_path):
    nib.save(nib.Nifti1Image(np.full((16, 16, 16), 40.0, np.float32), np.eye(4)), str(tmp_path / "ct.nii.gz"))
    lab = np.zeros((16, 16, 16), np.int16); lab[4:10, 4:10, 4:10] = 6
    nib.save(nib.Nifti1Image(lab, np.eye(4)), str(tmp_path / "gt.nii.gz"))
    auto = np.zeros((16, 16, 16), bool); auto[4:10, 4:10, 4:10] = True
    save_mask(tmp_path / "a.npy", auto)
    liver = OrganMask(prompt_source="totalseg", deployed="auto", mask_path=str(tmp_path / "a.npy"),
                      auto_mask_path=str(tmp_path / "a.npy"), box_prompt=[2, 2, 2, 12, 12, 12],
                      prompt_mask_dice=0.9, boundary_entropy=0.2, volume_ml=0.216)
    seg = SegmentationOutput(case_id="p1", ct_path=str(tmp_path / "ct.nii.gz"), spacing=(1.0, 1.0, 1.0),
                             shape=(16, 16, 16),
                             organs={"liver": liver, "spleen": OrganMask(), "kidney_left": OrganMask(),
                                     "stomach": OrganMask()})
    write_json_atomic(tmp_path / "seg" / "p1" / "seg.json", seg.model_dump_json())
    cfg = DevConfig(cases=[DevCase(patient="p1", fold=0, ct=str(tmp_path / "ct.nii.gz"), gt=str(tmp_path / "gt.nii.gz"))])
    df = build_feature_table(cfg, tmp_path / "seg", {"liver": "auto"}, lambda fold: BoxRefiner(), n_jitter=3)
    assert list(df.organ) == ["liver"] and df.attrs["skipped"] == 3
    row = df.iloc[0]
    assert row.dice == 1.0 and row.hu_mean == 40.0 and row.patient == "p1"
    assert set(FEATURE_NAMES) <= set(df.columns)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/eval/test_feature_table.py -v` → FAIL

- [ ] **Step 3: Implement** `mars/eval/feature_table.py`

```python
from pathlib import Path

import numpy as np
import pandas as pd

from mars.agents.segmentation.deploy import apply_deploy
from mars.agents.uncertainty.features import compute_features
from mars.contracts import ORGANS, SegmentationOutput
from mars.eval.dev_config import DevConfig, load_gt
from mars.metrics import dice
from mars.volumes import load_ct, load_mask


def build_feature_table(cfg: DevConfig, seg_dir, deploy: dict[str, str], make_refiner,
                        n_jitter: int = 8, seed: int = 0) -> pd.DataFrame:
    rows, skipped, refiners = [], 0, {}
    for case in cfg.cases:
        seg_json = (Path(seg_dir) / case.key / "seg.json").read_text()
        seg = apply_deploy(SegmentationOutput.model_validate_json(seg_json), deploy)
        if case.fold not in refiners:
            refiners[case.fold] = make_refiner(case.fold)
        ct, spacing = load_ct(seg.ct_path)
        gt = load_gt(case, cfg.gt_format)
        rng = np.random.default_rng(seed)
        for organ in ORGANS:
            om = seg.organs[organ]
            try:
                feats = compute_features(ct, spacing, om, refiners[case.fold], rng, n_jitter)
            except ValueError:               # no prompt, or refinement failed: not calibratable data
                skipped += 1
                continue
            rows.append({"patient": case.patient, "case": case.key, "fold": case.fold, "organ": organ,
                         "dice": dice(load_mask(om.mask_path), gt[organ]), **feats})
    df = pd.DataFrame(rows)
    df.attrs["skipped"] = skipped
    return df
```

`scripts/make_feature_table.py`:
```python
import argparse
import json
from pathlib import Path

from mars.agents.segmentation.refiner import load_refiner
from mars.eval.dev_config import DevConfig
from mars.eval.feature_table import build_feature_table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seg-dir", required=True)
    p.add_argument("--deploy", default="configs/deploy.json")
    p.add_argument("--out", required=True)
    p.add_argument("--n-jitter", type=int, default=8)
    a = p.parse_args()
    cfg = DevConfig.from_file(a.config)
    deploy = json.loads(Path(a.deploy).read_text())
    df = build_feature_table(cfg, a.seg_dir, deploy, lambda f: load_refiner(cfg.ckpts[f], "cuda"), a.n_jitter)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} organ-cases written; {df.attrs['skipped']} skipped (no prompt / no refinement)")


if __name__ == "__main__":
    main()
```

**Note:** `DevCase.key` (the per-case directory name: `case_id` if set, else `patient`) is defined in Plan 2 Task 5 and used by `run_dev_seg` in Plan 2 Task 6. It lets the external set list each patient once per fold checkpoint (`patient="s0123", case_id="s0123_f2", fold=2`) without directory collisions, while bootstraps still cluster on `patient`.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/eval -v` → all PASS

- [ ] **Step 5: Commit**

```bash
git add mars scripts tests
git commit -m "feat(unc): offline feature table builder"
```

---

### Task 6: Feasibility, nested evaluation, fitting, external evaluation

**Files:**
- Create: `mars/eval/nested.py`, `scripts/feasibility.py`, `scripts/fit_calibration.py`, `scripts/evaluate_nested.py`, `scripts/evaluate_external.py`
- Test: `tests/eval/test_nested.py`

**Interfaces:**
- Consumes: `Calibrator`, `metrics.*`, `cluster_bootstrap_ci`, `is_calibratable`
- Produces (all in `mars/eval/nested.py`):
  - `feasibility_table(table: pd.DataFrame, d_star: dict[str, float]) -> pd.DataFrame` — per organ: `n, n_pos, n_neg, calibratable, dice_q10, dice_median, dice_q90`
  - `assert_disjoint(train: pd.DataFrame, test: pd.DataFrame) -> None` (raises `AssertionError` if any patient is in both)
  - `evaluate_nested(table, d_star, alpha, n_repeats=20, n_splits=5, seed=0) -> tuple[pd.DataFrame, pd.DataFrame]` — `(rows, taus)`; rows columns `repeat, patient, organ, dice, y, p_ok, tau, accepted`; taus columns `repeat, fold, organ, tau`
  - `evaluate_frozen(table, cal: Calibrator, d_star) -> pd.DataFrame` — same columns as `rows`, with `repeat = table["fold"]` (each fold model's prediction is one "repeat")
  - `summarize(rows, taus, n_boot=1000, seed=0) -> pd.DataFrame` — per organ: `n_patients, auroc, auroc_lo, auroc_hi, ece, ece_lo, ece_hi, brier, aurc, coverage, coverage_lo, coverage_hi, precision, precision_lo, precision_hi, tau_median, tau_q25, tau_q75`. Per patient, `p_ok` and the accept rate are averaged over repeats; a patient counts as accepted if the accept rate ≥ 0.5.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd
import pytest

from mars.agents.uncertainty.calibrator import Calibrator
from mars.contracts import ORGANS
from mars.eval.nested import (
    assert_disjoint, evaluate_frozen, evaluate_nested, feasibility_table, summarize,
)
from tests.agents.uncertainty.synth import D_STAR, synthetic_table


def test_feasibility_table_flags_degenerate_organ():
    df = synthetic_table()
    df.loc[df.organ == "stomach", "dice"] = 0.5
    t = feasibility_table(df, D_STAR).set_index("organ")
    assert t.loc["stomach", "n_pos"] == 0 and not t.loc["stomach", "calibratable"]
    assert t.loc["liver", "calibratable"] and t.loc["liver", "n"] == 60
    assert t.loc["stomach", "dice_median"] == pytest.approx(0.5)


def test_assert_disjoint():
    a, b = pd.DataFrame({"patient": ["p1", "p2"]}), pd.DataFrame({"patient": ["p2"]})
    with pytest.raises(AssertionError, match="p2"):
        assert_disjoint(a, b)
    assert_disjoint(a, pd.DataFrame({"patient": ["p3"]}))


def test_each_patient_is_held_out_exactly_once_per_repeat():
    df = synthetic_table(n=40)
    rows, taus = evaluate_nested(df, D_STAR, alpha=0.9, n_repeats=3, n_splits=5)
    counts = rows.groupby(["repeat", "patient", "organ"]).size()
    assert (counts == 1).all() and len(counts) == 3 * 40 * len(ORGANS)
    assert set(taus.organ) == set(ORGANS) and len(taus) == 3 * 5 * len(ORGANS)


def test_summarize_has_expected_columns_and_sane_values():
    df = synthetic_table(n=60)
    rows, taus = evaluate_nested(df, D_STAR, alpha=0.9, n_repeats=3, n_splits=5)
    s = summarize(rows, taus, n_boot=50).set_index("organ")
    assert set(ORGANS) == set(s.index)
    assert {"auroc", "ece", "ece_lo", "ece_hi", "coverage", "precision", "tau_median", "tau_q25", "tau_q75",
            "brier", "aurc", "n_patients"} <= set(s.columns)
    assert s.loc["liver", "n_patients"] == 60
    assert s.loc["liver", "auroc"] > 0.9 and 0.0 <= s.loc["liver", "coverage"] <= 1.0


def test_degenerate_organ_is_summarised_without_crashing():
    df = synthetic_table(n=40)
    df.loc[df.organ == "stomach", "dice"] = 0.5
    rows, taus = evaluate_nested(df, D_STAR, alpha=0.9, n_repeats=2, n_splits=5)
    s = summarize(rows, taus, n_boot=20).set_index("organ")
    assert s.loc["stomach", "n_patients"] == 0 and np.isnan(s.loc["stomach", "auroc"])


def test_evaluate_frozen_uses_fold_as_repeat():
    train = synthetic_table(seed=0)
    cal = Calibrator().fit(train, D_STAR); cal.fit_tau(train, D_STAR, 0.9)
    ext = synthetic_table(n=20, seed=5)
    rows = evaluate_frozen(ext, cal, D_STAR)
    assert len(rows) == len(ext) and set(rows["repeat"]) == set(ext["fold"])
    assert {"p_ok", "tau", "accepted", "y"} <= set(rows.columns)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/eval/test_nested.py -v` → FAIL

- [ ] **Step 3: Implement** `mars/eval/nested.py`

```python
import numpy as np
import pandas as pd

from mars.agents.uncertainty.calibrator import Calibrator, is_calibratable
from mars.contracts import ORGANS
from mars.eval.metrics import aurc, auroc, brier, ece
from mars.metrics import cluster_bootstrap_ci


def feasibility_table(table: pd.DataFrame, d_star: dict[str, float]) -> pd.DataFrame:
    y = Calibrator.labels(table, d_star)
    rows = []
    for organ in ORGANS:
        m = table["organ"] == organ
        d = table.loc[m, "dice"]
        n_pos, n_neg = int(y[m].sum()), int((1 - y[m]).sum())
        rows.append({"organ": organ, "n": int(m.sum()), "n_pos": n_pos, "n_neg": n_neg,
                     "calibratable": is_calibratable(n_pos, n_neg),
                     "dice_q10": d.quantile(0.1) if len(d) else np.nan,
                     "dice_median": d.median() if len(d) else np.nan,
                     "dice_q90": d.quantile(0.9) if len(d) else np.nan})
    return pd.DataFrame(rows)


def assert_disjoint(train: pd.DataFrame, test: pd.DataFrame) -> None:
    overlap = set(train["patient"]) & set(test["patient"])
    assert not overlap, f"patients in both fit and evaluation sets: {sorted(overlap)}"


def _predict_rows(cal: Calibrator, test: pd.DataFrame, d_star, repeat) -> pd.DataFrame:
    out = test[["patient", "organ", "dice"]].copy()
    out["repeat"] = repeat
    out["y"] = Calibrator.labels(test, d_star).to_numpy()
    out["p_ok"] = cal.predict_df(test)
    out["tau"] = out["organ"].map(cal.tau).astype(float)
    out["accepted"] = out["p_ok"] >= out["tau"]          # NaN compares False
    return out


def evaluate_nested(table, d_star, alpha, n_repeats=20, n_splits=5, seed=0):
    patients = np.array(sorted(table["patient"].unique()))
    rows, taus = [], []
    for r in range(n_repeats):
        perm = np.random.default_rng(seed + r).permutation(patients)
        for k, held in enumerate(np.array_split(perm, n_splits)):
            test = table[table["patient"].isin(held)]
            train = table[~table["patient"].isin(held)]
            assert_disjoint(train, test)
            cal = Calibrator().fit(train, d_star)
            cal.fit_tau(train, d_star, alpha)
            rows.append(_predict_rows(cal, test, d_star, r))
            taus += [{"repeat": r, "fold": k, "organ": o, "tau": cal.tau[o]} for o in ORGANS]
    return pd.concat(rows, ignore_index=True), pd.DataFrame(taus)


def evaluate_frozen(table, cal: Calibrator, d_star) -> pd.DataFrame:
    return _predict_rows(cal, table, d_star, table["fold"].to_numpy())


def summarize(rows: pd.DataFrame, taus: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    out = []
    for organ in ORGANS:
        g = rows[(rows["organ"] == organ) & rows["p_ok"].notna()]
        t = taus[taus["organ"] == organ]["tau"].astype(float).dropna()
        rec = {"organ": organ, "n_patients": g["patient"].nunique(),
               "tau_median": t.median() if len(t) else np.nan,
               "tau_q25": t.quantile(0.25) if len(t) else np.nan,
               "tau_q75": t.quantile(0.75) if len(t) else np.nan}
        stats = {"auroc": lambda d: auroc(d.p_ok.to_numpy(), d.y.to_numpy()),
                 "ece": lambda d: ece(d.p_ok.to_numpy(), d.y.to_numpy()),
                 "coverage": lambda d: float(d.accepted.mean()),
                 # numpy indexing: bootstrap samples have duplicate index labels
                 "precision": lambda d: (float(d.y.to_numpy()[d.accepted.to_numpy()].mean())
                                         if d.accepted.any() else np.nan)}
        if g.empty:
            for name in (*stats, "brier", "aurc"):
                rec[name] = np.nan
            for name in stats:
                rec[f"{name}_lo"] = rec[f"{name}_hi"] = np.nan
            out.append(rec)
            continue
        pooled = (g.groupby("patient").agg(p_ok=("p_ok", "mean"), accept_rate=("accepted", "mean"),
                                           y=("y", "first"), dice=("dice", "first")).reset_index())
        pooled["accepted"] = pooled["accept_rate"] >= 0.5
        rec["brier"] = brier(pooled.p_ok, pooled.y)
        rec["aurc"] = aurc(pooled.p_ok, pooled.dice)
        for name, fn in stats.items():
            rec[name], rec[f"{name}_lo"], rec[f"{name}_hi"] = cluster_bootstrap_ci(pooled, fn, n_boot, seed)
        out.append(rec)
    return pd.DataFrame(out)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/eval/test_nested.py -v` → all PASS. If the `auroc > 0.9` assertion fails on the synthetic data, print `summarize(...)` and check the synthetic generator before loosening anything.

- [ ] **Step 5: Write the four scripts**

`scripts/feasibility.py`:
```python
import argparse
import json

import pandas as pd

from mars.contracts import ORGANS
from mars.eval.nested import feasibility_table


def main():
    p = argparse.ArgumentParser(description="Step 0: can each organ reach d*? (run BEFORE committing targets.json)")
    p.add_argument("--table", required=True)
    p.add_argument("--d-star", type=float, default=0.9, help="candidate target applied to every organ")
    a = p.parse_args()
    t = feasibility_table(pd.read_csv(a.table), {o: a.d_star for o in ORGANS})
    print(t.round(3).to_string(index=False))
    print("\nCalibratable needs >= 8 positives and >= 8 negatives. For organs that fail, apply the one permitted")
    print("adjustment (spec 6.2): lower d* once to the development-set median Dice, then commit configs/targets.json:")
    print(json.dumps({o: a.d_star for o in ORGANS}, indent=2))


if __name__ == "__main__":
    main()
```

`scripts/fit_calibration.py`:
```python
import argparse
import subprocess

import pandas as pd

from mars.agents.uncertainty.calibrator import Calibrator
from mars.config import load_targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", required=True)
    p.add_argument("--alpha", type=float, default=0.9, help="accepted-precision target")
    p.add_argument("--out", default="configs/calibration.json")
    a = p.parse_args()
    d_star, df = load_targets(), pd.read_csv(a.table)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    cal = Calibrator().fit(df, d_star, version=f"dev-{commit or 'nogit'}")
    cal.fit_tau(df, d_star, a.alpha)
    cal.save(a.out)
    print(f"calibratable: {cal.calibratable}\ntau: {cal.tau}\nwrote {a.out}")


if __name__ == "__main__":
    main()
```

`scripts/evaluate_nested.py`:
```python
import argparse
from pathlib import Path

import pandas as pd

from mars.config import load_targets
from mars.eval.nested import evaluate_nested, summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", required=True)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--out", required=True, help="output directory")
    a = p.parse_args()
    rows, taus = evaluate_nested(pd.read_csv(a.table), load_targets(), a.alpha, a.repeats)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "heldout_rows.csv", index=False)
    taus.to_csv(out / "taus.csv", index=False)
    summary = summarize(rows, taus)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.round(3).to_string(index=False))
    print("\nn is small: read the CIs, not just the point estimates.")


if __name__ == "__main__":
    main()
```

`scripts/evaluate_external.py`:
```python
import argparse
from pathlib import Path

import pandas as pd

from mars.agents.uncertainty.calibrator import Calibrator
from mars.config import load_targets
from mars.eval.nested import evaluate_frozen, summarize


def main():
    p = argparse.ArgumentParser(description="Run ONCE, after configs are frozen and committed")
    p.add_argument("--table", required=True, help="feature table of the external set (one row per patient, organ, fold)")
    p.add_argument("--calibration", default="configs/calibration.json")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cal, d_star = Calibrator.load(a.calibration), load_targets()
    rows = evaluate_frozen(pd.read_csv(a.table), cal, d_star)
    taus = pd.DataFrame([{"repeat": 0, "fold": 0, "organ": o, "tau": t} for o, t in cal.tau.items()])
    summary = summarize(rows, taus)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    summary.to_csv(Path(a.out) / "external_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Smoke-run the scripts on synthetic data**

Run:
```bash
python - <<'EOF'
from tests.agents.uncertainty.synth import synthetic_table
synthetic_table().to_csv("/tmp/synth_table.csv", index=False)
EOF
python scripts/feasibility.py --table /tmp/synth_table.csv --d-star 0.85
```
Expected: a table with four organs, all `calibratable True`.

- [ ] **Step 7: Commit**

```bash
git add mars scripts tests
git commit -m "feat(unc): feasibility check, nested CV, fitting and external evaluation"
```

---

## After this plan

Run Runbook steps A3–A5 on the lab. Then:
1. Read `outputs/dev/nested/summary.csv`; write the headline sentence per organ: "When `p_ok ≥ τ`, Dice ≥ d* in X% [CI] of accepted cases at Y% coverage."
2. Report `tau_q25`–`tau_q75`. If wide, say the threshold is unstable at this sample size.
3. Report the ablations (spec §7.6). They are re-runs of the same scripts with different inputs:
   - **Auto-model-only baseline:** build the feature table with a `deploy.json` that says `"auto"` for every organ, then run `evaluate_nested`. This shows what the pipeline achieves without Slide-SAM's mask.
   - **Oracle vs auto prompts:** run `make_dev_seg` once with `OraclePromptSource` (Plan 2 Task 8) and compare Dice and calibration quality.
   - **Each signal alone / leave-one-signal-out:** pass a reduced feature list (temporarily edit `FEATURE_NAMES` or drop columns before `Calibrator.fit`).
   - **Uncalibrated vs calibrated:** use the raw `jitter_dice` as the score (`p_ok` column replaced) in `summarize`.
   - **Shared vs per-organ slopes:** fit one `Calibrator` per organ (loop over `table[table.organ == o]`).

**Known simplification:** `τ` is chosen on in-sample training predictions. With six features and L2 this is slightly optimistic; the held-out accepted precision reported by `summarize` is the honest number, and it is what the write-up should cite.
