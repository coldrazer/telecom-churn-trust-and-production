"""The serving contract.

A model in a notebook has no contract. A model in production has exactly one:
given this input schema, produce this output schema, within this latency, and
fail loudly when the input is not what it was trained on.

``ChurnScorer`` is that contract in one object. It bundles the fitted
pipeline, the decision threshold, the calibration temperature, the column
contract and a compact profile of the training distribution -- everything
needed to score *and* to notice that the world has moved since training.

The bundle is a single joblib file on purpose, and **that bundle is what the
registry stores and what serving loads**. Partial deployments -- model in one
place, preprocessing in another, threshold in a config file -- are how a
"reproducible" model ends up scoring differently in production than it did in
the notebook that approved it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from .features import required_input_columns
from .schema import validate_dataframe

DEFAULT_NUMERIC_DRIFT: tuple[str, ...] = ("tenure", "MonthlyCharges", "TotalCharges")
DEFAULT_CATEGORICAL_DRIFT: tuple[str, ...] = (
    "Contract", "InternetService", "PaymentMethod", "PaperlessBilling",
)

_PSI_EPS = 1e-6


# --------------------------------------------------------------------------
# Drift detection
# --------------------------------------------------------------------------


def _reference_edges(expected: np.ndarray, n_bins: int) -> np.ndarray | None:
    """Quantile bin edges of the *reference* sample (None if too few distinct values)."""
    quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(quantiles) < 3:
        return None
    return np.concatenate(([-np.inf], quantiles[1:-1], [np.inf]))


def _bin_shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.histogram(values, bins=edges)[0].astype(float)
    return counts / counts.sum()


def _psi_from_shares(expected_share: np.ndarray, actual_share: np.ndarray) -> float:
    # Floor the proportions so an empty bin contributes a large-but-finite
    # value instead of inf, which would make every chart useless.
    e = np.clip(expected_share, _PSI_EPS, None)
    a = np.clip(actual_share, _PSI_EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def population_stability_index(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
) -> float:
    """PSI between two samples of a single numeric feature.

    Conventions: < 0.1 no meaningful shift, 0.1-0.25 moderate, > 0.25
    significant. The convention is industry folklore rather than theory, but
    it is a useful tripwire -- it is designed to be noisy and late rather than
    precise and early.

    Bins are quantile-based on the *reference* sample, so the buckets mean the
    same thing in both periods. Using fresh quantiles each week would hide
    exactly the drift you are trying to detect.
    """
    expected = pd.Series(expected).dropna().to_numpy()
    actual = pd.Series(actual).dropna().to_numpy()
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    edges = _reference_edges(expected, n_bins)
    if edges is None:
        return 0.0
    return _psi_from_shares(_bin_shares(expected, edges), _bin_shares(actual, edges))


def categorical_shift(
    expected: pd.Series,
    actual: pd.Series,
    *,
    categories: list[str] | None = None,
) -> float:
    """Total-variation distance between two categorical distributions."""
    cats = categories or sorted(set(expected.dropna().unique()) | set(actual.dropna().unique()))
    e = expected.value_counts(normalize=True).reindex(cats).fillna(0.0)
    a = actual.value_counts(normalize=True).reindex(cats).fillna(0.0)
    return float(0.5 * (e - a).abs().sum())


@dataclass(frozen=True)
class ReferenceProfile:
    """A compact summary of the training distribution -- no customer rows.

    Drift monitoring needs only the reference *shape*: quantile bin edges and
    the share of training rows in each bin for numeric features, and category
    proportions for categorical ones. Storing that (rather than the training
    frame) keeps the deployed bundle small and keeps raw customer records out
    of a model artefact that gets copied around.
    """

    n_rows: int
    numeric_edges: dict[str, np.ndarray]
    numeric_shares: dict[str, np.ndarray]
    categorical_shares: dict[str, dict[str, float]]

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        numeric_features: tuple[str, ...] | list[str] | None = None,
        categorical_features: tuple[str, ...] | list[str] | None = None,
        n_bins: int = 10,
    ) -> ReferenceProfile:
        numeric_features = list(numeric_features or DEFAULT_NUMERIC_DRIFT)
        categorical_features = list(categorical_features or DEFAULT_CATEGORICAL_DRIFT)
        edges: dict[str, np.ndarray] = {}
        shares: dict[str, np.ndarray] = {}
        for col in numeric_features:
            if col not in frame:
                continue
            values = pd.to_numeric(frame[col], errors="coerce").dropna().to_numpy()
            col_edges = _reference_edges(values, n_bins) if len(values) else None
            if col_edges is not None:
                edges[col] = col_edges
                shares[col] = _bin_shares(values, col_edges)
        cats = {
            col: frame[col].value_counts(normalize=True).to_dict()
            for col in categorical_features if col in frame
        }
        return cls(n_rows=len(frame), numeric_edges=edges, numeric_shares=shares,
                   categorical_shares=cats)


@dataclass
class DriftReport:
    """Per-feature drift plus an overall verdict."""

    table: pd.DataFrame
    n_flagged: int
    verdict: str

    @property
    def degraded(self) -> bool:
        return self.n_flagged > 0


def detect_drift(
    reference: pd.DataFrame | ReferenceProfile,
    actual: pd.DataFrame,
    *,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    psi_warn: float = 0.10,
    psi_alert: float = 0.25,
    tv_warn: float = 0.10,
    tv_alert: float = 0.30,
) -> DriftReport:
    """Compare a live batch against the training reference distribution.

    ``reference`` is either the training frame or a :class:`ReferenceProfile`
    built from it (what a deployed scorer carries). Both go through the same
    code path, so a profile gives exactly the numbers the frame would.

    Deliberately compares *features*, not predictions. Prediction drift tells
    you something broke downstream; feature drift tells you whether the
    monitoring signal itself is still valid. (It also cannot see a change in
    the *relationship* between features and churn -- that needs labels.)
    """
    numeric_features = list(numeric_features or DEFAULT_NUMERIC_DRIFT)
    categorical_features = list(categorical_features or DEFAULT_CATEGORICAL_DRIFT)
    profile = (
        reference if isinstance(reference, ReferenceProfile)
        else ReferenceProfile.from_frame(
            reference, numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
    )

    rows = []
    for col in numeric_features:
        if col not in profile.numeric_edges or col not in actual:
            continue
        values = pd.to_numeric(actual[col], errors="coerce").dropna().to_numpy()
        if len(values) == 0:
            continue
        psi = _psi_from_shares(
            profile.numeric_shares[col], _bin_shares(values, profile.numeric_edges[col])
        )
        status = "alert" if psi >= psi_alert else ("warn" if psi >= psi_warn else "ok")
        rows.append({"feature": col, "kind": "numeric", "statistic": "PSI",
                     "value": psi, "status": status})

    for col in categorical_features:
        if col not in profile.categorical_shares or col not in actual:
            continue
        ref = pd.Series(profile.categorical_shares[col], dtype=float)
        live = actual[col].value_counts(normalize=True)
        cats = sorted(set(ref.index) | set(live.index))
        tv = float(0.5 * (ref.reindex(cats).fillna(0.0) - live.reindex(cats).fillna(0.0)).abs().sum())
        status = "alert" if tv >= tv_alert else ("warn" if tv >= tv_warn else "ok")
        rows.append({"feature": col, "kind": "categorical", "statistic": "TV dist",
                     "value": tv, "status": status})

    table = pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)
    n_alert = int((table["status"] == "alert").sum()) if len(table) else 0
    n_warn = int((table["status"] == "warn").sum()) if len(table) else 0

    if n_alert:
        verdict = f"ALERT: {n_alert} feature(s) materially shifted. Investigate before trusting scores."
    elif n_warn:
        verdict = f"WARN: {n_warn} feature(s) drifting. Add to watchlist."
    else:
        verdict = "OK: live batch is statistically consistent with training data."

    return DriftReport(table=table, n_flagged=n_alert, verdict=verdict)


# --------------------------------------------------------------------------
# Scoring contract
# --------------------------------------------------------------------------


@dataclass
class ScoringResult:
    """Output of a scoring call, with the latency that produced it.

    ``churn_probability`` is the *calibrated* probability (identical to
    ``raw_probability`` while the temperature is 1.0). ``flagged`` and
    ``risk_band`` are derived from it. ``quarantined`` holds rows that failed
    the data contract and were excluded (only when scoring with
    ``on_invalid="quarantine"``).
    """

    customer_ids: list[str]
    churn_probability: np.ndarray
    flagged: np.ndarray
    risk_band: list[str]
    threshold: float
    latency_ms: float
    raw_probability: np.ndarray | None = None
    quarantined: pd.DataFrame | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "customerID": self.customer_ids,
                "churn_probability": self.churn_probability,
                "flagged": self.flagged,
                "risk_band": self.risk_band,
            }
        )

    def __len__(self) -> int:
        return len(self.customer_ids)


def _band(p: float, threshold: float) -> str:
    if p >= max(threshold + 0.15, 0.85):
        return "high"
    if p >= threshold:
        return "medium"
    if p >= threshold / 2:
        return "low"
    return "minimal"


class ChurnScorer:
    """The thing a serving endpoint actually calls.

    Responsibilities, in order:
      1. refuse input that violates the data contract (structural breaks block
         the whole batch; row-level defects raise, or are quarantined when the
         caller asks for that policy)
      2. produce calibrated scores *and* the decision, with the threshold
         baked in
      3. record how long it took (contract validation included)
      4. never silently repair bad input -- a silent repair in a serving path
         is how a model ends up scoring garbage and reporting confidence

    The required input columns default to exactly what the model reads
    (:func:`~churnml.features.required_input_columns`): the label and the
    customer identifier are *not* required, so the scorer works on real,
    unlabelled traffic.
    """

    def __init__(
        self,
        model: BaseEstimator | Pipeline,
        threshold: float,
        *,
        feature_columns: list[str] | None = None,
        model_version: str = "unknown",
        training_reference: pd.DataFrame | ReferenceProfile | None = None,
        metadata: dict[str, Any] | None = None,
        temperature: float = 1.0,
    ) -> None:
        self.model = model
        self.threshold = float(threshold)
        self.feature_columns = list(feature_columns) if feature_columns is not None else required_input_columns()
        self.model_version = model_version
        self._reference = (
            ReferenceProfile.from_frame(training_reference)
            if isinstance(training_reference, pd.DataFrame)
            else training_reference
        )
        self.metadata = metadata or {}
        self.temperature = 1.0
        self.set_temperature(temperature)

    # -- contract gate ----------------------------------------------------
    def validate_input(
        self,
        batch: pd.DataFrame,
        *,
        on_invalid: Literal["raise", "quarantine"] = "raise",
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Apply the data contract. Returns ``(clean_batch, quarantined_rows)``.

        Structural breaks (a missing column, a category the model has never
        seen) always block the batch: they mean an upstream system changed, so
        every score from this batch would be an extrapolation. Row-level
        defects (an impossible tenure, a duplicate customer, inconsistent
        billing totals) raise by default, or are set aside with a reason when
        ``on_invalid="quarantine"`` so one poison row cannot stop a nightly run.
        """
        if batch.empty:
            raise ValueError("empty batch")
        missing = [c for c in self.feature_columns if c not in batch.columns]
        if missing:
            raise ValueError(f"input missing required columns: {missing}")
        nulls = batch[self.feature_columns].isna().sum()
        bad = nulls[nulls > 0]
        if len(bad) and not self.metadata.get("tolerate_nulls", False):
            raise ValueError(f"unexpected nulls in required columns: {bad.to_dict()}")

        report = validate_dataframe(batch, mode="serving", quarantine_bad=(on_invalid == "quarantine"))
        errors = [f for f in report.findings if f.severity == "error"]
        structural = [f for f in errors if f.check.startswith(("domain.", "schema."))]
        if structural:
            raise ValueError(
                "Data contract violated (structural; batch blocked):\n  "
                + "\n  ".join(str(f) for f in structural)
            )
        if not errors:
            return batch, None
        if on_invalid == "raise":
            report.raise_if_failed()
        quarantined = report.quarantine
        clean = batch.drop(index=quarantined.index)
        if clean.empty:
            raise ValueError("every row in the batch failed the data contract")
        return clean, quarantined

    # -- scoring ----------------------------------------------------------
    def score(
        self,
        batch: pd.DataFrame,
        *,
        return_input: bool = False,
        on_invalid: Literal["raise", "quarantine"] = "raise",
    ) -> ScoringResult:
        t0 = time.perf_counter()
        clean, quarantined = self.validate_input(batch, on_invalid=on_invalid)

        ids = (
            clean["customerID"].astype(str).tolist()
            if "customerID" in clean.columns
            else [str(i) for i in clean.index]
        )

        X = clean[self.feature_columns]
        raw = np.asarray(self.model.predict_proba(X)[:, 1], dtype=float)
        proba = np.asarray(self.calibrated_probability(raw), dtype=float)
        flagged = proba >= self.threshold

        result = ScoringResult(
            customer_ids=ids,
            churn_probability=proba,
            raw_probability=raw,
            flagged=flagged,
            risk_band=[_band(p, self.threshold) for p in proba],
            threshold=self.threshold,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            quarantined=quarantined,
        )
        if return_input:
            result.extra["n_rows"] = len(clean)
        return result

    # -- monitoring hook --------------------------------------------------
    def check_drift(self, batch: pd.DataFrame) -> DriftReport:
        if self._reference is None:
            raise RuntimeError("no training reference stored; cannot check drift")
        return detect_drift(self._reference, batch)

    # -- calibration ------------------------------------------------------
    def set_temperature(self, temperature: float) -> None:
        """Set the calibration temperature that :meth:`score` applies.

        Calibration is a monotone transform of the logit, so ranking is
        untouched while the probability scale changes: ``T > 1`` softens,
        ``T < 1`` sharpens, ``T = 1`` is the identity. The decision threshold
        lives on the *calibrated* scale, so changing ``T`` means re-deriving
        the threshold.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.metadata["temperature"] = self.temperature

    def calibrated_probability(self, proba: np.ndarray | float) -> np.ndarray | float:
        t = self.temperature
        p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p)) / t
        out = 1.0 / (1.0 + np.exp(-logit))
        return float(out) if np.isscalar(proba) or out.ndim == 0 else out

    # -- persistence ------------------------------------------------------
    def save(self, path) -> None:
        import joblib

        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path) -> ChurnScorer:
        import joblib

        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} contains {type(obj).__name__}, not ChurnScorer")
        return obj


# --------------------------------------------------------------------------
# Latency
# --------------------------------------------------------------------------


def benchmark_latency(
    scorer: ChurnScorer,
    frame: pd.DataFrame,
    sizes: tuple[int, ...] = (100, 1_000, 10_000, 50_000),
    *,
    repeats: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Median wall-clock time to score batches of EXACTLY ``n`` rows.

    Rows are resampled *with replacement* up to each requested size, the
    identifier and label are dropped (resampling creates duplicate IDs, and
    neither is needed to score), one warm-up call is excluded, and every timed
    call is checked to have scored the number of rows it claims. Timings
    include contract validation, because that is part of what serving costs.
    """
    inputs = frame.drop(columns=[c for c in ("customerID", "Churn", "_target", "_row_id") if c in frame.columns])
    scorer.score(inputs.head(min(len(inputs), 200)))          # warm-up, discarded
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        batch = inputs.sample(n=n, replace=True, random_state=int(rng.integers(0, 2**31 - 1)))
        times = []
        for _ in range(repeats):
            result = scorer.score(batch)
            if len(result) != n:                               # the check that was missing
                raise AssertionError(f"benchmark scored {len(result)} rows, expected {n}")
            times.append(result.latency_ms)
        median_ms = float(np.median(times))
        rows.append(
            {
                "rows": n,
                "rows_scored": len(result),
                "median_ms": median_ms,
                "p95_ms": float(np.percentile(times, 95)),
                "us_per_row": median_ms / n * 1000.0,
            }
        )
    return pd.DataFrame(rows)
