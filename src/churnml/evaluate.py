"""Evaluation: metrics, uncertainty, calibration, thresholds, slices.

The organising principle here is that **a point estimate without an interval
is not a result.** Every headline number this project reports comes with a
bootstrap interval, because the difference between two models on 1,400
validation rows is frequently smaller than the noise in measuring it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import (
    MONTHLY_VALUE,
    RETENTION_HORIZON_MONTHS,
    RETENTION_OFFER_COST,
    RETENTION_SUCCESS_RATE,
)

PRIMARY_METRIC = "pr_auc"  # average precision; see `metric_report` docstring


# --------------------------------------------------------------------------
# Core metrics
# --------------------------------------------------------------------------


def metric_report(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Standard binary metrics at a given decision threshold.

    We report average precision rather than ROC-AUC as the headline. On
    imbalanced problems (26.5% positives here) ROC-AUC flatters weak models by
    rewarding them for correctly ranking easy negatives. Average precision
    answers the operational question: of the customers who *actually* churn,
    what fraction have we ranked in our top slice?
    """
    y_pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def metric_frame(
    y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5
) -> pd.DataFrame:
    return pd.DataFrame([metric_report(y_true, proba, threshold)])


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------


def bootstrap_ci(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    metric: str = "roc_auc",
    threshold: float = 0.5,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a metric.

    Resamples *rows* with replacement, which is the correct unit here: rows
    are independent, features within a row are not.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    def _score(idx: np.ndarray) -> float:
        if metric in ("roc_auc", "pr_auc"):
            yt, pp = y_true[idx], proba[idx]
            if yt.sum() == 0 or yt.sum() == len(yt):
                return np.nan
            return (
                roc_auc_score(yt, pp) if metric == "roc_auc"
                else average_precision_score(yt, pp)
            )
        yt = y_true[idx]
        pp = (proba[idx] >= threshold).astype(int)
        if yt.sum() == 0:
            return np.nan
        if metric == "precision":
            return precision_score(yt, pp, zero_division=0)
        if metric == "recall":
            return recall_score(yt, pp, zero_division=0)
        if metric == "brier":
            return brier_score_loss(yt, proba[idx])
        return float((pp == yt).mean())

    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        stats[b] = _score(rng.integers(0, n, n))

    stats = stats[~np.isnan(stats)]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_delta(
    y_true: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    *,
    metric: str = "roc_auc",
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap on the *difference* between two models.

    Pairing is the point. Because both models score the same rows, the
    bootstrap resamples rows jointly and cancels the row-to-row variation that
    dominates the marginal intervals. The resulting interval on the
    difference is far tighter, and is the only honest way to answer "is model
    A actually better than model B?".
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true = np.asarray(y_true)

    def _m(y: np.ndarray, p: np.ndarray) -> float:
        if metric == "roc_auc":
            return roc_auc_score(y, p)
        if metric == "pr_auc":
            return average_precision_score(y, p)
        return roc_auc_score(y, p)

    deltas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.sum() in (0, len(yt)):
            deltas[b] = np.nan
            continue
        deltas[b] = _m(yt, proba_a[idx]) - _m(yt, proba_b[idx])

    deltas = deltas[~np.isnan(deltas)]
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {
        "delta": float(_m(y_true, proba_a) - _m(y_true, proba_b)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_better": float((deltas > 0).mean()),
        "significant": bool(lo > 0 or hi < 0),
    }


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def calibration_table(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Observed churn rate per predicted-probability bin.

    A perfectly calibrated model has ``observed == predicted`` on every row.
    The *direction* of miscalibration differs by model family -- boosted trees
    often over-predict in the top bin, a regularised logistic regression can
    under-predict there -- so read the gap column rather than assuming it.
    """
    df = pd.DataFrame({"y_true": np.asarray(y_true), "proba": np.asarray(proba)})
    # quantile bins: equal-count, so sparse high-probability bins are not empty
    df["bin"] = pd.qcut(df["proba"], q=n_bins, duplicates="drop")
    out = (
        df.groupby("bin", observed=True)
        .agg(
            n=("y_true", "size"),
            mean_predicted=("proba", "mean"),
            observed_rate=("y_true", "mean"),
        )
        .reset_index(drop=True)
    )
    out["gap"] = out["mean_predicted"] - out["observed_rate"]
    out["abs_gap"] = out["gap"].abs()
    return out


def expected_calibration_error(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> float:
    """Single-number calibration summary (sample-weighted mean |gap|)."""
    cal = calibration_table(y_true, proba, n_bins)
    total = cal["n"].sum()
    return float((cal["abs_gap"] * cal["n"]).sum() / total)


def fit_temperature(
    y_true: np.ndarray,
    proba: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    """Fit the calibration temperature by minimising log-loss on a validation set.

    Temperature scaling divides the logit by ``T``: ``T > 1`` softens
    over-confident scores, ``T < 1`` sharpens under-confident ones, and
    ``T = 1`` means the scores were already well scaled. It is a one-parameter
    monotone transform, so ranking cannot change. Returns ``(best_T, curve)``
    where ``curve`` is the log-loss at every grid point.
    """
    from sklearn.metrics import log_loss

    grid = np.round(np.arange(0.5, 3.01, 0.05), 3) if grid is None else np.asarray(grid)
    p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p))
    losses = [log_loss(np.asarray(y_true), 1.0 / (1.0 + np.exp(-logits / t))) for t in grid]
    curve = pd.DataFrame({"temperature": grid, "log_loss": losses})
    return float(curve.loc[curve["log_loss"].idxmin(), "temperature"]), curve


# --------------------------------------------------------------------------
# Threshold economics
# --------------------------------------------------------------------------


def threshold_sweep(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
    offer_cost: float = RETENTION_OFFER_COST,
    success_rate: float = RETENTION_SUCCESS_RATE,
    monthly_value: float = MONTHLY_VALUE,
    horizon_months: int = RETENTION_HORIZON_MONTHS,
) -> pd.DataFrame:
    """Sweep the decision threshold and attach business consequences.

    The economics are **illustrative**, and the notebook says so. What is not
    illustrative is the logic: a customer is worth contacting when
    ``P(churn) x success x value > offer cost``, so the best threshold is set
    by the cost/benefit ratio. Cheap offers push it well below 0.5, expensive
    offers push it above 0.5; 0.5 is optimal only by coincidence. See
    :func:`threshold_sensitivity`.
    """
    thresholds = (
        thresholds if thresholds is not None
        else np.round(np.arange(0.05, 0.96, 0.01), 3)
    )
    rows = []
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        n_flagged = int(y_pred.sum())

        expected_saves = tp * success_rate
        gross_value = expected_saves * monthly_value * horizon_months
        total_cost = n_flagged * offer_cost

        rows.append(
            {
                "threshold": t,
                "n_flagged": n_flagged,
                "pct_flagged": n_flagged / len(y_true),
                "tp": tp, "fp": fp, "fn": fn,
                "precision": tp / n_flagged if n_flagged else np.nan,
                "recall": tp / (tp + fn) if (tp + fn) else np.nan,
                "accuracy_flagged": tp / n_flagged if n_flagged else np.nan,
                "expected_saves": expected_saves,
                "offer_cost_total": total_cost,
                "retained_value": gross_value,
                "net_value": gross_value - total_cost,
            }
        )
    return pd.DataFrame(rows)


def optimal_threshold(sweep: pd.DataFrame) -> float:
    """Highest net-value threshold. Ties break toward flagging fewer people."""
    best = sweep.loc[sweep["net_value"].idxmax()]
    tied = sweep.loc[np.isclose(sweep["net_value"], best["net_value"], rtol=1e-9)]
    return float(tied["threshold"].max())


def net_value_at(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    *,
    offer_cost: float = RETENTION_OFFER_COST,
    success_rate: float = RETENTION_SUCCESS_RATE,
    monthly_value: float = MONTHLY_VALUE,
    horizon_months: int = RETENTION_HORIZON_MONTHS,
) -> float:
    """Net value of flagging everyone at or above ``threshold`` (same formula as the sweep)."""
    y_true = np.asarray(y_true)
    flagged = np.asarray(proba) >= threshold
    true_positives = int((flagged & (y_true == 1)).sum())
    gross = true_positives * success_rate * monthly_value * horizon_months
    return float(gross - int(flagged.sum()) * offer_cost)


def net_value_delta_ci(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold_a: float,
    threshold_b: float,
    *,
    n_boot: int = 2000,
    seed: int = 42,
    **economics: float,
) -> dict[str, float]:
    """Bootstrap interval for ``net_value(a) - net_value(b)`` on ONE sample.

    A threshold chosen on validation data will look better there than it
    deserves to (it was picked *because* it looked best). The honest check is
    to evaluate both thresholds on rows that played no part in the choice, and
    to report how uncertain the difference is: a gain whose interval spans zero
    is a hypothesis, not a result.
    """
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    n = len(y_true)
    deltas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[b] = (
            net_value_at(y_true[idx], proba[idx], threshold_a, **economics)
            - net_value_at(y_true[idx], proba[idx], threshold_b, **economics)
        )
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {
        "delta": net_value_at(y_true, proba, threshold_a, **economics)
        - net_value_at(y_true, proba, threshold_b, **economics),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_a_better": float((deltas > 0).mean()),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def threshold_sensitivity(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    offer_costs: tuple[float, ...] = (25.0, 50.0, 100.0, 200.0, 300.0),
    success_rates: tuple[float, ...] = (0.2, 0.4, 0.6),
    **fixed: float,
) -> pd.DataFrame:
    """Where does the net-value-optimal threshold sit as the economics change?

    The point of the grid is to stop anyone reading the optimum of ONE set of
    illustrative costs as a property of the model. It is a property of the
    cost/benefit ratio.
    """
    rows = []
    for cost in offer_costs:
        for success in success_rates:
            sweep = threshold_sweep(
                y_true, proba, offer_cost=cost, success_rate=success, **fixed
            )
            rows.append(
                {"offer_cost": cost, "success_rate": success,
                 "optimal_threshold": optimal_threshold(sweep)}
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Slice analysis
# --------------------------------------------------------------------------


def slice_report(
    y_true: np.ndarray,
    proba: np.ndarray,
    frame: pd.DataFrame,
    slices: dict[str, pd.Series],
    *,
    threshold: float = 0.5,
    min_n: int = 30,
    top_n: int | None = None,
    sort_by: str = "error_rate",
    ascending: bool = False,
) -> pd.DataFrame:
    """Error rate and metric quality within each subgroup *level*.

    Aggregate accuracy is an average over segments that may differ wildly in
    size and risk. A model can look fine overall while being close to useless
    for the segment you most need to act on.

    **Raw error rate is not a quality measure across segments.** A segment
    where 55% of customers churn can hardly score below ~45% error even for a
    model that ranks perfectly, while a segment with 3% churn gets ~3% error
    from a model that never flags anyone. So each row also reports:

    - ``baseline_error``: the error of always predicting the segment's majority
      class (``min(churn_rate, 1 - churn_rate)``),
    - ``skill_vs_majority``: ``1 - error_rate / baseline_error``, i.e. how much
      of that baseline error the model removes (0 = no better than ignoring
      the features, 1 = perfect),
    - ``auc``: within-segment ranking quality (NaN if a single class).

    ``slices`` maps a slice name to a *label* column (e.g. ``{"Contract":
    frame["Contract"]}``); this function expands each column into one row per
    distinct level.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    y_pred = (proba >= threshold).astype(int)
    err = (y_pred != y_true).astype(int)
    correct_pos = ((y_pred == 1) & (y_true == 1))

    rows = []
    for name, series in slices.items():
        s = pd.Series(series).reset_index(drop=True)
        for level in pd.unique(s.dropna()):
            # .eq() yields a guaranteed-bool Series. (Calling .to_numpy() on a
            # pandas 3 string/categorical column gives an *object* array whose
            # .sum() concatenates strings instead of counting.)
            mask = s.eq(level).to_numpy(dtype=bool)
            n = int(mask.sum())
            if n < min_n:
                continue
            pos = y_true[mask]
            n_pos = int(pos.sum())
            n_flagged = int(y_pred[mask].sum())
            churn_rate = float(pos.mean())
            error_rate = float(err[mask].mean())
            baseline = min(churn_rate, 1.0 - churn_rate)
            rows.append(
                {
                    "slice": name,
                    "level": str(level),
                    "n": n,
                    "share": n / len(y_true),
                    "churn_rate": churn_rate,
                    "mean_pred": float(proba[mask].mean()),
                    "error_rate": error_rate,
                    "baseline_error": baseline,
                    "skill_vs_majority": (
                        1.0 - error_rate / baseline if baseline > 0 else float("nan")
                    ),
                    "auc": (
                        float(roc_auc_score(pos, proba[mask]))
                        if 0 < n_pos < n else float("nan")
                    ),
                    "recall": float(correct_pos[mask].sum() / max(n_pos, 1)),
                    "precision": float(correct_pos[mask].sum() / max(n_flagged, 1)),
                }
            )

    out = pd.DataFrame(rows).sort_values(sort_by, ascending=ascending)
    return out.head(top_n).reset_index(drop=True) if top_n else out.reset_index(drop=True)


def default_slices(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Business-meaningful segments, including tenure bands.

    Tenure bands are derived from domain knowledge rather than quantiles so
    the boundaries mean something operationally (a "first 6 months" cohort
    behaves differently from a "3+ years" cohort).
    """
    tenure = pd.to_numeric(frame["tenure"], errors="coerce")
    return {
        "Contract": frame["Contract"],
        "InternetService": frame["InternetService"],
        "PaperlessBilling": frame["PaperlessBilling"],
        "PaymentMethod": frame["PaymentMethod"],
        "SeniorCitizen": frame["SeniorCitizen"].map({0: "not senior", 1: "senior"}),
        "tenure_band": pd.cut(
            tenure,
            bins=[-1, 6, 12, 24, 36, 72],
            labels=["0-6 mo", "7-12 mo", "13-24 mo", "25-36 mo", "37+ mo"],
        ),
        "charges_band": pd.qcut(
            pd.to_numeric(frame["MonthlyCharges"], errors="coerce"),
            q=4, labels=["low", "mid-low", "mid-high", "high"],
        ),
    }


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------


@dataclass
class CVResult:
    """Per-fold scores for several models evaluated on IDENTICAL folds.

    Using the same folds for every model is what makes the comparison *paired*:
    fold-to-fold difficulty cancels in the per-fold differences.
    """

    scores: dict[str, list[float]]            # per-fold PR-AUC
    summary: pd.DataFrame
    roc_scores: dict[str, list[float]] = field(default_factory=dict)
    test_train_ratio: float = 0.25            # n_test / n_train within one fold

    def best(self) -> str:
        return str(self.summary.sort_values("pr_auc_mean", ascending=False).index[0])

    def spread(self, key: str) -> float:
        """Range (max minus min) of the per-fold PR-AUC.

        This is the *range*, not a standard error. It is a crude noise scale
        for eyeballing; use :meth:`paired_difference` to decide whether a gap
        between two models is real.
        """
        return float(self.summary.loc[key, "pr_auc_max"] - self.summary.loc[key, "pr_auc_min"])

    def paired_difference(
        self, model_a: str, model_b: str, metric: str = "pr_auc"
    ) -> dict[str, float]:
        """Is ``model_a`` better than ``model_b``? Corrected resampled t-test.

        Repeated k-fold folds overlap in their training data, so the per-fold
        differences are positively correlated and a plain paired t-test is
        over-confident. Nadeau & Bengio (2003) correct the variance by
        ``1/k + n_test/n_train``, which is what this does.
        """
        from scipy import stats

        source = self.scores if metric == "pr_auc" else self.roc_scores
        diff = np.asarray(source[model_a]) - np.asarray(source[model_b])
        k = len(diff)
        variance = float(diff.var(ddof=1)) * (1.0 / k + self.test_train_ratio)
        se = float(np.sqrt(variance))
        mean = float(diff.mean())
        if se == 0.0:
            t_stat, p_value = (float("inf") if mean else 0.0), (0.0 if mean else 1.0)
        else:
            t_stat = mean / se
            p_value = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=k - 1)))
        half_width = float(stats.t.ppf(0.975, df=k - 1)) * se
        return {
            "mean_diff": mean,
            "ci_low": mean - half_width,
            "ci_high": mean + half_width,
            "t": float(t_stat),
            "p_value": p_value,
            "significant": bool(p_value < 0.05),
        }


def cross_validate(pipelines: dict[str, object], X: pd.DataFrame, y: np.ndarray,
                   *, n_splits: int = 5, n_repeats: int = 5, seed: int = 42) -> CVResult:
    """Repeated stratified CV over several pipelines.

    Repeated (not just k-fold) because the fold-to-fold spread is itself the
    interesting quantity: a 0.3-point gap between two models sitting inside a
    1.0-point spread is not a real difference.
    """
    from sklearn.base import clone
    from sklearn.model_selection import RepeatedStratifiedKFold

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    y = np.asarray(y)

    scores: dict[str, list[float]] = {}
    roc_scores: dict[str, list[float]] = {}
    for key, pipe in pipelines.items():
        fold_scores, fold_roc = [], []
        for train_idx, val_idx in cv.split(X, y):
            m = clone(pipe)
            m.fit(X.iloc[train_idx], y[train_idx])
            proba = m.predict_proba(X.iloc[val_idx])[:, 1]
            fold_scores.append(average_precision_score(y[val_idx], proba))
            fold_roc.append(roc_auc_score(y[val_idx], proba))
        scores[key] = fold_scores
        roc_scores[key] = fold_roc

    summary = pd.DataFrame(
        {
            "pr_auc_mean": {k: float(np.mean(v)) for k, v in scores.items()},
            "pr_auc_std": {k: float(np.std(v, ddof=1)) for k, v in scores.items()},
            "pr_auc_min": {k: float(np.min(v)) for k, v in scores.items()},
            "pr_auc_max": {k: float(np.max(v)) for k, v in scores.items()},
            "roc_auc_mean": {k: float(np.mean(v)) for k, v in roc_scores.items()},
            "roc_auc_std": {k: float(np.std(v, ddof=1)) for k, v in roc_scores.items()},
        }
    )
    return CVResult(
        scores=scores,
        summary=summary,
        roc_scores=roc_scores,
        test_train_ratio=1.0 / (n_splits - 1),
    )
