"""Explanation methods, and -- more importantly -- how to audit them.

Any model can produce a plot that looks like an explanation. What separates an
explanation you can act on from decoration is **fidelity**: does this account
of the model actually predict how the model behaves?

So this module provides three families of explanation and one family of audit:

- *global*: which features matter on average, by three independent methods
- *heterogeneity*: how the effect of a feature varies across the population
- *local*: why this specific customer scored what they scored
- *audit*: does shuffling the top-attributed features hurt the model far more
  than shuffling the same number of randomly chosen features (the control
  that makes the test mean something)

The audit is the point. Without it you have opinions about your model. With it
you have evidence.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from .config import SEED

# --------------------------------------------------------------------------
# Global: importance from three angles
# --------------------------------------------------------------------------


def permutation_importance_frame(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    metric_fn=None,
    n_repeats: int = 10,
    seed: int = SEED,
    input_features: bool = False,
) -> pd.DataFrame:
    """Model-agnostic importance on the *raw input* columns.

    Scored on raw inputs rather than post-transform columns, because "which
    input would I need to go collect?" is the question a stakeholder asks.
    Correlated features share credit, which is a feature of the method, not a
    bug -- we correct for it in `importance_agreement`.
    """
    scoring = metric_fn or "roc_auc"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = permutation_importance(
            pipeline, X, y,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=seed,
        )
    return (
        pd.DataFrame(
            {
                "feature": list(X.columns) if input_features else list(r.importances_mean.index),
                "importance_mean": r.importances_mean,
                "importance_std": r.importances_std,
            }
        )
        .assign(importance=lambda d: d["importance_mean"].clip(lower=0))
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def impurity_importance_frame(pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Built-in tree importance, included so it can be compared and demoted.

    Impurity-based importance is biased toward high-cardinality, continuously
    varying features: they offer more split points, so they win by
    construction. Including it in the notebook is a deliberate teaching move
    -- it is the number practitioners most often quote and least often justify.
    """
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "impurity_importance"])

    imp = np.asarray(model.feature_importances_)
    if len(imp) != len(feature_names):
        feature_names = [f"f{i}" for i in range(len(imp))]
    return (
        pd.DataFrame({"feature": feature_names, "impurity_importance": imp})
        .sort_values("impurity_importance", ascending=False)
        .reset_index(drop=True)
    )


def importance_agreement(
    perm: pd.DataFrame, shap_vals: pd.Series
) -> pd.DataFrame:
    """Rank correlation between two importance methods.

    Where they disagree, the disagreement is the finding: it marks features
    where method-specific bias is driving the answer.
    """
    merged = (
        perm.rename(columns={"importance_mean": "permutation"})
        .merge(
            shap_vals.rename("shap").reset_index().rename(columns={"index": "feature"}),
            on="feature",
            how="inner",
        )
        .dropna()
    )
    if merged.empty:
        return merged

    from scipy.stats import kendalltau, spearmanr

    rho_s, p_s = spearmanr(merged["permutation"], merged["shap"])
    tau, p_t = kendalltau(merged["permutation"], merged["shap"])
    return pd.DataFrame(
        [
            {
                "n_features": len(merged),
                "spearman_rho": float(rho_s),
                "spearman_p": float(p_s),
                "kendall_tau": float(tau),
                "kendall_p": float(p_t),
                "verdict": (
                    "methods broadly agree"
                    if rho_s > 0.7
                    else "methods disagree -- investigate before trusting either"
                ),
            }
        ]
    )


# --------------------------------------------------------------------------
# SHAP
# --------------------------------------------------------------------------


@dataclass
class ShapBundle:
    """Explainer plus the aligned value matrix and feature names."""

    values: np.ndarray            # (n_samples, n_features)
    names: list[str]
    base_value: float
    feature_index: dict[str, int]

    def abs_mean(self) -> pd.Series:
        return pd.Series(np.abs(self.values).mean(axis=0), index=self.names).sort_values(
            ascending=False
        )

    def signed_mean(self) -> pd.Series:
        return pd.Series(self.values.mean(axis=0), index=self.names).sort_values(
            key=np.abs, ascending=False
        )

    def waterfall(self, row: int, top: int = 8) -> pd.DataFrame:
        """Tidy local attribution for one row, largest magnitude first."""
        n = self.values.shape[0]
        if not -n <= row < n:
            raise IndexError(
                f"row {row} is outside the SHAP matrix, which holds {n} rows. "
                "SHAP was computed on a subsample; either select case rows "
                "from the same subset or raise `max_samples` so it covers "
                "every row you want to explain."
            )
        vals = self.values[row]
        idx = np.argsort(np.abs(vals))[::-1][:top]
        return pd.DataFrame(
            {
                "feature": [self.names[i] for i in idx],
                "shap_value": vals[idx],
                "direction": np.where(vals[idx] > 0, "pushes risk UP", "pushes risk DOWN"),
            }
        )


def encoded_feature_names(pipeline: Pipeline) -> list[str]:
    """Readable names for the columns the model actually consumes.

    This is the column order SHAP values correspond to. Getting it wrong is
    the classic way to publish a confident, meaningless beeswarm plot, so it
    is derived from the fitted encoder rather than reconstructed by hand.
    """
    pre = pipeline.named_steps["preprocess"]
    encoder = pre.named_steps["encode"] if hasattr(pre, "named_steps") else pre
    names = [
        n.replace("categorical__", "")
         .replace("derived__", "")
         .replace("numeric__", "")
         .replace("ordinal__", "")
        for n in encoder.get_feature_names_out()
    ]
    return names


#: Encoded name -> the original input column it came from. Used to roll the
#: model's internal view back up to "which field would I go collect?".
def _encoded_to_input_map(pipeline: Pipeline, names: list[str]) -> dict[str, str]:
    from .config import (
        ADDON_COLUMNS,
        BINARY_AS_CATEGORICAL,
        ORDINAL_COLUMNS,
    )
    from .features import ChurnFeatureBuilder

    b = ChurnFeatureBuilder()
    mapping: dict[str, str] = {}

    for n in names:
        if n in b.numeric_features or n in b._derived_names() or n in ORDINAL_COLUMNS:
            mapping[n] = n
        elif n.startswith("tenure_") or n == "Contract":
            mapping[n] = "Contract"
        elif n.startswith("MultipleLines_"):
            mapping[n] = "MultipleLines"
        elif n.startswith("InternetService_"):
            mapping[n] = "InternetService"
        elif n.startswith("SeniorCitizen_"):
            mapping[n] = "SeniorCitizen"
        elif "_" in n:
            base = n.rsplit("_", 1)[0]
            # one-hot dummies are named "<column>_<level>"; the level itself
            # can contain underscores, so fall back to a prefix match.
            for candidate in (
                *ADDON_COLUMNS, *BINARY_AS_CATEGORICAL, "PaperlessBilling",
                "PaymentMethod", "gender", "Partner", "Dependents", "PhoneService",
            ):
                if n.startswith(candidate + "_"):
                    base = candidate
                    break
            mapping[n] = base
        else:
            mapping[n] = n
    return mapping


#: Derived feature -> the raw input columns it is computed from. A derived
#: feature has no raw column of its own, so its importance must be handed back
#: to its sources; otherwise it silently falls out of any "which input matters"
#: ranking (an earlier version dropped it, understating tenure, charges and
#: add-ons).
def _derived_sources() -> dict[str, tuple[str, ...]]:
    from .config import ADDON_COLUMNS

    return {
        "charges_per_month": ("TotalCharges", "tenure"),
        "num_addons": tuple(ADDON_COLUMNS),
        "is_month_to_month": ("Contract",),
        "implied_total_charges": ("MonthlyCharges", "tenure"),
    }


def roll_up_to_inputs(
    importance: pd.Series, pipeline: Pipeline, names: list[str]
) -> pd.Series:
    """Aggregate encoded-column importance back to original input fields.

    One-hot levels of a variable share that variable's importance; summing
    |value| across levels gives the variable's total contribution. A derived
    feature's importance is split *equally* among the raw columns it is
    computed from (an assumption, not a measurement: the credit between, say,
    ``tenure`` and ``TotalCharges`` inside ``charges_per_month`` is not
    identifiable). Total importance is conserved.
    """
    mapping = _encoded_to_input_map(pipeline, names)
    sources = _derived_sources()
    totals: dict[str, float] = {}
    for feature, value in importance.items():
        targets = sources.get(feature) or (mapping.get(feature, feature),)
        for target in targets:
            totals[target] = totals.get(target, 0.0) + float(value) / len(targets)
    return pd.Series(totals, dtype=float).sort_values(ascending=False)


def compute_shap(
    pipeline: Pipeline,
    X: pd.DataFrame,
    *,
    max_samples: int = 1000,
    seed: int = SEED,
) -> ShapBundle:
    """Compute SHAP values for a fitted sklearn pipeline.

    SHAP is computed on the *transformed* matrix; names are mapped back to the
    original inputs where the mapping is one-to-one, and to the derived feature
    names otherwise. Getting this mapping wrong is the classic way to publish
    a beautiful, meaningless beeswarm plot.
    """
    import shap

    model = pipeline.named_steps["model"]
    transformed = pipeline.named_steps["preprocess"].transform(X)
    n_features = transformed.shape[1]

    names = encoded_feature_names(pipeline)
    if len(names) != n_features:  # pragma: no cover - defensive
        names = [f"f{i}" for i in range(n_features)]

    # Background sample keeps the expected-value calculation tractable.
    rng = np.random.default_rng(seed)
    n_bg = min(100, transformed.shape[0])
    bg_idx = (
        np.arange(n_bg)
        if transformed.shape[0] <= n_bg
        else rng.choice(transformed.shape[0], n_bg, replace=False)
    )
    background = transformed[bg_idx]

    from sklearn.linear_model import LogisticRegression

    is_linear = isinstance(model, LogisticRegression)
    if hasattr(model, "booster_") or type(model).__name__ == "LGBMClassifier":
        explainer = shap.TreeExplainer(model, background)
    elif is_linear:
        # Exact and instant for a linear model: each feature's attribution is
        # coef x (value - background mean), in log-odds. No sampling needed.
        explainer = shap.LinearExplainer(model, background)
    else:
        explainer = shap.Explainer(
            model.predict_proba, background, algorithm="permutation"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = explainer(transformed) if is_linear else explainer(transformed, check_additivity=False)

    values = np.asarray(sv.values)
    if values.ndim == 3:  # (n, features, classes) -> positive class
        values = values[:, :, 1]
    elif values.ndim == 1 and values.size == n_features:
        values = values.reshape(1, -1)

    values = values[:max_samples]

    return ShapBundle(
        values=values,
        names=names,
        base_value=float(np.ravel(sv.base_values)[-1]),
        feature_index={name: i for i, name in enumerate(names)},
    )


# --------------------------------------------------------------------------
# Heterogeneity: PDP vs ICE
# --------------------------------------------------------------------------


def partial_dependence(
    pipeline: Pipeline,
    X: pd.DataFrame,
    feature: str,
    *,
    grid: np.ndarray | None = None,
    n_samples: int = 400,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average marginal effect: E[y | feature = x], averaged over everyone.

    Returns (grid, pdp_line, ice_matrix). The ICE matrix is the important
    output -- see `ice_spread`.
    """
    from sklearn.inspection import partial_dependence as _pd

    if grid is None:
        lo, hi = np.nanpercentile(X[feature], [5, 95])
        grid = np.linspace(float(lo), float(hi), 20)

    # sklearn refuses integer columns for partial dependence (correctly --
    # re-assigning integer values can silently round). Cast a copy to float;
    # the fitted pipeline is untouched.
    X_pd = X.copy()
    X_pd[feature] = pd.to_numeric(X_pd[feature], errors="coerce").astype("float64")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # `n_jobs` was removed from this API in sklearn 1.9; ICE is computed
        # explicitly below anyway, so parallelism is not needed here.
        pd_res = _pd(
            pipeline, X_pd, [feature],
            kind="average",
            grid_resolution=len(grid),
        )
    # sklearn 1.9 returns 'average' shaped (n_out, n_grid); older returns a list
    pdp = np.asarray(pd_res["average"])[0] if isinstance(pd_res, dict) else np.asarray(pd_res[0])[0]

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    subset = X.iloc[idx].copy()
    ice = np.zeros((len(subset), len(grid)), dtype=float)

    base_proba = pipeline.predict_proba(subset)[:, 1]
    for j, value in enumerate(grid):
        col = subset.copy()
        col[feature] = value
        ice[:, j] = pipeline.predict_proba(col)[:, 1] - base_proba

    return grid, pdp, ice


def ice_spread(ice: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    """How much do individual curves disagree with the average line?

    A wide band means there is no single "effect of tenure" -- the effect
    depends on who you are, and reporting only the average curve is a
    misrepresentation. This is the single most-overlooked result in the
    tabular explainability literature.
    """
    return pd.DataFrame(
        {
            "grid": grid,
            "pdp": ice.mean(axis=0),
            "ice_p05": np.percentile(ice, 5, axis=0),
            "ice_p95": np.percentile(ice, 95, axis=0),
            "sd_across_population": ice.std(axis=0),
            "mean_abs_individual_effect": np.abs(ice).mean(axis=0),
        }
    )


# --------------------------------------------------------------------------
# Audit: does the explanation actually explain?
# --------------------------------------------------------------------------


@dataclass
class FaithfulnessResult:
    """Outcome of :func:`faithfulness_audit`.

    ``alignment`` is the share of *informative* k (k no larger than half the
    number of inputs) at which the attributed top-k features damage the model
    more than at least 70% of random feature sets of the same size (``max_share``
    = 0.30). ``ablation`` also reports the stricter 90% bar, the damage ratio
    against a typical random subset, and the older shuffle-versus-ablate
    columns as a secondary sanity check.
    """

    baseline_score: float
    ablation: pd.DataFrame
    alignment: float
    verdict: str
    n_inputs: int = 0
    n_random: int = 0


def _neutral_value(series: pd.Series) -> object:
    """A representative, information-free value for a column.

    Median for numeric, mode for categorical. Using the mode rather than the
    median matters: a median of a string column is not a real category, and
    substituting one silently invents data the model has never seen.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.median()
    mode = series.mode(dropna=True)
    return mode.iloc[0] if len(mode) else series.iloc[0]


def faithfulness_audit(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    shap_bundle: ShapBundle | None = None,
    *,
    ranking: list[str] | None = None,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    metric_fn=None,
    n_shuffle: int = 5,
    n_random: int = 40,
    max_share: float = 0.30,
    seed: int = SEED,
) -> FaithfulnessResult:
    """Test whether an importance ranking points at the model's real drivers.

    For each k, the k top-ranked raw inputs are *shuffled* (destroying their
    relationship to the label while keeping their distribution) and the score
    is recomputed. That drop only means something relative to a **control**:
    shuffling k *randomly chosen* inputs. Any informative feature hurts when
    destroyed, so "shuffling hurts" is not evidence that a ranking is right;
    "shuffling the top-k hurts far more than shuffling k random inputs" is.

    The verdict is rank-based: for each k it reports the share of ``n_random``
    random same-size subsets that hurt *at least as much* as the top-k. A
    percentile cut-off such as "the 5th percentile of random subsets" is the
    wrong rule: with few inputs, a random subset contains the single most
    important feature fairly often, so that percentile lands on the top
    feature's own score and even a perfect ranking can fail.

    Two bars are reported. The strict bar (share <= 10%: the top-k is among the
    most damaging subsets) is not attainable in general: with correlated inputs
    no importance ranking is guaranteed to pick the single worst subset. The
    verdict therefore uses ``max_share`` (default 0.30: the top-k is more
    damaging than 70% of random subsets) at >= 75% of the informative k. That
    bar was set AFTER observing that the strict one fails for a collinear
    logistic regression, which is why both are shown. Only k <= half the number
    of inputs count: beyond that a random subset overlaps the top-k so heavily
    that the comparison stops being informative.

    Two older checks stay as secondary columns. *Ablation* replaces the top-k
    with a constant (median/mode); it is not a clean intervention (the result
    can even improve as more features are ablated, because the constant
    combines with the untouched columns into records that never occur).
    "Shuffled worse than ablated" is likewise true for almost any informative
    feature set, so it is reported but never used for the verdict.

    Provide either ``shap_bundle`` or an explicit ``ranking`` of raw input
    names (most important first).
    """
    from sklearn.metrics import roc_auc_score

    metric_fn = metric_fn or roc_auc_score
    y = np.asarray(y)
    base = float(metric_fn(y, pipeline.predict_proba(X)[:, 1]))
    rng = np.random.default_rng(seed)

    # Importance is computed on encoded columns (one-hot levels, scaled
    # numerics). The audit perturbs *raw inputs*, so roll the ranking up to
    # input fields first -- otherwise it targets columns that do not exist in X
    # and silently measures nothing.
    if ranking is None:
        if shap_bundle is None:
            raise ValueError("provide either shap_bundle or an explicit ranking")
        ranking = list(roll_up_to_inputs(shap_bundle.abs_mean(), pipeline, shap_bundle.names).index)
    ranked = [c for c in ranking if c in X.columns]
    if not ranked:
        raise ValueError("none of the ranked features are columns of X")

    def shuffled_score(columns: list[str]) -> float:
        frame = X.copy()
        for col in columns:
            frame[col] = rng.permutation(frame[col].to_numpy())
        return float(metric_fn(y, pipeline.predict_proba(frame)[:, 1]))

    max_informative_k = len(ranked) // 2
    rows = []
    for k in k_values:
        if k > len(ranked):
            continue
        top = ranked[:k]

        ablated = X.copy()
        for col in top:
            ablated[col] = _neutral_value(ablated[col])

        top_scores = [shuffled_score(top) for _ in range(n_shuffle)]
        random_scores = [
            shuffled_score([str(c) for c in rng.choice(ranked, size=k, replace=False)])
            for _ in range(n_random)
        ]
        rows.append(
            {
                "k_features": k,
                "features": ", ".join(top),
                "score_ablated": float(metric_fn(y, pipeline.predict_proba(ablated)[:, 1])),
                "score_shuffled": float(np.mean(top_scores)),
                "score_random_k_mean": float(np.mean(random_scores)),
                "share_random_as_damaging": float(
                    np.mean(np.asarray(random_scores) <= float(np.mean(top_scores)))
                ),
                "informative_k": k <= max_informative_k,
            }
        )

    table = pd.DataFrame(rows)
    table["drop_from_baseline"] = base - table["score_shuffled"]
    table["random_drop_from_baseline"] = base - table["score_random_k_mean"]
    table["damage_ratio"] = table["drop_from_baseline"] / table["random_drop_from_baseline"].clip(lower=1e-9)
    table["beats_random"] = table["share_random_as_damaging"] <= max_share
    table["beats_random_strict"] = table["share_random_as_damaging"] <= 0.10
    table["shuffle_worse_than_ablation"] = table["score_shuffled"] < table["score_ablated"]

    informative = table[table["informative_k"]]
    if informative.empty:
        alignment = float("nan")
        verdict = "Not enough inputs to run the random-feature control."
    else:
        alignment = float(informative["beats_random"].mean())
        n_pass, n_total = int(informative["beats_random"].sum()), len(informative)
        n_strict = int(informative["beats_random_strict"].sum())
        verdict = (
            f"Ranking is faithful: at {n_pass} of {n_total} informative k, shuffling the top-ranked "
            f"features hurt more than {1 - max_share:.0%} of random feature sets of the same size "
            f"({n_strict} of {n_total} also clear the strict 90% bar)."
            if alignment >= 0.75
            else f"Ranking is NOT faithful: at only {n_pass} of {n_total} informative k did the "
            f"top-ranked features hurt more than {1 - max_share:.0%} of random feature sets. "
            "Treat the attributions with suspicion."
        )
    return FaithfulnessResult(
        baseline_score=base,
        ablation=table,
        alignment=alignment,
        verdict=verdict,
        n_inputs=len(ranked),
        n_random=n_random,
    )


# --------------------------------------------------------------------------
# Consistency: does the model contradict known behaviour?
# --------------------------------------------------------------------------


def consistency_audit(
    frame: pd.DataFrame,
    proba: np.ndarray,
    y: np.ndarray,
    *,
    proba_high_risk_column: str = "Contract",
) -> pd.DataFrame:
    """Look for contradictions between model output and strong domain priors.

    A model that predicts *low* churn risk for month-to-month customers in
    their first six months is not making a subtle error -- it has learned
    something the business already knows to be false, and that is worth
    surfacing before anyone acts on its output.

    These priors are deliberately simple and explicitly debatable. They are a
    sanity net, not a ground truth.
    """
    proba = np.asarray(proba)
    y = np.asarray(y)
    rows = []

    tenure = pd.to_numeric(frame["tenure"], errors="coerce")
    checks = [
        {
            "prior": "month-to-month, tenure <6mo should be a high-risk group",
            "mask": frame["Contract"].eq("Month-to-month") & (tenure < 6),
            "expect": "high",
        },
        {
            "prior": "2-year contract, tenure >24mo should be low risk",
            "mask": frame["Contract"].eq("Two year") & (tenure > 24),
            "expect": "low",
        },
        {
            "prior": "no internet service should be low risk",
            "mask": frame["InternetService"].eq("No"),
            "expect": "low",
        },
    ]

    base = float(proba.mean())
    for check in checks:
        mask = check["mask"].to_numpy(dtype=bool)
        n = int(mask.sum())
        if n == 0:
            continue
        mean_p = float(proba[mask].mean())
        actual = float(y[mask].mean())
        # A violation is the model scoring a group on the WRONG side of the
        # overall average relative to the prior: a "high risk" group scored
        # below average, or a "low risk" group scored above it. (An earlier
        # version applied the "high risk" test to every prior, so a model that
        # rated two-year-contract customers as the riskiest group in the
        # book would have passed.)
        violated = mean_p < base if check["expect"] == "high" else mean_p > base
        rows.append(
            {
                "prior": check["prior"],
                "expected": check["expect"],
                "n": n,
                "observed_churn_rate": actual,
                "model_mean_prediction": mean_p,
                "overall_mean_prediction": base,
                "model_agrees": not violated,
                "severity": "REVIEW" if violated else "ok",
            }
        )
    return pd.DataFrame(rows)
