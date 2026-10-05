"""Model factories.

Two model families on purpose. A single model tells you what *that* model
does; comparing an interpretable linear model against a gradient-boosted one
tells you whether the complexity is earning anything, and -- more usefully for
this project -- gives you two independent explanations of the same data to
cross-check against each other.

If the boosted model cannot beat a well-specified logistic regression, its
SHAP values are explanations of noise, and no amount of plotting makes them
right.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import LGBM_PARAMS, LGBM_REGULARISED_PARAMS, LOGREG_C, LOGREG_MAX_ITER, SEED
from .features import build_preprocessor


@dataclass(frozen=True)
class NamedModel:
    """A model plus the metadata the notebook needs to report on it."""

    key: str
    display_name: str
    estimator: BaseEstimator
    family: str
    interprets: str

    @property
    def pipeline(self) -> Pipeline:
        return self.pipeline_

    def __post_init__(self) -> None:
        object.__setattr__(self, "pipeline_", self.estimator)


def _wrap(model: BaseEstimator, **preprocessor_kwargs: object) -> Pipeline:
    """Preprocess + model in one serialisable object."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor(**preprocessor_kwargs)),  # type: ignore[arg-type]
            ("model", model),
        ]
    )


def build_logreg(
    *,
    C: float = LOGREG_C,
    max_iter: int = LOGREG_MAX_ITER,
    class_weight: str | None = None,
    preprocessor_kwargs: dict[str, object] | None = None,
) -> Pipeline:
    """L2-regularised logistic regression -- the interpretability floor.

    Coefficients are directly readable as log-odds and comparable in
    magnitude across standardised inputs, which is what makes this model the
    control condition for every explanation in the notebook.

    Model hyper-parameters are explicit keyword arguments; preprocessing
    options go in ``preprocessor_kwargs``. Routing both through ``**kwargs``
    is how ``C=0.02`` silently ends up as an argument to the imputer.
    """
    return _wrap(
        LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=SEED,
        ),
        **(preprocessor_kwargs or {}),
    )


def build_lgbm(
    *,
    preprocessor_kwargs: dict[str, object] | None = None,
    **model_kwargs: object,
) -> Pipeline:
    """LightGBM -- the performance model whose behaviour we then interrogate.

    Defaults come from ``config.LGBM_PARAMS``; any keyword overrides them.
    """
    from lightgbm import LGBMClassifier

    return _wrap(
        LGBMClassifier(**{**LGBM_PARAMS, **model_kwargs}),
        **(preprocessor_kwargs or {}),
    )


def build_lgbm_regularised(
    *,
    preprocessor_kwargs: dict[str, object] | None = None,
    **model_kwargs: object,
) -> Pipeline:
    """A small, regularised LightGBM (8 leaves, large leaves, shrinkage, L2).

    The fair opponent for the linear model. The project's starting LightGBM
    configuration (400 trees x 31 leaves) is generous for ~4,500 rows, so a
    gap against it measures over-parameterisation, not the ceiling of boosting.
    These parameters are fixed by reasoning about the data size, not searched.
    """
    from lightgbm import LGBMClassifier

    return _wrap(
        LGBMClassifier(**{**LGBM_REGULARISED_PARAMS, **model_kwargs}),
        **(preprocessor_kwargs or {}),
    )


def build_random_forest(
    *,
    n_estimators: int = 400,
    min_samples_leaf: int = 5,
    n_jobs: int = -1,
    preprocessor_kwargs: dict[str, object] | None = None,
) -> Pipeline:
    """A third opinion with very different inductive bias.

    Included because model-agnostic importance (permutation, SHAP) should be
    compared against a model that cannot use the same shortcuts LightGBM can.
    Agreement across three families is evidence; agreement within one is not.
    """
    return _wrap(
        RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=SEED,
        ),
        **(preprocessor_kwargs or {}),
    )


#: Display order in the notebook and the model card.
MODEL_BUILDERS = {
    "logreg": ("Logistic regression", "linear", build_logreg),
    "lgbm": ("LightGBM", "boosting", build_lgbm),
    "lgbm_small": ("LightGBM (regularised)", "boosting", build_lgbm_regularised),
    "rf": ("Random forest", "bagging", build_random_forest),
}


def named_models(include: tuple[str, ...] = ("logreg", "lgbm")) -> list[NamedModel]:
    out = []
    for key in include:
        display, family, builder = MODEL_BUILDERS[key]
        out.append(
            NamedModel(
                key=key,
                display_name=display,
                estimator=builder(),
                family=family,
                interprets={
                    "logreg": "sign and magnitude of each standardised input",
                    "lgbm": "conditional risk contribution per split path",
                    "lgbm_small": "conditional risk contribution per split path",
                    "rf": "averaged impurity reduction",
                }[key],
            )
        )
    return out


def extract_linear_coefficients(
    pipeline: Pipeline, *, top_n: int = 15
) -> object:
    """Return a tidy DataFrame of logistic-regression coefficients.

    Only meaningful for the logreg pipeline; returns an empty frame otherwise
    rather than raising, so the notebook can call it uniformly.
    """
    import pandas as pd

    model = pipeline.named_steps["model"]
    pre = pipeline.named_steps["preprocess"]

    if not hasattr(model, "coef_"):
        return pd.DataFrame(columns=["feature", "coef", "abs_coef"])

    names = list(pre.get_feature_names_out())
    coefs = np.asarray(model.coef_).ravel()

    if len(names) != len(coefs):
        # Ordinal/derived columns can change width; fall back to positional
        # labels rather than crashing mid-notebook.
        names = [f"f{i}" for i in range(len(coefs))]

    tidy = pd.DataFrame({"feature": names, "coef": coefs})
    tidy["abs_coef"] = tidy["coef"].abs()
    tidy["direction"] = np.where(tidy["coef"] > 0, "increases risk", "decreases risk")
    return tidy.reindex(tidy["abs_coef"].sort_values(ascending=False).index).head(top_n)
