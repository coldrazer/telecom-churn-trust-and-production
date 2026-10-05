"""Central configuration.

Every tunable in this project lives here. Nothing downstream hard-codes a path,
a seed, a threshold or a column name. That is the single change that most
reliably separates a notebook you can rerun in six months from one you cannot.

The config object is frozen: if a notebook mutates it, the run is invalid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

#: One seed for everything: split, model init, any library that takes one.
#: Note this controls *reproducibility*, not *stability* -- see README on why
#: a single seed still cannot tell you whether a 0.4-point gap is real.
SEED: int = 42


def project_root() -> Path:
    """Repository root, resolved relative to this file rather than the cwd.

    A notebook is launched from ``notebooks/``, pytest from the root, and the
    IDE from anywhere. Deriving paths from the working directory is the most
    common way a professional repo breaks in front of a reviewer.
    """
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Every filesystem location in one place."""

    root: Path = field(default_factory=project_root)

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def versions(self) -> Path:
        """Append-only model registry: one directory per promoted version."""
        return self.artifacts / "versions"

    @property
    def raw_csv(self) -> Path:
        return self.data / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    def ensure(self) -> Paths:
        """Create output directories. Safe to call repeatedly."""
        for p in (self.artifacts, self.reports, self.figures, self.versions):
            p.mkdir(parents=True, exist_ok=True)
        return self


PATHS = Paths()


# --------------------------------------------------------------------------
# Dataset schema knowledge
# --------------------------------------------------------------------------

TARGET: str = "Churn"
POSITIVE_LABEL: str = "Yes"
NEGATIVE_LABEL: str = "No"

#: Row-identifying columns. These MUST NOT reach a model: a model that can see
#: customerID will happily memorise the training customers, and the resulting
#: "explanations" will be nonsense that looks authoritative.
IDENTIFIER_COLUMNS: tuple[str, ...] = ("customerID",)

#: Columns stored as text but semantically numeric. The classic Telco trap:
#: TotalCharges arrives as `object` dtype with 11 whitespace-only values.
#: All 11 belong to customers with tenure 0 -- new signups who have not been
#: billed yet -- and none of them churned. Their total is a structural zero,
#: not missing data. Because the blanks are strings rather than nulls,
#: `dropna()` misses them, and a model that sees an object column silently
#: drops the whole feature.
NUMERIC_COERCIONS: dict[str, str] = {"TotalCharges": "float64"}

#: Add-on services that are only meaningful when the customer has internet.
#: "No internet service" is *structural* missingness -- the question was never
#: applicable -- not a behavioural answer. Encoding it as a third category
#: splits every genuinely-internet population and hurts both fit and
#: explanation. `features.StructuralMissingEncoder` collapses it to null.
ADDON_COLUMNS: tuple[str, ...] = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)

#: The value in the add-on columns that means "not applicable", not "declined".
NOT_APPLICABLE: str = "No internet service"

#: Columns stored as 0/1 integers but semantically categorical. Left as int
#: they get treated as continuous and the SHAP plot implies a smooth
#: relationship between "0 senior" and "1 senior" that does not exist.
BINARY_AS_CATEGORICAL: tuple[str, ...] = ("SeniorCitizen",)

#: Ordinal feature. The levels have a real order, so ordinal encoding carries
#: genuine information that one-hot throws away.
ORDINAL_COLUMNS: dict[str, list[str]] = {
    "Contract": ["Month-to-month", "One year", "Two year"]
}


# --------------------------------------------------------------------------
# Split / evaluation
# --------------------------------------------------------------------------

#: Held-out set is touched exactly once, at the very end.
TEST_SIZE: float = 0.20

#: Repeated stratified CV on the *training* portion only. One fold's score is
#: a single noisy number; the spread across folds is what tells you whether a
#: difference between two models is real or noise.
CV_SPLITS: int = 5
CV_REPEATS: int = 5

#: Decision threshold is a business parameter, not a model parameter. It is
#: selected on validation data and reported alongside 0.5 for contrast.
DEFAULT_THRESHOLD: float = 0.50


# --------------------------------------------------------------------------
# Model hyper-parameters
# --------------------------------------------------------------------------

#: Logistic regression is included deliberately: it is the *interpretability
#: floor*. If a tree ensemble cannot beat a well-specified linear model on
#: this data, the extra complexity is not earning its cost and its SHAP values
#: are explanations of noise.
LOGREG_C: float = 1.0
LOGREG_MAX_ITER: int = 2000

#: The project's *starting configuration* for LightGBM: 400 trees of up to 31
#: leaves. These are hand-chosen values, NOT the library defaults (which are
#: 100 trees and a 0.1 learning rate), and they are generous for ~4,500
#: training rows. Notebook 01 therefore compares logistic regression against
#: this configuration *and* against ``LGBM_REGULARISED_PARAMS`` below, because
#: "linear beats boosting" is only a claim about a specific configuration.
LGBM_PARAMS: dict[str, object] = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": 4,
    "verbose": -1,
}

#: A deliberately small, regularised LightGBM. Chosen once, by reasoning about
#: the data size rather than by searching: few leaves, a large minimum leaf
#: size, shrinkage and L2. It is NOT tuned, so any advantage it has over
#: ``LGBM_PARAMS`` reflects over-parameterisation, not a hyper-parameter search.
LGBM_REGULARISED_PARAMS: dict[str, object] = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 8,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.0,
    "reg_lambda": 5.0,
    "random_state": SEED,
    "n_jobs": 4,
    "verbose": -1,
}

#: Business economics for the threshold sweep. These are *illustrative* and
#: stated as such. The location of the optimal threshold depends entirely on
#: them (cheap offers push it well below 0.5, expensive ones above it), so the
#: notebook reports a sensitivity grid rather than a single "right" cut-off.
RETENTION_OFFER_COST: float = 100.0
#: Fraction of a would-be churner retained when offered.
RETENTION_SUCCESS_RATE: float = 0.40
#: Months of revenue preserved per retained customer.
MONTHLY_VALUE: float = 70.0
RETENTION_HORIZON_MONTHS: int = 12


@dataclass(frozen=True)
class Config:
    """Frozen bundle passed to every function that needs settings."""

    seed: int = SEED
    paths: Paths = PATHS
    test_size: float = TEST_SIZE
    cv_splits: int = CV_SPLITS
    cv_repeats: int = CV_REPEATS
    threshold: float = DEFAULT_THRESHOLD

    def with_(self, **overrides: object) -> Config:
        """Return a modified copy. Config itself stays immutable."""
        from dataclasses import replace

        return replace(self, **overrides)  # type: ignore[arg-type]


CONFIG = Config()
