"""Feature engineering as composable, serialisable transformers.

Two rules:

1. Every transform is a class with ``fit``/``transform``, so it can live
   inside a :class:`~sklearn.pipeline.Pipeline` and be serialised with the
   model. A transformation applied in a notebook cell *cannot* be
   reproduced at inference time -- that gap is where production bugs live.

2. Every transform takes a DataFrame and returns a DataFrame with stable
   column names. Positional arrays are how feature-name mismatches become
   silent prediction corruption.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .config import (
    ADDON_COLUMNS,
    BINARY_AS_CATEGORICAL,
    IDENTIFIER_COLUMNS,
    NOT_APPLICABLE,
    ORDINAL_COLUMNS,
)

#: Bookkeeping columns that must never reach a model.
NON_FEATURE_COLUMNS: tuple[str, ...] = (*IDENTIFIER_COLUMNS, "Churn", "_target", "_row_id")


class StructuralMissingEncoder(BaseEstimator, TransformerMixin):
    """Collapse "not applicable" answers to true missing values.

    The six add-on services (``OnlineSecurity`` ... ``StreamingMovies``)
    carry a third state, ``"No internet service"``. It reads like a category
    but it is not: it records that the question never applied.

    Left as its own level, it splits every genuinely-subscribed population in
    two and teaches the model that "declined" and "no internet" are
    meaningfully different behaviours. They are not, once you condition on
    internet service.

    This encoder maps the state to null and lets imputation handle it, so the
    model learns "among internet customers, did they take security?" -- the
    question actually being asked.
    """

    def __init__(self, columns: tuple[str, ...] = ADDON_COLUMNS,
                 na_value: str = NOT_APPLICABLE) -> None:
        self.columns = columns
        self.na_value = na_value

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> StructuralMissingEncoder:
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        present = [c for c in self.columns if c in out.columns]
        out[present] = out[present].replace(self.na_value, np.nan)
        # `string` dtype so the null survives to the imputer instead of
        # being silently re-labelled as the literal string "nan".
        for c in present:
            out[c] = out[c].astype("string")
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        feats = self.feature_names_in_ if input_features is None else np.asarray(
            input_features, dtype=object
        )
        return feats


class ChurnFeatureBuilder(BaseEstimator, TransformerMixin):
    """Drop bookkeeping columns and add derived features.

    Every derived column earns its place by being something a domain expert
    would actually ask for:

    ``charges_per_month``
        TotalCharges divided by tenure. The gap between this and
        MonthlyCharges is a billable, actionable discrepancy -- the customer
        is being charged more than their usage history implies.
    ``num_addons``
        Depth of product adoption. A scalar summary of six correlated flags.
    ``is_month_to_month``
        The single strongest contract signal, made directly addressable
        rather than left to the model to discover from one-hot splits.
    ``implied_total_charges``
        MonthlyCharges multiplied by tenure: what the customer would have been
        billed to date at today's price. A coarse lifetime-value proxy. (It is
        a product, not an average, so it is not named like one.)
    """

    def __init__(self, drop_columns: tuple[str, ...] = NON_FEATURE_COLUMNS) -> None:
        self.drop_columns = drop_columns

    # -- feature lists are resolved lazily so fit can record them ---------
    @property
    def numeric_features(self) -> list[str]:
        return ["tenure", "MonthlyCharges", "TotalCharges"]

    @property
    def ordinal_features(self) -> list[str]:
        return list(ORDINAL_COLUMNS)

    @property
    def categorical_features(self) -> list[str]:
        base = [
            "gender", "Partner", "Dependents", "PhoneService",
            "MultipleLines", "InternetService", "PaperlessBilling",
            "PaymentMethod",
        ]
        return base + list(BINARY_AS_CATEGORICAL)

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> ChurnFeatureBuilder:
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        missing = [
            c for c in (self.numeric_features + self.ordinal_features + self.categorical_features)
            if c not in self.feature_names_in_
        ]
        if missing:
            raise ValueError(f"input is missing expected columns: {missing}")
        self.derived_features_ = self._derived_names()
        return self

    @staticmethod
    def _derived_names() -> list[str]:
        return [
            "charges_per_month",
            "num_addons",
            "is_month_to_month",
            "implied_total_charges",
        ]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c in self.drop_columns:
            if c in out.columns:
                out = out.drop(columns=[c])

        # -- structural missingness -------------------------------------
        # "No internet service" is not a behavioural answer, so it becomes a
        # true null and the imputer handles it. Doing this here (rather than as
        # a separate ColumnTransformer entry) guarantees it happens before
        # num_addons is computed, so a customer without internet is counted as
        # having zero add-ons rather than being skipped.
        addons = [c for c in ADDON_COLUMNS if c in out.columns]
        if addons:
            out[addons] = out[addons].replace(NOT_APPLICABLE, np.nan)
            for c in addons:
                out[c] = out[c].astype("string")

        # -- derived features ------------------------------------------
        # Assignment order MUST match `_derived_names()`, because that method
        # is the contract get_feature_names_out() publishes. If the two
        # disagree, every downstream feature name is attached to the wrong
        # column -- silently.
        # tenure can be 0 (a brand-new signup); guard the divide explicitly
        # rather than relying on pandas' inf propagation.
        tenure = pd.to_numeric(out["tenure"], errors="coerce").fillna(0.0)
        total = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)
        monthly = pd.to_numeric(out["MonthlyCharges"], errors="coerce").fillna(0.0)

        out["charges_per_month"] = total / (tenure + 1.0)
        out["num_addons"] = (
            out[addons].eq("Yes").sum(axis=1).astype("float64")
            if addons
            else 0.0
        )
        out["is_month_to_month"] = out["Contract"].eq("Month-to-month").astype("float64")
        out["implied_total_charges"] = monthly * tenure
        return out

    @property
    def raw_input_columns(self) -> list[str]:
        """The canonical raw input columns, in a stable order.

        Used as a fallback in ``get_feature_names_out`` when the transformer
        has not been fitted. sklearn calls that method on the *first* step of a
        Pipeline without supplying an upstream name list, so relying on
        ``feature_names_in_`` there raises AttributeError.
        """
        from .config import ADDON_COLUMNS, TARGET

        return [
            "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
            "tenure", "PhoneService", "MultipleLines", "InternetService",
            *ADDON_COLUMNS, "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges", TARGET,
        ]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is None:
            base = getattr(
                self, "feature_names_in_",
                np.asarray(self.raw_input_columns, dtype=object),
            )
        else:
            base = np.asarray(input_features, dtype=object)
        kept = [f for f in base if f not in self.drop_columns]
        return np.asarray(list(kept) + self._derived_names(), dtype=object)


def build_preprocessor(
    *,
    drop_na_impute: bool = True,
    one_hot_min_frequency: int | None = 3,
) -> Pipeline:
    """Assemble the full preprocessing pipeline.

    Two ordered stages, and the order matters:

    1. ``engineer`` -- derive columns and drop bookkeeping, applied to the
       whole frame.
    2. ``encode`` -- encode/impute/scale the specific column groups.

    A :class:`ColumnTransformer` cannot add a column, so the deriving step has
    to be a separate stage that runs first. Chaining them in a ``Pipeline``
    (rather than inventing derived columns in a notebook cell) is what keeps
    the transform serialisable and therefore reproducible at inference time.

    ``one_hot_min_frequency`` groups levels below the threshold into a single
    ``infrequent_sklearn`` category. With 7k rows and 2-4 per level, this
    stops the model fitting noise on categories with almost no support --
    a quiet form of overfitting that inflates apparent feature importance.
    """
    from sklearn.impute import SimpleImputer

    builder = ChurnFeatureBuilder()

    steps_num = []
    if drop_na_impute:
        steps_num.append(("impute", SimpleImputer(strategy="median")))
    steps_num.append(("scale", StandardScaler()))

    steps_cat = []
    if drop_na_impute:
        steps_cat.append(("impute", SimpleImputer(strategy="most_frequent")))
    cat_encoder = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=one_hot_min_frequency,
        sparse_output=False,
    )
    steps_cat.append(("onehot", cat_encoder))

    ordinal_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[[*ORDINAL_COLUMNS[c]] for c in builder.ordinal_features],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    encode = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps_num), builder.numeric_features),
            ("ordinal", ordinal_pipe, builder.ordinal_features),
            ("categorical", Pipeline(steps_cat), builder.categorical_features),
            ("derived", "passthrough", builder._derived_names()),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return Pipeline([("engineer", builder), ("encode", encode)])


def required_input_columns() -> list[str]:
    """The raw columns a scoring batch must contain -- and nothing more.

    This is the *serving* contract, and it is deliberately narrower than the
    training frame: it excludes the identifier (``customerID``, optional at
    serving time) and the label (``Churn``), which does not exist for a
    customer who has not churned yet. A serving schema that demands the label
    can only ever be exercised on already-labelled data.

    The six add-on columns are required even though they feed only the derived
    ``num_addons``: ``ChurnFeatureBuilder`` silently treats absent add-on
    columns as "zero add-ons", which is exactly the kind of quiet repair a
    serving path must refuse.
    """
    builder = ChurnFeatureBuilder()
    cols: list[str] = []
    for c in (
        *builder.numeric_features,
        *builder.ordinal_features,
        *builder.categorical_features,
        *ADDON_COLUMNS,
    ):
        if c not in cols:
            cols.append(c)
    return cols


def get_encoder(preprocessor: Pipeline) -> ColumnTransformer:
    """Reach the ColumnTransformer inside a fitted preprocessing Pipeline."""
    return preprocessor.named_steps["encode"]


def feature_importance_names(preprocessor: Pipeline) -> list[str]:
    """Readable output column names, after dropping the stage prefixes."""
    encoder = get_encoder(preprocessor)
    return [
        n.replace("categorical__", "")
         .replace("derived__", "")
         .replace("numeric__", "")
         .replace("ordinal__", "")
        for n in encoder.get_feature_names_out()
    ]
