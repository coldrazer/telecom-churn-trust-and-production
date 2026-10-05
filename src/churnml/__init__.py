"""churnml -- a small, explicit ML library for the churn showcase.

Notebooks hold the narrative. This package holds the logic.

The split exists so that (a) the analysis is unit-testable outside a notebook
context, (b) two notebooks share one definition of "what a clean split is", and
(c) anything used at serving time is importable and importable *only* from
tested code.
"""

from __future__ import annotations

__version__ = "1.1.0"

from .config import (  # noqa: F401
    ADDON_COLUMNS,
    CONFIG,
    PATHS,
    SEED,
    TARGET,
    Config,
    Paths,
)
from .data import (  # noqa: F401
    Dataset,
    Splits,
    assert_no_leakage,
    clean,
    load_and_prepare,
    load_raw,
    make_splits,
)
from .evaluate import (  # noqa: F401
    bootstrap_ci,
    calibration_table,
    cross_validate,
    default_slices,
    expected_calibration_error,
    fit_temperature,
    metric_report,
    net_value_at,
    net_value_delta_ci,
    optimal_threshold,
    paired_bootstrap_delta,
    slice_report,
    threshold_sensitivity,
    threshold_sweep,
)
from .explain import (  # noqa: F401
    ShapBundle,
    compute_shap,
    consistency_audit,
    faithfulness_audit,
    ice_spread,
    impurity_importance_frame,
    partial_dependence,
    permutation_importance_frame,
)
from .features import (  # noqa: F401
    ChurnFeatureBuilder,
    StructuralMissingEncoder,
    build_preprocessor,
    required_input_columns,
)
from .models import (  # noqa: F401
    MODEL_BUILDERS,
    build_lgbm,
    build_lgbm_regularised,
    build_logreg,
    build_random_forest,
    extract_linear_coefficients,
    named_models,
)
from .pipeline import (  # noqa: F401
    ChurnScorer,
    DriftReport,
    ReferenceProfile,
    ScoringResult,
    benchmark_latency,
    detect_drift,
    population_stability_index,
)
from .registry import (  # noqa: F401
    GateResult,
    IntegrityError,
    PromotionBlocked,
    QualityGate,
    latest,
    list_versions,
    load_active,
    promote,
    promotion_log,
    register_model,
    rollback,
)
from .schema import ValidationReport, environment_manifest, validate_dataframe  # noqa: F401

__all__ = [n for n in dir() if not n.startswith("_")]
