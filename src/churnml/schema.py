"""Data contracts.

A schema is the cheapest bug prevention available in ML and the most
frequently skipped. It costs ~60 lines and catches the class of failure that
otherwise surfaces three months later as "the model got worse and nobody knows
why" -- usually because a column silently changed dtype, a category was
renamed upstream, or a feed started emitting nulls.

Design choice: validation returns a *report*, not an exception. A production
pipeline quarantines bad rows and alerts; a notebook that crashes on the first
violation teaches you nothing. `strict=True` switches to fail-fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from .config import ADDON_COLUMNS, NOT_APPLICABLE, TARGET

Severity = Literal["ok", "warning", "error"]

#: Columns that must be present, with the categorical domain we expect.
#: Anything outside these values is a schema break, not a rare customer.
EXPECTED_DOMAINS: dict[str, set[str]] = {
    "gender": {"Female", "Male"},
    "Partner": {"No", "Yes"},
    "Dependents": {"No", "Yes"},
    "PhoneService": {"No", "Yes"},
    "MultipleLines": {"No", "No phone service", "Yes"},
    "InternetService": {"No", "DSL", "Fiber optic"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"No", "Yes"},
    "PaymentMethod": {
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    },
    TARGET: {"No", "Yes"},
    **{col: {"No", "Yes", NOT_APPLICABLE} for col in ADDON_COLUMNS},
}

#: Numeric ranges outside which a row is almost certainly corrupt.
EXPECTED_RANGES: dict[str, tuple[float, float]] = {
    "SeniorCitizen": (0, 1),
    "tenure": (0, 72),
    "MonthlyCharges": (0, 200),
    "TotalCharges": (0, 10000),
}

REQUIRED_COLUMNS: tuple[str, ...] = (
    *EXPECTED_DOMAINS,
    "customerID",
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
)

#: Columns a *scoring* batch may omit. The label does not exist for a customer
#: who has not churned yet, and the identifier is bookkeeping, not a feature.
SERVING_OPTIONAL_COLUMNS: tuple[str, ...] = ("customerID", TARGET)

#: Plausible band for ``TotalCharges / (tenure x MonthlyCharges)``. Customers
#: are billed roughly their monthly price for every month of tenure, so in the
#: real data this ratio sits between 0.69 and 1.57 (plan changes explain the
#: spread). The band below never fires on a real row, but it catches records
#: that were edited inconsistently -- for example tenure shortened while
#: TotalCharges is left untouched, which no real customer can produce.
CHARGES_CONSISTENCY_BAND: tuple[float, float] = (0.5, 2.0)

ValidationMode = Literal["training", "serving"]


@dataclass
class Finding:
    """One validation observation, with enough detail to act on."""

    severity: Severity
    check: str
    message: str
    n_rows: int = 0

    def __str__(self) -> str:  # pragma: no cover - display only
        icon = {"ok": "PASS", "warning": "WARN", "error": "FAIL"}[self.severity]
        suffix = f" ({self.n_rows} rows)" if self.n_rows else ""
        return f"[{icon}] {self.check}: {self.message}{suffix}"


@dataclass
class ValidationReport:
    """Structured outcome of a validation pass."""

    findings: list[Finding] = field(default_factory=list)
    quarantine: pd.DataFrame | None = None

    def add(
        self, severity: Severity, check: str, message: str, n_rows: int = 0
    ) -> None:
        self.findings.append(Finding(severity, check, message, n_rows))

    @property
    def n_errors(self) -> int:
        return sum(f.severity == "error" for f in self.findings)

    @property
    def n_warnings(self) -> int:
        return sum(f.severity == "warning" for f in self.findings)

    @property
    def ok(self) -> bool:
        return self.n_errors == 0

    def raise_if_failed(self) -> None:
        """Fail-fast gate for production. The notebook calls this too."""
        if not self.ok:
            lines = [str(f) for f in self.findings if f.severity == "error"]
            raise ValueError(
                "Data contract violated:\n  " + "\n  ".join(lines)
            )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "severity": f.severity,
                    "check": f.check,
                    "message": f.message,
                    "n_rows": f.n_rows,
                }
                for f in self.findings
            ]
        )

    def __str__(self) -> str:  # pragma: no cover - display only
        head = (
            f"ValidationReport: {self.n_errors} error(s), "
            f"{self.n_warnings} warning(s)"
        )
        return "\n".join([head, *(str(f) for f in self.findings)])


def validate_dataframe(
    df: pd.DataFrame,
    *,
    quarantine_bad: bool = False,
    mode: ValidationMode = "training",
) -> ValidationReport:
    """Validate raw customer data against the contract.

    Checks, in order of how badly they hurt when missed:
      1. required columns present
      2. identifier uniqueness (duplicates silently corrupt every metric)
      3. null counts
      4. categorical domain drift
      5. numeric ranges
      6. cross-field consistency (add-ons without internet; TotalCharges
         against tenure x MonthlyCharges)

    ``mode="training"`` demands the full 21-column frame. ``mode="serving"``
    is the contract for a *scoring* batch: the label and the identifier are
    optional (a customer who has not churned yet has no label), and every
    other check is identical, so the same rules guard training and serving.
    """
    report = ValidationReport()
    required = tuple(
        c for c in REQUIRED_COLUMNS
        if mode == "training" or c not in SERVING_OPTIONAL_COLUMNS
    )

    # -- 1. structure ----------------------------------------------------
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.add(
            "error",
            "schema.columns",
            f"missing required columns: {missing}",
            n_rows=len(df),
        )
        return report
    report.add("ok", "schema.columns", f"all {len(required)} required columns present")

    bad_mask = pd.Series(False, index=df.index)
    reasons = pd.Series("", index=df.index, dtype="object")

    def flag(mask: pd.Series, check: str) -> None:
        """Mark rows as bad and remember WHICH check rejected them."""
        nonlocal bad_mask
        mask = mask.fillna(False).astype(bool)
        bad_mask = bad_mask | mask
        reasons.loc[mask] = reasons.loc[mask] + check + "; "

    # -- 2. identifier ----------------------------------------------------
    if "customerID" in df.columns:
        dupes = int(df["customerID"].duplicated().sum())
        if dupes:
            flag(df["customerID"].duplicated(keep="first"), "id.uniqueness")
            report.add(
                "error",
                "id.uniqueness",
                f"customerID is not unique; {dupes} duplicate row(s)",
                n_rows=dupes,
            )
        else:
            report.add("ok", "id.uniqueness", "customerID unique")

    # -- 3. nulls ---------------------------------------------------------
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        report.add(
            "warning",
            "nulls.count",
            f"nulls in {nulls.to_dict()}",
            n_rows=int(nulls.sum()),
        )
    else:
        report.add("ok", "nulls.count", "no nulls in any column")

    # -- 4. categorical domains ------------------------------------------
    for col, domain in EXPECTED_DOMAINS.items():
        if col not in df.columns:      # only the label can be absent (serving mode)
            continue
        seen = set(df[col].dropna().unique())
        unexpected = seen - domain
        if unexpected:
            n_bad = int(df[col].isin(unexpected).sum())
            flag(df[col].isin(unexpected), f"domain.{col}")
            report.add(
                "error",
                f"domain.{col}",
                f"unexpected value(s) {sorted(unexpected)}; expected {sorted(domain)}",
                n_rows=n_bad,
            )

    if not any(f.severity == "error" and f.check.startswith("domain.") for f in report.findings):
        report.add("ok", "domain.categorical", "all categorical values in domain")

    # -- 5. numeric ranges -------------------------------------------------
    for col, (lo, hi) in EXPECTED_RANGES.items():
        s = pd.to_numeric(df[col], errors="coerce")
        n_bad = int(((s < lo) | (s > hi)).sum())
        if n_bad:
            flag((s < lo) | (s > hi), f"range.{col}")
            report.add(
                "error",
                f"range.{col}",
                f"{n_bad} value(s) outside [{lo}, {hi}]",
                n_rows=n_bad,
            )
    if not any(f.check.startswith("range.") for f in report.findings):
        report.add("ok", "range.numeric", "all numeric values within range")

    # -- 6. cross-field consistency ---------------------------------------
    # A customer with no internet cannot have declined a security add-on.
    # If these disagree, the record is internally inconsistent and one of the
    # two fields is stale.
    addon_cols = list(ADDON_COLUMNS)
    no_internet = df["InternetService"].eq("No")
    inconsistent = no_internet & df[addon_cols].eq("Yes").any(axis=1)
    n_inconsistent = int(inconsistent.sum())
    if n_inconsistent:
        report.add(
            "warning",
            "consistency.addons_without_internet",
            f"{n_inconsistent} record(s) have no internet but an active add-on",
            n_rows=n_inconsistent,
        )
    else:
        report.add(
            "ok",
            "consistency.addons_without_internet",
            "add-on flags consistent with InternetService",
        )

    # TotalCharges is the running sum of what a customer was billed, so it
    # must be roughly tenure x MonthlyCharges. A record that violates this was
    # edited inconsistently (or corrupted upstream): scoring it produces a
    # confident number from a customer who cannot exist. Rows with tenure 0 are
    # skipped -- their total is legitimately blank/zero.
    tenure = pd.to_numeric(df["tenure"], errors="coerce")
    monthly = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    total = pd.to_numeric(df["TotalCharges"], errors="coerce")
    billed = (tenure > 0) & (monthly > 0) & total.notna()
    ratio = total / (tenure * monthly)
    lo, hi = CHARGES_CONSISTENCY_BAND
    off_band = billed & ((ratio < lo) | (ratio > hi))
    n_off_band = int(off_band.sum())
    if n_off_band:
        flag(off_band, "consistency.total_vs_tenure")
        report.add(
            "error",
            "consistency.total_vs_tenure",
            f"{n_off_band} record(s) where TotalCharges / (tenure x MonthlyCharges) "
            f"falls outside [{lo}, {hi}] (real data: 0.69-1.57)",
            n_rows=n_off_band,
        )
    else:
        report.add(
            "ok",
            "consistency.total_vs_tenure",
            "TotalCharges consistent with tenure x MonthlyCharges",
        )

    # -- whitespace-only numerics ----------------------------------------
    # TotalCharges arrives as object dtype with 11 cells that are ' ' -- a
    # blank string, which is NOT caught by isna(). They belong to brand-new
    # signups (tenure 0). Strings rather than nulls is the single most common
    # way this dataset is silently mis-handled.
    for col in ("TotalCharges",):
        coerced = pd.to_numeric(df[col], errors="coerce")
        n_uncoercible = int(coerced.isna().sum() - df[col].isna().sum())
        if n_uncoercible:
            report.add(
                "warning",
                f"coercion.{col}",
                f"{n_uncoercible} non-numeric cell(s) (blank strings) coerced to NaN",
                n_rows=n_uncoercible,
            )

    if quarantine_bad and bad_mask.any():
        report.quarantine = df.loc[bad_mask].copy()
        report.quarantine["quarantine_reason"] = reasons.loc[bad_mask].str.rstrip("; ")

    return report


def environment_manifest() -> dict[str, Any]:
    """Capture the exact software stack used to produce a result.

    Reproducibility is not a claim, it is a file. If a metric moves six months
    from now, this dict is the first thing anyone asks for.
    """
    import platform
    import sys
    from importlib.metadata import PackageNotFoundError, version

    packages = [
        "numpy", "pandas", "scikit-learn", "scipy", "lightgbm",
        "xgboost", "shap", "matplotlib", "joblib", "pydantic",
    ]
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:  # pragma: no cover
            versions[pkg] = "not installed"

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }
