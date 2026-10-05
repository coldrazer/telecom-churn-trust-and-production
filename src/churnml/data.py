"""Loading, cleaning and splitting.

The design rule throughout: **anything that learns from data lives behind a
fit call.** Cleaning the whole frame before splitting is the most common
silent leak in tabular projects -- it lets validation rows inform a statistic
you then report as generalisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    IDENTIFIER_COLUMNS,
    NUMERIC_COERCIONS,
    PATHS,
    POSITIVE_LABEL,
    TARGET,
    Config,
)
from .schema import ValidationReport, environment_manifest, validate_dataframe


@dataclass
class Dataset:
    """A materialised dataset plus the provenance needed to trust it."""

    frame: pd.DataFrame
    report: ValidationReport
    source: Path

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"<Dataset {self.n_rows} rows x {len(self.frame.columns)} cols>"


@dataclass
class Splits:
    """Train / validation / test, split on the index so rows stay traceable."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def indices(self) -> dict[str, pd.Index]:
        return {"train": self.train.index, "val": self.val.index, "test": self.test.index}


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the raw CSV exactly as delivered -- no cleaning here.

    Keeping load and clean separate is deliberate. It means the raw bytes are
    always recoverable, and every cleaning decision is explicit and reviewable
    rather than hidden inside a chain of inplace mutations.
    """
    path = Path(path) if path is not None else PATHS.raw_csv
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Expected the IBM Telco Customer "
            "Churn CSV (WA_Fn-UseC_-Telco-Customer-Churn.csv)."
        )
    return pd.read_csv(path)


# --------------------------------------------------------------------------
# Clean
# --------------------------------------------------------------------------


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply documented cleaning. Returns (frame, log) for the notebook.

    Every step is logged with a before/after count. Silent cleaning is how
    datasets lose 300 rows and nobody notices for six months.
    """
    df = raw.copy()
    log: list[str] = []
    n0 = len(df)

    # -- 1. numeric coercion ---------------------------------------------
    # TotalCharges is `object` dtype. The 11 whitespace cells belong to
    # customers with tenure 0 -- new signups who have not been billed yet --
    # and none of them churned. Their total is *structurally* zero, not
    # missing at random, so we set it to 0.0 rather than dropping the rows or
    # imputing a mean. The log states what the data actually shows, computed
    # from the rows rather than asserted, so a different extract that breaks
    # this assumption is visible immediately.
    for col, dtype in NUMERIC_COERCIONS.items():
        if col not in df.columns:
            continue
        as_object = pd.to_numeric(df[col], errors="coerce")
        blank = as_object.isna() & df[col].notna()
        n_blank = int(blank.sum())
        if n_blank:
            tenure_zero = int((pd.to_numeric(df.loc[blank, "tenure"], errors="coerce") == 0).sum())
            churned = int((df.loc[blank, TARGET].astype("string").str.strip() == POSITIVE_LABEL).sum())
            df[col] = as_object.fillna(0.0)
            log.append(
                f"{col}: coerced to {dtype}; {n_blank} blank cell(s) -> 0.0 "
                f"({tenure_zero} of {n_blank} have tenure 0, {churned} churned: "
                "not-yet-billed new signups, a structural zero rather than missing data)"
            )
        else:
            df[col] = as_object.astype(dtype)
            log.append(f"{col}: cast to {dtype}")

    # -- 2. categorical hygiene -----------------------------------------
    for col in [*IDENTIFIER_COLUMNS, *(c for c in df.columns if df[c].dtype == object)]:
        if col in IDENTIFIER_COLUMNS:
            continue
        stripped = df[col].astype("string").str.strip()
        n_changed = int((stripped != df[col]).sum())
        if n_changed:
            df[col] = stripped
            log.append(f"{col}: stripped whitespace from {n_changed} value(s)")
        else:
            df[col] = stripped

    # -- 3. target --------------------------------------------------------
    df[TARGET] = df[TARGET].astype("string").str.strip()
    y = (df[TARGET] == POSITIVE_LABEL).astype(int)
    log.append(f"{TARGET}: binarised to 0/1, positive rate {y.mean():.4f}")

    df["_target"] = y
    df["_row_id"] = np.arange(len(df))

    # -- 4. drop unusable -------------------------------------------------
    before = len(df)
    df = df.loc[df[TARGET].notna()].copy()
    dropped = before - len(df)
    if dropped:
        log.append(f"dropped {dropped} row(s) with unusable target")

    log.insert(0, f"loaded {n0} rows -> {len(df)} rows after cleaning")
    return df, log


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------


def make_splits(
    df: pd.DataFrame, config: Config | None = None
) -> Splits:
    """Stratified train/val/test split with leakage assertions.

    The assertions here are the point of the function. A split that silently
    duplicates rows across boundaries produces a validation score that is
    slightly too high, which is exactly the number a reviewer will trust.
    """
    config = config or Config()

    idx = df.index.to_numpy()
    y = df["_target"].to_numpy()

    train_idx, test_idx = train_test_split(
        idx,
        test_size=config.test_size,
        stratify=y,
        random_state=config.seed,
    )
    # The validation set is carved from train only, so the test set is never
    # used for any selection decision -- not threshold choice, not model
    # choice, not early stopping.
    #
    # train_test_split returns *values* from its input array in [train, test]
    # order -- it does not return positions into it. So `val_idx` already
    # holds original frame labels and must not be re-indexed through
    # train_idx. (Verified against sklearn 1.9 rather than assumed.)
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.20,
        stratify=y[train_idx],
        random_state=config.seed,
    )

    splits = Splits(
        train=df.loc[train_idx].copy(),
        val=df.loc[val_idx].copy(),
        test=df.loc[test_idx].copy(),
    )
    assert_no_leakage(splits)
    return splits


def assert_no_leakage(splits: Splits) -> None:
    """Fail loudly if the split is not clean. No exceptions to this rule."""
    train_ids = set(splits.train["customerID"])
    val_ids = set(splits.val["customerID"])
    test_ids = set(splits.test["customerID"])

    overlap = {
        "train/val": train_ids & val_ids,
        "train/test": train_ids & test_ids,
        "val/test": val_ids & test_ids,
    }
    dirty = {k: v for k, v in overlap.items() if v}
    if dirty:
        # ASCII only: these messages end up in logs and terminals whose
        # encoding may not carry symbols like the set-intersection glyph.
        raise AssertionError(
            "Customer leakage across splits: "
            + ", ".join(f"{k}={len(v)}" for k, v in dirty.items())
        )

    n = splits.sizes()
    total = sum(n.values())
    assert n["test"] > 0 and n["val"] > 0 and n["train"] > 0, f"empty split: {n}"

    # Class balance should be preserved by stratification.
    for name, part in splits.__dict__.items():
        rate = part["_target"].mean()
        assert 0.20 < rate < 0.35, (
            f"{name} churn rate {rate:.3f} drifted outside the expected band; "
            "stratification looks broken"
        )

    assert len(train_ids) + len(val_ids) + len(test_ids) == total, (
        "row counts do not reconcile with unique customer counts"
    )


def load_and_prepare(
    path: Path | None = None, config: Config | None = None
) -> tuple[Dataset, Splits, list[str]]:
    """One-call entry point used by both notebooks."""
    config = config or Config()
    raw = load_raw(path)
    report = validate_dataframe(raw)
    frame, log = clean(raw)
    splits = make_splits(frame, config)
    return Dataset(frame=frame, report=report, source=PATHS.raw_csv), splits, log


def manifest() -> dict[str, object]:
    return environment_manifest()
