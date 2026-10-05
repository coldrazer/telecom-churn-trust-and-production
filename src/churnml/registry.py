"""Model registry: versioned artifacts, manifests, a real gate, and model cards.

The gap between "I trained a model" and "there is a model I can deploy and
roll back" is mostly bookkeeping. This module is that bookkeeping:

- every bundle gets a SHA-256 recorded next to it, and promotion and loading
  both *verify* it, so "which bytes are serving?" has an exact, checked answer
- versions are sequential (``v001``, ``v002`` ...) and never overwritten;
  ``register_model`` only ever creates the next one
- every version carries a manifest with the data hash, package versions, the
  selection metrics that justified it and -- separately -- held-out test metrics
- a :class:`QualityGate` can be handed to :func:`promote`, which then REFUSES to
  promote a version whose recorded metrics miss the floor (overriding requires
  an explicit ``force`` and a written reason, both logged)
- every promotion and rollback is appended to ``promotion_log.jsonl``
- rollback re-activates the previous version's exact bytes; it never retrains
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from .config import PATHS, Paths

PROMOTION_LOG = "promotion_log.jsonl"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _portable_path(path: Path | str) -> str:
    """Path relative to the project root, so manifests never embed a home directory."""
    p = Path(path).resolve()
    try:
        return p.relative_to(PATHS.root.resolve()).as_posix()
    except ValueError:
        return p.name


# --------------------------------------------------------------------------
# Quality gate
# --------------------------------------------------------------------------


class PromotionBlocked(Exception):  # noqa: N818  (public name; reads as an event, not an error class)
    """Raised by :func:`promote` when a version fails the quality gate."""

    def __init__(self, version: str, failures: list[str]) -> None:
        self.version = version
        self.failures = list(failures)
        super().__init__(
            f"promotion of {version} blocked by the quality gate: " + "; ".join(self.failures)
        )


class IntegrityError(Exception):
    """Raised when a bundle's bytes do not match the checksum recorded for it."""


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: list[str]
    floors: dict[str, float]
    ceilings: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "failures": self.failures,
                "floors": self.floors, "ceilings": self.ceilings}


@dataclass(frozen=True)
class QualityGate:
    """Minimum bar a version's recorded metrics must clear to be promoted.

    ``floors`` are metrics that must be at least the given value (e.g. ROC-AUC),
    ``ceilings`` metrics that must be at most it (e.g. Brier score). A metric
    that is required but absent counts as a failure: a gate that passes when
    the evidence is missing is not a gate.

    The gate is a *floor*, not a ranking: a version can pass and still be worse
    than the one it would replace, which is why rollback exists.
    """

    floors: dict[str, float] = field(default_factory=dict)
    ceilings: dict[str, float] = field(default_factory=dict)

    def evaluate(self, metrics: dict[str, float]) -> GateResult:
        failures: list[str] = []
        for name, floor in self.floors.items():
            value = metrics.get(name)
            if value is None:
                failures.append(f"{name} not reported (floor {floor})")
            elif value < floor:
                failures.append(f"{name} = {value:.4f} is below the floor {floor}")
        for name, ceiling in self.ceilings.items():
            value = metrics.get(name)
            if value is None:
                failures.append(f"{name} not reported (ceiling {ceiling})")
            elif value > ceiling:
                failures.append(f"{name} = {value:.4f} is above the ceiling {ceiling}")
        return GateResult(
            passed=not failures, failures=failures,
            floors=dict(self.floors), ceilings=dict(self.ceilings),
        )


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


@dataclass
class Manifest:
    """Everything needed to know whether a result can be reproduced."""

    version: str
    created_utc: str
    model_key: str
    model_family: str
    data_sha256: str
    data_path: str
    n_train: int
    n_val: int
    n_test: int
    seed: int
    threshold: float
    metrics: dict[str, float] = field(default_factory=dict)
    selection_split: str = "validation"
    test_metrics: dict[str, float] = field(default_factory=dict)
    metric_intervals: dict[str, list[float]] = field(default_factory=dict)
    gate: dict[str, Any] = field(default_factory=dict)
    temperature: float = 1.0
    bundle_sha256: str = ""
    packages: dict[str, str] = field(default_factory=dict)
    python: str = ""
    platform: str = ""
    feature_names: list[str] = field(default_factory=list)
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)


def next_version(registry_dir: Path) -> str:
    """Sequential version string: v001, v002, ... (never reuses a number)."""
    registry_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(registry_dir.glob("v*"))
    n = 0
    for path in existing:
        stem = path.name.split(".")[0]
        if stem.startswith("v") and stem[1:].isdigit():
            n = max(n, int(stem[1:]))
    return f"v{n + 1:03d}"


@dataclass
class RegisteredModel:
    version: str
    dir: Path

    @property
    def bundle(self) -> Path:
        return self.dir / "model.joblib"

    @property
    def manifest(self) -> Path:
        return self.dir / "manifest.json"

    @property
    def card(self) -> Path:
        return self.dir / "MODEL_CARD.md"

    @property
    def checksums(self) -> Path:
        return self.dir / "CHECKSUMS.txt"

    def load_manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest.read_text(encoding="utf-8"))

    def load(self) -> Any:
        return joblib.load(self.bundle)

    def exists(self) -> bool:
        return self.bundle.exists() and self.manifest.exists()

    def verify(self) -> str:
        """Check the bundle against its recorded checksum; return the SHA-256."""
        actual = sha256_file(self.bundle)
        if not self.checksums.exists():
            raise IntegrityError(f"{self.version} has no CHECKSUMS.txt to verify against")
        recorded = self.checksums.read_text(encoding="utf-8").split()[0]
        if actual != recorded:
            raise IntegrityError(
                f"{self.version} bundle does not match its checksum "
                f"(recorded {recorded[:12]}..., actual {actual[:12]}...)"
            )
        return actual


# --------------------------------------------------------------------------
# Register
# --------------------------------------------------------------------------


def register_model(
    obj: Any,
    *,
    model_key: str,
    model_family: str,
    metrics: dict[str, float],
    data_path: Path,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    threshold: float,
    feature_names: list[str] | None = None,
    metric_intervals: dict[str, list[float]] | None = None,
    notes: str = "",
    versions_dir: Path | None = None,
    test_metrics: dict[str, float] | None = None,
    selection_split: str = "validation",
    gate: QualityGate | None = None,
    temperature: float = 1.0,
) -> RegisteredModel:
    """Write a versioned artifact plus manifest, checksum and model card.

    ``metrics`` are the *selection* metrics (computed on the split used to
    choose this model and its threshold); ``test_metrics`` are the untouched
    held-out numbers. Keeping them apart in the manifest and the card is what
    stops a selection-set score being read as an honest held-out estimate.
    If a ``gate`` is given, its verdict on ``metrics`` is recorded (promotion
    is where it is *enforced*).
    """
    from .schema import environment_manifest

    versions_dir = versions_dir or (PATHS.versions)
    version = next_version(versions_dir)
    target = versions_dir / version
    target.mkdir(parents=True, exist_ok=True)

    bundle_path = target / "model.joblib"
    joblib.dump(obj, bundle_path, compress=3)
    bundle_sha = sha256_file(bundle_path)

    env = environment_manifest()
    clean_metrics = {k: float(v) for k, v in metrics.items()}
    manifest = Manifest(
        version=version,
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        model_key=model_key,
        model_family=model_family,
        data_sha256=sha256_file(Path(data_path)),
        data_path=_portable_path(data_path),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=seed,
        threshold=threshold,
        metrics=clean_metrics,
        selection_split=selection_split,
        test_metrics={k: float(v) for k, v in (test_metrics or {}).items()},
        metric_intervals=metric_intervals or {},
        gate=gate.evaluate(clean_metrics).to_dict() if gate else {},
        temperature=float(temperature),
        bundle_sha256=bundle_sha,
        packages=env["packages"],
        python=env["python"],
        platform=env["platform"],
        feature_names=feature_names or [],
        notes=notes,
    )
    (target / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")
    (target / "MODEL_CARD.md").write_text(render_model_card(manifest), encoding="utf-8")
    (target / "CHECKSUMS.txt").write_text(f"{bundle_sha}  model.joblib\n", encoding="utf-8")
    return RegisteredModel(version=version, dir=target)


def _metric_rows(metrics: dict[str, float]) -> str:
    return "\n".join(f"| {k} | {v:.4f} |" for k, v in metrics.items())


def render_model_card(manifest: Manifest) -> str:
    """Render a human-readable model card.

    Kept deliberately blunt about limits, and *evidence-based*: the calibration
    section reports what was measured (or says nothing was), instead of
    asserting a property the model may not have. A card that oversells --
    or undersells with boilerplate -- transfers the wrong confidence to
    whoever reads it.
    """
    intervals = (
        "\n".join(
            f"| {k} | [{lo:.4f}, {hi:.4f}] |"
            for k, (lo, hi) in manifest.metric_intervals.items()
        )
        or "| _none reported_ | |"
    )
    test_block = (
        f"| Metric | Value |\n|---|---|\n{_metric_rows(manifest.test_metrics)}"
        if manifest.test_metrics
        else "_Not recorded at registration. Do not quote the selection metrics above as held-out performance._"
    )
    ece = manifest.metrics.get("ece")
    calibration = (
        f"Expected calibration error {ece:.4f} on the {manifest.selection_split} split, "
        f"with calibration temperature {manifest.temperature:g} "
        "(1 means the raw probabilities were left unchanged)."
        if ece is not None
        else "Calibration was not assessed when this version was registered; "
        "treat the probabilities as scores, not as likelihoods."
    )
    if manifest.gate:
        verdict = "PASSED" if manifest.gate.get("passed") else "FAILED"
        gate_block = f"{verdict} at registration."
        for failure in manifest.gate.get("failures", []):
            gate_block += f"\n- {failure}"
    else:
        gate_block = "No gate was evaluated at registration."
    packages = "\n".join(f"- `{k}` {v}" for k, v in sorted(manifest.packages.items()))

    return f"""# Model card -- {manifest.model_key} {manifest.version}

**Generated:** {manifest.created_utc}
**Family:** {manifest.model_family}
**Decision threshold:** {manifest.threshold} (on the calibrated probability scale)
**Bundle SHA-256:** `{manifest.bundle_sha256}`

## Intended use
Rank telecom subscribers by predicted probability of churn in the next
billing period, and flag customers at or above the decision threshold for a
retention offer.

## Out of scope
- Not approved for individual pricing or eligibility decisions.
- Not validated on data from any provider other than the one in `data_path`.
- Does not establish that an offer *causes* a customer to stay; that needs a
  randomised holdout.

## Data
- Source: `{manifest.data_path}` (relative to the project root)
- SHA256: `{manifest.data_sha256}`
- Rows: train {manifest.n_train} / val {manifest.n_val} / test {manifest.n_test}

## Performance

### Selection metrics ({manifest.selection_split} split)
These were used to choose this model, its threshold and the quality gate, so
they are optimistic by construction.

| Metric | Value |
|---|---|
{_metric_rows(manifest.metrics)}

95% bootstrap intervals ({manifest.selection_split} split):
| Metric | Interval |
|---|---|
{intervals}

### Held-out (test) metrics
{test_block}

## Calibration
{calibration}

## Promotion gate
{gate_block}

## Reproducibility
- Seed: {manifest.seed}
- Python: {manifest.python}
- Packages:
{packages}

## Limitations
- Trained on a single cross-sectional snapshot. No temporal validation, so
  performance under concept drift is unknown.
- Segment-level error rates differ materially, partly because base rates do.
  Do not deploy on a population unlike the training distribution without a
  slice report.
- `gender` and `SeniorCitizen` are model inputs; no fairness analysis was done.
{manifest.notes}
"""


# --------------------------------------------------------------------------
# Query
# --------------------------------------------------------------------------


def list_versions(versions_dir: Path | None = None) -> list[RegisteredModel]:
    versions_dir = versions_dir or PATHS.versions
    if not versions_dir.exists():
        return []
    out = []
    for d in sorted(versions_dir.glob("v*")):
        if (d / "manifest.json").exists():
            out.append(RegisteredModel(version=d.name, dir=d))
    return out


def latest(versions_dir: Path | None = None) -> RegisteredModel | None:
    versions = list_versions(versions_dir)
    return versions[-1] if versions else None


def promotion_log(versions_dir: Path | None = None) -> list[dict[str, Any]]:
    """Every promotion and rollback, oldest first."""
    log_path = (versions_dir or PATHS.versions) / PROMOTION_LOG
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------
# Promote / load / rollback
# --------------------------------------------------------------------------


def _activate(
    model: RegisteredModel,
    versions_dir: Path,
    *,
    event: str,
    gate_result: GateResult | None = None,
    forced: bool = False,
    reason: str = "",
) -> Path:
    """Copy a *verified* bundle to ``active.joblib`` and record the event."""
    sha = model.verify()
    active = versions_dir / "active.joblib"
    shutil.copy2(model.bundle, active)
    (versions_dir / "active.version").write_text(model.version, encoding="utf-8")
    (versions_dir / "active.sha256").write_text(sha, encoding="utf-8")
    entry = {
        "utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "event": event,
        "version": model.version,
        "bundle_sha256": sha,
        "gate_passed": None if gate_result is None else gate_result.passed,
        "forced": forced,
        "reason": reason,
    }
    with open(versions_dir / PROMOTION_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return active


def promote(
    version: str,
    *,
    versions_dir: Path | None = None,
    gate: QualityGate | None = None,
    force: bool = False,
    reason: str = "",
) -> Path:
    """Make a version the active one (``active.joblib``).

    Serving code points at a stable path, so promotion never requires a code
    change -- only a registry write. Promotion first verifies the bundle's
    checksum. If a ``gate`` is supplied it is evaluated against the metrics
    recorded in the version's manifest; a failing version raises
    :class:`PromotionBlocked` unless ``force=True`` *and* a ``reason`` is given
    (both are written to the promotion log).
    """
    versions_dir = versions_dir or PATHS.versions
    src = RegisteredModel(version=version, dir=versions_dir / version)
    if not src.exists():
        raise FileNotFoundError(f"{version} is not in the registry")

    gate_result = gate.evaluate(src.load_manifest().get("metrics", {})) if gate else None
    forced = False
    if gate_result is not None and not gate_result.passed:
        if not force:
            raise PromotionBlocked(version, gate_result.failures)
        if not reason.strip():
            raise ValueError("forcing a promotion past the quality gate requires a written reason")
        forced = True
    return _activate(src, versions_dir, event="promote", gate_result=gate_result, forced=forced, reason=reason)


def load_active(paths: Paths | None = None, *, versions_dir: Path | None = None, verify: bool = True):
    """Load whatever is currently promoted. Raises if nothing is promoted.

    With ``verify=True`` (the default) the bytes of ``active.joblib`` are
    checked against the checksum recorded at promotion, so a file that was
    swapped, truncated or corrupted after promotion is refused rather than
    served.
    """
    root = versions_dir or (paths or PATHS).versions
    active = root / "active.joblib"
    if not active.exists():
        raise FileNotFoundError(
            f"No promoted model at {active}. Run the training job first, then "
            "promote a version."
        )
    recorded = root / "active.sha256"
    if verify and recorded.exists() and sha256_file(active) != recorded.read_text(encoding="utf-8").strip():
        raise IntegrityError(f"{active} does not match the checksum recorded at promotion")
    return joblib.load(active)


def rollback(
    current: str | None = None, *, versions_dir: Path | None = None
) -> str | None:
    """Re-activate the version that was serving *before* ``current``.

    "Before" means the previously **promoted** version, read from the promotion
    log -- not merely the previous number in the registry. A version that was
    registered but blocked by the gate was never approved, so rolling back onto
    it would swap one problem for another. (Without a log, e.g. a registry
    built by hand, it falls back to the preceding registered version.)

    Rollback must not retrain -- a retrained model is a *different* model, and
    the point of rollback is to get the exact artefact that was working
    yesterday. The target's checksum is verified before it is activated, and
    the gate is deliberately not re-applied: a rollback target was already
    approved, and a tightened gate must not trap you on a bad version.
    """
    versions_dir = versions_dir or PATHS.versions
    versions = list_versions(versions_dir)
    names = [v.version for v in versions]
    if not names:
        return None

    if current is None:
        current_file = versions_dir / "active.version"
        current = current_file.read_text(encoding="utf-8").strip() if current_file.exists() else None

    target = None
    history = [e["version"] for e in promotion_log(versions_dir) if e["event"] in ("promote", "rollback")]
    if current in history:
        last_seen = max(i for i, v in enumerate(history) if v == current)
        earlier = [v for v in history[:last_seen] if v != current and v in names]
        target = earlier[-1] if earlier else None
    if target is None:
        if current not in names:
            target = names[-2] if len(names) > 1 else names[-1]
        else:
            i = names.index(current)
            target = names[i - 1] if i > 0 else names[i]
    _activate(
        RegisteredModel(version=target, dir=versions_dir / target),
        versions_dir,
        event="rollback",
        reason=f"rolled back from {current}",
    )
    return target
