"""Core types and the check registry.

Every verification check in this harness is a pure function of the artifacts a
run left on disk.  Checks never touch the model, never touch the network, and
never trust anything the run *said* about itself that they can recompute.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional

Status = Literal["pass", "fail", "warn", "skip", "error"]

TIERS = {
    "T0": "environment — is the compute real and as configured",
    "T1": "plumbing — did the intervention actually happen",
    "T2": "statistics — is the effect distinguishable from nothing",
    "T3": "integrity — do the artifacts agree with each other",
    "T4": "claims — do the reported numbers follow from the records",
}


@dataclass
class CheckResult:
    """One falsifiable verdict.  `evidence` holds the numbers it rests on."""

    id: str
    tier: str
    status: Status
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)
    remedy: str = ""

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"unknown tier {self.tier!r}")
        if self.status not in ("pass", "fail", "warn", "skip", "error"):
            raise ValueError(f"unknown status {self.status!r}")


@dataclass
class VerifyReport:
    run_id: str
    results: list[CheckResult] = field(default_factory=list)

    @property
    def failed(self) -> list[CheckResult]:
        return [r for r in self.results if r.status in ("fail", "error")]

    @property
    def ok(self) -> bool:
        return not self.failed

    def by_id(self, check_id: str) -> Optional[CheckResult]:
        for r in self.results:
            if r.id == check_id:
                return r
        return None

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.results:
            out[r.status] = out.get(r.status, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "counts": self.counts(),
                "ok": self.ok, "results": [asdict(r) for r in self.results]}


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

CheckFn = Callable[["RunArtifacts"], "CheckResult | list[CheckResult]"]


@dataclass
class Check:
    id: str
    tier: str
    fn: CheckFn
    doc: str = ""
    needs: tuple[str, ...] = ()   # artifact keys this check requires; skip if absent


CHECKS: dict[str, Check] = {}


def check(check_id: str, tier: str, needs: Iterable[str] = ()) -> Callable[[CheckFn], CheckFn]:
    """Register a check.  Ids are stable and referenced by tests — never rename."""

    def deco(fn: CheckFn) -> CheckFn:
        if check_id in CHECKS:
            raise ValueError(f"duplicate check id {check_id!r}")
        CHECKS[check_id] = Check(check_id, tier, fn, (fn.__doc__ or "").strip(),
                                 tuple(needs))
        return fn

    return deco


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


# --------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------

@dataclass
class RunArtifacts:
    """Lazy reader over one run directory.  Missing pieces are None, not errors."""

    run_dir: Path
    manifest: dict[str, Any] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    claims: dict[str, Any] = field(default_factory=dict)
    _npz: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, run_dir: str | Path) -> "RunArtifacts":
        run_dir = Path(run_dir)
        manifest: dict[str, Any] = {}
        mp = run_dir / "manifest.json"
        if mp.exists():
            manifest = json.loads(mp.read_text())
        records: list[dict[str, Any]] = []
        rp = run_dir / "records.jsonl"
        if rp.exists():
            for line in rp.read_text().splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        claims: dict[str, Any] = {}
        cp = run_dir / "claims.json"
        if cp.exists():
            claims = json.loads(cp.read_text())
        return cls(run_dir=run_dir, manifest=manifest, records=records, claims=claims)

    def npz(self, name: str):
        """Load `<run_dir>/<name>.npz` once.  Returns None if absent."""
        if name not in self._npz:
            import numpy as np
            p = self.run_dir / f"{name}.npz"
            self._npz[name] = np.load(p) if p.exists() else None
        return self._npz[name]

    def cfg(self, *path: str, default: Any = None) -> Any:
        node: Any = self.manifest
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def sibling(self, role: str) -> Optional["RunArtifacts"]:
        """A companion run this one declares, e.g. 'baseline', 'shuffled',
        'random_direction', 'alpha_zero', 'sign_flip', 'replay'.

        manifest['companions'] maps role -> path (relative to run_dir's parent).
        """
        rel = self.cfg("companions", role)
        if not rel:
            return None
        p = (self.run_dir.parent / rel)
        return RunArtifacts.load(p) if p.exists() else None
