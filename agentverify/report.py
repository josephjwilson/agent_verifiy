"""Execute the check registry over one run directory and render the verdict.

Importing this module is what makes the registry complete: every module under
`checks/` is imported here and the resulting id set is asserted against the
contract's list.  A check that quietly stopped registering would otherwise just
shrink coverage while making the report look cleaner than it is.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import math
import pkgutil
import traceback
import warnings
from typing import Any, Iterable, Optional

from . import checks as _checks_pkg
from .types import CHECKS, TIERS, CheckResult, RunArtifacts, VerifyReport

# The four the contract names; any other module dropped into checks/ is imported
# too, because registration only happens on import and a module never imported
# is a coverage hole nothing would report.
_REQUIRED_CHECK_MODULES = ("environment", "plumbing", "statistical", "integrity")


def _import_check_modules() -> tuple[list[str], list[str]]:
    """Import every module under `checks/` for its `@check` side effects."""
    found = {m.name for m in pkgutil.iter_modules(_checks_pkg.__path__)}
    imported: list[str] = []
    failures: list[str] = []
    for name in sorted(found | set(_REQUIRED_CHECK_MODULES)):
        try:
            importlib.import_module(f"{__package__}.checks.{name}")
            imported.append(name)
        except Exception as exc:  # missing, unimportable, or syntactically broken
            failures.append(f"checks/{name}.py: {type(exc).__name__}: {exc}")
    return imported, failures


CHECK_MODULES, _MODULE_FAILURES = _import_check_modules()

# --------------------------------------------------------------------------
# the contract's check list, transcribed from CONTRACT.md
# --------------------------------------------------------------------------

EXPECTED_CHECKS: dict[str, str] = {
    # T0 — checks/environment.py
    "t0.cuda_available": "T0",
    "t0.torch_build_matches_driver": "T0",
    "t0.params_on_device": "T0",
    "t0.dtype_as_configured": "T0",
    "t0.no_cpu_fallback": "T0",
    "t0.offline_mode": "T0",
    "t0.versions_recorded": "T0",
    # T1 — checks/plumbing.py
    "t1.hook_fired": "T1",
    "t1.vector_finite_nonzero": "T1",
    "t1.vector_dim_matches_model": "T1",
    "t1.activation_delta_matches_alpha": "T1",
    "t1.no_effect_before_layer": "T1",
    "t1.effect_after_layer": "T1",
    "t1.outputs_differ_from_baseline": "T1",
    "t1.completions_finite": "T1",
    "t1.layer_index_convention": "T1",
    # T2 — checks/statistical.py
    "t2.sample_size_adequate": "T2",
    "t2.metric_not_degenerate": "T2",
    "t2.effect_ci_excludes_zero": "T2",
    "t2.shuffled_label_null": "T2",
    "t2.selection_declared": "T2",
    # T3/T4 — checks/integrity.py
    "t3.artifact_hashes_match": "T3",
    "t3.record_count_matches": "T3",
    "t3.record_hashes_match": "T3",
    "t3.schema_complete": "T3",
    "t3.leakage": "T3",
    "t3.git_recorded": "T3",
    "t4.claims_recomputable": "T4",
    "t4.claim_n_matches_records": "T4",
    "t4.no_unsupported_claims": "T4",
    "t4.claim_direction_matches": "T4",
}

CHECK_IDS: tuple[str, ...] = tuple(EXPECTED_CHECKS)


def _assert_registry_matches_contract() -> None:
    """Coverage is only meaningful if it is the coverage the contract promised.

    Missing coverage is fatal: a contracted check that never registered would
    otherwise just make the report look cleaner than the run deserves.  An
    *extra* check costs no coverage, so it warns instead of bricking the tool.
    """
    registered = set(CHECKS)
    expected = set(EXPECTED_CHECKS)
    problems: list[str] = list(_MODULE_FAILURES)
    missing = sorted(expected - registered)
    extra = sorted(registered - expected)
    if missing:
        problems.append(f"never registered: {missing}")
    mistiered = sorted(
        f"{cid} is {CHECKS[cid].tier}, contract says {EXPECTED_CHECKS[cid]}"
        for cid in expected & registered
        if CHECKS[cid].tier != EXPECTED_CHECKS[cid]
    )
    if mistiered:
        problems.append(f"wrong tier: {mistiered}")
    if extra:
        warnings.warn(
            f"agentverify: checks registered that CONTRACT.md does not list: {extra}"
            " — they will run, but no planted fault covers them",
            RuntimeWarning, stacklevel=2,
        )
    if problems:
        raise ImportError(
            "agentverify check registry disagrees with CONTRACT.md — "
            + "; ".join(problems)
        )


_assert_registry_matches_contract()


# --------------------------------------------------------------------------
# `needs` resolution
# --------------------------------------------------------------------------

# A `needs` entry names an artifact or a manifest path.  Resolution is
# deliberately generous: a check that runs when its input is thin still has to
# skip on its own, but a check wrongly skipped here would silently stop being
# able to fail.
_ARTIFACT_NEEDS: dict[str, Any] = {
    "manifest": lambda a: bool(a.manifest),
    "manifest.json": lambda a: bool(a.manifest),
    "records": lambda a: bool(a.records),
    "records.jsonl": lambda a: bool(a.records),
    "claims": lambda a: bool(a.claims),
    "claims.json": lambda a: bool(a.claims),
    "acts": lambda a: a.npz("acts") is not None,
    "acts.npz": lambda a: a.npz("acts") is not None,
    "vector": lambda a: a.npz("vector") is not None or a.cfg("vector") is not None,
    "vector.npz": lambda a: a.npz("vector") is not None,
}


def need_present(art: RunArtifacts, need: str) -> bool:
    """True if `need` — an artifact name, `companion:<role>`, or a dotted
    manifest path — is available on disk.

    A dotted path may be written from the manifest root (`hook.fires_observed`)
    or with the manifest named (`manifest.hook.fires_observed`); both resolve,
    because a spelling that resolved to nothing would skip its check forever.
    """
    need = need.strip()
    fn = _ARTIFACT_NEEDS.get(need)
    try:
        if fn is not None:
            return bool(fn(art))
        if need.startswith("companion:"):
            return art.sibling(need.split(":", 1)[1]) is not None
        path = [p for p in need.split(".") if p]
        if art.cfg(*path) is not None:
            return True
        if len(path) > 1 and path[0] == "manifest":
            rest = path[2:] if path[1] == "json" else path[1:]
            return bool(rest) and art.cfg(*rest) is not None
        return False
    except Exception:
        # A file present but unreadable is a finding, not a reason to skip:
        # let the check run and report it.
        return True


def missing_needs(art: RunArtifacts, needs: Iterable[str]) -> list[str]:
    return [n for n in needs if not need_present(art, n)]


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def _normalise_tiers(only_tiers: Optional[Iterable[str] | str]) -> Optional[set[str]]:
    if only_tiers is None:
        return None
    if isinstance(only_tiers, str):
        only_tiers = [only_tiers]
    tiers = {str(t).strip().upper() for t in only_tiers}
    unknown = sorted(tiers - set(TIERS))
    if unknown:
        raise ValueError(f"unknown tier(s) {unknown}; known tiers: {sorted(TIERS)}")
    return tiers


def _error_result(chk, exc: BaseException) -> CheckResult:
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return CheckResult(
        id=chk.id,
        tier=chk.tier,
        status="error",
        summary=f"check raised {type(exc).__name__}: {exc}",
        evidence={"exception": type(exc).__name__, "message": str(exc),
                  "traceback": "".join(tb[-4:]).strip()},
        remedy="Fix the check itself — an erroring check verifies nothing.",
    )


def run_checks(art: RunArtifacts, only_tiers: Optional[Iterable[str] | str] = None) -> VerifyReport:
    """Run every registered check against `art`, in stable id order.

    A crashing check becomes an `error` result rather than taking the suite
    down, and `error` counts against the run exactly like `fail` does.
    """
    tiers = _normalise_tiers(only_tiers)
    run_id = art.cfg("run_id") or art.run_dir.name
    rep = VerifyReport(run_id=str(run_id))

    for check_id in sorted(CHECKS):
        chk = CHECKS[check_id]
        if tiers is not None and chk.tier not in tiers:
            continue
        try:
            absent = missing_needs(art, chk.needs)
            if absent:
                rep.results.append(CheckResult(
                    id=chk.id, tier=chk.tier, status="skip",
                    summary=f"input absent: {', '.join(absent)}",
                    evidence={"missing_needs": absent, "needs": list(chk.needs)},
                    remedy=f"Produce {', '.join(absent)} in the run dir to make this check decidable.",
                ))
                continue
            out = chk.fn(art)
        except Exception as exc:  # a broken check must be loud, never silent
            rep.results.append(_error_result(chk, exc))
            continue
        rep.results.extend(_coerce(chk, out))

    return rep


def _coerce(chk, out: Any) -> list[CheckResult]:
    """Accept a CheckResult or a list of them; anything else is a check bug."""
    if isinstance(out, CheckResult):
        return [out]
    if isinstance(out, (list, tuple)):
        results = list(out)
        bad = [r for r in results if not isinstance(r, CheckResult)]
        if not results:
            return [CheckResult(
                id=chk.id, tier=chk.tier, status="error",
                summary="check returned no results",
                evidence={"returned": repr(out)[:200]},
                remedy="Return a CheckResult (or a non-empty list of them).",
            )]
        if bad:
            return [CheckResult(
                id=chk.id, tier=chk.tier, status="error",
                summary=f"check returned {len(bad)} non-CheckResult item(s)",
                evidence={"types": sorted({type(b).__name__ for b in bad})},
                remedy="Return CheckResult objects only.",
            )]
        return results
    return [CheckResult(
        id=chk.id, tier=chk.tier, status="error",
        summary=f"check returned {type(out).__name__}, expected CheckResult",
        evidence={"returned": repr(out)[:200]},
        remedy="Return a CheckResult (or a list of them).",
    )]


def apply_strict(rep: VerifyReport) -> VerifyReport:
    """Return a copy of `rep` with every `warn` promoted to `fail`."""
    out = VerifyReport(run_id=rep.run_id)
    for r in rep.results:
        if r.status == "warn":
            r = dataclasses.replace(
                r, status="fail",
                evidence={**r.evidence, "promoted_by_strict": True},
            )
        out.results.append(r)
    return out


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

_STATUS_ORDER = ("fail", "error", "warn", "pass", "skip")
_MARK = {"pass": "PASS", "fail": "FAIL", "warn": "WARN", "skip": "SKIP", "error": "ERR "}


def verdict(rep: VerifyReport) -> str:
    """FAIL beats everything; a run where nothing could be decided is not a pass."""
    if rep.failed:
        return "FAIL"
    if not any(r.status == "pass" for r in rep.results):
        return "INCONCLUSIVE"
    return "PASS"


def _fmt_value(v: Any, limit: int = 200) -> str:
    ndim = getattr(v, "ndim", None)  # numpy values read better as Python ones
    if ndim is not None:
        try:
            v = v.item() if ndim == 0 else v.tolist()
        except Exception:
            pass
    if isinstance(v, bool) or v is None:
        return json.dumps(v)
    if isinstance(v, float):
        return f"{v:.1f}" if v.is_integer() and abs(v) < 1e15 else f"{v:.6g}"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        s = " ".join(v.split())
        return s if len(s) <= limit else s[:limit] + f"... (+{len(s) - limit} chars)"
    if isinstance(v, (list, tuple)):
        head = [_fmt_value(x, 60) for x in list(v)[:8]]
        tail = f", ... (+{len(v) - 8} more)" if len(v) > 8 else ""
        return "[" + ", ".join(head) + tail + "]"
    try:
        s = json.dumps(v, default=_json_default)
    except Exception:
        s = repr(v)
    return s if len(s) <= limit else s[:limit] + "..."


def _fmt_detail(r: CheckResult) -> list[str]:
    lines: list[str] = []
    if r.evidence:
        lines.append("")
        lines.append("Evidence:")
        for k, v in r.evidence.items():
            if k == "traceback" and isinstance(v, str):
                continue
            lines.append(f"- `{k}`: {_fmt_value(v)}")
        tb = r.evidence.get("traceback")
        if isinstance(tb, str) and tb:  # a harness bug is only useful in full
            lines += ["", "```", tb, "```"]
    if r.remedy:
        lines.append("")
        lines.append(f"Remedy: {r.remedy}")
    return lines


def to_markdown(rep: VerifyReport) -> str:
    """A report a human skims in ten seconds: verdict, counts, failures, then
    everything else one line per check."""
    counts = rep.counts()
    tally = ", ".join(f"{counts[s]} {s}" for s in _STATUS_ORDER if counts.get(s))
    lines = [
        f"# agentverify — {rep.run_id}",
        "",
        f"**VERDICT: {verdict(rep)}** — {tally or 'no checks ran'} "
        f"({len(rep.results)} checks)",
    ]
    if verdict(rep) == "INCONCLUSIVE":
        lines.append("")
        lines.append("> No check reached a verdict — this run proves nothing. "
                     "Check that the run directory holds a complete set of artifacts.")

    # real failures before harness errors; the run's problems outrank ours
    bad = sorted((r for r in rep.results if r.status in ("fail", "error")),
                 key=lambda r: (r.status != "fail", r.id))
    if bad:
        lines += ["", f"## Failures ({len(bad)})"]
        for r in bad:
            lines += ["", f"### {_MARK[r.status].strip()} `{r.id}` — {r.tier} {TIERS[r.tier]}",
                      "", r.summary]
            lines += _fmt_detail(r)

    warned = [r for r in rep.results if r.status == "warn"]
    if warned:
        lines += ["", f"## Warnings ({len(warned)})"]
        for r in warned:
            lines.append(f"- `{r.id}` — {r.summary}"
                         + (f"  _Remedy: {r.remedy}_" if r.remedy else ""))

    lines += ["", "## All checks by tier"]
    for tier, desc in TIERS.items():
        rows = [r for r in rep.results if r.tier == tier]
        if not rows:
            continue
        lines += ["", f"### {tier} — {desc}", ""]
        for r in rows:
            lines.append(f"- `[{_MARK[r.status]}]` `{r.id}` — {r.summary}")

    return "\n".join(lines) + "\n"


def _json_default(o: Any) -> Any:
    """numpy scalars and arrays routinely land in `evidence`."""
    for attr in ("item", "tolist"):
        fn = getattr(o, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=str)
    return str(o)


def _jsonable(o: Any) -> Any:
    """Normalise evidence into strict JSON.

    Two things reach `evidence` that `json.dumps` gets wrong: numpy scalars and
    arrays (a TypeError), and non-finite floats — which it happily writes as
    bare `NaN`/`Infinity`, tokens no JSON parser outside Python accepts.  A
    finiteness check is *exactly* the check whose evidence is a NaN, so the
    report that matters most is the one that would not parse.
    """
    if o is None or isinstance(o, (bool, str, int)):
        return o
    if isinstance(o, float):  # also covers numpy float64
        return o if math.isfinite(o) else ("NaN" if math.isnan(o) else
                                           ("Infinity" if o > 0 else "-Infinity"))
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (set, frozenset)):
        return [_jsonable(v) for v in sorted(o, key=str)]
    for attr in ("item", "tolist"):  # numpy scalars, then numpy arrays
        fn = getattr(o, attr, None)
        if callable(fn):
            try:
                return _jsonable(fn())
            except Exception:
                pass
    return str(o)


def to_json(rep: VerifyReport) -> str:
    payload = _jsonable(rep.to_dict())
    payload["verdict"] = verdict(rep)  # `ok` alone cannot say INCONCLUSIVE
    return json.dumps(payload, indent=2, allow_nan=False)
