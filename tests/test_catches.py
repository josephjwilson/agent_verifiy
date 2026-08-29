"""The harness has to be able to fail.

Every fault in the contract's table is planted into a known-good run and the
check that owns it must come back `fail` — not `skip`, not `warn`.  The clean
fixture is the control: it must pass everything, and it must actually *exercise*
every check a fault targets, otherwise the fault tests below prove nothing.

Runs on CPU in seconds: no model, no GPU, no network.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The harness is documented to run offline (CLAUDE.md); a shell that forgot to
# export it must not be mistaken for a run that hit the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/workspace/hf-cache")

from agentverify import report as report_mod  # noqa: E402  (imports every checks/ module)
from agentverify.types import CHECKS, RunArtifacts, sha256_file, sha256_text  # noqa: E402
from tests.faults import FAULT_CHECKS, plant, synthetic_run  # noqa: E402

CONTRACT = (ROOT / "CONTRACT.md").read_text()


# --------------------------------------------------------------------------
# fixtures — one clean family, one planted family per fault, built once
# --------------------------------------------------------------------------

class _Bench:
    """Builds the clean run once and memoises one report per fault, so the whole
    module costs one fixture build plus one verify pass per fault."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.clean = synthetic_run(root / "clean")
        self._runs: dict[str, Path] = {}
        self._reports: dict[str, object] = {}

    def run_dir(self, fault: str | None) -> Path:
        if fault is None:
            return self.clean
        if fault not in self._runs:
            self._runs[fault] = plant(fault, self.clean,
                                      self.root / f"fault-{fault}" / self.clean.name)
        return self._runs[fault]

    def report(self, fault: str | None = None):
        key = fault or "__clean__"
        if key not in self._reports:
            self._reports[key] = report_mod.run_checks(
                RunArtifacts.load(self.run_dir(fault)))
        return self._reports[key]


@pytest.fixture(scope="session")
def bench(tmp_path_factory) -> _Bench:
    return _Bench(tmp_path_factory.mktemp("agentverify-catches"))


def _describe(results, statuses=("fail", "error")) -> str:
    rows = [f"    {r.id} [{r.status}] {r.summary}" for r in results if r.status in statuses]
    return "\n".join(rows) if rows else "    (none)"


# --------------------------------------------------------------------------
# the control: the fixture itself, then the clean run through the registry
# --------------------------------------------------------------------------

def test_fixture_is_self_consistent(bench: _Bench) -> None:
    """Re-derive the fixture's invariants without going through the harness.
    If this fails, the fixture is broken and every other test here is noise."""
    art = RunArtifacts.load(bench.clean)
    man = art.manifest

    for name, digest in man["hashes"].items():
        assert digest == sha256_file(bench.clean / name), f"stale hash for {name}"
    for rec in art.records:
        assert rec["completion_sha256"] == sha256_text(rec["completion"])
    assert len(art.records) == man["counts"]["n_records"] == man["counts"]["n_eval"]

    acts, vec = art.npz("acts"), art.npz("vector")
    layer, alpha = man["config"]["layer"], man["config"]["alpha"]
    v = vec["v"]
    layers = [int(x) for x in acts["layers"]]
    base, steered = acts["base"], acts["steered"]
    assert base.shape == steered.shape == (len(layers), len(art.records), v.shape[0])
    for k, lay in enumerate(layers):
        delta = np.linalg.norm(steered[k] - base[k], axis=-1)
        if lay < layer:
            assert np.array_equal(steered[k], base[k]), f"layer {lay} moved below L"
        elif lay == layer:
            assert np.allclose(delta, abs(alpha) * np.linalg.norm(v), rtol=1e-4)
        else:
            assert float(delta.mean()) > 1e-3, f"layer {lay} shows no propagation"

    baseline = art.sibling("baseline")
    assert baseline is not None and len(baseline.records) == len(art.records)
    seen = {r["prompt_id"]: r["completion_sha256"] for r in baseline.records}
    differing = sum(1 for r in art.records if seen[r["prompt_id"]] != r["completion_sha256"])
    assert differing >= 0.5 * len(art.records), "steering barely moved the outputs"

    rate = lambda recs: sum(int(r["score"]["sycophantic"]) for r in recs) / len(recs)
    claim = art.claims["claims"][0]
    assert claim["baseline_value"] == rate(baseline.records)
    assert claim["treatment_value"] == rate(art.records)
    assert claim["n"] == len(art.records)


def test_clean_run_passes(bench: _Bench) -> None:
    rep = bench.report()
    bad = [r for r in rep.results if r.status in ("fail", "error")]
    assert not bad, ("clean synthetic run must pass every check:\n"
                     + _describe(rep.results))
    assert rep.ok


def test_clean_run_exercises_every_targeted_check(bench: _Bench) -> None:
    """A check that merely skips on the clean run cannot prove anything about
    the fault that targets it."""
    rep = bench.report()
    not_run = []
    for check_id in sorted(set(FAULT_CHECKS.values())):
        res = rep.by_id(check_id)
        if res is None or res.status != "pass":
            not_run.append(f"    {check_id} -> {res.status if res else 'MISSING'}")
    assert not not_run, ("these checks never actually ran on the clean fixture:\n"
                         + "\n".join(not_run))


# --------------------------------------------------------------------------
# one planted fault per row of the contract's fault table
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fault,check_id", sorted(FAULT_CHECKS.items()))
def test_planted_fault_is_caught(bench: _Bench, fault: str, check_id: str) -> None:
    rep = bench.report(fault)
    res = rep.by_id(check_id)
    assert res is not None, f"{check_id} is not registered at all"
    assert res.status == "fail", (
        f"fault {fault!r} must make {check_id} fail, got {res.status!r}: "
        f"{res.summary}\n  evidence: {res.evidence}\n"
        f"  other failures in this run:\n{_describe(rep.results)}")


def test_no_collateral_errors(bench: _Bench) -> None:
    """A check that crashes on bad input is itself a bug: broken artifacts are
    exactly what these checks exist to read."""
    crashed = []
    for fault in sorted(FAULT_CHECKS):
        rep = bench.report(fault)
        for res in rep.results:
            if res.status == "error":
                crashed.append(f"    {fault}: {res.id} -> {res.summary}")
    assert not crashed, "checks errored instead of failing cleanly:\n" + "\n".join(crashed)


# --------------------------------------------------------------------------
# the registry and the fault table must match the contract, not a second copy
# --------------------------------------------------------------------------

def _contract_check_ids() -> list[str]:
    start = CONTRACT.index("Complete check id list")
    end = CONTRACT.index("### `report.py`", start)
    return re.findall(r"^- `(t\d\.[A-Za-z0-9_]+)`", CONTRACT[start:end], re.M)


def _contract_fault_table() -> dict[str, str]:
    rows = re.findall(r"^\|\s*`([A-Za-z0-9_]+)`\s*\|\s*`(t\d\.[A-Za-z0-9_]+)`\s*\|",
                      CONTRACT, re.M)
    return dict(rows)


def test_all_contract_ids_registered() -> None:
    contract_ids = _contract_check_ids()
    assert len(contract_ids) > 30, "CONTRACT.md check list failed to parse"
    assert len(contract_ids) == len(set(contract_ids)), "duplicate id in CONTRACT.md"
    registered = set(CHECKS)
    assert registered == set(contract_ids), (
        f"missing from registry: {sorted(set(contract_ids) - registered)}; "
        f"not in CONTRACT.md: {sorted(registered - set(contract_ids))}")


def test_fault_table_matches_contract() -> None:
    table = _contract_fault_table()
    assert table, "CONTRACT.md fault table failed to parse"
    assert FAULT_CHECKS == table, (
        f"faults.py disagrees with CONTRACT.md: "
        f"only in faults.py {sorted(set(FAULT_CHECKS) - set(table))}, "
        f"only in CONTRACT.md {sorted(set(table) - set(FAULT_CHECKS))}, "
        f"different target "
        f"{sorted(k for k in set(table) & set(FAULT_CHECKS) if table[k] != FAULT_CHECKS[k])}")


def test_check_ids_agree_with_their_tier() -> None:
    """`t2.foo` registered under T1 would send its result to the wrong tier gate."""
    for check_id, chk in CHECKS.items():
        assert chk.tier in ("T0", "T1", "T2", "T3", "T4"), check_id
        assert check_id.startswith(chk.tier.lower() + "."), (check_id, chk.tier)
