"""T1 — did the intervention actually happen?

A steering run can look perfect and have done nothing: a hook registered on the
wrong module, a vector of zeros, an alpha that never reached the forward pass.
Every check here re-derives the perturbation from the arrays and the completions
on disk; ``manifest['hook']``, ``manifest['vector']['norm']`` and the recorded
completion hashes are treated as claims to be audited, not as inputs to trust.
"""
from __future__ import annotations

import math
import re
from typing import Any, Optional

from ..types import CheckResult, RunArtifacts, check, sha256_text

# ||steered - base|| at the hooked layer should equal |alpha| * ||v||.  bf16
# carries ~8 mantissa bits (rel. eps ~4e-3) and the two runs round independently,
# so a few percent of slack is physics; 15% still fails any hook that adds the
# vector at half, double or zero scale.
ALPHA_REL_TOL = 0.15
# With alpha = 0 the residual stream must be untouched, up to storage rounding
# relative to the activation's own magnitude.
ZERO_DELTA_REL = 1e-3
# Below this fraction of changed completions the intervention did nothing.
MIN_DIFF_FRACTION = 0.05
WARN_DIFF_FRACTION = 0.20
MIN_VECTOR_NORM = 1e-6
# Tolerance on the manifest's self-reported ||v|| vs. the recomputed one.
NORM_SELF_REPORT_TOL = 1e-3

_DEGENERATE_TOKENS = {"nan", "-nan", "+nan", "inf", "-inf", "+inf",
                      "infinity", "-infinity", "none", "null", "<nan>"}


# --------------------------------------------------------------------------
# coercion helpers — a partial manifest must produce a verdict, not a traceback
# --------------------------------------------------------------------------

def _num(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return None
    return None


def _int(v: Any) -> Optional[int]:
    n = _num(v)
    if n is None or not math.isfinite(n) or n != int(n):
        return None
    return int(n)


def _as_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "y", "on"):
            return True
        if s in ("false", "0", "no", "n", "off"):
            return False
    return None


def _alpha(art: RunArtifacts) -> Optional[float]:
    return _num(art.cfg("config", "alpha"))


def _layer(art: RunArtifacts) -> Optional[int]:
    return _int(art.cfg("config", "layer"))


def _enabled(art: RunArtifacts) -> Optional[bool]:
    return _as_bool(art.cfg("config", "steering_enabled"))


def _intervenes(art: RunArtifacts) -> bool:
    """True when this run is supposed to have perturbed anything."""
    en = _enabled(art)
    a = _alpha(art)
    if en is False:
        return False
    if a is not None and a == 0.0:
        return False
    return bool(en) or (a is not None and a != 0.0)


# --------------------------------------------------------------------------
# artifact readers
# --------------------------------------------------------------------------

def _vector_info(art: RunArtifacts) -> dict[str, Any]:
    """Recompute ||v|| and d from vector.npz; the manifest is only a fallback."""
    info: dict[str, Any] = {
        "key": art.cfg("vector", "key", default="v") or "v",
        "declared_norm": _num(art.cfg("vector", "norm")),
        "declared_dim": _int(art.cfg("vector", "dim")),
        "declared_finite": art.cfg("vector", "finite"),
        "norm": None, "dim": None, "all_finite": None, "n_nonfinite": None,
        "source": None, "problem": None,
    }
    try:
        z = art.npz("vector")
    except Exception as exc:  # a corrupt npz is present-and-wrong, not absent
        info["problem"] = f"vector.npz is unreadable: {exc}"
        return info
    if z is None:
        if info["declared_norm"] is not None:
            info.update(norm=info["declared_norm"], dim=info["declared_dim"],
                        source="manifest")
        return info
    try:
        files = list(z.files)
        if info["key"] not in files:
            info["problem"] = (f"vector.npz has no key {info['key']!r} "
                               f"(keys: {sorted(files)})")
            return info
        import numpy as np
        v = np.asarray(z[info["key"]], dtype=np.float64).reshape(-1)
    except Exception as exc:
        info["problem"] = f"vector.npz is unreadable: {exc}"
        return info
    finite = np.isfinite(v)
    info.update(
        norm=float(np.linalg.norm(v[finite])) if finite.any() else 0.0,
        dim=int(v.size),
        all_finite=bool(finite.all()),
        n_nonfinite=int((~finite).sum()),
        max_abs=float(np.abs(v[finite]).max()) if finite.any() else 0.0,
        source="vector.npz",
    )
    return info


def _load_acts(art: RunArtifacts) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """(data, problem).  data is None with problem None only when acts.npz is absent."""
    try:
        z = art.npz("acts")
    except Exception as exc:
        return None, f"acts.npz is unreadable: {exc}"
    if z is None:
        return None, None
    try:
        files = set(z.files)
    except Exception as exc:
        return None, f"acts.npz is unreadable: {exc}"
    missing = [k for k in ("layers", "base", "steered") if k not in files]
    if missing:
        return None, (f"acts.npz is missing key(s) {missing}; "
                      f"present keys are {sorted(files)}")
    try:
        import numpy as np
        layers = np.asarray(z["layers"]).reshape(-1)
        base = np.asarray(z["base"], dtype=np.float64)
        steered = np.asarray(z["steered"], dtype=np.float64)
    except Exception as exc:
        return None, f"acts.npz arrays are unreadable: {exc}"
    if base.ndim != 3 or steered.ndim != 3:
        return None, (f"acts base/steered must be [n_layers, n_eval, d_model]; got "
                      f"{tuple(base.shape)} and {tuple(steered.shape)}")
    if base.shape != steered.shape:
        return None, (f"acts base {tuple(base.shape)} and steered "
                      f"{tuple(steered.shape)} have different shapes")
    if int(layers.shape[0]) != int(base.shape[0]):
        return None, (f"acts.layers has {int(layers.shape[0])} entries but base has "
                      f"{int(base.shape[0])} layer slices")
    try:
        idx = [int(x) for x in layers.tolist()]
    except Exception as exc:
        return None, f"acts.layers is not an integer array: {exc}"
    return {"layers": idx, "base": base, "steered": steered}, None


def _acts_or_result(art: RunArtifacts, check_id: str) -> tuple[Optional[dict[str, Any]],
                                                              Optional[CheckResult]]:
    data, problem = _load_acts(art)
    if problem:
        # A run that never captured activations writes no file at all; a file that
        # is there but unusable is a defect, unless nothing was steered anyway.
        status = "skip" if _enabled(art) is False else "fail"
        return None, CheckResult(check_id, "T1", status, problem,
                                 {"acts_path": str(art.run_dir / "acts.npz")},
                                 "acts.npz must hold layers[n], base and steered "
                                 "[n_layers, n_eval, d_model] float32")
    if data is None:
        return None, CheckResult(check_id, "T1", "skip",
                                 "acts.npz absent — this run captured no activations",
                                 {"acts_path": str(art.run_dir / "acts.npz")})
    return data, None


def _delta_stats(data: dict[str, Any]) -> dict[str, Any]:
    """Per-captured-layer summary of steered - base.  Exact zeros stay exact."""
    import numpy as np
    diff = data["steered"] - data["base"]
    absd = np.abs(diff)
    norms = np.linalg.norm(diff, axis=2)              # [n_layers, n_eval]
    base_norms = np.linalg.norm(data["base"], axis=2)
    return {
        "layers": list(data["layers"]),
        "max_abs": [float(x) for x in absd.max(axis=(1, 2))],
        "n_nonzero": [int(x) for x in (absd > 0).sum(axis=(1, 2))],
        "median_norm": [float(x) for x in np.median(norms, axis=1)],
        "mean_norm": [float(x) for x in norms.mean(axis=1)],
        "min_norm": [float(x) for x in norms.min(axis=1)],
        "max_norm": [float(x) for x in norms.max(axis=1)],
        "median_base_norm": [float(x) for x in np.median(base_norms, axis=1)],
        "n_eval": int(diff.shape[1]),
        "d_model": int(diff.shape[2]),
        "n_nonfinite": int((~np.isfinite(diff)).sum()),
    }


def _layer_pos(layers: list[int], target: int) -> Optional[int]:
    for i, l in enumerate(layers):
        if l == target:
            return i
    return None


def _digests(art: RunArtifacts, prefer: Optional[str] = None) -> tuple[dict[str, str],
                                                                      dict[str, Any]]:
    """prompt_id -> sha256 of the completion, recomputed from the text where present."""
    by_cond: dict[Any, list[dict[str, Any]]] = {}
    for r in art.records:
        if isinstance(r, dict):
            by_cond.setdefault(r.get("condition"), []).append(r)
    if prefer is not None and by_cond.get(prefer):
        rows, cond = by_cond[prefer], prefer
    elif len(by_cond) == 1:
        cond = next(iter(by_cond))
        rows = by_cond[cond]
    else:
        rows = [r for rs in by_cond.values() for r in rs]
        cond = "*mixed*"
    out: dict[str, str] = {}
    meta: dict[str, Any] = {"condition_used": cond, "n_rows": len(rows),
                            "conditions_present": sorted(str(c) for c in by_cond),
                            "n_declared_hash_used": 0, "n_no_completion": 0,
                            "n_duplicate_prompt_ids": 0}
    for i, r in enumerate(rows):
        pid = r.get("prompt_id")
        if pid is None:
            pid = f"idx:{r.get('idx')}" if r.get("idx") is not None else f"row:{i}"
        pid = str(pid)
        text = r.get("completion")
        if isinstance(text, str):
            dig = sha256_text(text)
        elif isinstance(r.get("completion_sha256"), str):
            dig = r["completion_sha256"]
            meta["n_declared_hash_used"] += 1
        else:
            meta["n_no_completion"] += 1
            continue
        if pid in out:
            meta["n_duplicate_prompt_ids"] += 1
            continue
        out[pid] = dig
    meta["n_prompt_ids"] = len(out)
    return out, meta


def _compare(a: dict[str, str], b: dict[str, str]) -> dict[str, Any]:
    common = sorted(set(a) & set(b))
    same = [p for p in common if a[p] == b[p]]
    n = len(common)
    return {"n_common": n, "n_identical": len(same), "n_differ": n - len(same),
            "fraction_differ": round((n - len(same)) / n, 6) if n else None,
            "identical_examples": same[:5],
            "only_in_a": sorted(set(a) - set(b))[:5],
            "only_in_b": sorted(set(b) - set(a))[:5]}


def _companion(art: RunArtifacts, role: str) -> tuple[Optional[RunArtifacts], Optional[str]]:
    """(sibling, note).  A companion that is declared but missing is worth saying
    out loud — it is a broken reference, not simply an experiment nobody ran."""
    rel = art.cfg("companions", role)
    if not rel:
        return None, f"no {role} companion is declared in manifest['companions']"
    sib = art.sibling(role)
    if sib is None:
        return None, (f"companion {role}={rel!r} is declared but no run directory "
                      f"exists at {art.run_dir.parent / rel}")
    if not sib.manifest and not sib.records:
        return None, (f"companion {role}={rel!r} exists at {sib.run_dir} but holds "
                      "neither a manifest nor records")
    return sib, None


# --------------------------------------------------------------------------
# T1 checks
# --------------------------------------------------------------------------

@check("t1.hook_fired", "T1")
def hook_fired(art: RunArtifacts) -> CheckResult:
    """A hook that registered but was never called is the quietest way to measure nothing."""
    if not art.manifest:
        return CheckResult("t1.hook_fired", "T1", "skip", "no manifest.json", {})
    hook = art.cfg("hook", default=None)
    en, alpha = _enabled(art), _alpha(art)
    ev: dict[str, Any] = {"steering_enabled": en, "alpha": alpha, "hook": hook,
                          "n_eval": art.cfg("counts", "n_eval"),
                          "n_records": len(art.records)}
    if hook is None:
        if _intervenes(art):
            return CheckResult("t1.hook_fired", "T1", "fail",
                               "steering is enabled but the manifest records no hook "
                               "block, so nothing shows the intervention ever ran", ev,
                               "record hook.module_path, fires_expected and the hook's "
                               "own .fires counter as fires_observed")
        return CheckResult("t1.hook_fired", "T1", "skip",
                           "no hook block and no steering configured", ev)
    if not isinstance(hook, dict):
        return CheckResult("t1.hook_fired", "T1", "fail",
                           f"manifest['hook'] is {type(hook).__name__}, expected a mapping", ev)
    obs, exp = _num(hook.get("fires_observed")), _num(hook.get("fires_expected"))
    ev["fires_observed"], ev["fires_expected"] = obs, exp
    if obs is None:
        if _intervenes(art):
            return CheckResult("t1.hook_fired", "T1", "fail",
                               "hook.fires_observed not recorded for a steered run", ev,
                               "SteeringHook.fires must be read after generation and written out")
        return CheckResult("t1.hook_fired", "T1", "skip",
                           "hook.fires_observed not recorded and no steering configured", ev)
    if en is False or (alpha is not None and alpha == 0.0 and en is not True):
        if obs > 0:
            return CheckResult("t1.hook_fired", "T1", "fail",
                               f"steering is off (steering_enabled={en}, alpha={alpha}) "
                               f"but the hook fired {obs:g} times — this run is not a "
                               "clean baseline", ev,
                               "detach the hook for baseline runs; a contaminated "
                               "baseline makes every downstream delta meaningless")
        return CheckResult("t1.hook_fired", "T1", "pass",
                           "steering off and the hook never fired", ev)
    if obs <= 0:
        return CheckResult("t1.hook_fired", "T1", "fail",
                           f"hook fired {obs:g} times while steering_enabled={en}, "
                           f"alpha={alpha} — the intervention never ran", ev,
                           "the module returned by resolve_layer_module must be the one "
                           "the forward pass actually calls; register_forward_hook on a "
                           "module that is bypassed never fires")
    if exp is not None and obs != exp:
        return CheckResult("t1.hook_fired", "T1", "fail",
                           f"hook fired {obs:g} times, expected {exp:g} — the "
                           "intervention did not cover every forward pass", ev,
                           "a mismatch usually means some prompts were generated "
                           "outside the hook's context manager")
    if exp is None:
        return CheckResult("t1.hook_fired", "T1", "warn",
                           f"hook fired {obs:g} times but fires_expected was not "
                           "recorded, so coverage cannot be checked", ev)
    return CheckResult("t1.hook_fired", "T1", "pass",
                       f"hook fired {obs:g}/{exp:g} times", ev)


@check("t1.vector_finite_nonzero", "T1")
def vector_finite_nonzero(art: RunArtifacts) -> CheckResult:
    """Adding a zero (or NaN) vector is a no-op (or a wipe-out) dressed as an experiment."""
    info = _vector_info(art)
    ev: dict[str, Any] = {k: v for k, v in info.items() if k != "problem"}
    ev["min_norm"] = MIN_VECTOR_NORM
    if info["problem"]:
        return CheckResult("t1.vector_finite_nonzero", "T1", "fail", info["problem"], ev,
                           "vector.npz must hold the steering vector under "
                           f"key {info['key']!r}")
    if info["norm"] is None:
        return CheckResult("t1.vector_finite_nonzero", "T1", "skip",
                           "no vector.npz and no manifest['vector']['norm']", ev)

    if info["source"] == "vector.npz":
        if not info["all_finite"]:
            return CheckResult("t1.vector_finite_nonzero", "T1", "fail",
                               f"{info['n_nonfinite']}/{info['dim']} vector components "
                               "are NaN or inf", ev,
                               "a non-finite component poisons every steered forward pass; "
                               "check for empty contrast batches or a division by zero "
                               "in extract_vector")
        if info["norm"] <= MIN_VECTOR_NORM:
            return CheckResult("t1.vector_finite_nonzero", "T1", "fail",
                               f"||v|| = {info['norm']:.6g} <= {MIN_VECTOR_NORM} — the "
                               "steering vector is effectively zero, so alpha*v adds "
                               "nothing whatever the manifest claims", ev,
                               "positive and negative contrast activations cancelled: "
                               "check that the pair labels are not identical and that "
                               "the last-prompt-token index is right")
        dec = info["declared_norm"]
        if dec is not None and abs(dec - info["norm"]) > NORM_SELF_REPORT_TOL * max(1.0, abs(dec)):
            return CheckResult("t1.vector_finite_nonzero", "T1", "fail",
                               f"manifest declares ||v|| = {dec:.6g} but the vector in "
                               f"vector.npz has norm {info['norm']:.6g}", ev,
                               "the manifest was not written from the vector that was used")
        return CheckResult("t1.vector_finite_nonzero", "T1", "pass",
                           f"||v|| = {info['norm']:.6g} over {info['dim']} finite "
                           "components", ev)

    # Only the self-report survives: it can still be falsified, not confirmed.
    if not math.isfinite(info["norm"]) or info["norm"] <= MIN_VECTOR_NORM:
        return CheckResult("t1.vector_finite_nonzero", "T1", "fail",
                           f"manifest declares ||v|| = {info['norm']!r}, which is zero "
                           "or non-finite", ev)
    return CheckResult("t1.vector_finite_nonzero", "T1", "warn",
                       f"vector.npz absent; only the manifest's ||v|| = "
                       f"{info['norm']:.6g} could be inspected", ev,
                       "write vector.npz so the vector can be recomputed independently")


@check("t1.vector_dim_matches_model", "T1")
def vector_dim_matches_model(art: RunArtifacts) -> CheckResult:
    """A vector of the wrong width means it came from another model or another layer."""
    info = _vector_info(art)
    sources: dict[str, int] = {}
    if info["declared_dim"] is not None:
        sources["manifest.vector.dim"] = int(info["declared_dim"])
    if info["dim"] is not None and info["source"] == "vector.npz":
        sources["vector.npz"] = int(info["dim"])
    data, problem = _load_acts(art)
    if data is not None:
        sources["acts.npz d_model"] = int(data["base"].shape[2])
    ev: dict[str, Any] = {"d_model_by_source": sources, "acts_problem": problem,
                          "vector_problem": info["problem"]}
    if len(sources) < 2:
        return CheckResult("t1.vector_dim_matches_model", "T1", "skip",
                           "fewer than two independent d_model sources "
                           f"({sorted(sources)})", ev)
    distinct = sorted(set(sources.values()))
    if len(distinct) > 1:
        return CheckResult("t1.vector_dim_matches_model", "T1", "fail",
                           f"d_model disagrees across artifacts: {sources}", ev,
                           "the steering vector, the captured activations and the "
                           "manifest must all describe the same model width")
    return CheckResult("t1.vector_dim_matches_model", "T1", "pass",
                       f"d_model = {distinct[0]} agrees across {sorted(sources)}", ev)


@check("t1.activation_delta_matches_alpha", "T1")
def activation_delta_matches_alpha(art: RunArtifacts) -> CheckResult:
    """||steered - base|| at layer L must equal |alpha|*||v||, or the hook lied about scale."""
    cid = "t1.activation_delta_matches_alpha"
    data, res = _acts_or_result(art, cid)
    if res is not None:
        return res
    st = _delta_stats(data)
    L, alpha = _layer(art), _alpha(art)
    info = _vector_info(art)
    ev: dict[str, Any] = {
        "layer": L, "alpha": alpha, "steering_enabled": _enabled(art),
        "v_norm": info["norm"], "v_norm_source": info["source"],
        "rel_tol": ALPHA_REL_TOL, "zero_rel_tol": ZERO_DELTA_REL,
        "captured_layers": st["layers"], "n_eval": st["n_eval"],
        "d_model": st["d_model"],
        "median_delta_norm_by_layer": {str(l): round(v, 6) for l, v
                                       in zip(st["layers"], st["median_norm"])},
    }
    if st["n_nonfinite"]:
        ev["n_nonfinite_delta"] = st["n_nonfinite"]
        return CheckResult(cid, "T1", "fail",
                           f"{st['n_nonfinite']} non-finite entries in steered - base", ev)
    if L is None:
        return CheckResult(cid, "T1", "skip", "config.layer not recorded", ev)
    i = _layer_pos(st["layers"], L)
    if i is None:
        return CheckResult(cid, "T1", "skip",
                           f"layer {L} is not among the captured layers {st['layers']}", ev)
    measured = st["median_norm"][i]
    ev.update(measured_median_delta_norm=round(measured, 6),
              measured_mean_delta_norm=round(st["mean_norm"][i], 6),
              measured_min_delta_norm=round(st["min_norm"][i], 6),
              measured_max_delta_norm=round(st["max_norm"][i], 6),
              median_base_norm_at_layer=round(st["median_base_norm"][i], 6))

    if not _intervenes(art):
        floor = ZERO_DELTA_REL * max(st["median_base_norm"][i], 1e-12)
        ev["allowed_delta_norm"] = round(floor, 9)
        if measured <= floor:
            return CheckResult(cid, "T1", "pass",
                               f"no intervention configured (alpha={alpha}, "
                               f"steering_enabled={_enabled(art)}) and none measured at "
                               f"layer {L}", ev)
        return CheckResult(cid, "T1", "fail",
                           f"alpha={alpha} / steering_enabled={_enabled(art)} means no "
                           f"perturbation, but layer {L} moved by {measured:.6g} "
                           f"(allowed {floor:.3g})", ev,
                           "something added to the residual stream in a run that "
                           "declared no intervention")
    if alpha is None:
        return CheckResult(cid, "T1", "skip", "config.alpha not recorded", ev)
    if info["norm"] is None:
        return CheckResult(cid, "T1", "skip",
                           "||v|| is unknown (no vector.npz, no manifest vector.norm)", ev)

    expected = abs(alpha) * float(info["norm"])
    ev["expected_delta_norm"] = round(expected, 6)
    ev["ratio_by_layer"] = ({str(l): round(v / expected, 6) for l, v
                             in zip(st["layers"], st["median_norm"])}
                            if expected > 0 else None)
    if expected <= 0:
        return CheckResult(cid, "T1", "fail",
                           f"|alpha|*||v|| = {expected:.6g}: this run cannot have "
                           "steered anything", ev,
                           "see t1.vector_finite_nonzero")
    ratio = measured / expected
    ev["ratio"] = round(ratio, 6)
    spread = (st["max_norm"][i] - st["min_norm"][i]) / expected
    ev["per_item_norm_spread"] = round(spread, 6)

    if abs(ratio - 1.0) > ALPHA_REL_TOL:
        if ratio < 0.02:
            why = (f"layer {L} barely moved ({measured:.6g} vs an expected "
                   f"{expected:.6g}) — alpha*v never reached the residual stream")
        else:
            why = (f"layer {L} moved {measured:.6g}, which is {ratio:.4g}x the "
                   f"expected |alpha|*||v|| = {expected:.6g}")
        return CheckResult(cid, "T1", "fail", why, ev,
                           f"tolerance is {ALPHA_REL_TOL:.0%} relative; compare "
                           "ratio_by_layer — a ratio of ~1 at a neighbouring layer means "
                           "an off-by-one in the layer convention (see "
                           "t1.layer_index_convention), a constant factor means alpha "
                           "was applied twice or not at all")
    if spread > 0.5:
        return CheckResult(cid, "T1", "warn",
                           f"median delta matches |alpha|*||v|| (ratio {ratio:.4g}) but "
                           f"per-item norms vary by {spread:.3g} of it — the addition is "
                           "not the same constant vector for every prompt", ev)
    return CheckResult(cid, "T1", "pass",
                       f"||steered-base|| at layer {L} = {measured:.6g} vs "
                       f"|alpha|*||v|| = {expected:.6g} (ratio {ratio:.4g}, "
                       f"tol {ALPHA_REL_TOL:.0%})", ev)


@check("t1.no_effect_before_layer", "T1")
def no_effect_before_layer(art: RunArtifacts) -> CheckResult:
    """Nothing upstream of the hook can move; if it did, the hook is not where it claims."""
    cid = "t1.no_effect_before_layer"
    data, res = _acts_or_result(art, cid)
    if res is not None:
        return res
    st = _delta_stats(data)
    L = _layer(art)
    ev: dict[str, Any] = {"layer": L, "captured_layers": st["layers"],
                          "n_eval": st["n_eval"], "d_model": st["d_model"]}
    if L is None:
        return CheckResult(cid, "T1", "skip", "config.layer not recorded", ev)
    before = [(l, i) for i, l in enumerate(st["layers"]) if l < L]
    ev["layers_before"] = [l for l, _ in before]
    if not before:
        return CheckResult(cid, "T1", "skip",
                           f"no captured layer is below layer {L}", ev)
    ev["max_abs_diff_by_layer"] = {str(l): st["max_abs"][i] for l, i in before}
    ev["n_nonzero_elements_by_layer"] = {str(l): st["n_nonzero"][i] for l, i in before}
    offenders = [(l, st["max_abs"][i], st["n_nonzero"][i]) for l, i in before
                 if st["max_abs"][i] != 0.0]
    if offenders:
        l0, m0, n0 = offenders[0]
        ev["first_offending_layer"] = l0
        ev["n_offending_layers"] = len(offenders)
        return CheckResult(
            cid, "T1", "fail",
            f"{len(offenders)} captured layer(s) below {L} differ between base and "
            f"steered; layer {l0} has {n0} changed elements, max |diff| {m0:.6g}",
            ev,
            "the two runs must be bitwise identical upstream of the hook. Look at: "
            "(a) the hooked module — resolve_layer_module must return block L itself, "
            "not the whole model or an ancestor that runs earlier; (b) the hidden-state "
            "offset — hidden_states[i] is the INPUT to block i, so block L's output is "
            "index L+1, and capturing at the wrong index makes an on-target hook look "
            "like an upstream leak; (c) the two runs themselves — a different seed, "
            "batch size, padding side or prompt order changes the activations before "
            "any steering happens.")
    return CheckResult(cid, "T1", "pass",
                       f"all {len(before)} captured layer(s) below {L} are bitwise "
                       "identical between base and steered", ev)


@check("t1.effect_after_layer", "T1")
def effect_after_layer(art: RunArtifacts) -> CheckResult:
    """If the perturbation does not propagate downstream it never entered the graph."""
    cid = "t1.effect_after_layer"
    data, res = _acts_or_result(art, cid)
    if res is not None:
        return res
    st = _delta_stats(data)
    L = _layer(art)
    ev: dict[str, Any] = {"layer": L, "captured_layers": st["layers"],
                          "steering_enabled": _enabled(art), "alpha": _alpha(art)}
    if L is None:
        return CheckResult(cid, "T1", "skip", "config.layer not recorded", ev)
    after = [(l, i) for i, l in enumerate(st["layers"]) if l > L]
    ev["layers_after"] = [l for l, _ in after]
    if not after:
        return CheckResult(cid, "T1", "skip",
                           f"no captured layer is above layer {L}", ev)
    ev["max_abs_diff_by_layer"] = {str(l): st["max_abs"][i] for l, i in after}
    ev["median_delta_norm_by_layer"] = {str(l): round(st["median_norm"][i], 6)
                                        for l, i in after}
    moved = [l for l, i in after if st["max_abs"][i] > 0.0]
    ev["n_layers_moved"] = len(moved)
    if not _intervenes(art):
        if moved:
            return CheckResult(cid, "T1", "fail",
                               f"no intervention configured, yet layers {moved} above "
                               f"{L} differ between base and steered", ev)
        return CheckResult(cid, "T1", "skip",
                           "no intervention configured, so no downstream effect is "
                           "expected", ev)
    if not moved:
        return CheckResult(cid, "T1", "fail",
                           f"every captured layer above {L} is bitwise identical between "
                           "base and steered — the perturbation did not propagate", ev,
                           "the hook fired on a tensor that is discarded (a copy, or a "
                           "module whose output is not the residual stream), or the "
                           "steered activations were written from the baseline pass")
    return CheckResult(cid, "T1", "pass",
                       f"{len(moved)}/{len(after)} captured layer(s) above {L} changed",
                       ev)


@check("t1.outputs_differ_from_baseline", "T1")
def outputs_differ_from_baseline(art: RunArtifacts) -> CheckResult:
    """Identical completions with a nonzero alpha means the hook perturbed nothing."""
    cid = "t1.outputs_differ_from_baseline"
    alpha, en = _alpha(art), _enabled(art)
    ev: dict[str, Any] = {"alpha": alpha, "steering_enabled": en,
                          "fail_below_fraction_differ": MIN_DIFF_FRACTION,
                          "warn_below_fraction_differ": WARN_DIFF_FRACTION}
    if not _intervenes(art):
        return CheckResult(cid, "T1", "skip",
                           f"no intervention configured (alpha={alpha}, "
                           f"steering_enabled={en}); identical outputs are expected", ev)
    mine, meta = _digests(art, prefer="steered")
    ev["self"] = meta
    if not mine:
        return CheckResult(cid, "T1", "skip", "this run has no completions to compare", ev)

    sib, note = _companion(art, "baseline")
    ev["baseline_companion"] = note or str(sib.run_dir)
    if sib is not None and sib.records:
        theirs, bmeta = _digests(sib, prefer="baseline")
        ev["baseline_source"] = f"companion {art.cfg('companions', 'baseline')!r}"
    else:
        rows = [r for r in art.records
                if isinstance(r, dict) and r.get("condition") == "baseline"]
        if not rows:
            why = note or "the baseline companion holds no records"
            return CheckResult(cid, "T1", "skip",
                               f"nothing to compare against: {why}, and this run holds "
                               "no baseline-condition records", ev)
        theirs, bmeta = _digests(RunArtifacts(run_dir=art.run_dir, manifest=art.manifest,
                                              records=rows), prefer="baseline")
        ev["baseline_source"] = "baseline-condition records in this run"
    ev["baseline"] = bmeta
    cmp = _compare(mine, theirs)
    ev.update(cmp)
    if not cmp["n_common"]:
        return CheckResult(cid, "T1", "fail",
                           "steered and baseline records share no prompt_id, so no "
                           "paired comparison is possible", ev,
                           "prompt_id must be stable across runs")
    frac = cmp["fraction_differ"]
    if frac < MIN_DIFF_FRACTION:
        return CheckResult(cid, "T1", "fail",
                           f"only {cmp['n_differ']}/{cmp['n_common']} completions "
                           f"({frac:.1%}) differ from baseline while alpha={alpha} — the "
                           "intervention changed nothing the model produced", ev,
                           "a hook attached to the wrong module, or added to a tensor "
                           "the forward pass discards, fires and still leaves generation "
                           "byte-identical; cross-check t1.activation_delta_matches_alpha")
    if frac < WARN_DIFF_FRACTION:
        return CheckResult(cid, "T1", "warn",
                           f"only {cmp['n_differ']}/{cmp['n_common']} completions "
                           f"({frac:.1%}) differ from baseline at alpha={alpha}", ev)
    return CheckResult(cid, "T1", "pass",
                       f"{cmp['n_differ']}/{cmp['n_common']} completions ({frac:.1%}) "
                       "differ from baseline", ev)


@check("t1.alpha_zero_is_identity", "T1")
def alpha_zero_is_identity(art: RunArtifacts) -> CheckResult:
    """alpha=0 must reproduce the baseline exactly; anything else is a leaking hook."""
    cid = "t1.alpha_zero_is_identity"
    ev: dict[str, Any] = {"companions": art.cfg("companions", default=None)}
    sib, note = _companion(art, "alpha_zero")
    if sib is None:
        return CheckResult(cid, "T1", "skip", note, ev)
    ev["alpha_zero_run"] = str(sib.run_dir)
    sib_alpha = _num(sib.cfg("config", "alpha"))
    ev["alpha_zero_alpha"] = sib_alpha
    ev["alpha_zero_steering_enabled"] = _enabled(sib)
    if sib_alpha is None:
        return CheckResult(cid, "T1", "fail",
                           "the alpha_zero companion records no config.alpha", ev)
    if sib_alpha != 0.0:
        return CheckResult(cid, "T1", "fail",
                           f"the run declared as the alpha_zero control has "
                           f"alpha={sib_alpha}", ev,
                           "the alpha=0 control must run the hook with alpha exactly 0")

    ref, bnote = _companion(art, "baseline")
    ev["baseline_companion"] = bnote or str(ref.run_dir)
    if ref is not None and ref.records:
        ev["baseline_run"] = str(ref.run_dir)
    elif _enabled(art) is False or (_alpha(art) == 0.0):
        ref = art
        ev["baseline_run"] = f"{art.run_dir} (this run is itself unsteered)"
    else:
        return CheckResult(cid, "T1", "skip",
                           f"the alpha_zero run cannot be checked for identity: "
                           f"{bnote or 'the baseline companion has no records'}", ev)
    a, ameta = _digests(sib, prefer="steered")
    b, bmeta = _digests(ref, prefer="baseline")
    ev["alpha_zero"], ev["baseline"] = ameta, bmeta
    if not a or not b:
        return CheckResult(cid, "T1", "skip",
                           "the alpha_zero or baseline run has no completions", ev)
    cmp = _compare(a, b)
    ev.update(cmp)
    if not cmp["n_common"]:
        return CheckResult(cid, "T1", "fail",
                           "the alpha_zero and baseline runs share no prompt_id", ev,
                           "prompt_id must be stable across runs")
    if cmp["n_differ"]:
        ev["differing_examples"] = [p for p in sorted(set(a) & set(b))
                                    if a[p] != b[p]][:5]
        return CheckResult(cid, "T1", "fail",
                           f"{cmp['n_differ']}/{cmp['n_common']} alpha=0 completions "
                           "differ from baseline — the hook changes the forward pass "
                           "even when it adds nothing", ev,
                           "look for a dtype cast, an in-place write, or a device round "
                           "trip inside the hook; also check both runs used the same "
                           "seed, batch size and prompt order")
    return CheckResult(cid, "T1", "pass",
                       f"all {cmp['n_common']} alpha=0 completion hashes match baseline",
                       ev)


@check("t1.sign_flip_differs", "T1")
def sign_flip_differs(art: RunArtifacts) -> CheckResult:
    """-alpha must not reproduce +alpha; if it does, alpha never reached the addition."""
    cid = "t1.sign_flip_differs"
    alpha = _alpha(art)
    ev: dict[str, Any] = {"alpha": alpha, "companions": art.cfg("companions", default=None),
                          "min_fraction_differ": MIN_DIFF_FRACTION}
    sib, note = _companion(art, "sign_flip")
    if sib is None:
        return CheckResult(cid, "T1", "skip", note, ev)
    ev["sign_flip_run"] = str(sib.run_dir)
    sib_alpha = _num(sib.cfg("config", "alpha"))
    ev["sign_flip_alpha"] = sib_alpha
    if alpha is not None and alpha == 0.0:
        return CheckResult(cid, "T1", "skip",
                           "this run has alpha=0, so a sign flip is a no-op", ev)
    if alpha is not None and sib_alpha is not None and sib_alpha != -alpha:
        return CheckResult(cid, "T1", "fail",
                           f"the run declared as the sign_flip control has "
                           f"alpha={sib_alpha}, not {-alpha}", ev)
    a, ameta = _digests(art, prefer="steered")
    b, bmeta = _digests(sib, prefer="steered")
    ev["self"], ev["sign_flip"] = ameta, bmeta
    if not a or not b:
        return CheckResult(cid, "T1", "skip",
                           "this run or its sign_flip companion has no completions", ev)
    cmp = _compare(a, b)
    ev.update(cmp)
    if not cmp["n_common"]:
        return CheckResult(cid, "T1", "fail",
                           "this run and the sign_flip companion share no prompt_id", ev,
                           "prompt_id must be stable across runs")
    frac = cmp["fraction_differ"]
    if frac < MIN_DIFF_FRACTION:
        return CheckResult(cid, "T1", "fail",
                           f"only {cmp['n_differ']}/{cmp['n_common']} completions "
                           f"({frac:.1%}) differ between alpha={alpha} and "
                           f"alpha={sib_alpha} — the sign of alpha does not reach the "
                           "residual stream", ev,
                           "check that the hook multiplies by the signed alpha rather "
                           "than by abs(alpha) or by a value captured at import time")
    return CheckResult(cid, "T1", "pass",
                       f"{cmp['n_differ']}/{cmp['n_common']} completions ({frac:.1%}) "
                       f"differ between alpha={alpha} and alpha={sib_alpha}", ev)


@check("t1.completions_finite", "T1")
def completions_finite(art: RunArtifacts) -> CheckResult:
    """Empty or NaN-poisoned generations score as data and quietly move the metric."""
    cid = "t1.completions_finite"
    if not art.records:
        return CheckResult(cid, "T1", "skip", "records.jsonl absent or empty",
                           {"records_path": str(art.run_dir / "records.jsonl")})
    max_new = _int(art.cfg("config", "max_new_tokens"))
    bad: dict[str, list[Any]] = {"nonfinite_logits": [], "empty_completion": [],
                                 "degenerate_text": [], "no_token_count": [],
                                 "nonpositive_tokens": [], "over_max_tokens": [],
                                 "nonfinite_act_norm": []}
    for i, r in enumerate(art.records):
        if not isinstance(r, dict):
            bad["empty_completion"].append(f"row:{i}")
            continue
        tag = r.get("prompt_id", r.get("idx", f"row:{i}"))
        fin = _as_bool(r.get("finite_logits"))
        if fin is False:
            bad["nonfinite_logits"].append(tag)
        text = r.get("completion")
        if not isinstance(text, str) or not text.strip():
            bad["empty_completion"].append(tag)
        else:
            toks = [t for t in re.split(r"[\s,]+", text.strip().lower()) if t]
            if toks and all(t in _DEGENERATE_TOKENS for t in toks):
                bad["degenerate_text"].append(tag)
        n = _num(r.get("n_new_tokens"))
        if n is None:
            bad["no_token_count"].append(tag)
        elif n <= 0:
            bad["nonpositive_tokens"].append(tag)
        elif max_new is not None and n > max_new:
            bad["over_max_tokens"].append(tag)
        an = r.get("act_norm")
        if an is not None and (_num(an) is None or not math.isfinite(_num(an))):
            bad["nonfinite_act_norm"].append(tag)

    counts = {k: len(v) for k, v in bad.items()}
    ev: dict[str, Any] = {"n_records": len(art.records), "counts": counts,
                          "max_new_tokens": max_new,
                          "examples": {k: v[:5] for k, v in bad.items() if v}}
    hard = [k for k in ("nonfinite_logits", "empty_completion", "degenerate_text",
                        "nonpositive_tokens", "over_max_tokens", "nonfinite_act_norm")
            if counts[k]]
    if hard:
        return CheckResult(cid, "T1", "fail",
                           "; ".join(f"{counts[k]}/{len(art.records)} records {k}"
                                     for k in hard), ev,
                           "non-finite logits usually mean the steering vector blew the "
                           "residual stream up: lower alpha, and check that the vector "
                           "is finite (t1.vector_finite_nonzero). Empty completions mean "
                           "generation stopped immediately — check the chat template and "
                           "that BatchEncoding is splatted into generate(**enc, ...)")
    if counts["no_token_count"]:
        return CheckResult(cid, "T1", "warn",
                           f"{counts['no_token_count']}/{len(art.records)} records carry "
                           "no n_new_tokens", ev)
    return CheckResult(cid, "T1", "pass",
                       f"all {len(art.records)} completions non-empty, finite and within "
                       "the token budget", ev)


@check("t1.layer_index_convention", "T1")
def layer_index_convention(art: RunArtifacts) -> CheckResult:
    """hidden_states[i] is the INPUT to block i — an off-by-one here steers the wrong layer."""
    cid = "t1.layer_index_convention"
    L = _layer(art)
    module_path = art.cfg("hook", "module_path")
    hook_layer = None
    if isinstance(module_path, str):
        m = re.search(r"(?:^|\.)layers?\.(\d+)", module_path) or \
            re.search(r"(\d+)(?!.*\d)", module_path)
        if m:
            hook_layer = int(m.group(1))
    rec_layers = sorted({int(v) for v in (_int(r.get("act_layer"))
                                          for r in art.records if isinstance(r, dict))
                         if v is not None})
    data, _problem = _load_acts(art)
    captured = list(data["layers"]) if data is not None else None
    ev: dict[str, Any] = {
        "config.layer": L, "vector.layer": _int(art.cfg("vector", "layer")),
        "hook.module_path": module_path, "hook_layer_from_module_path": hook_layer,
        "record_act_layers": rec_layers, "captured_layers": captured,
    }
    known = [v for v in (L, ev["vector.layer"], hook_layer) if v is not None]
    if not known and not rec_layers:
        return CheckResult(cid, "T1", "skip",
                           "no layer index recorded anywhere (config.layer, "
                           "vector.layer, hook.module_path, records[].act_layer)", ev)
    if L is None:
        return CheckResult(cid, "T1", "skip",
                           "config.layer not recorded; nothing to reconcile against", ev)

    disagree: list[str] = []
    if ev["vector.layer"] is not None and ev["vector.layer"] != L:
        disagree.append(f"manifest.vector.layer={ev['vector.layer']}")
    if hook_layer is not None and hook_layer != L:
        disagree.append(f"hook.module_path={module_path!r} (block {hook_layer})")
    if rec_layers and rec_layers != [L]:
        disagree.append(f"records[].act_layer={rec_layers}")
    if disagree:
        return CheckResult(cid, "T1", "fail",
                           f"config.layer={L} but " + ", ".join(disagree), ev,
                           "one index is derived from another somewhere: the hooked "
                           "module, the vector's layer and the recorded act_layer must "
                           "all name block L, with hidden_states index L+1 as its output")

    # The arrays settle the convention argument: if the vector landed one block off,
    # the delta shows up at L-1 or L+1 with exactly the expected magnitude.
    if data is not None and _intervenes(art):
        st = _delta_stats(data)
        info = _vector_info(art)
        alpha = _alpha(art)
        if captured is not None and L not in captured:
            ev["note"] = f"layer {L} was not captured"
            return CheckResult(cid, "T1", "warn",
                               f"layer indices agree on {L}, but acts.npz captured "
                               f"{captured} and not {L}", ev)
        if alpha is not None and info["norm"]:
            expected = abs(alpha) * float(info["norm"])
            ratios = {l: (st["median_norm"][i] / expected if expected > 0 else None)
                      for i, l in enumerate(st["layers"])}
            ev["expected_delta_norm"] = round(expected, 6)
            ev["ratio_by_layer"] = {str(l): round(v, 6) for l, v in ratios.items()
                                    if v is not None}
            at_l = ratios.get(L)
            if at_l is not None and at_l < 0.02:
                off = [l for l in (L - 1, L + 1)
                       if ratios.get(l) is not None
                       and abs(ratios[l] - 1.0) <= ALPHA_REL_TOL]
                if off:
                    return CheckResult(
                        cid, "T1", "fail",
                        f"the perturbation lands at layer {off[0]}, not the configured "
                        f"layer {L} — an off-by-one in the layer convention", ev,
                        "hidden_states has n_layers+1 entries and hidden_states[i] is "
                        "the INPUT to block i, so the OUTPUT of block L is index L+1; "
                        "hidden_state_index(L) and resolve_layer_module(L) must agree "
                        "on that")
    return CheckResult(cid, "T1", "pass",
                       f"every recorded layer index names block {L}", ev)
