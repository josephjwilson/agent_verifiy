"""T3 — do the artifacts agree with each other; T4 — do the reported numbers
follow from the records.

T3 recomputes every hash and count the manifest asserts about itself.  T4 is the
anti-overclaim tier: it takes ``claims.json`` — the run's own report — and
recomputes each metric claim from ``records.jsonl``.  The sycophancy rate is a
ratio of integer counts, so the recomputation is exact: a claimed value that does
not reproduce is a fabricated or stale number, and the evidence carries both the
claimed and the recomputed value so the reader can see the gap.

Record reading is imported from ``checks/statistical.py`` on purpose — T2 and T4
must define "the sycophancy rate" identically, or the audit audits nothing.

`needs` is left empty on every check here for the same reason as in T2: each
check decides for itself whether its input is absent (skip) or present and wrong
(fail).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from ..types import CheckResult, RunArtifacts, check, sha256_file, sha256_text
from .statistical import (_effect, as_record, baseline_records, count_scored, num,
                          pair_by_prompt, sibling_of, treatment_records,
                          undetermined_rate)

CLAIM_TOL = 1e-6          # claims are exact rationals of counts; the slack is float noise
MAX_EXAMPLES = 5          # how many offending items to name in evidence
MIN_LEAK_CHARS = 24       # shorter shared strings are common English, not leakage
SIG_ALPHA = 0.05          # used only to flag a claimed/recomputed significance flip

CLAIM_KINDS = ("metric_value", "metric_delta", "qualitative")

# Metrics records.jsonl can actually produce.  Anything else is an overclaim by
# construction, and t4.no_unsupported_claims says so.
METRIC_ALIASES = {
    "sycophancy_rate": "sycophancy_rate",
    "sycophancy": "sycophancy_rate",
    "sycophancy_frac": "sycophancy_rate",
    "sycophancy_fraction": "sycophancy_rate",
    "sycophancy_count": "sycophancy_count",
    "n_sycophantic": "sycophancy_count",
    "n": "n_records",
    "n_records": "n_records",
    "count": "n_records",
    "undetermined_rate": "undetermined_rate",
    "mean_new_tokens": "mean_new_tokens",
    "n_new_tokens": "mean_new_tokens",
}

REQUIRED_KEYS = (
    "schema_version", "run_id", "created_utc", "git_sha",
    "config.model_id", "config.dtype", "config.device", "config.layer",
    "config.alpha", "config.seed", "config.max_new_tokens", "config.n_eval",
    "config.steering_enabled", "config.vector_source",
    "env.python", "env.torch", "env.transformers", "env.cuda_available",
    "counts.n_eval", "counts.n_records", "artifacts.records",
)
STEERED_KEYS = (
    "vector.path", "vector.layer", "vector.dim", "vector.norm",
    "hook.module_path", "hook.fires_expected", "hook.fires_observed",
)
RECOMMENDED_KEYS = (
    "git_dirty", "placement.n_params", "placement.param_devices",
    "placement.param_dtypes", "placement.matmul_tflops", "timing.load_s",
    "timing.generate_s", "hashes", "metrics",
)
KEY_TYPES: dict[str, tuple] = {
    "config.layer": (int,), "config.alpha": (int, float), "config.seed": (int,),
    "config.max_new_tokens": (int,), "config.n_eval": (int,),
    "config.steering_enabled": (bool,), "config.model_id": (str,),
    "config.vector_source": (str,), "counts.n_eval": (int,), "counts.n_records": (int,),
    "env.cuda_available": (bool,), "run_id": (str,), "git_sha": (str,),
    "vector.dim": (int,), "vector.layer": (int,), "vector.norm": (int, float),
    "hook.fires_expected": (int,), "hook.fires_observed": (int,),
}


def _res(cid: str, tier: str, status: str, summary: str,
         evidence: Optional[dict] = None, remedy: str = "") -> CheckResult:
    return CheckResult(id=cid, tier=tier, status=status, summary=summary,
                       evidence=evidence or {}, remedy=remedy)


def _get(manifest: Any, path: str) -> tuple[bool, Any]:
    """(present, value) for a dotted manifest path; None/'' counts as absent."""
    node = manifest
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    if node is None or (isinstance(node, str) and node.strip() == ""):
        return False, None
    if isinstance(node, (dict, list)) and len(node) == 0:
        return False, None
    return True, node


def _completion_hash(rec: Any) -> tuple[Optional[str], Optional[str]]:
    """(recomputed, recorded) — the text always wins; that is the point of T3."""
    if not isinstance(rec, dict):
        return None, None
    recorded = rec.get("completion_sha256")
    recorded = recorded.strip().lower() if isinstance(recorded, str) else None
    text = rec.get("completion")
    got = sha256_text(text) if isinstance(text, str) else None
    return got, recorded


def _rec_id(rec: Any, i: int) -> str:
    if isinstance(rec, dict):
        for k in ("prompt_id", "idx"):
            if rec.get(k) is not None:
                return f"{k}={rec[k]}"
    return f"line={i}"


# --------------------------------------------------------------------------
# T3 — integrity
# --------------------------------------------------------------------------

@check("t3.artifact_hashes_match", "T3")
def artifact_hashes_match(art: RunArtifacts) -> CheckResult:
    """Recompute sha256 of every file the manifest hashes; tampering shows up here."""
    cid = "t3.artifact_hashes_match"
    hashes = art.cfg("hashes", default=None)
    if not hashes:
        return _res(cid, "T3", "skip", "manifest records no hashes block")
    if not isinstance(hashes, dict):
        return _res(cid, "T3", "fail", "manifest.hashes is not an object",
                    {"hashes_type": type(hashes).__name__})

    mismatched, missing, malformed, ok = [], [], [], []
    for name, want in sorted(hashes.items()):
        path = art.run_dir / str(name)
        if not isinstance(want, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", want.strip()):
            malformed.append({"file": name, "recorded": want})
            continue
        if not path.exists():
            missing.append(name)
            continue
        try:
            got = sha256_file(path)
        except Exception as exc:
            malformed.append({"file": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if got != want.strip().lower():
            mismatched.append({"file": name, "recorded": want.strip().lower(),
                               "recomputed": got, "size_bytes": path.stat().st_size})
        else:
            ok.append(name)

    declared = art.cfg("artifacts", default={}) or {}
    uncovered = sorted({str(v) for v in declared.values() if isinstance(v, str)}
                       - set(map(str, hashes.keys()))) if isinstance(declared, dict) else []
    uncovered = [u for u in uncovered if (art.run_dir / u).exists()]
    ev = {"verified": ok, "mismatched": mismatched, "missing": missing,
          "malformed": malformed, "unhashed_artifacts": uncovered,
          "vector_sha256_in_manifest": art.cfg("vector", "sha256")}

    if mismatched or missing or malformed:
        bits = []
        if mismatched:
            bits.append(f"{len(mismatched)} file(s) do not match their recorded sha256: "
                        + ", ".join(m["file"] for m in mismatched))
        if missing:
            bits.append(f"{len(missing)} hashed file(s) absent: " + ", ".join(missing))
        if malformed:
            bits.append(f"{len(malformed)} unusable hash entr(ies)")
        return _res(cid, "T3", "fail", "; ".join(bits), ev,
                    "the artifacts on disk are not the ones this manifest describes")
    if uncovered:
        return _res(cid, "T3", "warn",
                    f"{len(ok)} hash(es) verified, but declared artifact(s) {uncovered} "
                    "carry no recorded hash", ev)
    return _res(cid, "T3", "pass", f"{len(ok)} artifact hash(es) verified", ev)


@check("t3.record_count_matches", "T3")
def record_count_matches(art: RunArtifacts) -> CheckResult:
    """len(records) == counts.n_records == counts.n_eval."""
    cid = "t3.record_count_matches"
    counts = art.cfg("counts", default={}) or {}
    recs = list(art.records or [])
    rp = art.run_dir / "records.jsonl"
    if not recs and not counts and not rp.exists():
        return _res(cid, "T3", "skip", "neither records.jsonl nor a counts block is present")

    n_lines = len(recs)
    ids = [_rec_id(r, i) for i, r in enumerate(recs)]
    n_ids = len({i for i in ids if not i.startswith("line=")}) or n_lines
    conds = sorted({str(as_record(r).get("condition", "")) for r in recs})
    ev = {"n_lines": n_lines, "n_distinct_ids": n_ids, "conditions": conds,
          "counts.n_records": counts.get("n_records"), "counts.n_eval": counts.get("n_eval"),
          "config.n_eval": art.cfg("config", "n_eval"),
          "records_file_present": rp.exists()}

    problems: list[str] = []
    if not rp.exists() and counts:
        problems.append("manifest declares counts but records.jsonl is absent")

    def as_int(v):
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    n_rec_declared = as_int(counts.get("n_records"))
    n_eval_declared = as_int(counts.get("n_eval"))
    cfg_eval = as_int(art.cfg("config", "n_eval"))

    if n_rec_declared is not None and n_rec_declared != n_lines:
        problems.append(f"counts.n_records={n_rec_declared} but records.jsonl has {n_lines} lines")
    if n_eval_declared is not None and n_eval_declared not in (n_lines, n_ids):
        problems.append(f"counts.n_eval={n_eval_declared} but the file holds {n_lines} records "
                        f"over {n_ids} distinct items")
    if cfg_eval is not None and n_eval_declared is not None and cfg_eval != n_eval_declared:
        problems.append(f"config.n_eval={cfg_eval} disagrees with counts.n_eval={n_eval_declared}")
    if (n_rec_declared is not None and n_eval_declared is not None
            and n_rec_declared != n_eval_declared and n_ids != n_eval_declared):
        problems.append(f"counts.n_records={n_rec_declared} != counts.n_eval={n_eval_declared} "
                        f"and the {n_ids} distinct prompt ids explain neither")

    if problems:
        return _res(cid, "T3", "fail", "; ".join(problems), ev,
                    "the counts block does not describe the records that exist")
    if n_rec_declared is None and n_eval_declared is None:
        return _res(cid, "T3", "warn",
                    f"records.jsonl holds {n_lines} records but the manifest declares no counts",
                    ev)
    return _res(cid, "T3", "pass",
                f"{n_lines} records over {n_ids} distinct items, counts agree", ev)


@check("t3.record_hashes_match", "T3")
def record_hashes_match(art: RunArtifacts) -> CheckResult:
    """completion_sha256 must be the sha256 of the completion text beside it."""
    cid = "t3.record_hashes_match"
    recs = list(art.records or [])
    if not recs:
        return _res(cid, "T3", "skip", "no records to hash")
    hashed = [r for r in recs if isinstance(r, dict) and isinstance(r.get("completion_sha256"), str)]
    if not hashed:
        return _res(cid, "T3", "skip", "no record carries completion_sha256")

    bad, unhashable, unrecorded = [], [], []
    for i, rec in enumerate(recs):
        got, recorded = _completion_hash(rec)
        if recorded is None:
            unrecorded.append(_rec_id(rec, i))
            continue
        if got is None:
            unhashable.append(_rec_id(rec, i))
            continue
        if got != recorded:
            bad.append({"record": _rec_id(rec, i), "recorded": recorded, "recomputed": got})

    ev = {"n_records": len(recs), "n_hashed": len(hashed),
          "mismatched": bad[:MAX_EXAMPLES], "n_mismatched": len(bad),
          "records_without_hash": unrecorded[:MAX_EXAMPLES],
          "records_without_completion_text": unhashable[:MAX_EXAMPLES]}
    if bad or unhashable:
        bits = []
        if bad:
            bits.append(f"{len(bad)} completion(s) do not hash to their recorded "
                        f"completion_sha256 ({', '.join(b['record'] for b in bad[:MAX_EXAMPLES])})")
        if unhashable:
            bits.append(f"{len(unhashable)} record(s) record a hash but carry no completion text")
        return _res(cid, "T3", "fail", "; ".join(bits), ev,
                    "the completions in records.jsonl are not the ones that were hashed")
    if unrecorded:
        return _res(cid, "T3", "fail",
                    f"{len(recs) - len(unrecorded)} completion hash(es) verified but "
                    f"{len(unrecorded)} record(s) carry no completion_sha256 and cannot be "
                    "checked at all", ev,
                    "hash every completion when writing records.jsonl")
    return _res(cid, "T3", "pass", f"{len(hashed)} completion hash(es) verified", ev)


@check("t3.schema_complete", "T3")
def schema_complete(art: RunArtifacts) -> CheckResult:
    """Every manifest key the contract requires is present and of the right type."""
    cid = "t3.schema_complete"
    if not art.manifest:
        return _res(cid, "T3", "skip", "no manifest.json in the run dir")

    required = list(REQUIRED_KEYS)
    if art.cfg("config", "steering_enabled", default=False):
        required += list(STEERED_KEYS)
    missing = [k for k in required if not _get(art.manifest, k)[0]]
    wrong_type = []
    for k in required:
        present, val = _get(art.manifest, k)
        if not present or k not in KEY_TYPES:
            continue
        want = KEY_TYPES[k]
        if want == (bool,) and not isinstance(val, bool):
            wrong_type.append({"key": k, "want": "bool", "got": type(val).__name__})
        elif want != (bool,) and isinstance(val, bool):
            wrong_type.append({"key": k, "want": "/".join(t.__name__ for t in want),
                               "got": "bool"})
        elif want == (int,) and isinstance(val, float) and not float(val).is_integer():
            wrong_type.append({"key": k, "want": "int", "got": f"float {val}"})
        elif not isinstance(val, want) and not (want == (int,) and isinstance(val, float)):
            wrong_type.append({"key": k, "want": "/".join(t.__name__ for t in want),
                               "got": type(val).__name__})
    absent_recommended = [k for k in RECOMMENDED_KEYS if not _get(art.manifest, k)[0]]
    ev = {"missing_required": missing, "wrong_type": wrong_type,
          "missing_recommended": absent_recommended,
          "steering_enabled": art.cfg("config", "steering_enabled"),
          "n_required_checked": len(required)}

    if missing or wrong_type:
        bits = []
        if missing:
            bits.append(f"{len(missing)} required manifest key(s) missing: {missing}")
        if wrong_type:
            bits.append(f"{len(wrong_type)} key(s) of the wrong type: {wrong_type}")
        return _res(cid, "T3", "fail", "; ".join(bits), ev,
                    "runner.py must write every field of the manifest schema")
    if absent_recommended:
        return _res(cid, "T3", "warn",
                    f"all {len(required)} required keys present; optional provenance keys "
                    f"missing: {absent_recommended}", ev)
    return _res(cid, "T3", "pass", f"all {len(required)} required manifest keys present", ev)


def _data_module():
    try:
        from .. import data
    except Exception:
        return None
    return data


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _pair_strings(pairs: Any) -> list[tuple[str, str]]:
    """(pair id, text) for every long free-text field of a contrast pair."""
    out: list[tuple[str, str]] = []
    for i, p in enumerate(pairs or []):
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id", i))
        for key in ("positive", "negative", "prompt", "text", "question"):
            v = p.get(key)
            if isinstance(v, str) and len(_norm_text(v)) >= MIN_LEAK_CHARS:
                out.append((f"{pid}.{key}", _norm_text(v)))
    return out


def _item_strings(items: Any) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for i, it in enumerate(items or []):
        if not isinstance(it, dict):
            continue
        iid = str(it.get("id", i))
        for key in ("question", "prompt", "text"):
            v = it.get(key)
            if isinstance(v, str) and len(_norm_text(v)) >= MIN_LEAK_CHARS:
                out.append((f"{iid}.{key}", _norm_text(v)))
    return out


def _pairs_on_disk(art: RunArtifacts) -> tuple[list, str]:
    """A run may ship the contrast set it actually used; prefer it over the library."""
    for name in ("pairs.json", "contrast_pairs.json"):
        p = art.run_dir / name
        if not p.exists():
            continue
        try:
            doc = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(doc, dict):
            doc = doc.get("pairs") or doc.get("contrast_pairs")
        if isinstance(doc, list):
            return doc, name
    return [], ""


@check("t3.leakage", "T3")
def leakage(art: RunArtifacts) -> CheckResult:
    """Vector-building text must not appear in the eval text, on disk or in data.py."""
    cid = "t3.leakage"
    data = _data_module()
    cfg = art.cfg("config", default={}) or {}
    seed = cfg.get("seed", 0) if isinstance(cfg.get("seed", 0), int) else 0

    pairs, pairs_src = _pairs_on_disk(art)
    pairs_from_disk = bool(pairs)
    items: list = []
    lib_note = ""
    if data is not None:
        if not pairs:
            try:
                pairs = data.contrast_pairs(n=cfg.get("n_pairs"), seed=seed,
                                            shuffle_labels=bool(cfg.get("label_shuffled", False)))
                pairs_src = "data.contrast_pairs()"
            except Exception as exc:
                pairs = list(getattr(data, "CONTRAST_PAIRS", []) or [])
                pairs_src = "data.CONTRAST_PAIRS"
                lib_note = f"data.contrast_pairs() raised {type(exc).__name__}: {exc}"
        try:
            items = data.eval_items(n=cfg.get("n_eval"), seed=seed,
                                   pressure=bool(cfg.get("pressure", True)))
        except Exception:
            items = list(getattr(data, "EVAL_ITEMS", []) or [])

    recs = list(art.records or [])
    rec_items = [{"id": _rec_id(r, i), "prompt": r.get("prompt")}
                 for i, r in enumerate(recs) if isinstance(r, dict)]
    if not recs and not pairs_from_disk:
        return _res(cid, "T3", "skip",
                    "this run contributes no text to compare: no records.jsonl and no "
                    "contrast-pair file in the run dir",
                    {"data_importable": data is not None})
    if not pairs:
        return _res(cid, "T3", "skip",
                    "no contrast-pair text available (data.py unimportable and the run ships "
                    "no pairs file)", {"data_importable": data is not None, "note": lib_note})

    ev: dict[str, Any] = {"pairs_source": pairs_src, "n_pairs": len(pairs),
                          "n_library_items": len(items), "n_records": len(rec_items),
                          "note": lib_note}

    # 1. the library's own disjointness contract
    lib_ok, lib_offenders = True, []
    if data is not None and items and hasattr(data, "disjoint"):
        try:
            lib_ok, lib_offenders = data.disjoint(pairs, items)
            lib_ok = bool(lib_ok)
            lib_offenders = [str(x) for x in (lib_offenders or [])]
        except Exception as exc:
            lib_ok, lib_offenders = True, []
            ev["disjoint_error"] = f"data.disjoint raised {type(exc).__name__}: {exc}"
    ev["data_disjoint_ok"] = lib_ok
    ev["data_disjoint_offenders"] = lib_offenders[:MAX_EXAMPLES]

    # 2. the text actually on disk: eval prompts vs the contrast set that was used
    pair_texts = _pair_strings(pairs)
    eval_texts = _item_strings(items) + _item_strings(rec_items)
    overlaps = []
    for eid, etext in eval_texts:
        for pid, ptext in pair_texts:
            if ptext in etext or etext in ptext:
                overlaps.append({"eval": eid, "pair": pid,
                                 "shared": ptext[:120] if ptext in etext else etext[:120]})
                break
    ev["n_overlaps"] = len(overlaps)
    ev["overlaps"] = overlaps[:MAX_EXAMPLES]
    ev["n_texts_compared"] = len(pair_texts) * max(len(eval_texts), 1)

    if not lib_ok or overlaps:
        bits = []
        if overlaps:
            bits.append(f"{len(overlaps)} eval text(s) share wording with the contrast pairs "
                        f"(e.g. {overlaps[0]['eval']} vs {overlaps[0]['pair']})")
        if not lib_ok:
            bits.append(f"data.disjoint() reports overlap: {lib_offenders[:MAX_EXAMPLES]}")
        return _res(cid, "T3", "fail", "; ".join(bits), ev,
                    "the vector was built on text the eval reuses; the effect can be "
                    "memorisation of the eval items")
    if not eval_texts:
        return _res(cid, "T3", "skip", "no eval text to compare against the contrast pairs", ev)
    return _res(cid, "T3", "pass",
                f"{len(pair_texts)} contrast-pair strings disjoint from {len(eval_texts)} "
                "eval strings", ev)


@check("t3.git_recorded", "T3")
def git_recorded(art: RunArtifacts) -> CheckResult:
    """Provenance: a run that cannot name its commit cannot be reproduced."""
    cid = "t3.git_recorded"
    if not art.manifest:
        return _res(cid, "T3", "skip", "no manifest.json in the run dir")
    sha = art.cfg("git_sha")
    dirty = art.cfg("git_dirty")
    ev = {"git_sha": sha, "git_dirty": dirty}
    if not isinstance(sha, str) or not sha.strip():
        return _res(cid, "T3", "fail", "manifest records no git_sha", ev,
                    "record the commit the run was produced from")
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", sha.strip()):
        return _res(cid, "T3", "fail",
                    f"git_sha {sha!r} is not a commit hash", ev,
                    "record the real commit hash, not a placeholder")
    if dirty is None:
        return _res(cid, "T3", "warn",
                    f"git_sha {sha[:12]} recorded but git_dirty is not; the working tree "
                    "state at run time is unknown", ev)
    if dirty:
        return _res(cid, "T3", "warn",
                    f"run was produced from a dirty tree at {sha[:12]}; the recorded commit "
                    "is not the code that ran", ev,
                    "commit before running, or treat this run as unreproducible")
    return _res(cid, "T3", "pass", f"clean tree at {sha[:12]}", ev)


# --------------------------------------------------------------------------
# T4 — claims
# --------------------------------------------------------------------------

def _claims_doc(art: RunArtifacts) -> tuple[Optional[dict], str, str]:
    """(doc, state, message) with state in {ok, absent, malformed}.

    Read defensively off disk: a report that is not valid JSON is a finding, and
    it must arrive as a failed check rather than a traceback.
    """
    path = art.run_dir / "claims.json"
    doc: Any = art.claims if isinstance(art.claims, dict) and art.claims else None
    if doc is None:
        if not path.exists():
            return None, "absent", "no claims.json in the run dir"
        try:
            doc = json.loads(path.read_text())
        except Exception as exc:
            return None, "malformed", f"claims.json is not valid JSON: {type(exc).__name__}: {exc}"
    if not isinstance(doc, dict):
        return None, "malformed", f"claims.json holds a {type(doc).__name__}, not an object"
    claims = doc.get("claims")
    if claims is None:
        return None, "malformed", "claims.json has no 'claims' key"
    if not isinstance(claims, list):
        return None, "malformed", f"claims.json 'claims' is a {type(claims).__name__}, not a list"
    bad = [i for i, c in enumerate(claims) if not isinstance(c, dict)]
    if bad:
        return None, "malformed", f"claims entries at {bad} are not objects"
    return doc, "ok", ""


def _canonical_metric(claim: dict) -> Optional[str]:
    m = claim.get("metric")
    if not isinstance(m, str) or not m.strip():
        return None
    return METRIC_ALIASES.get(m.strip().lower())


def _compute_metric(metric: str, records: list) -> tuple[Optional[float], int, str]:
    """(value, n_used, reason_if_none) — recomputed from records, never read back
    from the manifest."""
    if not records:
        return None, 0, "no records in the arm this claim is about"
    if metric in ("sycophancy_rate", "sycophancy_count"):
        n_syc, n_scored = count_scored(records)
        if n_scored == 0:
            return None, 0, "no record carries score.sycophantic"
        if metric == "sycophancy_count":
            return float(n_syc), n_scored, ""
        return n_syc / n_scored, n_scored, ""
    if metric == "n_records":
        return float(len(records)), len(records), ""
    if metric == "undetermined_rate":
        v = undetermined_rate(records)
        if v is None:
            return None, 0, "no record carries score.detail"
        return v, len(records), ""
    if metric == "mean_new_tokens":
        vals = [r.get("n_new_tokens") for r in records if isinstance(r, dict)]
        vals = [float(v) for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not vals:
            return None, 0, "no record carries n_new_tokens"
        return sum(vals) / len(vals), len(vals), ""
    return None, 0, f"metric {metric!r} has no recomputation rule"


def _alt_rate(metric: str, records: list) -> Optional[float]:
    """The same rate over *all* records rather than the scored ones — reported when
    the two denominators disagree, so a partially scored file is named, not guessed."""
    if metric != "sycophancy_rate" or not records:
        return None
    n_syc, n_scored = count_scored(records)
    if n_scored == 0 or n_scored == len(records):
        return None
    return n_syc / len(records)


def _sibling_dirs(art: RunArtifacts) -> list[Path]:
    try:
        return sorted(p for p in art.run_dir.parent.iterdir() if p.is_dir())
    except Exception:
        return []


def _load_dir(path: Path) -> Optional[RunArtifacts]:
    try:
        return RunArtifacts.load(path)
    except Exception:
        return None


def _baseline_for_claim(art: RunArtifacts, claim: dict) -> tuple[list, str, bool]:
    """Resolve the arm a delta claim measures against, and say whether the run the
    claim *named* is the one that was found.  A claim pointing at a baseline that
    is not on disk is auditing itself against something nobody can see."""
    name = claim.get("baseline_run")
    if isinstance(name, str) and name.strip():
        name = name.strip()
        cand = art.run_dir.parent / name
        if cand.is_dir():
            sib = _load_dir(cand)
            if sib is not None and sib.records:
                recs = [r for r in sib.records
                        if str(as_record(r).get("condition", "")) == "baseline"] or list(sib.records)
                return recs, f"run dir {name!r}", True
        for p in _sibling_dirs(art):
            mp = p / "manifest.json"
            if not mp.exists():
                continue
            try:
                rid = (json.loads(mp.read_text()) or {}).get("run_id")
            except Exception:
                continue
            if rid == name:
                sib = _load_dir(p)
                if sib is not None and sib.records:
                    recs = [r for r in sib.records
                            if str(as_record(r).get("condition", "")) == "baseline"] or list(sib.records)
                    return recs, f"run_id {name!r} in dir {p.name!r}", True
        recs, label = baseline_records(art)
        if recs:
            return recs, f"{label} (claim names baseline_run {name!r}, which is not on disk)", False
        return [], f"baseline_run {name!r} is not on disk and this run has no baseline arm", False
    recs, label = baseline_records(art)
    return recs, label, True


def _claimed(claim: dict, *keys: str) -> tuple[Optional[str], Optional[float]]:
    for k in keys:
        if k in claim:
            v = claim[k]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return k, float(v)
    return None, None


def _comparisons(art: RunArtifacts, claim: dict, metric: str) -> tuple[list[dict], dict]:
    """Every (claimed field, recomputed value) pair this claim exposes."""
    kind = claim.get("kind")
    treat, tlabel = treatment_records(art, condition=claim.get("condition"))
    got_t, n_t, why_t = _compute_metric(metric, treat)
    ctx: dict[str, Any] = {"metric": metric, "treatment_source": tlabel,
                           "n_treatment": len(treat), "n_used": n_t,
                           "recomputed_treatment": num(got_t)}
    rows: list[dict] = []

    if kind == "metric_value":
        field, val = _claimed(claim, "value", "metric_value", "treatment_value", "claimed_value")
        rows.append({"field": field or "value", "claimed": val, "recomputed": got_t,
                     "alt": _alt_rate(metric, treat), "why": why_t, "n": n_t})
        return rows, ctx

    base, blabel, resolved = _baseline_for_claim(art, claim)
    got_b, n_b, why_b = _compute_metric(metric, base)
    ctx.update({"baseline_source": blabel, "n_baseline": len(base),
                "baseline_named": claim.get("baseline_run"), "baseline_resolved": resolved,
                "recomputed_baseline": num(got_b)})
    f_t, v_t = _claimed(claim, "treatment_value", "value", "after", "steered_value")
    f_b, v_b = _claimed(claim, "baseline_value", "before", "baseline")
    rows.append({"field": f_t or "treatment_value", "claimed": v_t, "recomputed": got_t,
                 "alt": _alt_rate(metric, treat), "why": why_t, "n": n_t})
    rows.append({"field": f_b or "baseline_value", "claimed": v_b, "recomputed": got_b,
                 "alt": _alt_rate(metric, base), "why": why_b, "n": n_b})
    f_d, v_d = _claimed(claim, "delta", "difference", "change")
    if f_d is not None:
        got_d = None if (got_t is None or got_b is None) else got_t - got_b
        rows.append({"field": f_d, "claimed": v_d, "recomputed": got_d,
                     "alt": None, "why": why_t or why_b, "n": min(n_t, n_b)})
    ctx["recomputed_delta"] = num(None if (got_t is None or got_b is None) else got_t - got_b)
    return rows, ctx


@check("t4.claims_recomputable", "T4")
def claims_recomputable(art: RunArtifacts) -> CheckResult:
    """Recompute every metric claim from records.jsonl; the match must be exact."""
    cid = "t4.claims_recomputable"
    doc, state, msg = _claims_doc(art)
    if state == "absent":
        return _res(cid, "T4", "skip", msg)
    if state == "malformed":
        return _res(cid, "T4", "fail", msg, {"claims_path": str(art.run_dir / "claims.json")},
                    "claims.json must be {'run_id': ..., 'claims': [ ... ]}")

    claims = doc["claims"]
    metric_claims = [c for c in claims if c.get("kind") in ("metric_value", "metric_delta")]
    if not metric_claims:
        return _res(cid, "T4", "skip",
                    f"claims.json carries {len(claims)} claim(s), none of them recomputable "
                    "metric claims", {"kinds": [c.get("kind") for c in claims]})

    mismatches, verified, unverifiable, warns = [], [], [], []
    malformed_fields: list[dict] = []
    details = []
    for i, claim in enumerate(metric_claims):
        ident = str(claim.get("id") or f"claim[{i}]")
        bad = [{"field": k, "value": repr(claim[k])[:80]}
               for k in ("value", "treatment_value", "baseline_value", "delta")
               if k in claim and (isinstance(claim[k], bool)
                                  or not isinstance(claim[k], (int, float)))]
        if bad:
            malformed_fields.append({"claim": ident, "fields": bad})
        metric = _canonical_metric(claim)
        if metric is None:
            unverifiable.append({"claim": ident, "reason":
                                 f"metric {claim.get('metric')!r} is not derivable from "
                                 "records — see t4.no_unsupported_claims"})
            continue
        rows, ctx = _comparisons(art, claim, metric)
        details.append({"claim": ident, **ctx})
        if ctx.get("baseline_resolved") is False:
            warns.append({"claim": ident, "baseline_named": ctx.get("baseline_named"),
                          "used_instead": ctx.get("baseline_source"),
                          "note": "the baseline run this claim names is not on disk"})
        for row in rows:
            claimed, got, alt = row["claimed"], row["recomputed"], row["alt"]
            if claimed is None:
                continue
            if got is None:
                unverifiable.append({"claim": ident, "field": row["field"],
                                     "claimed": num(claimed),
                                     "reason": row["why"] or "not recomputable from records"})
                continue
            if abs(claimed - got) <= CLAIM_TOL:
                verified.append({"claim": ident, "field": row["field"], "value": num(got)})
            elif alt is not None and abs(claimed - alt) <= CLAIM_TOL:
                warns.append({"claim": ident, "field": row["field"], "claimed": num(claimed),
                              "over_scored_records": num(got), "over_all_records": num(alt),
                              "note": "claim matches only if unscored records count as 0"})
            else:
                mismatches.append({"claim": ident, "field": row["field"],
                                   "claimed": num(claimed), "recomputed": num(got),
                                   "abs_error": num(abs(claimed - got)),
                                   "n": row["n"], "metric": metric,
                                   "statement": str(claim.get("statement", ""))[:200]})

    ev = {"tolerance": CLAIM_TOL, "n_metric_claims": len(metric_claims),
          "mismatches": mismatches, "verified": verified, "malformed_fields": malformed_fields,
          "unverifiable": unverifiable, "warnings": warns, "recomputation": details}
    ev["p_value_notes"] = _p_value_notes(art, metric_claims)

    if mismatches or malformed_fields:
        bits = []
        if mismatches:
            first = mismatches[0]
            bits.append(f"{len(mismatches)} claimed value(s) do not reproduce from "
                        f"records.jsonl: {first['claim']}.{first['field']} claims "
                        f"{first['claimed']} but the records give {first['recomputed']}"
                        + (f" (and {len(mismatches) - 1} more)" if len(mismatches) > 1 else ""))
        if malformed_fields:
            bits.append(f"{len(malformed_fields)} claim(s) state a non-numeric value where a "
                        f"number is required: {malformed_fields[:MAX_EXAMPLES]}")
        return _res(cid, "T4", "fail", "; ".join(bits), ev,
                    "the reported number is fabricated, stale, or not a number; regenerate "
                    "claims.json from the records that exist")
    if not verified:
        return _res(cid, "T4", "skip",
                    f"no metric claim could be recomputed ({len(unverifiable)} unverifiable)", ev)
    if warns or ev["p_value_notes"]:
        return _res(cid, "T4", "warn",
                    f"{len(verified)} claimed value(s) reproduce exactly, with "
                    f"{len(warns) + len(ev['p_value_notes'])} caveat(s)", ev)
    return _res(cid, "T4", "pass",
                f"{len(verified)} claimed value(s) reproduce exactly from records.jsonl "
                f"(tol={CLAIM_TOL:g})", ev)


def _p_value_notes(art: RunArtifacts, claims: list[dict]) -> list[dict]:
    """A claimed p-value cannot be reproduced exactly (the method is not recorded),
    but a claimed significance the paired records do not support is worth saying."""
    notes: list[dict] = []
    claimed_ps = [(str(c.get("id") or "claim"), float(c["p_value"])) for c in claims
                  if isinstance(c.get("p_value"), (int, float)) and not isinstance(c.get("p_value"), bool)]
    if not claimed_ps:
        return notes
    eff = _effect(art)
    if not eff["ok"] or eff["p_value"] is None:
        return notes
    for ident, p in claimed_ps:
        if (p < SIG_ALPHA) != (eff["p_value"] < SIG_ALPHA):
            notes.append({"claim": ident, "claimed_p": num(p),
                          "recomputed_p": num(eff["p_value"]),
                          "source": eff["p_source"], "alpha": SIG_ALPHA,
                          "note": "claimed and recomputed significance disagree"})
    return notes


@check("t4.claim_n_matches_records", "T4")
def claim_n_matches_records(art: RunArtifacts) -> CheckResult:
    """A claim's n must be a count the records can actually supply."""
    cid = "t4.claim_n_matches_records"
    doc, state, msg = _claims_doc(art)
    if state == "absent":
        return _res(cid, "T4", "skip", msg)
    if state == "malformed":
        return _res(cid, "T4", "fail", msg)

    rows, bad = [], []
    for i, claim in enumerate(doc["claims"]):
        n = claim.get("n")
        ident = str(claim.get("id") or f"claim[{i}]")
        if "n" not in claim:
            continue
        if not isinstance(n, (int, float)) or isinstance(n, bool):
            detail = {"claim": ident, "claimed_n": repr(n)[:40], "allowed": [],
                      "problem": "n is not a number"}
            rows.append(detail)
            bad.append(detail)
            continue
        treat, tlabel = treatment_records(art, condition=claim.get("condition"))
        _, n_scored = count_scored(treat)
        allowed = {len(treat), n_scored}
        detail: dict[str, Any] = {"claim": ident, "claimed_n": int(n),
                                  "n_treatment_records": len(treat), "n_scored": n_scored,
                                  "treatment_source": tlabel}
        if claim.get("kind") == "metric_delta":
            base, blabel, resolved = _baseline_for_claim(art, claim)
            a, b, used, info = pair_by_prompt(treat, base)
            allowed |= {len(base), len(used), info["n_matched"]}
            detail.update({"n_baseline_records": len(base), "n_paired": len(used),
                           "baseline_source": blabel, "baseline_resolved": resolved})
        allowed.discard(0)
        detail["allowed"] = sorted(allowed)
        rows.append(detail)
        if int(n) not in allowed:
            bad.append(detail)

    ev = {"claims_with_n": rows, "mismatched": bad}
    if not rows:
        return _res(cid, "T4", "skip", "no claim declares an n", ev)
    if bad:
        first = bad[0]
        return _res(cid, "T4", "fail",
                    f"{len(bad)} claim(s) declare an n the records cannot supply: "
                    f"{first['claim']} claims n={first['claimed_n']}"
                    + (f" ({first['problem']})" if first.get("problem")
                       else f", records offer {first['allowed']}"), ev,
                    "the claim was computed over a different set of records than the one on disk")
    return _res(cid, "T4", "pass", f"n matches the records for all {len(rows)} claim(s)", ev)


@check("t4.no_unsupported_claims", "T4")
def no_unsupported_claims(art: RunArtifacts) -> CheckResult:
    """A claim may only name a metric records.jsonl can actually produce."""
    cid = "t4.no_unsupported_claims"
    doc, state, msg = _claims_doc(art)
    if state == "absent":
        return _res(cid, "T4", "skip", msg)
    if state == "malformed":
        return _res(cid, "T4", "fail", msg)

    claims = doc["claims"]
    if not claims:
        return _res(cid, "T4", "skip", "claims.json declares no claims")

    problems, okd = [], []
    for i, claim in enumerate(claims):
        ident = str(claim.get("id") or f"claim[{i}]")
        kind = claim.get("kind")
        if kind not in CLAIM_KINDS:
            problems.append({"claim": ident, "problem": f"unknown claim kind {kind!r}",
                             "allowed": list(CLAIM_KINDS)})
            continue
        raw = claim.get("metric")
        if kind in ("metric_value", "metric_delta") and not (isinstance(raw, str) and raw.strip()):
            problems.append({"claim": ident,
                             "problem": f"kind={kind} but the claim names no metric"})
            continue
        if not (isinstance(raw, str) and raw.strip()):
            okd.append({"claim": ident, "kind": kind, "metric": None})
            continue
        metric = _canonical_metric(claim)
        if metric is None:
            problems.append({"claim": ident, "problem":
                             f"metric {raw!r} cannot be produced from records.jsonl",
                             "records_can_produce": sorted(set(METRIC_ALIASES.values())),
                             "statement": str(claim.get("statement", ""))[:200]})
            continue
        treat, _ = treatment_records(art, condition=claim.get("condition"))
        val, n_used, why = _compute_metric(metric, treat)
        if val is None:
            problems.append({"claim": ident, "problem":
                             f"metric {raw!r} is known but these records cannot produce it: {why}"})
            continue
        okd.append({"claim": ident, "kind": kind, "metric": metric,
                    "recomputed": num(val), "n": n_used})

    ev = {"supported_metrics": sorted(set(METRIC_ALIASES.values())),
          "unsupported": problems, "supported_claims": okd, "n_claims": len(claims)}
    if problems:
        first = problems[0]
        return _res(cid, "T4", "fail",
                    f"{len(problems)} claim(s) rest on something the records do not contain: "
                    f"{first['claim']}: {first['problem']}", ev,
                    "measure the metric and write it into records.jsonl, or drop the claim")
    return _res(cid, "T4", "pass",
                f"all {len(claims)} claim(s) name metrics the records can produce", ev)


@check("t4.claim_direction_matches", "T4")
def claim_direction_matches(art: RunArtifacts) -> CheckResult:
    """The sign of a claimed delta must be the sign the records produce."""
    cid = "t4.claim_direction_matches"
    doc, state, msg = _claims_doc(art)
    if state == "absent":
        return _res(cid, "T4", "skip", msg)
    if state == "malformed":
        return _res(cid, "T4", "fail", msg)

    deltas = [c for c in doc["claims"] if c.get("kind") == "metric_delta"]
    if not deltas:
        return _res(cid, "T4", "skip", "no metric_delta claim to orient")

    flipped, checked, unverifiable = [], [], []
    for i, claim in enumerate(deltas):
        ident = str(claim.get("id") or f"claim[{i}]")
        metric = _canonical_metric(claim)
        if metric is None:
            unverifiable.append({"claim": ident, "reason": "unsupported metric"})
            continue
        treat, tlabel = treatment_records(art, condition=claim.get("condition"))
        base, blabel, resolved = _baseline_for_claim(art, claim)
        got_t, _, why_t = _compute_metric(metric, treat)
        got_b, _, why_b = _compute_metric(metric, base)
        _, v_t = _claimed(claim, "treatment_value", "value", "after", "steered_value")
        _, v_b = _claimed(claim, "baseline_value", "before", "baseline")
        f_d, v_d = _claimed(claim, "delta", "difference", "change")
        claimed_delta = v_d if f_d is not None else (
            None if (v_t is None or v_b is None) else v_t - v_b)
        if claimed_delta is None:
            unverifiable.append({"claim": ident, "reason": "claim states no delta"})
            continue
        if got_t is None or got_b is None:
            unverifiable.append({"claim": ident,
                                 "reason": why_t or why_b or "delta not recomputable",
                                 "baseline_source": blabel})
            continue
        real_delta = got_t - got_b
        row = {"claim": ident, "metric": metric, "claimed_delta": num(claimed_delta),
               "recomputed_delta": num(real_delta), "claimed_treatment": num(v_t),
               "claimed_baseline": num(v_b), "recomputed_treatment": num(got_t),
               "recomputed_baseline": num(got_b), "treatment_source": tlabel,
               "baseline_source": blabel, "baseline_resolved": resolved,
               "statement": str(claim.get("statement", ""))[:200]}
        checked.append(row)
        if abs(claimed_delta) <= CLAIM_TOL:
            continue
        if abs(real_delta) <= CLAIM_TOL:
            row["problem"] = "the records show no change at all"
            flipped.append(row)
        elif (claimed_delta > 0) != (real_delta > 0):
            row["problem"] = "claimed and recomputed deltas point in opposite directions"
            flipped.append(row)

    ev = {"checked": checked, "flipped": flipped, "unverifiable": unverifiable}
    if flipped:
        first = flipped[0]
        return _res(cid, "T4", "fail",
                    f"{len(flipped)} claim(s) point the wrong way: {first['claim']} claims "
                    f"{first['claimed_delta']:+.6g} but the records give "
                    f"{first['recomputed_delta']:+.6g} ({first['problem']})", ev,
                    "the reported direction of the effect is not the one in the records")
    if not checked:
        return _res(cid, "T4", "skip",
                    f"no metric_delta claim could be oriented ({len(unverifiable)} unverifiable)",
                    ev)
    return _res(cid, "T4", "pass",
                f"all {len(checked)} claimed delta(s) match the recomputed direction", ev)
