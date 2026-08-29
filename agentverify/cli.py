"""Command line: `python -m agentverify {preflight,run,verify,report}`.

Exit codes are the point of the tool, so they are fixed: 0 = verified,
1 = a check failed or errored, or nothing could be decided at all,
2 = the command could not be carried out (bad arguments, missing run
directory, a module that is not built yet, a fault in the harness itself).

Nothing else may exit 1: a harness that crashes must not be mistakable for a
run that failed verification, so every unexpected exception exits 2 with its
traceback.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Optional, Sequence

from . import report
from .types import RunArtifacts, VerifyReport

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2

_TIER_CHOICES = ("T0", "T1", "T2", "T3", "T4")

# The manifest["config"] keys settable from the command line, with their types.
_CONFIG_FLAGS: dict[str, type] = {
    "model_id": str,
    "dtype": str,
    "device": str,
    "layer": int,
    "alpha": float,
    "seed": int,
    "max_new_tokens": int,
    "n_eval": int,
    "n_pairs": int,
    "vector_source": str,
}
_BOOL_FLAGS = ("steering_enabled", "label_shuffled", "pressure")


class CliError(Exception):
    """Something the user must fix before the command can run at all."""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _load_run(run_dir: str | Path) -> RunArtifacts:
    """Refuse to 'verify' a directory that holds no run — silence must not read
    as a pass."""
    p = Path(run_dir)
    if not p.is_dir():
        raise CliError(f"no such run directory: {p}")
    if not (p / "manifest.json").is_file():
        raise CliError(f"{p} has no manifest.json — not a run directory")
    try:
        return RunArtifacts.load(p)
    except json.JSONDecodeError as exc:
        raise CliError(f"{p}: unreadable JSON artifact: {exc}") from exc


def _render(rep: VerifyReport, as_json: bool) -> str:
    return report.to_json(rep) if as_json else report.to_markdown(rep)


def _emit(text: str, out: Optional[str]) -> None:
    if out:
        Path(out).write_text(text)
        print(f"wrote {out}", file=sys.stderr)
    else:
        print(text)


def _import(name: str):
    """Import a sibling module, turning 'not built yet' into a usage error."""
    import importlib
    try:
        return importlib.import_module(f"{__package__}.{name}")
    except ImportError as exc:
        raise CliError(f"cannot import agentverify.{name}: {exc}") from exc


def _read_config_file(path: str) -> dict[str, Any]:
    """Accept either a flat mapping of RunConfig fields or a manifest-shaped
    object with the run config nested under "config"."""
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError as exc:
        raise CliError(f"no such config file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CliError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise CliError(f"{path}: expected a JSON object, got {type(data).__name__}")
    nested = data.get("config")
    if isinstance(nested, dict):
        # Manifest-shaped: config lives under "config" and the rest of the file
        # is what the *previous* run recorded (env, placement, hashes, ...).
        # Only the run-level keys carry over, so that re-running a finished run
        # — `--config <old>/manifest.json`, the `replay` companion — works.
        # `run_id` is deliberately NOT carried over: reusing a finished run's id
        # would overwrite the very run being replayed.  Pass --run-id.
        cfg = dict(nested)
        for key in ("out_dir", "companions"):
            if key in data:
                cfg[key] = data[key]
        return cfg
    return dict(data)


def _parse_companions(pairs: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise CliError(f"--companion expects ROLE=DIR, got {item!r}")
        role, _, directory = item.partition("=")
        role, directory = role.strip(), directory.strip()
        if not role or not directory:
            raise CliError(f"--companion expects ROLE=DIR, got {item!r}")
        out[role] = directory
    return out


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------

def _verdict_exit(rep: VerifyReport) -> int:
    """0 only for a report that actually decided something and liked it.

    An all-skip report is not a pass: it is a report that verified nothing, and
    a green exit code on it is precisely the silence this tool exists to break.
    """
    return EXIT_OK if report.verdict(rep) == "PASS" else EXIT_FAILED


def cmd_preflight(args: argparse.Namespace) -> int:
    env = _import("env")
    results = list(env.preflight(strict=args.strict))
    rep = VerifyReport(run_id="preflight", results=results)
    if args.strict:
        rep = report.apply_strict(rep)
    _emit(_render(rep, args.json), args.out)
    return _verdict_exit(rep)


def cmd_run(args: argparse.Namespace) -> int:
    runner = _import("runner")
    cfg: dict[str, Any] = {}
    if args.config:
        cfg.update(_read_config_file(args.config))

    for name in list(_CONFIG_FLAGS) + list(_BOOL_FLAGS) + ["run_id", "out_dir"]:
        value = getattr(args, name, None)
        if value is not None:
            cfg[name] = value
    if args.companion:
        companions = dict(cfg.get("companions") or {})
        companions.update(_parse_companions(args.companion))
        cfg["companions"] = companions

    fields = {f.name for f in dataclasses.fields(runner.RunConfig)}
    unknown = sorted(set(cfg) - fields)
    if unknown:
        raise CliError(
            f"RunConfig has no field(s) {unknown}; known fields: {sorted(fields)}")
    try:
        run_cfg = runner.RunConfig(**cfg)
    except TypeError as exc:  # missing required field
        raise CliError(f"incomplete run config: {exc}") from exc

    run_dir = Path(runner.run(run_cfg))
    if args.claims:
        claims = json.loads(Path(args.claims).read_text())
        if isinstance(claims, dict):
            claims = claims.get("claims", [])
        runner.write_claims(run_dir, list(claims))
    print(run_dir)
    return EXIT_OK


def cmd_verify(args: argparse.Namespace) -> int:
    art = _load_run(args.run)
    rep = report.run_checks(art, only_tiers=args.tier or None)
    if args.strict:
        rep = report.apply_strict(rep)
    _emit(_render(rep, args.json), args.out)
    return _verdict_exit(rep)


def cmd_report(args: argparse.Namespace) -> int:
    """Same checks as `verify`, but this command only formats — the exit code
    reports whether the report was produced, not whether the run is sound."""
    art = _load_run(args.run)
    rep = report.run_checks(art, only_tiers=args.tier or None)
    if args.strict:
        rep = report.apply_strict(rep)
    _emit(_render(rep, args.format == "json"), args.out)
    return EXIT_OK


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m agentverify",
        description="Verify a steering-vector run against the artifacts it left on disk.",
        epilog="exit codes: 0 verified, 1 a check failed or errored or nothing could be "
               "decided, 2 the command could not run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="{preflight,run,verify,report}")

    pre = sub.add_parser(
        "preflight", help="check this machine before spending GPU time on a run",
        description="Run the T0 environment checks against the live interpreter "
                    "(CUDA present, torch build matches the driver, offline mode).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pre.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True,
                     help="strict preflight: warnings count as failures")
    pre.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    pre.add_argument("--out", metavar="FILE", help="write the report to FILE instead of stdout")
    pre.set_defaults(func=cmd_preflight)

    run = sub.add_parser(
        "run", help="execute a run and write its artifacts",
        description="Execute a steering run via runner.run and write "
                    "manifest.json / records.jsonl / *.npz into the run directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    run.add_argument("--config", metavar="FILE",
                     help="JSON file of RunConfig fields; a finished run's manifest.json "
                          "is accepted too (its \"config\" block, out_dir and companions "
                          "are used, its run_id is not — pass --run-id). "
                          "Explicit flags below override it.")
    run.add_argument("--run-id", dest="run_id", help="run id, also the run directory name")
    run.add_argument("--out-dir", dest="out_dir", help="parent directory for the run directory")
    run.add_argument("--model-id", dest="model_id", help="HF model id (must be in the local cache)")
    run.add_argument("--dtype", help="bfloat16 | float16 | float32")
    run.add_argument("--device", help="cuda:0 | cpu")
    run.add_argument("--layer", type=int, help="block index to read the vector from and steer at")
    run.add_argument("--alpha", type=float, help="steering coefficient")
    run.add_argument("--seed", type=int, help="seed for torch, numpy and random")
    run.add_argument("--max-new-tokens", dest="max_new_tokens", type=int)
    run.add_argument("--n-eval", dest="n_eval", type=int, help="number of eval items")
    run.add_argument("--n-pairs", dest="n_pairs", type=int, help="number of contrast pairs")
    run.add_argument("--vector-source", dest="vector_source",
                     choices=("contrast_pairs_v1", "random_direction", "external"))
    run.add_argument("--steering-enabled", dest="steering_enabled",
                     action=argparse.BooleanOptionalAction, default=None,
                     help="attach the steering hook (--no-steering-enabled for a baseline run)")
    run.add_argument("--label-shuffled", dest="label_shuffled",
                     action=argparse.BooleanOptionalAction, default=None,
                     help="permute the contrast-pair labels: the null control")
    run.add_argument("--pressure", action=argparse.BooleanOptionalAction, default=None,
                     help="prefix eval items with the user's wrong position")
    run.add_argument("--companion", action="append", metavar="ROLE=DIR", default=[],
                     help="declare a companion run, e.g. baseline=A-baseline (repeatable)")
    run.add_argument("--claims", metavar="FILE",
                     help="JSON list of claims (or an object with a \"claims\" key) to "
                          "write as claims.json after the run")
    run.set_defaults(func=cmd_run)

    ver = sub.add_parser(
        "verify", help="run the checks over a run directory and set the exit code",
        description="Re-derive every check from the artifacts in DIR. "
                    "Exits 1 if any check failed or errored, and also if no check "
                    "reached a verdict — an all-skip run is not a pass.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ver.add_argument("--run", metavar="DIR", required=True, help="run directory to verify")
    ver.add_argument("--json", action="store_true",
                     help="emit the report as JSON (VerifyReport.to_dict() plus "
                          "\"verdict\") instead of markdown")
    ver.add_argument("--strict", action="store_true", help="promote every warn to fail")
    ver.add_argument("--tier", action="append", choices=_TIER_CHOICES, default=[],
                     help="restrict to one tier (repeatable); default is every tier")
    ver.add_argument("--out", metavar="FILE", help="write the report to FILE instead of stdout")
    ver.set_defaults(func=cmd_verify)

    rep = sub.add_parser(
        "report", help="render a run's report without setting a verdict exit code",
        description="Same checks as verify, rendered for reading. Always exits 0 "
                    "when the report could be produced.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    rep.add_argument("--run", metavar="DIR", required=True, help="run directory to report on")
    rep.add_argument("--format", choices=("markdown", "json"), default="markdown")
    rep.add_argument("--strict", action="store_true", help="promote every warn to fail")
    rep.add_argument("--tier", action="append", choices=_TIER_CHOICES, default=[],
                     help="restrict to one tier (repeatable); default is every tier")
    rep.add_argument("--out", metavar="FILE", help="write the report to FILE instead of stdout")
    rep.set_defaults(func=cmd_report)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except CliError as exc:
        print(f"agentverify: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except (OSError, ValueError) as exc:  # ValueError covers JSONDecodeError
        print(f"agentverify: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except Exception:
        # A crash in the harness must never land on exit 1, which is reserved
        # for "the run failed verification".  Say so, loudly, and exit 2.
        traceback.print_exc()
        print("agentverify: the harness itself crashed — no verdict was reached",
              file=sys.stderr)
        return EXIT_USAGE
