"""Execute one steering condition end to end and leave a run directory the
checks can audit.

Nothing written here is allowed to be a self-report the harness could recompute:
file hashes are taken from the bytes after they land on disk, hook fires are read
off the live hook object, timings off a clock, vector norm/dim off the array.
`manifest.json` is written LAST, so a crashed run leaves an obviously incomplete
directory rather than a plausible-looking lie.

The baseline (`steering_enabled=False`) and the alpha=0 control go down exactly
the same path as a live steered run — that is the whole point of the alpha_zero
companion, which is only evidence if nothing about it was special-cased.
"""
from __future__ import annotations

import contextlib
import json
import math
import random
import subprocess
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from .types import sha256_bytes, sha256_file, sha256_text

SCHEMA_VERSION = "1"

#: manifest["config"] is exactly these keys, in this order.
CONFIG_KEYS = (
    "model_id", "dtype", "device", "layer", "alpha", "seed", "max_new_tokens",
    "n_eval", "n_pairs", "steering_enabled", "label_shuffled", "vector_source",
    "pressure",
)
VECTOR_SOURCES = ("contrast_pairs_v1", "random_direction", "external")
COMPANION_ROLES = ("baseline", "alpha_zero", "sign_flip", "shuffled",
                   "random_direction", "replay")

# One sequence per forward pass while the hook is attached.  Two reasons: it
# makes hook.fires_expected exact whatever batching steering.py chose, and it
# removes pad-token effects that would otherwise stop alpha_zero from
# reproducing the baseline bit for bit.  Vector extraction runs unhooked, so it
# can batch.
ACT_BATCH_SIZE = 1
GEN_BATCH_SIZE = 1
PAIR_BATCH_SIZE = 8


@dataclass
class RunConfig:
    """The manifest["config"] block, plus where to write and who the siblings are.

    `companions` maps role -> sibling directory name under `out_dir`; the extra
    role "vector_from" names where an `external` vector is loaded from.
    """

    run_id: str
    out_dir: str | Path = "outputs"
    model_id: str = "Qwen/Qwen3-1.7B"
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    layer: int = 14
    alpha: float = 1.0
    seed: int = 0
    max_new_tokens: int = 48
    n_eval: int = 40
    n_pairs: int = 64
    steering_enabled: bool = True
    label_shuffled: bool = False
    vector_source: str = "contrast_pairs_v1"
    pressure: bool = True
    companions: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required")
        if self.vector_source not in VECTOR_SOURCES:
            raise ValueError(f"vector_source must be one of {VECTOR_SOURCES}")
        if self.layer < 0:
            raise ValueError("layer must be >= 0")
        if self.n_eval <= 0 or self.n_pairs <= 0 or self.max_new_tokens <= 0:
            raise ValueError("n_eval, n_pairs and max_new_tokens must be positive")
        self.alpha = float(self.alpha)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.run_id

    def config_block(self) -> dict[str, Any]:
        """The dict that goes to manifest["config"] — schema keys only."""
        raw = {
            "model_id": str(self.model_id), "dtype": str(self.dtype),
            "device": str(self.device), "layer": int(self.layer),
            "alpha": float(self.alpha), "seed": int(self.seed),
            "max_new_tokens": int(self.max_new_tokens), "n_eval": int(self.n_eval),
            "n_pairs": int(self.n_pairs),
            "steering_enabled": bool(self.steering_enabled),
            "label_shuffled": bool(self.label_shuffled),
            "vector_source": str(self.vector_source), "pressure": bool(self.pressure),
        }
        return {k: raw[k] for k in CONFIG_KEYS}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(obj: Any) -> Any:
    """Last-ditch coercion so a numpy scalar wandering in from another module
    cannot kill the manifest write after the GPU work is already done."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serialisable")


def _git_info() -> tuple[str, bool]:
    """git sha + dirty flag of the harness itself; empty sha if git says nothing."""
    root = Path(__file__).resolve().parents[1]
    def _git(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(("git", *args), cwd=root, capture_output=True,
                                 text=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    sha = _git("rev-parse", "HEAD") or ""
    status = _git("status", "--porcelain")
    return sha, bool(status)


def _render_prompt(item: dict[str, Any], pressure: bool) -> str:
    """The exact user turn the model sees; the wrong position lives in the
    pressure prefix, so dropping it is the no-pressure ablation."""
    if item.get("prompt"):
        return str(item["prompt"])
    parts = []
    if pressure and item.get("pressure_prefix"):
        parts.append(str(item["pressure_prefix"]).strip())
    parts.append(str(item.get("question", "")).strip())
    return " ".join(p for p in parts if p)


def _n_layers(model) -> int:
    n = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if isinstance(n, int) and n > 0:
        return n
    for path in ("model.layers", "layers", "transformer.h", "model.language_model.layers"):
        obj: Any = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            try:
                return len(obj)
            except TypeError:
                pass
    raise RuntimeError("cannot determine the model's layer count")


def _capture_layers(n_layers: int, target: int) -> list[int]:
    """A spread of layers that always brackets the steered one — T1 needs at
    least one layer below L (must be untouched) and one above (must change)."""
    cand = {0, n_layers - 1, target, max(0, target - 1), min(n_layers - 1, target + 1)}
    for frac in (0.25, 0.5, 0.75):
        cand.add(int(round(frac * (n_layers - 1))))
    return sorted(c for c in cand if 0 <= c < n_layers)


def _n_batches(n: int, batch_size: int) -> int:
    return math.ceil(n / max(1, batch_size)) if n > 0 else 0


def _expected_generate_fires(n_new: list[int], batch_size: int) -> int:
    """Forward passes generation schedules through the hooked block: one per
    decode step, and a batch keeps stepping until its longest member is done."""
    bs = max(1, batch_size)
    total = 0
    for i in range(0, len(n_new), bs):
        chunk = [int(x) for x in n_new[i:i + bs]]
        total += max(chunk) if chunk else 0
    return total


def _seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # greedy decoding is only reproducible if the kernels are too
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@contextlib.contextmanager
def _attached(hook, module) -> Iterator[Any]:
    """Hold the steering hook on the block output for one phase; a None hook is
    the unsteered path and must stay the same shape of call."""
    if hook is None:
        yield None
    else:
        with hook.attach(module) as h:
            yield h


def _fires(hook) -> int:
    return int(getattr(hook, "fires", 0)) if hook is not None else 0


def _sync(device: str) -> None:
    import torch
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_vram(device: str) -> int:
    import torch
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated(device))


def _external_vector_path(cfg: RunConfig) -> Path:
    ref = cfg.companions.get("vector_from")
    if not ref:
        raise ValueError("vector_source='external' needs companions['vector_from'] "
                         "naming a sibling run dir or an .npz path")
    p = Path(ref)
    if not p.is_absolute():
        p = cfg.run_dir.parent / p
    return p / "vector.npz" if p.is_dir() else p


def _load_vector_file(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path) as z:
        if "v" not in z:
            raise ValueError(f"{path} has no key 'v'")
        v = np.asarray(z["v"], dtype=np.float32).reshape(-1)
    return v, {"n_pairs": 0, "source_path": str(path)}


def _build_vector(cfg: RunConfig, model, tok, pairs, steering) -> tuple[np.ndarray, dict[str, Any]]:
    """The vector this run steers with.  Every source ends in the same place: a
    float32 [d_model] array plus whatever metadata the builder can vouch for."""
    external = cfg.vector_source == "external" or (
        cfg.vector_source == "random_direction" and cfg.companions.get("vector_from"))
    if external:
        v, meta = _load_vector_file(_external_vector_path(cfg))
    else:
        v, raw_meta = steering.extract_vector(model, tok, pairs, cfg.layer,
                                              batch_size=PAIR_BATCH_SIZE)
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        meta = dict(raw_meta or {})

    if cfg.vector_source == "random_direction":
        # A null is only a null if it is the same size as the thing it stands in
        # for, so match the real vector's norm and throw the direction away.
        rng = np.random.default_rng(cfg.seed)
        r = rng.standard_normal(v.shape[0]).astype(np.float32)
        norm = float(np.linalg.norm(r))
        if norm == 0.0:
            raise RuntimeError("degenerate random direction")
        matched = float(np.linalg.norm(v))
        v = (r / norm * matched).astype(np.float32)
        meta = dict(meta)
        meta["method"] = "random_direction_norm_matched"
        meta["matched_norm"] = matched
    return v, meta


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def run(cfg: RunConfig) -> Path:
    """Execute one condition and write a run dir that satisfies the manifest
    schema field for field.  Returns the directory."""
    # imported here so `import agentverify.runner` costs nothing on a CPU box
    import torch
    from . import data, env, scoring, steering

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    # a stale manifest from a previous crashed attempt would describe files that
    # no longer exist, so drop it before anything else is written
    (run_dir / "manifest.json").unlink(missing_ok=True)

    _seed_everything(cfg.seed)

    # Raw device throughput, measured on an idle card before the model owns it.
    # A measurement we could not take reads as "not fast" (0.0), never as
    # "assume it was fine" — t0.no_cpu_fallback should fire, not shrug.
    try:
        tflops = float(env.matmul_tflops(device=cfg.device))
    except Exception:
        tflops = 0.0
    if str(cfg.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t = time.perf_counter()
    model, tok = steering.load_model(cfg.model_id, dtype=cfg.dtype, device=cfg.device)
    _sync(cfg.device)
    load_s = time.perf_counter() - t

    n_layers = _n_layers(model)
    if not 0 <= cfg.layer < n_layers:
        raise ValueError(f"layer {cfg.layer} outside 0..{n_layers - 1}")
    module, module_path = steering.resolve_layer_module(model, cfg.layer)
    # resolved up front: a convention mismatch should stop the run before the
    # expensive part, and t1.layer_index_convention re-derives it from `layer`
    hs_index = int(steering.hidden_state_index(cfg.layer))
    layers = _capture_layers(n_layers, cfg.layer)

    pairs = data.contrast_pairs(n=cfg.n_pairs, seed=cfg.seed,
                                shuffle_labels=cfg.label_shuffled)
    items = data.eval_items(n=cfg.n_eval, seed=cfg.seed, pressure=cfg.pressure)
    prompts = [_render_prompt(it, cfg.pressure) for it in items]

    t = time.perf_counter()
    vec, vmeta = _build_vector(cfg, model, tok, pairs, steering)
    per_pair = vmeta.pop("per_pair", None)

    # Baseline and alpha=0 take this path unchanged: the hook is the only branch,
    # and both captures run the same code so their agreement is measured, not
    # assumed.
    hook = steering.SteeringHook(vec, cfg.alpha) if cfg.steering_enabled else None

    _seed_everything(cfg.seed)
    base_acts = np.asarray(
        steering.extract_activations(model, tok, prompts, layers, batch_size=ACT_BATCH_SIZE),
        dtype=np.float32)
    with _attached(hook, module):
        f0 = _fires(hook)
        steered_acts = np.asarray(
            steering.extract_activations(model, tok, prompts, layers,
                                         batch_size=ACT_BATCH_SIZE),
            dtype=np.float32)
        _sync(cfg.device)
        f1 = _fires(hook)
        extract_s = time.perf_counter() - t

        t = time.perf_counter()
        gen = list(steering.generate(model, tok, prompts,
                                     max_new_tokens=cfg.max_new_tokens, seed=cfg.seed,
                                     hook=hook, batch_size=GEN_BATCH_SIZE))
        _sync(cfg.device)
        generate_s = time.perf_counter() - t
        f2 = _fires(hook)
    fires_capture, fires_generate = f1 - f0, f2 - f1

    placement = dict(env.placement(model))
    placement["matmul_tflops"] = tflops
    placement.setdefault("peak_vram_bytes", _peak_vram(cfg.device))

    if len(gen) != len(prompts):
        raise RuntimeError(f"generate returned {len(gen)} of {len(prompts)} completions")
    for name, arr in (("base", base_acts), ("steered", steered_acts)):
        if arr.ndim != 3 or arr.shape[0] != len(layers) or arr.shape[1] != len(prompts):
            raise RuntimeError(f"{name} activations have shape {arr.shape}, expected "
                               f"[{len(layers)}, {len(prompts)}, d_model]")

    layer_row = layers.index(cfg.layer)
    condition = "steered" if cfg.steering_enabled else "baseline"
    records: list[dict[str, Any]] = []
    total_new = 0
    for i, (item, prompt, g) in enumerate(zip(items, prompts, gen)):
        completion = str(g.get("completion", ""))
        act = np.ascontiguousarray(steered_acts[layer_row, i], dtype=np.float32)
        n_new = int(g.get("n_new_tokens", 0))
        total_new += n_new
        records.append({
            "idx": i,
            "condition": condition,
            "prompt_id": str(item.get("id", f"eval-{i}")),
            "prompt": prompt,
            "pressure": bool(cfg.pressure),
            "user_position": item.get("user_position", "wrong"),
            "completion": completion,
            "completion_sha256": sha256_text(completion),
            "n_new_tokens": n_new,
            # absence of the flag is not evidence of finiteness
            "finite_logits": bool(g.get("finite_logits", False)),
            "score": scoring.score_completion(completion, item),
            "act_layer": int(cfg.layer),
            "act_sha256": sha256_bytes(act.tobytes()),
            "act_norm": float(np.linalg.norm(act)),
        })

    records_path = run_dir / "records.jsonl"
    with open(records_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, default=_jsonable) + "\n")

    vector_path = run_dir / "vector.npz"
    if per_pair is not None:
        np.savez(vector_path, v=vec, per_pair=np.asarray(per_pair, dtype=np.float32))
    else:
        np.savez(vector_path, v=vec)

    acts_path = run_dir / "acts.npz"
    np.savez(acts_path, layers=np.asarray(layers, dtype=np.int64),
             base=base_acts, steered=steered_acts)

    # hashes of the bytes that are actually on disk, taken after writing them
    hashes = {p.name: sha256_file(p) for p in (records_path, vector_path, acts_path)}

    vector_block = dict(vmeta)
    vector_block.update({
        "path": "vector.npz", "key": "v", "layer": int(cfg.layer),
        "dim": int(vec.shape[0]), "norm": float(np.linalg.norm(vec)),
        "sha256": hashes["vector.npz"],
        "n_pairs": int(vmeta.get("n_pairs", len(pairs))),
        "dtype": str(vec.dtype), "finite": bool(np.all(np.isfinite(vec))),
    })

    exp_capture = _n_batches(len(prompts), ACT_BATCH_SIZE) if hook is not None else 0
    exp_generate = (_expected_generate_fires([r["n_new_tokens"] for r in records],
                                             GEN_BATCH_SIZE)
                    if hook is not None else 0)
    git_sha, git_dirty = _git_info()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": cfg.run_id,
        "created_utc": _utc_now(),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "config": cfg.config_block(),
        "env": dict(env.fingerprint()),
        "placement": placement,
        "timing": {
            "load_s": round(load_s, 4), "extract_s": round(extract_s, 4),
            "generate_s": round(generate_s, 4),
            "tokens_per_s": round(total_new / generate_s, 4) if generate_s > 0 else 0.0,
        },
        "vector": vector_block,
        "hook": {
            "module_path": module_path,
            # expected = forward passes this run scheduled through the block:
            # one per captured sequence, plus one per decode step of each batch
            "fires_expected": exp_capture + exp_generate,
            "fires_observed": fires_capture + fires_generate,
            "fires_expected_capture": exp_capture,
            "fires_expected_generate": exp_generate,
            "fires_observed_capture": fires_capture,
            "fires_observed_generate": fires_generate,
            "attached": hook is not None,
            "alpha": float(cfg.alpha),
        },
        "artifacts": {"records": "records.jsonl", "acts": "acts.npz",
                      "vector": "vector.npz"},
        "hashes": hashes,
        "counts": {"n_eval": len(items), "n_records": len(records),
                   "n_pairs": len(pairs)},
        "companions": dict(cfg.companions),
        "metrics": {"sycophancy_rate": float(scoring.sycophancy_rate(records)),
                    "n": len(records)},
        "acts": {"layers": [int(x) for x in layers],
                 "d_model": int(base_acts.shape[2]), "position": "last_prompt_token",
                 "hidden_state_index": hs_index, "n_layers": int(n_layers),
                 "capture_batch_size": ACT_BATCH_SIZE},
    }

    # LAST: a manifest that exists means every file it hashes is already final
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_jsonable) + "\n", encoding="utf-8")
    return run_dir


def write_claims(run_dir: str | Path, claims: list[dict]) -> Path:
    """Record what the run asserts — the report T4 then tries to knock down."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name
    mp = run_dir / "manifest.json"
    if mp.exists():
        try:
            run_id = json.loads(mp.read_text()).get("run_id") or run_id
        except (json.JSONDecodeError, OSError):
            pass
    path = run_dir / "claims.json"
    path.write_text(json.dumps({"run_id": run_id, "claims": [dict(c) for c in claims]},
                               indent=2, default=_jsonable) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# companion set
# --------------------------------------------------------------------------

def companion_configs(cfg: RunConfig) -> dict[str, RunConfig]:
    """The six standard controls for `cfg`, as RunConfigs — not executed here.

    Each differs from the treatment in exactly one thing, so a difference in the
    result has exactly one explanation.
    """
    stem = cfg.run_id[:-len("-steered")] if cfg.run_id.endswith("-steered") else cfg.run_id
    ids = {role: f"{stem}-{role}" for role in COMPANION_ROLES}

    def _comp(extra: bool = True) -> dict[str, str]:
        c = {"baseline": ids["baseline"]} if extra else {}
        # an external vector has to stay the same vector across the whole set
        if cfg.companions.get("vector_from"):
            c["vector_from"] = cfg.companions["vector_from"]
        return c

    return {
        "baseline": replace(cfg, run_id=ids["baseline"], steering_enabled=False,
                            alpha=0.0, companions=_comp(extra=False)),
        "alpha_zero": replace(cfg, run_id=ids["alpha_zero"], steering_enabled=True,
                              alpha=0.0, companions=_comp()),
        "sign_flip": replace(cfg, run_id=ids["sign_flip"], alpha=-float(cfg.alpha),
                             companions=_comp()),
        "shuffled": replace(cfg, run_id=ids["shuffled"], label_shuffled=True,
                            companions=_comp()),
        "random_direction": replace(cfg, run_id=ids["random_direction"],
                                    vector_source="random_direction",
                                    companions=_comp()),
        # identical config, different directory: same seed must give same text
        "replay": replace(cfg, run_id=ids["replay"], companions=_comp()),
    }


def plan_run_set(cfg: RunConfig) -> dict[str, RunConfig]:
    """`cfg` wired to its controls, plus the controls: role -> RunConfig.

    A driver executes these with `run()` in any order; nothing runs here.
    """
    comps = companion_configs(cfg)
    primary = replace(cfg, companions={**cfg.companions,
                                       **{role: c.run_id for role, c in comps.items()}})
    return {"primary": primary, **comps}
