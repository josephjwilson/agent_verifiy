"""Environment fingerprint, device placement, and the T0 preflight.

This module exists because of one specific silent failure on this box: a torch
wheel built for cu130 on a driver-570 host reports ``cuda_available == False``,
every tensor quietly stays on the CPU, nothing raises, and the experiment is
~100x slow but still produces plausible-looking artifacts.  So nothing here
trusts a self-report it can measure: the driver number is read from the machine,
the matmul rate is timed, and parameter placement is walked tensor by tensor.

The T0 verdicts are pure functions of manifest blocks (``t0_results``) so that
``preflight`` on a live machine and the artifact checks in ``checks/environment``
apply the same rules to the same evidence.
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import torch

from .types import CheckResult

# A 4090 does ~170 TFLOP/s of bf16 dense matmul; a CPU does single-digit at
# best.  Anything under this on a machine that claims a GPU is the fallback.
CPU_FALLBACK_TFLOPS = 20.0

# Greedy decoding of a ~2B model on this card runs tens of tokens/s; on the CPU
# it is ~1.  A recorded rate under this floor is the same failure seen downstream.
TOKENS_PER_S_FLOOR = 5.0

T0_CHECK_IDS = (
    "t0.cuda_available",
    "t0.torch_build_matches_driver",
    "t0.params_on_device",
    "t0.dtype_as_configured",
    "t0.no_cpu_fallback",
    "t0.offline_mode",
    "t0.versions_recorded",
)

# Minimum NVIDIA driver *major* number for a given torch CUDA build tag.  CUDA
# 12.x is minor-version compatible from 525 up; 13.x moved the floor to 580,
# which is exactly why cu130 dies on this 570 host.
DRIVER_MIN_BY_TAG: dict[str, int] = {
    "cu118": 520, "cu121": 525, "cu124": 525, "cu126": 525, "cu128": 525,
    "cu130": 580,
}
DRIVER_MIN_BY_MAJOR: dict[int, int] = {11: 450, 12: 525, 13: 580}

_STATUS_RANK = {"skip": 0, "pass": 1, "warn": 2, "fail": 3, "error": 4}
_TRUTHY = {"1", "true", "yes", "on", "y", "t"}

_DTYPE_ALIASES = {
    "bf16": "torch.bfloat16", "bfloat16": "torch.bfloat16",
    "fp16": "torch.float16", "float16": "torch.float16", "half": "torch.float16",
    "fp32": "torch.float32", "float32": "torch.float32", "float": "torch.float32",
    "fp64": "torch.float64", "float64": "torch.float64", "double": "torch.float64",
    "int8": "torch.int8", "fp8": "torch.float8_e4m3fn",
}


# --------------------------------------------------------------------------
# parsing / probing
# --------------------------------------------------------------------------

def parse_torch_build(version: Optional[str]) -> dict[str, Any]:
    """Split ``2.12.0+cu126`` into its release and its CUDA build tag."""
    out: dict[str, Any] = {"version": version or "", "release": "", "cuda_tag": None,
                           "cuda": None, "cpu_only": False, "rocm": False}
    if not version:
        return out
    out["release"] = version.split("+", 1)[0]
    local = version.split("+", 1)[1] if "+" in version else ""
    m = re.search(r"cu(\d{2,4})", local) or re.search(r"cu(\d{2,4})", version)
    if m:
        digits = m.group(1)
        out["cuda_tag"] = f"cu{digits}"
        # torch tags are two-digit major + the rest: cu126 -> (12, 6), cu92 -> (9, 2).
        out["cuda"] = ((int(digits[0]), int(digits[1])) if len(digits) == 2
                       else (int(digits[:2]), int(digits[2:])))
    if "rocm" in local or "hip" in local:
        out["rocm"] = True
    if not out["cuda_tag"] and not out["rocm"]:
        out["cpu_only"] = ("cpu" in local) or (local == "")
    return out


def parse_driver_version(value: Any) -> Optional[tuple[int, ...]]:
    """``"570.195.03"`` -> ``(570, 195, 3)``.  None when unparseable."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return (int(value),)
    m = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", str(value))
    if not m:
        return None
    return tuple(int(g) for g in m.groups() if g is not None)


def driver_min_for(build: Any) -> Optional[int]:
    """Minimum driver major for a torch build tag ``"cu126"`` or a ``(12, 6)``."""
    if isinstance(build, str):
        tag = build if build.startswith("cu") else f"cu{build}"
        if tag in DRIVER_MIN_BY_TAG:
            return DRIVER_MIN_BY_TAG[tag]
        parsed = parse_torch_build(f"0+{tag}")["cuda"]
        return DRIVER_MIN_BY_MAJOR.get(parsed[0]) if parsed else None
    if isinstance(build, (tuple, list)) and build:
        tag = f"cu{build[0]}{build[1]}" if len(build) > 1 else None
        if tag and tag in DRIVER_MIN_BY_TAG:
            return DRIVER_MIN_BY_TAG[tag]
        return DRIVER_MIN_BY_MAJOR.get(int(build[0]))
    return None


def _pkg_version(name: str) -> str:
    """Version without importing the package (transformers import is not free)."""
    try:
        from importlib.metadata import version as _v
        return _v(name)
    except Exception:
        try:
            mod = __import__(name)
            return str(getattr(mod, "__version__", ""))
        except Exception:
            return ""


def nvidia_smi_driver_version() -> Optional[str]:
    """Driver number straight from nvidia-smi, or None if it is not there."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    first = out.stdout.strip().splitlines()
    return first[0].strip() if first and first[0].strip() else None


def _nvidia_smi_gpu_names() -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
    except Exception:
        return []
    if out.returncode != 0:
        return []
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def _proc_driver_version() -> Optional[str]:
    p = Path("/proc/driver/nvidia/version")
    try:
        text = p.read_text()
    except Exception:
        return None
    m = re.search(r"\b(\d+\.\d+(?:\.\d+)?)\b", text)
    return m.group(1) if m else None


def driver_probe() -> dict[str, Any]:
    """Everything we can learn about the NVIDIA driver *without* torch's help.

    Deliberately independent of ``torch.cuda.is_available()``: the trap is
    precisely a machine where a GPU and driver exist but torch cannot use them.
    """
    info: dict[str, Any] = {"driver_version": None, "driver_cuda_version": None,
                            "gpu_names": [], "source": None}
    fn = getattr(torch.cuda, "driver_version", None)
    if callable(fn):
        try:
            raw = fn()
        except Exception:
            raw = None
        if isinstance(raw, int) and raw >= 1000:
            # torch reports the CUDA driver *API* version (12080 -> 12.8), not
            # the 570.x number, so it lands in a different field.
            info["driver_cuda_version"] = f"{raw // 1000}.{(raw % 1000) // 10}"
        elif raw:
            info["driver_version"] = str(raw)
            info["source"] = "torch.cuda.driver_version"
    if info["driver_version"] is None:
        proc = _proc_driver_version()
        if proc:
            info["driver_version"] = proc
            info["source"] = "/proc/driver/nvidia/version"
    if info["driver_version"] is None:
        smi = nvidia_smi_driver_version()
        if smi:
            info["driver_version"] = smi
            info["source"] = "nvidia-smi"
    info["gpu_names"] = _nvidia_smi_gpu_names()
    return info


# --------------------------------------------------------------------------
# fingerprint / placement / speed
# --------------------------------------------------------------------------

def fingerprint() -> dict[str, Any]:
    """The ``manifest["env"]`` block for this interpreter, this instant."""
    cuda_ok = False
    try:
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False

    probe = driver_probe()
    device_name = ""
    if cuda_ok:
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = ""
    if not device_name and probe["gpu_names"]:
        # A name here with cuda_available False is the trap's signature.
        device_name = probe["gpu_names"][0]

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": _pkg_version("transformers"),
        "cuda_available": cuda_ok,
        "torch_cuda_version": torch.version.cuda or "",
        "device_name": device_name,
        "driver_version": probe["driver_version"] or "",
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", ""),
        "hf_home": os.environ.get("HF_HOME", ""),
    }


def placement(model) -> dict[str, Any]:
    """``manifest["placement"]`` minus ``matmul_tflops``: where the weights are.

    Counts are per parameter *tensor* (so one stranded tensor is visible),
    ``n_params`` is scalar elements.  'meta' shows up here when a model was
    built but never materialised — the other way to get CPU-speed nonsense.
    """
    devices: dict[str, int] = {}
    dtypes: dict[str, int] = {}
    n_params = 0
    cuda_indices: set[int] = set()
    for _, p in model.named_parameters():
        dev = str(p.device)
        devices[dev] = devices.get(dev, 0) + 1
        dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + 1
        if dev.startswith("cuda"):
            cuda_indices.add(p.device.index if p.device.index is not None else 0)
        n_params += int(p.numel())   # meta tensors have a shape but no storage

    peak = 0
    try:
        if torch.cuda.is_available():
            targets = sorted(cuda_indices) or [torch.cuda.current_device()]
            peak = max(int(torch.cuda.max_memory_allocated(i)) for i in targets)
    except Exception:
        peak = 0

    return {"n_params": n_params, "param_devices": devices, "param_dtypes": dtypes,
            "peak_vram_bytes": peak}


_WARMUP_SECONDS = 0.5   # long enough to bring an idle GPU up to clock


def benchmark_matmul(device: str = "cuda:0", n: int = 4096, iters: int = 8,
                     dtype: torch.dtype = torch.bfloat16) -> dict[str, Any]:
    """Timed dense matmul, with the device it *actually* ran on reported.

    Never pretends: if cuda was asked for and is unavailable, the work happens
    on the CPU and ``fell_back_to_cpu`` says so, which is the whole point — the
    resulting TFLOP/s is then supposed to look terrible.
    """
    requested = str(device)
    want_cuda = requested.startswith("cuda")
    have_cuda = False
    try:
        have_cuda = bool(torch.cuda.is_available())
    except Exception:
        have_cuda = False
    fell_back = bool(want_cuda and not have_cuda)
    used = torch.device(requested if (have_cuda or not want_cuda) else "cpu")

    if used.type != "cuda":
        # A CPU gemm at 4096 can take minutes on a weak host and we only need
        # the order of magnitude to tell CPU from tensor cores.
        n = min(n, 2048)
        iters = max(1, min(iters, 4))

    a = torch.randn(n, n, device=used, dtype=dtype)
    b = torch.randn(n, n, device=used, dtype=dtype)
    # Warm up on a CLOCK, not an iteration count.  An idle 4090 sits in P8 at
    # ~22 W and takes a few hundred ms to ramp; 4096^3 in bf16 is ~1 ms, so a
    # fixed handful of warmup iterations times the GPU at idle clocks and
    # reports ~20 TFLOP/s on hardware that sustains ~150.  That reads as a CPU
    # fallback and fails t0.no_cpu_fallback on a perfectly healthy machine —
    # a verification harness that cries wolf gets ignored, so it must not.
    if used.type == "cuda":
        _warm_until = time.perf_counter() + _WARMUP_SECONDS
        while time.perf_counter() < _warm_until:
            for _ in range(4):
                c = a @ b
            torch.cuda.synchronize(used)
    else:
        for _ in range(2):
            c = a @ b
    if used.type == "cuda":
        torch.cuda.synchronize(used)
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    if used.type == "cuda":
        torch.cuda.synchronize(used)
    elapsed = time.perf_counter() - t0
    del a, b, c

    flops = 2.0 * (n ** 3) * iters
    tflops = (flops / elapsed / 1e12) if elapsed > 0 else float("inf")
    return {"tflops": round(tflops, 3), "device_requested": requested,
            "device_used": str(used), "fell_back_to_cpu": fell_back,
            "n": n, "iters": iters, "elapsed_s": round(elapsed, 6),
            "dtype": str(dtype)}


def matmul_tflops(device: str = "cuda:0", n: int = 4096, iters: int = 8,
                  dtype: torch.dtype = torch.bfloat16) -> float:
    """Measured bf16 TFLOP/s.  Below ``CPU_FALLBACK_TFLOPS`` means no GPU."""
    return float(benchmark_matmul(device=device, n=n, iters=iters, dtype=dtype)["tflops"])


# --------------------------------------------------------------------------
# T0 verdicts — pure functions of manifest blocks
# --------------------------------------------------------------------------

def _soft(strict: bool) -> str:
    """Unverifiable-but-suspicious: hard failure under strict, flag otherwise."""
    return "fail" if strict else "warn"


def _worst(statuses: list[str]) -> str:
    return max(statuses, key=lambda s: _STATUS_RANK.get(s, 0)) if statuses else "skip"


def _norm_device(dev: Any) -> Optional[str]:
    if not dev:
        return None
    d = str(dev).strip().lower()
    if d == "cuda":
        return "cuda:0"
    if d.startswith("cuda:"):
        return d
    if d in ("cpu", "meta", "mps"):
        return d
    if d.isdigit():
        return f"cuda:{d}"
    return d


def _norm_dtype(dt: Any) -> Optional[str]:
    if dt is None:
        return None
    d = str(dt).strip().lower()
    if d in ("", "auto", "none"):
        return None
    if d.startswith("torch."):
        return d
    return _DTYPE_ALIASES.get(d, f"torch.{d}")


def _t0_cuda_available(env: dict, config: Optional[dict], strict: bool) -> CheckResult:
    """CUDA is actually usable, not merely installed."""
    tier = "T0"
    if not env or "cuda_available" not in env:
        return CheckResult("t0.cuda_available", tier, "skip",
                           "env.cuda_available not recorded")
    ok = bool(env.get("cuda_available"))
    dev = _norm_device((config or {}).get("device")) if config else None
    ev = {"cuda_available": env.get("cuda_available"),
          "device_name": env.get("device_name", ""),
          "configured_device": dev, "torch": env.get("torch", "")}
    if ok:
        return CheckResult("t0.cuda_available", tier, "pass",
                           f"CUDA available ({env.get('device_name') or 'unnamed device'})", ev)
    # Order matters: a GPU that torch cannot see is the trap, and declaring
    # device=cpu must not launder it into an excuse.
    if env.get("device_name"):
        return CheckResult(
            "t0.cuda_available", tier, "fail",
            f"a GPU is present ({env['device_name']}) but torch reports no CUDA", ev,
            "The documented trap: torch built for the wrong CUDA major silently "
            "falls back to CPU. See t0.torch_build_matches_driver.")
    if dev == "cpu":
        return CheckResult("t0.cuda_available", tier, "warn",
                           "no CUDA and no GPU detected; the run declared device=cpu", ev,
                           "CPU runs are ~100x slower; confirm this was intentional.")
    return CheckResult("t0.cuda_available", tier, _soft(strict),
                       "torch reports no CUDA and no GPU was detected", ev,
                       "Run on the GPU host, or set config.device='cpu' deliberately.")


def _t0_torch_build_matches_driver(env: dict, config: Optional[dict],
                                   strict: bool) -> CheckResult:
    """The documented trap: cu130 wheels need driver >= 580; this box is 570."""
    tier = "T0"
    cid = "t0.torch_build_matches_driver"
    if not env or not env.get("torch"):
        return CheckResult(cid, tier, "skip", "env.torch not recorded")
    build = parse_torch_build(env.get("torch"))
    drv_raw = env.get("driver_version") or ""
    drv = parse_driver_version(drv_raw)
    need = driver_min_for(build["cuda"] or build["cuda_tag"])
    ev = {"torch": env.get("torch"), "cuda_tag": build["cuda_tag"],
          "torch_cuda_version": env.get("torch_cuda_version", ""),
          "driver_version": drv_raw, "driver_major": drv[0] if drv else None,
          "driver_major_required": need, "cuda_available": env.get("cuda_available")}

    if build["rocm"]:
        return CheckResult(cid, tier, "warn", "torch is a ROCm build; no CUDA driver rule applies", ev)
    if build["cpu_only"] or not build["cuda_tag"]:
        if env.get("cuda_available"):
            return CheckResult(cid, tier, "fail",
                               f"torch {env['torch']} is a CPU-only build but env claims CUDA", ev,
                               "The manifest is internally inconsistent; regenerate it.")
        return CheckResult(cid, tier, _soft(strict),
                           f"torch {env['torch']} is a CPU-only build — no GPU compute possible", ev,
                           "Install a CUDA build matching this driver (cu126 here).")

    # Cross-check the wheel tag against the runtime CUDA version it reports.
    tcv = str(env.get("torch_cuda_version") or "").strip()
    if tcv and build["cuda"]:
        want = f"{build['cuda'][0]}.{build['cuda'][1]}"
        if not tcv.startswith(want):
            return CheckResult(cid, tier, "fail",
                               f"wheel tag {build['cuda_tag']} disagrees with torch_cuda_version {tcv}", ev,
                               "One of the two was edited by hand; the manifest cannot be trusted.")

    if drv and need:
        if drv[0] < need:
            return CheckResult(
                cid, tier, "fail",
                f"torch {build['cuda_tag']} needs driver >= {need}, host has {drv_raw}", ev,
                f"Reinstall torch for the CUDA major this driver runs "
                f"(driver {drv[0]} -> cu126 style build), or update the driver to >= {need}.")
        if not env.get("cuda_available"):
            return CheckResult(
                cid, tier, "warn",
                f"driver {drv_raw} satisfies {build['cuda_tag']} (>= {need}) yet CUDA is unavailable", ev,
                "Driver version is not the cause; check CUDA_VISIBLE_DEVICES and container GPU access.")
        return CheckResult(cid, tier, "pass",
                           f"torch {build['cuda_tag']} on driver {drv_raw} (>= {need})", ev)

    if drv and not need:
        return CheckResult(cid, tier, "warn",
                           f"no driver rule known for {build['cuda_tag']}; host driver is {drv_raw}", ev,
                           "Add the build tag to DRIVER_MIN_BY_TAG if this wheel is real.")

    # No driver number recorded: a working CUDA context is the stronger evidence.
    if env.get("cuda_available"):
        return CheckResult(cid, tier, "pass",
                           f"driver version unrecorded, but torch {build['cuda_tag']} initialised CUDA", ev)
    if env.get("device_name"):
        return CheckResult(cid, tier, "fail",
                           f"GPU {env['device_name']} present, torch {build['cuda_tag']} cannot use it", ev,
                           "Classic build/driver mismatch; check the driver major against "
                           f"the {build['cuda_tag']} requirement (>= {need}).")
    if _norm_device((config or {}).get("device")) == "cpu":
        return CheckResult(cid, tier, "skip",
                           "run declared device=cpu on a host with no GPU; no driver rule to apply", ev)
    return CheckResult(cid, tier, _soft(strict),
                       f"cannot verify torch {build['cuda_tag']} against the driver "
                       "(no driver_version, no CUDA)", ev,
                       "Record env.driver_version (nvidia-smi --query-gpu=driver_version).")


def _t0_params_on_device(placement_blk: Optional[dict], config: Optional[dict]) -> CheckResult:
    """Every weight on the configured device — one stranded tensor is enough."""
    tier, cid = "T0", "t0.params_on_device"
    devices = (placement_blk or {}).get("param_devices")
    if not devices:
        return CheckResult(cid, tier, "skip", "placement.param_devices not recorded")
    devices = {str(k): int(v) for k, v in devices.items()}
    want = _norm_device((config or {}).get("device")) if config else None
    seen = {_norm_device(k) or k: v for k, v in devices.items()}
    n_tensors = sum(seen.values())
    meta = {d: c for d, c in seen.items() if d == "meta"}
    ev = {"param_devices": seen, "configured_device": want,
          "n_param_tensors": n_tensors, "n_params": (placement_blk or {}).get("n_params")}

    if meta:
        return CheckResult(cid, tier, "fail",
                           f"{sum(meta.values())}/{n_tensors} parameter tensors are on 'meta'", ev,
                           "The model was never materialised; load with a real device_map.")
    if not want:
        return CheckResult(cid, tier, "skip", "config.device not recorded", ev)
    stray = {d: c for d, c in seen.items() if d != want}
    if stray:
        return CheckResult(cid, tier, "fail",
                           f"{sum(stray.values())}/{n_tensors} parameter tensors are not on {want}: {stray}",
                           ev, "Anything left on cpu runs at CPU speed with no error; "
                               "load with device_map={'': device} or call .to(device).")
    return CheckResult(cid, tier, "pass", f"all {n_tensors} parameter tensors on {want}", ev)


def _t0_dtype_as_configured(placement_blk: Optional[dict], config: Optional[dict]) -> CheckResult:
    """Weights are in the dtype the config claims (silent upcasts cost 2x)."""
    tier, cid = "T0", "t0.dtype_as_configured"
    dtypes = (placement_blk or {}).get("param_dtypes")
    if not dtypes:
        return CheckResult(cid, tier, "skip", "placement.param_dtypes not recorded")
    want = _norm_dtype((config or {}).get("dtype")) if config else None
    dtypes = {str(k): int(v) for k, v in dtypes.items()}
    ev = {"param_dtypes": dtypes, "configured_dtype": want}
    if not want:
        return CheckResult(cid, tier, "skip", "config.dtype not recorded", ev)
    stray = {d: c for d, c in dtypes.items() if _norm_dtype(d) != want}
    n_tensors = sum(dtypes.values())
    if stray:
        return CheckResult(cid, tier, "fail",
                           f"{sum(stray.values())}/{n_tensors} parameter tensors are not {want}: {stray}",
                           ev, "Load with the configured dtype; a silent fp32 upcast changes "
                               "both speed and numerics.")
    return CheckResult(cid, tier, "pass", f"all {n_tensors} parameter tensors are {want}", ev)


def _t0_no_cpu_fallback(placement_blk: Optional[dict], config: Optional[dict],
                        timing: Optional[dict], strict: bool) -> CheckResult:
    """Measured throughput consistent with a GPU, not a very patient CPU."""
    tier, cid = "T0", "t0.no_cpu_fallback"
    tf = (placement_blk or {}).get("matmul_tflops")
    tps = (timing or {}).get("tokens_per_s")
    if tf is None and tps is None:
        return CheckResult(cid, tier, "skip",
                           "neither placement.matmul_tflops nor timing.tokens_per_s recorded")
    want = _norm_device((config or {}).get("device")) if config else None
    on_cpu_by_design = (want == "cpu")
    ev = {"matmul_tflops": tf, "tflops_floor": CPU_FALLBACK_TFLOPS,
          "tokens_per_s": tps, "tokens_per_s_floor": TOKENS_PER_S_FLOOR,
          "configured_device": want}

    statuses: list[str] = []
    notes: list[str] = []
    if tf is not None:
        try:
            tfv = float(tf)
        except (TypeError, ValueError):
            tfv = float("nan")
        if not (tfv == tfv) or tfv <= 0:            # NaN or nonsense
            statuses.append("fail")
            notes.append(f"matmul_tflops is not a positive number ({tf!r})")
        elif tfv < CPU_FALLBACK_TFLOPS:
            statuses.append("warn" if on_cpu_by_design else "fail")
            notes.append(f"matmul {tfv:.2f} TFLOP/s < {CPU_FALLBACK_TFLOPS} floor")
        else:
            statuses.append("pass")
            notes.append(f"matmul {tfv:.2f} TFLOP/s")
    if tps is not None:
        try:
            tpsv = float(tps)
        except (TypeError, ValueError):
            tpsv = float("nan")
        if not (tpsv == tpsv) or tpsv <= 0:
            statuses.append("fail")
            notes.append(f"tokens_per_s is not a positive number ({tps!r})")
        elif tpsv < TOKENS_PER_S_FLOOR:
            statuses.append("warn" if on_cpu_by_design else "fail")
            notes.append(f"generation {tpsv:.2f} tok/s < {TOKENS_PER_S_FLOOR} floor")
        else:
            statuses.append("pass")
            notes.append(f"generation {tpsv:.2f} tok/s")

    status = _worst(statuses)
    remedy = ("These are CPU numbers. Check t0.cuda_available and "
              "t0.torch_build_matches_driver — the fallback is silent."
              if status == "fail" else "")
    return CheckResult(cid, tier, status, "; ".join(notes), ev, remedy)


def _t0_offline_mode(env: dict, strict: bool) -> CheckResult:
    """Offline hub + the shared cache: an online run is a different experiment."""
    tier, cid = "T0", "t0.offline_mode"
    if not env or ("hf_hub_offline" not in env and "hf_home" not in env):
        return CheckResult(cid, tier, "skip", "env.hf_hub_offline / env.hf_home not recorded")
    raw_off = env.get("hf_hub_offline")
    offline = (raw_off is True) or (str(raw_off).strip().lower() in _TRUTHY)
    home = str(env.get("hf_home") or "").strip()
    ev = {"hf_hub_offline": raw_off, "hf_home": home}

    statuses: list[str] = []
    notes: list[str] = []
    if offline:
        statuses.append("pass")
        notes.append("HF_HUB_OFFLINE set")
    else:
        statuses.append(_soft(strict))
        notes.append(f"HF_HUB_OFFLINE not set ({raw_off!r})")
    if not home:
        statuses.append(_soft(strict))
        notes.append("HF_HOME empty")
    elif not os.path.isabs(home):
        statuses.append("fail")
        notes.append(f"HF_HOME is not an absolute path ({home})")
    elif Path(home) == Path.home() / ".cache" / "huggingface":
        statuses.append("warn")
        notes.append(f"HF_HOME is the default user cache ({home}), not the volume")
    else:
        statuses.append("pass")
        notes.append(f"HF_HOME={home}")

    status = _worst(statuses)
    remedy = ("export HF_HUB_OFFLINE=1 and HF_HOME=/workspace/hf-cache before the run."
              if status in ("fail", "warn") else "")
    return CheckResult(cid, tier, status, "; ".join(notes), ev, remedy)


def _t0_versions_recorded(env: dict) -> CheckResult:
    """python/torch/transformers pinned in the manifest, or nothing is reproducible."""
    tier, cid = "T0", "t0.versions_recorded"
    if not env:
        return CheckResult(cid, tier, "skip", "no env block recorded")
    keys = ("python", "torch", "transformers")
    got = {k: str(env.get(k) or "").strip() for k in keys}
    missing = [k for k, v in got.items() if not v]
    if missing:
        return CheckResult(cid, tier, "fail", f"missing version(s): {', '.join(missing)}", got,
                           "Record env from agentverify.env.fingerprint() at run time.")
    return CheckResult(cid, tier, "pass",
                       f"python {got['python']}, torch {got['torch']}, transformers {got['transformers']}",
                       got)


def t0_results(env: Optional[dict], placement_blk: Optional[dict] = None,
               config: Optional[dict] = None, timing: Optional[dict] = None,
               strict: bool = True) -> list[CheckResult]:
    """Every T0 verdict for one set of manifest blocks, in stable id order.

    Shared by ``preflight`` (live machine) and the T0 artifact checks so both
    apply the same rules; anything genuinely unrecorded comes back as ``skip``.
    """
    env = env or {}
    return [
        _t0_cuda_available(env, config, strict),
        _t0_torch_build_matches_driver(env, config, strict),
        _t0_params_on_device(placement_blk, config),
        _t0_dtype_as_configured(placement_blk, config),
        _t0_no_cpu_fallback(placement_blk, config, timing, strict),
        _t0_offline_mode(env, strict),
        _t0_versions_recorded(env),
    ]


def preflight(strict: bool = True) -> list[CheckResult]:
    """Run T0 against this machine right now, with a freshly timed matmul.

    No model is loaded, so the placement/dtype verdicts honestly ``skip``
    instead of pretending; everything else is measured.
    """
    env = fingerprint()
    # Intent, not capitulation: if the host has a GPU at all we preflight as a
    # GPU run, so a card torch cannot reach fails instead of quietly becoming
    # "a CPU run that is working fine".
    device = "cuda:0" if (env["cuda_available"] or env["device_name"]) else "cpu"
    config = {"device": device, "dtype": None}
    try:
        bench = benchmark_matmul(device=device)
    except Exception as exc:                    # a broken CUDA context is itself the finding
        bench = {"tflops": None, "error": f"{type(exc).__name__}: {exc}",
                 "device_requested": device, "device_used": None}
    place = {"matmul_tflops": bench.get("tflops"), "benchmark": bench}
    results = t0_results(env, place, config, timing=None, strict=strict)
    if bench.get("tflops") is None:
        results = [r if r.id != "t0.no_cpu_fallback" else CheckResult(
            "t0.no_cpu_fallback", "T0", "fail",
            f"matmul benchmark did not run: {bench.get('error', 'unknown error')}",
            {"benchmark": bench},
            "The device is unusable; fix CUDA before trusting any timing.")
            for r in results]
    return results
