"""T0 — was the compute real, and was it the compute the manifest claims?

Every verdict here is re-derived from the recorded ``env`` / ``placement`` /
``timing`` blocks, so the same functions serve both ``verify`` (a finished run)
and ``env.preflight`` (a fingerprint of the live process).  The failure mode
these exist for is the documented cu126 trap: a torch built for a CUDA the
installed driver cannot run reports no CUDA, every tensor silently stays on the
CPU, and nothing crashes — the run just quietly measures nothing.
"""
from __future__ import annotations

import math
import re
from typing import Any, Optional

from ..types import CheckResult, RunArtifacts, check

try:  # env.py owns the constant; the contract pins its value, so a missing
    from ..env import CPU_FALLBACK_TFLOPS  # module (or an import cycle) is survivable.
except Exception:  # pragma: no cover — env.py may not exist yet
    CPU_FALLBACK_TFLOPS = 20.0

# A 1.7B model on a 4090 decodes at ~50 tok/s; the CPU fallback lands far below.
MIN_TOKENS_PER_S = 2.0
# Autoregressive decode cannot exceed this even batched — a larger number means
# the timing was fabricated or generation never ran.
MAX_TOKENS_PER_S = 20_000.0

# Minimum NVIDIA Linux driver per CUDA major, with minor-version compatibility.
# 13.x is the trap: it needs >= 580, this box runs 570.
_MIN_DRIVER: dict[int, tuple[int, ...]] = {
    11: (450, 80, 2),
    12: (525, 60, 13),
    13: (580, 65, 6),
}

_DTYPE_ALIASES = {
    "bf16": "bfloat16", "torch.bfloat16": "bfloat16",
    "fp16": "float16", "half": "float16", "torch.float16": "float16",
    "fp32": "float32", "float": "float32", "torch.float32": "float32",
    "fp64": "float64", "double": "float64", "torch.float64": "float64",
    "int8": "int8", "torch.int8": "int8", "fp8": "float8", "torch.uint8": "uint8",
}

_DTYPE_BYTES = {"float64": 8, "float32": 4, "bfloat16": 2, "float16": 2,
                "float8": 1, "int8": 1, "uint8": 1}


# --------------------------------------------------------------------------
# coercion helpers — a malformed manifest must produce a verdict, not a crash
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


def _as_bool(v: Any) -> Optional[bool]:
    """None means 'not readable as a boolean', which is itself a finding."""
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


def _ver(s: Any) -> Optional[tuple[int, ...]]:
    if not isinstance(s, str):
        return None
    parts = re.findall(r"\d+", s)
    return tuple(int(p) for p in parts[:4]) if parts else None


def _ge(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    n = max(len(a), len(b))
    return tuple(a + (0,) * n)[:n] >= tuple(b + (0,) * n)[:n]


def _cuda_tag(torch_version: Any) -> Optional[tuple[int, int]]:
    """'2.12.0+cu126' -> (12, 6).  None when the build carries no CUDA tag."""
    if not isinstance(torch_version, str):
        return None
    m = re.search(r"\+cu(\d{2,4})", torch_version)
    if not m:
        return None
    d = m.group(1)
    if len(d) == 2:
        return (int(d), 0)
    return (int(d[:2]), int(d[2:]))


def _norm_device(s: Any) -> Optional[str]:
    if not isinstance(s, str) or not s.strip():
        return None
    d = s.strip().lower()
    if d == "cuda":            # bare 'cuda' is the default index
        return "cuda:0"
    if d == "gpu":
        return "cuda:0"
    return d


def _norm_dtype(s: Any) -> Optional[str]:
    if not isinstance(s, str) or not s.strip():
        return None
    d = s.strip().lower()
    return _DTYPE_ALIASES.get(d, d)


def _int_counts(d: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            n = _num(v)
            if n is not None:
                out[str(k)] = n
    return out


# --------------------------------------------------------------------------
# T0 checks
# --------------------------------------------------------------------------

@check("t0.cuda_available", "T0")
def cuda_available(art: RunArtifacts) -> CheckResult:
    """A GPU experiment that ran on the CPU produces numbers that mean nothing."""
    raw = art.cfg("env", "cuda_available")
    device = _norm_device(art.cfg("config", "device"))
    ev: dict[str, Any] = {"env.cuda_available": raw, "config.device": device,
                          "device_name": art.cfg("env", "device_name")}
    if raw is None:
        return CheckResult("t0.cuda_available", "T0", "skip",
                           "env.cuda_available not recorded", ev,
                           "record torch.cuda.is_available() in manifest['env']")
    val = _as_bool(raw)
    if val is None:
        return CheckResult("t0.cuda_available", "T0", "fail",
                           f"env.cuda_available is not a boolean: {raw!r}", ev,
                           "write a real bool, not a placeholder")
    if not val:
        return CheckResult(
            "t0.cuda_available", "T0", "fail",
            "torch reports no CUDA — every tensor ran on the CPU"
            + (f" while config.device is {device!r}" if device and device.startswith("cuda") else ""),
            ev,
            "see t0.torch_build_matches_driver: a torch built for a CUDA newer "
            "than the driver supports reports no CUDA and silently falls back")
    name = art.cfg("env", "device_name")
    if not name:
        return CheckResult("t0.cuda_available", "T0", "warn",
                           "CUDA reported available but no device_name recorded", ev)
    return CheckResult("t0.cuda_available", "T0", "pass",
                       f"CUDA available on {name}", ev)


@check("t0.torch_build_matches_driver", "T0")
def torch_build_matches_driver(art: RunArtifacts) -> CheckResult:
    """The documented trap: a cu130 wheel on a 570 driver runs entirely on the CPU."""
    tv = art.cfg("env", "torch")
    driver = art.cfg("env", "driver_version")
    declared = art.cfg("env", "torch_cuda_version")
    avail = _as_bool(art.cfg("env", "cuda_available"))
    device = _norm_device(art.cfg("config", "device"))
    tag = _cuda_tag(tv)
    dv = _ver(driver)
    ev: dict[str, Any] = {
        "torch": tv, "torch_build_cuda": ".".join(str(x) for x in tag) if tag else None,
        "env.torch_cuda_version": declared, "driver_version": driver,
        "driver_parsed": list(dv) if dv else None,
        "cuda_available": avail, "config.device": device,
    }
    remedy = ("install the torch wheel built for the CUDA this driver runs, e.g.\n"
              "  pip install 'torch==2.12.0+cu126' "
              "--index-url https://download.pytorch.org/whl/cu126")

    if not isinstance(tv, str) or not tv.strip():
        return CheckResult("t0.torch_build_matches_driver", "T0", "skip",
                           "env.torch not recorded", ev)

    if re.search(r"\+(cpu|rocm|xpu)", tv):
        if (device or "").startswith("cuda") or avail is True:
            return CheckResult(
                "t0.torch_build_matches_driver", "T0", "fail",
                f"torch build {tv!r} has no CUDA support but the run claims CUDA "
                f"(device={device!r}, cuda_available={avail!r})", ev, remedy)
        return CheckResult("t0.torch_build_matches_driver", "T0", "warn",
                           f"torch build {tv!r} is not a CUDA build", ev, remedy)

    if tag is None:
        return CheckResult("t0.torch_build_matches_driver", "T0", "warn",
                           f"torch version {tv!r} carries no +cuXXX build tag; "
                           "build/driver compatibility cannot be verified", ev)

    reasons: list[str] = []
    declared_tag = _ver(declared)
    if declared_tag is not None and tuple(declared_tag[:2]) != tag:
        reasons.append(f"manifest torch_cuda_version {declared!r} disagrees with "
                       f"the build tag +cu{tag[0]}{tag[1]}")

    need = _MIN_DRIVER.get(tag[0])
    ev["min_driver_required"] = ".".join(str(x) for x in need) if need else None
    if need is None:
        return CheckResult("t0.torch_build_matches_driver", "T0", "warn",
                           f"no minimum-driver table entry for CUDA {tag[0]}.{tag[1]}; "
                           "cannot verify build/driver compatibility"
                           + (" — " + "; ".join(reasons) if reasons else ""), ev)
    if dv is None:
        status = "fail" if reasons else "warn"
        return CheckResult("t0.torch_build_matches_driver", "T0", status,
                           "driver_version not recorded; build/driver compatibility "
                           "cannot be verified"
                           + (" — " + "; ".join(reasons) if reasons else ""), ev,
                           "record `nvidia-smi --query-gpu=driver_version` in manifest['env']")

    ok = _ge(dv, need)
    ev["driver_supports_build"] = bool(ok)
    if not ok:
        return CheckResult(
            "t0.torch_build_matches_driver", "T0", "fail",
            f"torch is built for CUDA {tag[0]}.{tag[1]} which needs driver >= "
            f"{ev['min_driver_required']}, but the driver is {driver} — this box "
            "reports NO CUDA and every tensor silently stays on the CPU"
            + (" (also: " + "; ".join(reasons) + ")" if reasons else ""),
            ev, remedy)
    if reasons:
        return CheckResult("t0.torch_build_matches_driver", "T0", "fail",
                           "; ".join(reasons), ev,
                           "the env block was not written from the live torch build")
    return CheckResult("t0.torch_build_matches_driver", "T0", "pass",
                       f"torch cu{tag[0]}{tag[1]} runs on driver {driver} "
                       f"(needs >= {ev['min_driver_required']})", ev)


@check("t0.params_on_device", "T0")
def params_on_device(art: RunArtifacts) -> CheckResult:
    """Params left on cpu/meta mean the model never fully reached the accelerator."""
    raw = art.cfg("placement", "param_devices")
    want = _norm_device(art.cfg("config", "device"))
    ev: dict[str, Any] = {"config.device": want, "param_devices_raw": raw,
                          "n_params": art.cfg("placement", "n_params")}
    if raw is None:
        return CheckResult("t0.params_on_device", "T0", "skip",
                           "placement.param_devices not recorded", ev)
    if not isinstance(raw, dict):
        return CheckResult("t0.params_on_device", "T0", "fail",
                           f"placement.param_devices is {type(raw).__name__}, expected a mapping",
                           ev)
    counts: dict[str, float] = {}
    for k, v in raw.items():
        dev = _norm_device(k) or str(k)
        n = _num(v)
        if n is None:
            return CheckResult("t0.params_on_device", "T0", "fail",
                               f"placement.param_devices[{k!r}] is not a count: {v!r}", ev)
        counts[dev] = counts.get(dev, 0.0) + n
    total = sum(counts.values())
    ev["param_devices"] = counts
    ev["n_tensors_total"] = total
    if not counts or total <= 0:
        return CheckResult("t0.params_on_device", "T0", "fail",
                           "placement.param_devices records no parameters at all", ev,
                           "placement() must walk model.parameters() after the .to(device)")

    stranded = {d: n for d, n in counts.items() if d in ("cpu", "meta") and n > 0}
    if want is None:
        ev["n_distinct_devices"] = len(counts)
        if stranded:
            return CheckResult("t0.params_on_device", "T0", "fail",
                               f"{sum(stranded.values()):g}/{total:g} parameter tensors are on "
                               f"{sorted(stranded)} (config.device not recorded)", ev,
                               "meta/cpu params are never correct for a GPU run")
        return CheckResult("t0.params_on_device", "T0", "warn",
                           "config.device not recorded; params sit on "
                           f"{sorted(counts)} and none are on cpu/meta", ev)

    off = {d: n for d, n in counts.items() if d != want and n > 0}
    ev["n_off_device"] = sum(off.values())
    ev["off_device"] = off
    ev["fraction_off_device"] = round(sum(off.values()) / total, 6)
    if off:
        return CheckResult(
            "t0.params_on_device", "T0", "fail",
            f"{sum(off.values()):g}/{total:g} parameter tensors are not on {want}: {off}",
            ev,
            "load with an explicit device and assert every parameter's .device "
            "afterwards; cpu/meta leftovers make the run silently slow or wrong")
    return CheckResult("t0.params_on_device", "T0", "pass",
                       f"all {total:g} parameter tensors on {want}", ev)


@check("t0.dtype_as_configured", "T0")
def dtype_as_configured(art: RunArtifacts) -> CheckResult:
    """Loading in a wider dtype than configured changes both speed and numerics."""
    raw = art.cfg("placement", "param_dtypes")
    want = _norm_dtype(art.cfg("config", "dtype"))
    ev: dict[str, Any] = {"config.dtype": art.cfg("config", "dtype"),
                          "config.dtype_normalized": want, "param_dtypes_raw": raw}
    if raw is None:
        return CheckResult("t0.dtype_as_configured", "T0", "skip",
                           "placement.param_dtypes not recorded", ev)
    if not isinstance(raw, dict) or not raw:
        return CheckResult("t0.dtype_as_configured", "T0", "fail",
                           "placement.param_dtypes is empty or not a mapping", ev)
    if want is None:
        return CheckResult("t0.dtype_as_configured", "T0", "skip",
                           "config.dtype not recorded; nothing to compare against", ev)

    counts: dict[str, float] = {}
    for k, v in raw.items():
        dt = _norm_dtype(k) or str(k)
        n = _num(v)
        if n is None:
            return CheckResult("t0.dtype_as_configured", "T0", "fail",
                               f"placement.param_dtypes[{k!r}] is not a count: {v!r}", ev)
        counts[dt] = counts.get(dt, 0.0) + n
    total = sum(counts.values())
    ev["param_dtypes"] = counts
    ev["n_tensors_total"] = total
    if total <= 0:
        return CheckResult("t0.dtype_as_configured", "T0", "fail",
                           "placement.param_dtypes records no parameters at all", ev)
    dominant = max(counts.items(), key=lambda kv: kv[1])[0]
    off = {d: n for d, n in counts.items() if d != want and n > 0}
    ev["dominant_dtype"] = dominant
    ev["n_off_dtype"] = sum(off.values())

    # Cross-check: the weights must at least fit in VRAM at the configured width.
    n_params = _num(art.cfg("placement", "n_params"))
    peak = _num(art.cfg("placement", "peak_vram_bytes"))
    itemsize = _DTYPE_BYTES.get(want)
    vram_note = None
    if n_params and peak and itemsize and n_params > 0 and peak > 0:
        expected = n_params * itemsize
        ratio = peak / expected
        ev["expected_param_bytes"] = expected
        ev["peak_vram_bytes"] = peak
        ev["peak_over_param_bytes"] = round(ratio, 4)
        if ratio < 0.9:
            vram_note = (f"peak VRAM {peak:.3g} B is below the {expected:.3g} B the "
                         f"parameters need at {want} — the weights were not all resident")
        elif ratio >= 1.8:
            vram_note = (f"peak VRAM is {ratio:.2f}x the {want} parameter bytes — "
                         "wide enough to hide a float32 load")

    if dominant != want:
        return CheckResult("t0.dtype_as_configured", "T0", "fail",
                           f"most parameters are {dominant}, config.dtype says {want}",
                           ev, "pass torch_dtype through to from_pretrained and record "
                               "the dtypes actually observed on the loaded model")
    if off:
        return CheckResult("t0.dtype_as_configured", "T0", "warn",
                           f"{sum(off.values()):g}/{total:g} parameter tensors are not "
                           f"{want}: {off}" + (f"; {vram_note}" if vram_note else ""), ev)
    if vram_note:
        return CheckResult("t0.dtype_as_configured", "T0", "warn",
                           f"all parameters are {want}, but {vram_note}", ev)
    return CheckResult("t0.dtype_as_configured", "T0", "pass",
                       f"all {total:g} parameter tensors are {want}", ev)


@check("t0.no_cpu_fallback", "T0")
def no_cpu_fallback(art: RunArtifacts) -> CheckResult:
    """Measured throughput, not a flag, is what proves the GPU did the work."""
    tflops = _num(art.cfg("placement", "matmul_tflops"))
    tps = _num(art.cfg("timing", "tokens_per_s"))
    gen_s = _num(art.cfg("timing", "generate_s"))
    ev: dict[str, Any] = {
        "matmul_tflops": tflops, "cpu_fallback_tflops": CPU_FALLBACK_TFLOPS,
        "tokens_per_s": tps, "min_tokens_per_s": MIN_TOKENS_PER_S,
        "max_tokens_per_s": MAX_TOKENS_PER_S, "generate_s": gen_s,
        "device_name": art.cfg("env", "device_name"),
    }

    # Recompute throughput from the records rather than trusting timing.tokens_per_s.
    tok = [_num(r.get("n_new_tokens")) for r in art.records]
    tok = [t for t in tok if t is not None and t >= 0]
    recomputed = None
    if tok and gen_s and gen_s > 0:
        recomputed = sum(tok) / gen_s
        ev["n_new_tokens_total"] = sum(tok)
        ev["recomputed_tokens_per_s"] = round(recomputed, 4)

    if tflops is None and tps is None and recomputed is None:
        return CheckResult("t0.no_cpu_fallback", "T0", "skip",
                           "no throughput evidence recorded (placement.matmul_tflops, "
                           "timing.tokens_per_s, records all absent)", ev)

    fails: list[str] = []
    warns: list[str] = []
    if tflops is not None:
        if not math.isfinite(tflops) or tflops <= 0:
            fails.append(f"matmul_tflops is {tflops!r}")
        elif tflops < CPU_FALLBACK_TFLOPS:
            fails.append(f"matmul_tflops {tflops:.3g} < {CPU_FALLBACK_TFLOPS} — "
                         "this is CPU-class throughput on a machine that claims a 4090")
    else:
        warns.append("placement.matmul_tflops not recorded")

    for label, value in (("timing.tokens_per_s", tps),
                         ("throughput recomputed from records", recomputed)):
        if value is None:
            continue
        if not math.isfinite(value) or value <= 0:
            warns.append(f"{label} is {value!r}")
        elif value < MIN_TOKENS_PER_S:
            fails.append(f"{label} is {value:.4g} tok/s, below the {MIN_TOKENS_PER_S} "
                         "tok/s floor — generation ran on the CPU")
        elif value > MAX_TOKENS_PER_S:
            warns.append(f"{label} is {value:.4g} tok/s, physically impossible for "
                         "autoregressive decode — the timing is not measured")
    if tps is None and recomputed is None:
        warns.append("no token throughput recorded and none recomputable from records")

    if fails:
        return CheckResult("t0.no_cpu_fallback", "T0", "fail", "; ".join(fails), ev,
                           "check that torch.cuda.is_available() is true and that the "
                           "torch build matches the driver (t0.torch_build_matches_driver); "
                           "a cu130 wheel on driver 570 runs ~100x slower on the CPU")
    if warns:
        return CheckResult("t0.no_cpu_fallback", "T0", "warn", "; ".join(warns), ev)
    return CheckResult("t0.no_cpu_fallback", "T0", "pass",
                       f"matmul {tflops:.4g} TFLOP/s >= {CPU_FALLBACK_TFLOPS} and "
                       f"token throughput sane", ev)


@check("t0.offline_mode", "T0")
def offline_mode(art: RunArtifacts) -> CheckResult:
    """Offline + the volume cache is what makes a run reproducible on this box."""
    raw = art.cfg("env", "hf_hub_offline")
    home = art.cfg("env", "hf_home")
    ev: dict[str, Any] = {"hf_hub_offline": raw, "hf_home": home}
    if raw is None and home is None:
        return CheckResult("t0.offline_mode", "T0", "skip",
                           "neither hf_hub_offline nor hf_home recorded", ev)

    remedy = "export HF_HUB_OFFLINE=1 and HF_HOME=/workspace/hf-cache before the run"
    offline = _as_bool(raw)
    ev["hf_hub_offline_parsed"] = offline
    if raw is None or offline is None:
        return CheckResult("t0.offline_mode", "T0", "fail",
                           f"HF_HUB_OFFLINE not recorded as a set flag (got {raw!r}) — "
                           "the run could have reached the network", ev, remedy)
    if not offline:
        return CheckResult("t0.offline_mode", "T0", "fail",
                           "HF_HUB_OFFLINE is not set — weights may have come from the "
                           "network rather than the pinned cache", ev, remedy)

    if not isinstance(home, str) or not home.strip():
        return CheckResult("t0.offline_mode", "T0", "fail",
                           "HF_HUB_OFFLINE is set but HF_HOME is not recorded, so the "
                           "cache the run read cannot be identified", ev, remedy)
    home = home.strip()
    ev["hf_home"] = home
    if not home.startswith("/"):
        return CheckResult("t0.offline_mode", "T0", "fail",
                           f"HF_HOME {home!r} is not an absolute path", ev, remedy)
    if not home.startswith("/workspace"):
        return CheckResult("t0.offline_mode", "T0", "warn",
                           f"offline, but HF_HOME {home!r} is not on the /workspace "
                           "volume where the 53 GB model cache lives", ev, remedy)
    return CheckResult("t0.offline_mode", "T0", "pass",
                       f"offline with HF_HOME={home}", ev)


@check("t0.versions_recorded", "T0")
def versions_recorded(art: RunArtifacts) -> CheckResult:
    """Without exact versions the run cannot be reproduced or its traps diagnosed."""
    keys = ("python", "torch", "transformers")
    env = art.cfg("env", default=None)
    ev: dict[str, Any] = {k: art.cfg("env", k) for k in keys}
    if not isinstance(env, dict) or not env:
        return CheckResult("t0.versions_recorded", "T0", "skip",
                           "manifest['env'] block absent", ev)
    missing = [k for k in keys
               if not isinstance(ev[k], str) or not ev[k].strip()]
    unparsed = [k for k in keys
                if k not in missing and _ver(ev[k]) is None]
    ev["missing"] = missing
    ev["unparsable"] = unparsed
    ev["parsed"] = {k: list(_ver(ev[k]) or ()) for k in keys if k not in missing}
    if missing:
        return CheckResult("t0.versions_recorded", "T0", "fail",
                           f"missing or empty version(s): {missing}", ev,
                           "record sys.version_info, torch.__version__ and "
                           "transformers.__version__ in manifest['env']")
    if unparsed:
        return CheckResult("t0.versions_recorded", "T0", "fail",
                           f"version string(s) carry no version number: "
                           f"{ {k: ev[k] for k in unparsed} }", ev)
    return CheckResult("t0.versions_recorded", "T0", "pass",
                       f"python {ev['python']}, torch {ev['torch']}, "
                       f"transformers {ev['transformers']}", ev)
