"""Model loading, activation extraction, the steering hook, and generation.

Everything the T1 checks later re-derive — the vector, the activations in
`acts.npz`, the completions — is only as good as the layer bookkeeping done
here, so the index convention is stated once next to the two functions that
encode it (`resolve_layer_module` and `hidden_state_index`) and is never
restated differently anywhere else.

Nothing in this module reads a run directory or forms a verdict; it produces
the artifacts that the checks are then free to disbelieve.
"""
from __future__ import annotations

import contextlib
import copy
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
import transformers
from torch import nn

from .types import sha256_bytes

__all__ = [
    "DTYPES",
    "load_model",
    "resolve_layer_module",
    "hidden_state_index",
    "extract_activations",
    "extract_vector",
    "SteeringHook",
    "generate",
]

DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _resolve_dtype(dtype: str | torch.dtype) -> Any:
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).replace("torch.", "")
    if key == "auto":
        return "auto"
    if key not in DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}; use one of {sorted(DTYPES)} or 'auto'")
    return DTYPES[key]


def _ensure_pad_token(tok) -> int:
    """Batched work needs a pad id; borrowing EOS is the standard fallback."""
    if getattr(tok, "pad_token_id", None) is None:
        if getattr(tok, "eos_token", None) is None:
            raise ValueError("tokenizer has neither a pad token nor an eos token to borrow")
        tok.pad_token = tok.eos_token
    return int(tok.pad_token_id)


def _model_device(model) -> torch.device:
    for p in model.parameters():
        return p.device
    for b in model.buffers():
        return b.device
    return torch.device("cpu")


def load_model(model_id: str, dtype: str = "bfloat16", device: str = "cuda:0"):
    """Load a causal LM and its tokenizer, whole, onto one device.

    No `device_map`: a silently sharded or CPU-offloaded placement is precisely
    what `t0.params_on_device` exists to catch, so this either puts every
    parameter on `device` or raises here.  `.to(device)` is also the cheapest
    tripwire for the cu126/driver trap — without CUDA it fails loudly instead
    of running 100x slower on the CPU.
    """
    torch_dtype = _resolve_dtype(dtype)
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
    model.to(device)
    model.eval()
    _ensure_pad_token(tok)
    return model, tok


# --------------------------------------------------------------------------
# layers — the index convention lives here and only here
# --------------------------------------------------------------------------

def _configured_depth(model) -> int | None:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    configs = [cfg]
    get_text_config = getattr(cfg, "get_text_config", None)
    if callable(get_text_config):
        try:
            configs.append(get_text_config())
        except Exception:  # composite configs vary; depth is a hint, not a contract
            pass
    for candidate in configs:
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            value = getattr(candidate, attr, None)
            if isinstance(value, int) and value > 0:
                return value
    return None


def _decoder_layers(model) -> tuple[str, nn.ModuleList]:
    """The decoder block list and its dotted name exactly as `named_modules()` spells it.

    The name is what lands in `manifest["hook"]["module_path"]`, so it has to be
    resolvable from the model root ("model.layers.14"), not a guess.
    """
    depth = _configured_depth(model)
    candidates = [(name, mod) for name, mod in model.named_modules()
                  if isinstance(mod, nn.ModuleList) and len(mod) > 0 and name]
    if not candidates:
        raise RuntimeError(f"{type(model).__name__} exposes no nn.ModuleList of decoder blocks")
    exact = [c for c in candidates if len(c[1]) == depth] if depth else []
    pool = exact or candidates
    # shallowest path wins: the language stack sits above vision towers and adapters
    return min(pool, key=lambda c: (c[0].count("."), -len(c[1]), c[0]))


def _check_layer_arg(layer: Any) -> int:
    if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)):
        raise TypeError(f"layer must be an int block index, got {type(layer).__name__}")
    return int(layer)


def resolve_layer_module(model, layer: int) -> tuple[nn.Module, str]:
    """The decoder block whose OUTPUT is the residual stream at `layer`.

    LAYER INDEX CONVENTION (see `hidden_state_index` right below, which is the
    other half of it): `layer` is an absolute block index in [0, n_blocks).  The
    module returned here is the one whose output we perturb, and the same
    layer's activations are read at `hidden_state_index(layer)` — hooking one
    block and reading another is the single most common way a steering result
    turns out to describe a layer nobody touched.

    Negative indices are refused: the manifest records an absolute layer, and
    -1 would quietly mean a different block on a different model.
    """
    layer = _check_layer_arg(layer)
    name, blocks = _decoder_layers(model)
    if not 0 <= layer < len(blocks):
        raise IndexError(f"layer {layer} out of range: {type(model).__name__} has "
                         f"{len(blocks)} blocks (0..{len(blocks) - 1})")
    return blocks[layer], f"{name}.{layer}"


def hidden_state_index(layer: int) -> int:
    """Index into `output_hidden_states` holding the OUTPUT of block `layer`.

    `hidden_states` has n_blocks+1 entries: hidden_states[i] is the INPUT to
    block i, so hidden_states[0] is the embedding output and block L's output
    is at L+1.  `t1.layer_index_convention` asserts this offset; an off-by-one
    here measures the layer above or below the one that was steered.
    """
    layer = _check_layer_arg(layer)
    if layer < 0:
        raise ValueError("layer must be a non-negative absolute block index")
    return layer + 1


# --------------------------------------------------------------------------
# activations
# --------------------------------------------------------------------------

@contextlib.contextmanager
def _padding_side(tok, side: str) -> Iterator[None]:
    old = getattr(tok, "padding_side", None)
    tok.padding_side = side
    try:
        yield
    finally:
        if old is not None:
            tok.padding_side = old


def _max_length(tok, model) -> int | None:
    """A usable truncation length, ignoring the huge sentinels tokenizers carry."""
    limits = [getattr(tok, "model_max_length", None),
              getattr(getattr(model, "config", None), "max_position_embeddings", None)]
    usable = [int(v) for v in limits if isinstance(v, int) and 0 < v < 10_000_000]
    return min(usable) if usable else None


def _last_real_token(mask: torch.Tensor) -> torch.Tensor:
    """Index of the last non-pad token in each row — never assume it is -1.

    Right padding parks the real last token in the middle of the row, so reading
    position -1 blindly builds the vector out of pad tokens: a plausible-looking
    direction with nothing in it.
    """
    counts = mask.sum(dim=1)
    if bool((counts == 0).any()):
        empty = torch.nonzero(counts == 0).flatten().tolist()
        raise ValueError(f"rows {empty} tokenize to zero real tokens")
    flipped = torch.flip(mask.to(torch.int64), dims=[1])
    return mask.shape[1] - 1 - flipped.argmax(dim=1)


def extract_activations(model, tok, texts: Sequence[str], layers: Iterable[int],
                        batch_size: int = 8) -> np.ndarray:
    """Residual stream at the last real token of each text: `[len(layers), len(texts), d_model]`.

    Right padding on purpose.  A plain forward, unlike `generate`, numbers
    positions with a bare arange, so left-padded rows would get shifted RoPE
    positions and quietly wrong activations; with right padding every real token
    keeps its true position and `_last_real_token` finds the one we want.

    Texts are tokenized as-is (no chat template) so that whatever `data.py`
    authored is what the model sees.  Worth knowing: transformers ties
    `hidden_states[-1]` to `last_hidden_state`, i.e. the deepest block's entry
    is post-final-norm, so steering the last block will not show a clean
    alpha*||v|| delta there.
    """
    texts = [str(t) for t in texts]
    layers = [_check_layer_arg(l) for l in layers]
    if not texts:
        raise ValueError("extract_activations needs at least one text")
    if not layers:
        raise ValueError("extract_activations needs at least one layer")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    _, blocks = _decoder_layers(model)
    depth = len(blocks)
    for layer in layers:
        if not 0 <= layer < depth:
            raise IndexError(f"layer {layer} out of range: model has {depth} blocks")
    wanted = [hidden_state_index(layer) for layer in layers]

    device = _model_device(model)
    _ensure_pad_token(tok)
    max_len = _max_length(tok, model)
    model.eval()

    chunks: list[np.ndarray] = []
    with torch.inference_mode(), _padding_side(tok, "right"):
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            kwargs: dict[str, Any] = {"return_tensors": "pt", "padding": True}
            if max_len is not None:
                kwargs.update(truncation=True, max_length=max_len)
            enc = tok(batch, **kwargs)
            enc = {k: v.to(device) for k, v in enc.items()
                   if k in ("input_ids", "attention_mask")}
            mask = enc.get("attention_mask")
            if mask is None:
                mask = torch.ones_like(enc["input_ids"])
            out = model(**enc, output_hidden_states=True, use_cache=False)
            hidden = getattr(out, "hidden_states", None)
            if hidden is None:
                raise RuntimeError("model returned no hidden_states despite output_hidden_states=True")
            if len(hidden) != depth + 1:
                raise RuntimeError(
                    f"expected {depth + 1} hidden states for {depth} blocks, got {len(hidden)}; "
                    "the layer-index convention does not hold for this architecture")
            last = _last_real_token(mask)
            rows = torch.arange(mask.shape[0], device=last.device)
            picked = torch.stack([hidden[i][rows, last].float() for i in wanted], dim=0)
            chunks.append(picked.detach().to("cpu").numpy())

    return np.ascontiguousarray(np.concatenate(chunks, axis=1), dtype=np.float32)


def extract_vector(model, tok, pairs: Sequence[dict], layer: int,
                   batch_size: int = 8) -> tuple[np.ndarray, dict]:
    """Contrast vector: mean(positive) - mean(negative) at the last prompt token.

    Both halves go through one `extract_activations` call so neither side gets a
    different tokenization or padding treatment.  The returned dict is the
    `manifest["vector"]` block; its `sha256` is the hash of the vector's own
    float32 bytes (a content fingerprint that survives repacking) — the hash of
    the `vector.npz` container belongs to `manifest["hashes"]`, which only the
    writer of that file can compute.
    """
    pairs = list(pairs)
    if not pairs:
        raise ValueError("extract_vector needs at least one contrast pair")
    for i, pair in enumerate(pairs):
        missing = [k for k in ("positive", "negative") if not isinstance(pair.get(k), str)]
        if missing:
            raise ValueError(f"contrast pair {pair.get('id', i)!r} is missing text for {missing}")

    layer = _check_layer_arg(layer)
    texts = [p["positive"] for p in pairs] + [p["negative"] for p in pairs]
    acts = extract_activations(model, tok, texts, [layer], batch_size=batch_size)[0]

    n = len(pairs)
    # mean in float64: a 64-pair sum of bf16-derived activations loses real bits otherwise
    positive = acts[:n].astype(np.float64)
    negative = acts[n:].astype(np.float64)
    vector = np.ascontiguousarray((positive.mean(axis=0) - negative.mean(axis=0)),
                                  dtype=np.float32)

    meta = {
        "path": "vector.npz",
        "key": "v",
        "layer": layer,
        "dim": int(vector.shape[0]),
        "norm": float(np.linalg.norm(vector.astype(np.float64))),
        "sha256": sha256_bytes(vector.tobytes()),
        "n_pairs": n,
        "dtype": "float32",
        "finite": bool(np.isfinite(vector).all()),
    }
    return vector, meta


# --------------------------------------------------------------------------
# the intervention
# --------------------------------------------------------------------------

class SteeringHook:
    """Adds `alpha * vector` to a module's output, broadcast over all positions.

    `fires` counts real forward passes of the hooked module and nothing else —
    it is the entire evidence base for `t1.hook_fired`, so it is incremented in
    the hook body itself rather than inferred from how many prompts were sent.
    Note the arithmetic before predicting it: batched generation fires once for
    the prefill and once per decode step, so a batch of B prompts for T tokens
    fires up to 1+(T-1) times, not B times.  `prefill_rows` is the counter that
    does track one per prompt (rows seen on a multi-position pass), which is
    what `manifest["hook"]["fires_expected"] == n_eval` is counting.

    Used as a context manager so the hook comes off even on exception:

        with SteeringHook(v, alpha).attach(module) as h:
            ...                      # h.fires grows
    """

    def __init__(self, vector, alpha: float = 1.0) -> None:
        tensor = vector if torch.is_tensor(vector) else torch.as_tensor(np.asarray(vector))
        if tensor.dim() > 1 and min(tensor.shape) != 1:
            raise ValueError(f"steering vector must be 1-D, got shape {tuple(tensor.shape)}")
        # clone: as_tensor aliases the caller's numpy buffer, and a vector that can
        # change under the hook is a vector nobody can verify after the fact
        tensor = tensor.detach().to(torch.float32).reshape(-1).contiguous().clone()
        if tensor.numel() == 0:
            raise ValueError("steering vector is empty")
        self._vector = tensor
        self._alpha = float(alpha)
        self._delta = self._vector * self._alpha
        self._cast: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._handle = None
        self.module_type: str | None = None
        self.fires = 0
        self.prefill_rows = 0

    # `alpha` is a property so the pre-scaled delta and the device/dtype cache
    # can never drift out of sync with it during a sweep.
    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha = float(value)
        self._delta = self._vector * self._alpha
        self._cast.clear()

    @property
    def vector(self) -> torch.Tensor:
        return self._vector

    @property
    def attached(self) -> bool:
        return self._handle is not None

    def attach(self, module: nn.Module) -> "SteeringHook":
        """Register on `module`, returning self so it can be used as a context manager.

        `prepend=True` matters: transformers 5.x collects `output_hidden_states`
        with its own forward hooks on these same blocks, and hooks run in
        registration order with each one seeing the previous one's replacement
        output.  Running last would mean the recorded activations are the
        *unsteered* ones while the model consumes the steered ones — acts.npz
        would show a zero delta for a run that really was steered.
        """
        if self._handle is not None:
            raise RuntimeError("hook is already attached; detach before attaching again")
        if not isinstance(module, nn.Module):
            raise TypeError(f"expected an nn.Module to hook, got {type(module).__name__}")
        self._handle = module.register_forward_hook(self._forward_hook, prepend=True)
        self.module_type = type(module).__name__
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "SteeringHook":
        if self._handle is None:
            raise RuntimeError("SteeringHook used as a context manager without attach(module); "
                               "that would run the experiment with no intervention at all")
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.remove()
        return False

    def __repr__(self) -> str:
        return (f"SteeringHook(dim={self._vector.numel()}, alpha={self._alpha}, "
                f"attached={self.attached}, fires={self.fires}, "
                f"prefill_rows={self.prefill_rows})")

    def _aligned(self, ref: torch.Tensor) -> torch.Tensor:
        key = (ref.device, ref.dtype)
        cached = self._cast.get(key)
        if cached is None:
            # scale in float32 first, then cast: alpha*v rounded once, not twice
            cached = self._delta.to(device=ref.device, dtype=ref.dtype)
            self._cast[key] = cached
        return cached

    def _perturb(self, hidden: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(hidden):
            raise TypeError(f"hooked module produced {type(hidden).__name__}, not a tensor")
        if hidden.shape[-1] != self._vector.numel():
            raise RuntimeError(
                f"steering vector has dim {self._vector.numel()} but the hooked module "
                f"({self.module_type}) outputs dim {hidden.shape[-1]}")
        if hidden.dim() >= 3 and hidden.shape[-2] > 1:
            # a multi-position pass is a prefill (or a plain forward): one row per prompt
            self.prefill_rows += int(hidden.shape[0])
        # out-of-place: an in-place add would also rewrite whatever another hook
        # already captured from this same storage
        return hidden + self._aligned(hidden)

    def _forward_hook(self, module: nn.Module, args: tuple, output: Any) -> Any:
        self.fires += 1
        if isinstance(output, tuple):
            if not output:
                raise RuntimeError("hooked module returned an empty tuple")
            return (self._perturb(output[0]),) + tuple(output[1:])
        return self._perturb(output)


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

class _FiniteLogitsProbe:
    """Watches decoding logits and changes nothing.

    `finite_logits` in the records has to mean something, so it is measured at
    the source rather than guessed from the text.  Legitimately masked tokens
    are -inf, so only NaN, +inf, or a step with nothing finite left counts as
    broken.
    """

    def __init__(self) -> None:
        self.bad: torch.Tensor | None = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        bad = (torch.isnan(scores).any(dim=-1)
               | torch.isposinf(scores).any(dim=-1)
               | ~torch.isfinite(scores).any(dim=-1))
        self.bad = bad if self.bad is None else (self.bad | bad)
        return scores


def _eos_ids(gen_cfg, tok) -> set[int]:
    raw = getattr(gen_cfg, "eos_token_id", None)
    if raw is None:
        raw = getattr(tok, "eos_token_id", None)
    if raw is None:
        return set()
    if isinstance(raw, int):
        return {int(raw)}
    return {int(x) for x in raw if x is not None}


def _new_token_count(ids: list[int], pad_id: int | None, eos_ids: set[int]) -> int:
    """How many tokens were really generated before `generate` padded the row out."""
    for j, token in enumerate(ids):
        if token in eos_ids:
            return j + 1
    n = len(ids)
    if pad_id is not None and pad_id not in eos_ids:
        while n > 0 and ids[n - 1] == pad_id:
            n -= 1
    return n


# Qwen3 is a reasoning model and its chat template defaults to enable_thinking=True.
# Left on, every completion is `<think>\nOkay, the user is asking...` and, at a small
# max_new_tokens, contains no answer at all: the rule scorer marks the whole eval
# undetermined, both arms score 0.0, and the run reports a confident NULL EFFECT
# backed by complete, correctly-hashed artifacts.  Measured on this box 2026-08-29,
# see docs/measured_facts.md.  Templates that do not take the kwarg ignore it.
_THINKING_OFF = {"enable_thinking": False}


def _encode_prompts(tok, prompts: Sequence[str]) -> dict[str, torch.Tensor]:
    """Chat-format a batch of prompts for generation.

    transformers 5.x trap: `apply_chat_template(..., return_tensors="pt")` hands
    back a `BatchEncoding`, not a tensor.  Take `return_dict=True` and splat it
    into `generate(**enc, ...)`; passing it positionally gets you a bare
    `AttributeError` from `inputs_tensor.shape[0]`.
    """
    conversations = [[{"role": "user", "content": p}] for p in prompts]
    if getattr(tok, "chat_template", None):
        try:
            enc = tok.apply_chat_template(conversations, add_generation_prompt=True,
                                          tokenize=True, return_dict=True,
                                          return_tensors="pt", padding=True,
                                          **_THINKING_OFF)
        except (TypeError, ValueError):
            # older/stricter template paths: render to text, then tokenize once
            texts = [tok.apply_chat_template(c, add_generation_prompt=True, tokenize=False,
                                             **_THINKING_OFF)
                     for c in conversations]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    else:
        enc = tok(list(prompts), return_tensors="pt", padding=True)
    return {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}


def _greedy_config(model, max_new_tokens: int, pad_id: int):
    """A copy of the model's generation config with sampling switched off.

    Copied rather than mutated in place, and sampling knobs cleared rather than
    left dangling: Qwen3 ships `do_sample=True, temperature=0.6` by default, and
    a run that samples is not a run anyone can replay.  Everything goes on the
    config rather than into `generate(**kwargs)` — mixing the two is deprecated
    in transformers 5.x.
    """
    base = getattr(model, "generation_config", None)
    cfg = copy.deepcopy(base) if base is not None else transformers.GenerationConfig()
    cfg.do_sample = False
    cfg.num_beams = 1
    cfg.num_return_sequences = 1
    for knob in ("temperature", "top_p", "top_k", "typical_p", "min_p", "epsilon_cutoff"):
        if hasattr(cfg, knob):
            setattr(cfg, knob, None)
    cfg.max_new_tokens = int(max_new_tokens)
    cfg.return_dict_in_generate = True
    # unconditional: the id generate pads finished rows with has to be the id
    # `_new_token_count` strips, and we decode with the tokenizer, not the config
    cfg.pad_token_id = pad_id
    return cfg


def generate(model, tok, prompts: Sequence[str], max_new_tokens: int, seed: int,
             hook: SteeringHook | None = None, batch_size: int = 8) -> list[dict]:
    """Greedy, seeded, batched decoding -> one dict per prompt, in prompt order.

    Left padding, because a right-padded batch would have the decoder continue
    from pad tokens.  The prompt is stripped by slicing token ids, not by string
    surgery on the decoded text, so a completion that happens to repeat the
    prompt survives intact.

    A `hook` must already be attached: silently generating unsteered text from a
    hook nobody registered is the exact failure this harness exists to catch, so
    it is an error here rather than a surprise in the records.
    """
    prompts = [str(p) for p in prompts]
    if not prompts:
        return []
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if hook is not None and not hook.attached:
        raise ValueError("hook is not attached to any module; use "
                         "`with SteeringHook(v, alpha).attach(module) as h:` around generate()")

    pad_id = _ensure_pad_token(tok)
    gen_cfg = _greedy_config(model, max_new_tokens, pad_id)
    eos = _eos_ids(gen_cfg, tok)
    device = _model_device(model)
    model.eval()

    results: list[dict] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        # reseed per batch so results do not depend on how the prompts were chunked
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        with _padding_side(tok, "left"):
            enc = _encode_prompts(tok, batch)
        enc = {k: v.to(device) for k, v in enc.items()}
        prompt_len = int(enc["input_ids"].shape[1])

        probe = _FiniteLogitsProbe()
        with torch.inference_mode():
            out = model.generate(
                **enc,
                generation_config=gen_cfg,
                logits_processor=transformers.LogitsProcessorList([probe]),
            )
        sequences = out.sequences if hasattr(out, "sequences") else out
        new_ids = sequences[:, prompt_len:].detach().to("cpu")

        if probe.bad is None:
            finite = [True] * len(batch)
        else:
            finite = [not bool(b) for b in probe.bad.detach().to("cpu").tolist()]

        for row, ok in zip(new_ids.tolist(), finite):
            n_new = _new_token_count(row, pad_id, eos)
            results.append({
                "completion": tok.decode(row[:n_new], skip_special_tokens=True),
                "n_new_tokens": n_new,
                "finite_logits": bool(ok),
            })

    return results
