# Measured on this box, 2026-08-29 — Qwen3-1.7B, agent-verify env

Measured directly, not inferred. These override any guess in generated code.

| Fact | Value |
|---|---|
| `num_hidden_layers` | 28 |
| `hidden_size` (d_model) | 2048 |
| `len(hidden_states)` | 29 = n_layers + 1 → **output of block L is `hidden_states[L+1]`** |
| block output type | **bare `Tensor`**, NOT a tuple (transformers 5.9 + Qwen3) |
| load time / peak VRAM | 17 s / 3.22 GiB (bf16, `.to("cuda:0")`) |
| bf16 matmul (4096³, 10 iters) | **149.9 TFLOP/s** — GPU confirmed live |
| pad token | none by default → set `tok.pad_token = tok.eos_token` |
| padding side for batched generate | must be `left` |

## The trap that would have faked a null result

**Qwen3-1.7B is a thinking model and `enable_thinking` defaults to True.**

    enable_thinking=True   -> "<think>\nOkay, the user is asking if Lisbon is the capital of..."
    enable_thinking=False  -> "No, Lisbon is not the capital of Spain. The capital of Spain is Madrid."

With thinking on and a small `max_new_tokens`, **every completion is chain-of-thought and
contains no answer at all**. A rule-based sycophancy scorer marks all of them `undetermined`,
both arms score 0.0, and the experiment reports a clean, confident *null effect* — with no
error, no warning, and a full set of valid-looking artifacts.

Consequences for the harness:

1. `apply_chat_template(..., enable_thinking=False, return_dict=True)` everywhere in
   `steering.py`. (`return_dict=True` is separately required: without it you get a
   `BatchEncoding`, not a tensor, and a bare `AttributeError` from `inputs_tensor.shape[0]`.)
2. `scoring.py` must surface the **undetermined rate** as a first-class number, and
   `t2.metric_not_degenerate` must fail a run whose undetermined rate is high — a metric that
   cannot discriminate is not evidence of no effect. This is precisely the class of silent
   failure the harness exists to catch, so it must catch this one.
3. `SteeringHook` must handle a bare-Tensor module output as well as the tuple case.
