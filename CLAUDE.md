# agent_verifiy — project conventions

Agent verification workflow harness for steering vectors.
**Status: scaffold only. The task spec has not been written down yet — get it
from the human before building the harness.**

## Environment (verified 2026-08-29, do not re-derive)

Conda env **`agent-verify`**, NOT `forking-jspace`.

    source /workspace/activate_verify.sh        # activates env, cds here
    bash   /workspace/claude_verify.sh          # new Claude session, tmux 'verify'

- python 3.11.15, **torch 2.12.0+cu126**, transformers 5.9.0
- `HF_HOME=/workspace/hf-cache` — 53 GB already cached, work OFFLINE
  (`HF_HUB_OFFLINE=1`): Qwen3-1.7B/4B/8B, Gemma-4 E2B/E4B/12B/31B,
  ARC, GSM8K, GSM-Hard.

### The cu126 gotcha — the thing that will waste your afternoon
This machine is a **local RTX 4090, driver 570.195 (CUDA 12.8)**. The sibling
env `forking-jspace` carries torch 2.12.0**+cu130**, which needs driver >= 580:
on this box it reports `NO CUDA` and every tensor silently stays on the CPU —
no crash, just 100x slow. `agent-verify` is a `--clone` of it with torch swapped
to the same version built for **cu126**, which this driver runs.

Rebuild from scratch if ever lost:

    conda create -n agent-verify --clone forking-jspace --override-channels -c conda-forge -y
    $HOME/miniconda3/envs/agent-verify/bin/python -m pip install 'torch==2.12.0+cu126' \
        --index-url https://download.pytorch.org/whl/cu126

Measured after the swap: bf16 matmul **170 TFLOP/s** (4090 dense peak, tensor
cores engaged); Qwen3-1.7B loads to `cuda:0` in 12 s, 4.07 GiB peak, 29 hidden
-state layers at dim 2048 exposed via `output_hidden_states=True`.

### transformers 5.x gotcha
`apply_chat_template(..., return_tensors="pt")` returns a `BatchEncoding`, not a
tensor. Pass `return_dict=True` and splat it: `model.generate(**enc, ...)`.
Without it you get a bare `AttributeError` from `inputs_tensor.shape[0]`.

## GPU
One RTX 4090, 24 GiB, sm_89. Run GPU work directly — there is no job queue on
this project. Check `nvidia-smi` first; one card, so don't stack big jobs.

## Relationship to /workspace/geometry_of_reasoning
Separate project, separate env, separate lineage. Its CLAUDE.md is long and
binding and does NOT apply here (in particular its no-direct-GPU queue rule).
Never edit that repo or its `forking-jspace` env from this one.

## Verification rules
Before reporting any result as a success:
- Run the random_direction control at the same alpha and report its number alongside the learned-vector number. Never report one without the other.
- Report undetermined_rate next to every sycophancy_rate. If undetermined_rate is high, the metric is not measuring what it claims - say so.
- Print 5 randomly selected raw completions and state whether they are coherent.
- Write one paragraph headed "Dumbest way this could be wrong" and run the cheapest check that would rule it out.
Every number you write must cite the file and key it came from.
