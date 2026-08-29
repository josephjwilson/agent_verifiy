# agentverify

A verification harness for AI-assisted research, plus a worked example of what it catches.
It assumes the experiment may be silently broken and the write-up may be generous: 31 checks
(T0 environment, T1 plumbing, T2 statistics, T3 integrity, T4 claims) re-derive every reported
number from artifacts on disk. `tests/` plants 19 known faults and asserts each trips its named
check — a harness that cannot fail catches nothing.

    python -m pytest tests/ -q                    # 26 tests, CPU only, ~2s

## Reproduce the worked example

    source /workspace/activate_verify.sh
    export HF_HUB_OFFLINE=1

    python -m agentverify run --run-id A-baseline --layer 14 --alpha 0.0 \
        --n-eval 20 --n-pairs 32 --max-new-tokens 40 --no-steering-enabled
    python -m agentverify run --run-id A-steered-alpha8 --layer 14 --alpha 8.0 \
        --n-eval 20 --n-pairs 32 --max-new-tokens 40 --companion baseline=A-baseline
    python -m agentverify run --run-id C-random-alpha8 --layer 14 --alpha 8.0 \
        --n-eval 20 --n-pairs 32 --max-new-tokens 40 --companion baseline=A-baseline \
        --vector-source random_direction
    python -m agentverify verify --run outputs/A-steered-alpha8

All four commands above were executed in this environment and work as written. All runs are
**committed**, so `verify` reproduces the verdict from a fresh clone with no GPU and no model
download. `acts.npz` is kept (it is most of the repo's 31 MB) because three T1 checks read it:
`t1.activation_delta_matches_alpha`, `t1.no_effect_before_layer`, `t1.effect_after_layer`. Outputs land in `outputs/<run-id>/`: `records.jsonl` (per-item prompt, completion, score),
`manifest.json` (config, env, vector norm, hashes), `vector.npz`, `acts.npz`, `claims.json`.
Session transcripts are in `transcripts/`.

**The catch.** Learned vector and norm-matched random direction, both at alpha 8 (applied norm
329.34), give identical `sycophancy_rate` 0.000 with `undetermined_rate` 1.000: every completion
is degenerate repetition, scored as a flawless cure. The learned direction does no work.

## Known limitations
- n=20 in every run here, below the n>=30 threshold `t2.sample_size_adequate` enforces.
- The alpha=1 random-direction control was **not** run; only alpha=8.
- A `skip` on `t2.random_direction_null` means no run declared a `random_direction` companion,
  not that the control is unnecessary.
- `t2.metric_not_degenerate` gates only the baseline arm, so a fully degenerate treatment arm
  still passes it — the gap this example exposes, not yet fixed.
- Run A's numbers predate a `scoring.py` edit and no longer reproduce; A and B baselines are
  not comparable.
