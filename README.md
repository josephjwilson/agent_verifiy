# agent_verifiy

Agent verification workflow harness for steering vectors.

## Quick start

    bash /workspace/claude_verify.sh     # Claude session, env + repo wired up

or, for a plain shell:

    source /workspace/activate_verify.sh
    python -c "import torch; print(torch.cuda.get_device_name(0))"

Environment details, the cu126/driver trap, and the model cache are documented
in [CLAUDE.md](CLAUDE.md).
