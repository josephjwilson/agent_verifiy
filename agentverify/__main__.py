"""`python -m agentverify` entry point."""
from __future__ import annotations

import sys
import traceback

EXIT_USAGE = 2


def _entry() -> int:
    """Import the CLI late so that a harness that cannot even load says so.

    `agentverify.cli` pulls in `report`, which imports every module under
    `checks/` and refuses to load if a contracted check id went missing.  That
    failure is a fault in the harness, not a verdict on anyone's run, so it must
    not exit 1 — exit 1 is reserved for "this run failed verification".
    """
    try:
        from .cli import main
    except Exception:
        traceback.print_exc()
        print("agentverify: the harness failed to load — this is a fault in "
              "agentverify itself, not a verdict on any run", file=sys.stderr)
        return EXIT_USAGE
    return main()


if __name__ == "__main__":
    sys.exit(_entry())
