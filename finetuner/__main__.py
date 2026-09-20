from __future__ import annotations

import os
import sys

from finetuner.reloader import CHILD_ENV, run_reloader


def _strip_flags() -> None:
    sys.argv = [sys.argv[0], *[arg for arg in sys.argv[1:] if arg not in {"--reload", "--no-reload"}]]


def main() -> None:
    frozen = bool(getattr(sys, "frozen", False))
    child = os.environ.get(CHILD_ENV) == "1"
    no_reload = "--no-reload" in sys.argv
    if frozen or child or no_reload:
        from finetuner.app import main as app_main

        _strip_flags()
        app_main()
        return
    raise SystemExit(run_reloader())


if __name__ == "__main__":
    main()
