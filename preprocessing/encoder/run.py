#!/usr/bin/env python3
"""Top-level launcher for the Jev/TMCAD experiments.

This file is intentionally kept at the project root so the complete workflow can
be run with commands such as::

    python run.py scan --dataset "D:\\Postdoc\\research\\jepa_cad\\TMCAD"
    python run.py extract --workers 1
    python run.py jev --splits train val test --max-per-class 20
    python run.py evaluate
    python run.py qualitative

or, for the end-to-end pilot::

    python run.py all --dataset "D:\\Postdoc\\research\\jepa_cad\\TMCAD" \
        --pilot-per-class 20 --workers 1

The launcher adds ``src`` to ``sys.path`` so it can find the local package even
before ``pip install -e .`` has been run. Project dependencies must still be
installed as described in README.md.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.jev_tmcad.cli import build_parser  # noqa: E402
from src.jev_tmcad.utils import setup_logging  # noqa: E402


def _print_quick_start(parser) -> None:
    parser.print_help()
    print(
        "\nRecommended Windows workflow:\n"
        '  python run.py scan --dataset "D:\\Postdoc\\research\\jepa_cad\\TMCAD"\n'
        "  python run.py extract --workers 1\n"
        "  python run.py inspect-state -n 3\n"
        "  python run.py jev --splits test --max-per-class 2 --dry-run\n"
        "  python run.py jev --splits train val test --max-per-class 20\n"
        "  python run.py evaluate\n"
        "  python run.py qualitative\n"
        "\nEnd-to-end pilot:\n"
        '  python run.py all --dataset "D:\\Postdoc\\research\\jepa_cad\\TMCAD" '
        "--pilot-per-class 20 --workers 1\n"
        "\nPut the TypeSafe key in .env as TYPESAFE_API_KEY=... before real Jev calls.\n"
    )


def main() -> None:
    parser = build_parser()
    parser.prog = "python run.py"

    # Running a paid API workflow implicitly would be unsafe. With no command,
    # show the available commands and a concrete quick-start instead.
    if len(sys.argv) == 1:
        _print_quick_start(parser)
        return

    args = parser.parse_args()
    setup_logging(args.log_level)
    args.func(args)


if __name__ == "__main__":
    main()
