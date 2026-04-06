from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from relbench_experiment import parse_args, run_relbench  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_relbench(args)


if __name__ == "__main__":
    main()
