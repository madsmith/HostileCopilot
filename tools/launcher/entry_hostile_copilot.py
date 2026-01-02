from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from bootstrapper import bootstrap_and_run


def main() -> int:
    return bootstrap_and_run(target="hostile_copilot", args=sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
