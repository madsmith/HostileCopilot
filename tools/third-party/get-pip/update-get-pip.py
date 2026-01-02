import urllib.request
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    target = here / "get-pip.py"

    url = "https://bootstrap.pypa.io/get-pip.py"
    with urllib.request.urlopen(url) as resp:
        data = resp.read()

    target.write_bytes(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
