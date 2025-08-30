import json
import sys
from collections import defaultdict
from pathlib import Path

def main() -> int:
    # Default to project-root locations.json (script is in scripts/)
    default_path = Path(__file__).resolve().parents[1] / "locations.json"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    by_code = defaultdict(list)
    # data is expected to be { "<key>": { "id": ..., "name": ..., "code": ... }, ... }
    for key, loc in data.items():
        code = loc.get("code")
        if code:
            by_code[code].append((key, loc.get("name"), loc.get("id")))

    dupes = {code: entries for code, entries in by_code.items() if len(entries) > 1}

    if not dupes:
        print("No duplicate codes found.")
        return 0

    for code, entries in sorted(dupes.items(), key=lambda x: str(x[0])):
        print(f"Code: {code} ({len(entries)} occurrences)")
        for key, name, loc_id in entries:
            print(f"  - key={key} name={name} id={loc_id}")
        print()
    return 0

if __name__ == "__main__":
    sys.exit(main())