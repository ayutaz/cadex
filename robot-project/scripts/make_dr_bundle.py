"""実機との差に強くする学習バンドルを作る（q8/dr_bundle.py、docs/experiment-results.md §11）。

入力: robot/mjcf/mini-r2-p7-eff/cut/{cut.xml, walk-task.json}
出力: artifacts/mini-r2-p7-eff-dr/cut/{cut.xml, walk-task.json, dr.json}

使い方:
  uv run python scripts/make_dr_bundle.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from q8 import dr_bundle  # noqa: E402
from q8.paths import ARTIFACTS, MJCF_DIR  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=MJCF_DIR / "mini-r2-p7-eff" / "cut")
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "mini-r2-p7-eff-dr" / "cut")
    args = parser.parse_args()
    summary = dr_bundle.build(args.source, args.out)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
