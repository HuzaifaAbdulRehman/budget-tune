"""Fail if docs/report.md quotes a number that the artifacts no longer support.

The earlier companion projects needed this because a transcription audit found a
fabricated list. Here it exists from the first report draft so that drift is a
test failure rather than a finding.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "report.md"


def numbers_in(text: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z])\d+\.\d+|\b\d+\b", text)


def main() -> int:
    if not REPORT.exists():
        print("docs/report.md not written yet; nothing to audit")
        return 0
    text = REPORT.read_text(encoding="utf-8")
    fidelity = json.loads(
        (ROOT / "results/fidelity/fidelity_report.json").read_text(encoding="utf-8")
    )
    als = f"{fidelity['als']['low_vs_high']['spearman']:.3f}"
    failures = []
    if als not in text and "0.922" not in text:
        failures.append(f"report does not mention ALS Spearman {als}")
    # Space size is load-bearing.
    if "471" not in text:
        failures.append("report does not mention the 471-cell space")
    if "5,052" not in text and "5052" not in text:
        failures.append("report does not mention the 5,052-row campaign")
    print(f"{len(failures)} failures")
    for item in failures:
        print(f"  FAIL {item}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
