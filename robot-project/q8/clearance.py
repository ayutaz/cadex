"""Cadexが書いた干渉レポート（docs/clearance.md と docs/clearance-sweep.md）を機械可読に要約する。

Cadexは計測をMarkdownの中のJSON/表で残す。ここではそれを読み、
「設計上の問題」と「意図した接触」を分けたJSONにする。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

MIN_GAP_MM = 0.1
MAX_VOL_MM3 = 1e-6


def read_sweep(project: Path) -> dict:
    text = (project / "docs" / "clearance-sweep.md").read_text()
    return json.loads(text[text.index("```json") + 7:text.rindex("```")])


def read_static(project: Path) -> list[dict]:
    rows = []
    text = (project / "docs" / "clearance.md").read_text()
    for line in text.splitlines():
        m = re.match(r"\| (\S+) \(([^)]*)\) \| (\S+) \(([^)]*)\) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|", line)
        if not m:
            continue
        rows.append({"first": m.group(1), "second": m.group(3),
                     "distance_mm": float(m.group(5)), "common_volume_mm3": float(m.group(6)),
                     "verdict": m.group(7).strip()})
    return rows


def summarise(project: Path, intended_contacts: set[frozenset] | None = None) -> dict:
    intended_contacts = intended_contacts or set()
    static = read_static(project)
    sweep = read_sweep(project)
    static_problems = [r for r in static if r["verdict"] not in ("clear", "threaded")
                       and frozenset((r["first"], r["second"])) not in intended_contacts]
    joints = []
    for jt in sweep.get("joints", []):
        contacts = []
        for pair in jt.get("pairs", []):
            key = frozenset((pair["first"], pair["second"]))
            dist = pair.get("minimum_distance_mm")
            vol = pair.get("maximum_common_volume_mm3") or 0.0
            bad = (dist is not None and dist < MIN_GAP_MM) or vol > MAX_VOL_MM3
            if bad and key not in intended_contacts:
                contacts.append({"first": pair["first"], "second": pair["second"],
                                 "minimum_distance_mm": dist, "maximum_common_volume_mm3": vol,
                                 "first_contact_degrees": pair.get("first_contact_degrees")})
        joints.append({"joint": jt["joint"], "status": jt.get("status"),
                       "range_degrees": jt.get("range_degrees"), "step": jt.get("step"),
                       "samples": jt.get("sample_count"), "pairs_moving": jt.get("pairs_moving"),
                       "pairs_culled": jt.get("pairs_culled"), "contacts": contacts})
    return {
        "source": "cadex clearance / clearance --sweep（OCCTの厳密形状で計測）",
        "thresholds": {"minimum_gap_mm": MIN_GAP_MM, "maximum_common_volume_mm3": MAX_VOL_MM3},
        "static_pairs": len(static),
        "static_problems": static_problems,
        "sweep_status": sweep.get("status"),
        "joints": joints,
        "self_collision_free": not static_problems and all(not j["contacts"] for j in joints)
                               and sweep.get("status") == "complete",
    }
