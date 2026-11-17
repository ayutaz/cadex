"""製造の検証（印刷可否の簡易チェック）と部品表（BOM）。"""

from __future__ import annotations

import json
from pathlib import Path

from .stl import mesh_stats

PRINTED = ("chassis", "cover", "thigh_", "shin_", "foot_")
# 一般的な家庭用FDM（Bambu Lab A1 / Ender-3 系）の造形範囲 [mm]
BED_MM = (220.0, 220.0, 250.0)
NOZZLE_MM = 0.4


def is_printed(stem: str) -> bool:
    return stem.startswith(PRINTED)


def printability(mesh_dir: Path, params: dict) -> dict:
    parts = {}
    for stl in sorted(mesh_dir.glob("*.stl")):
        if not is_printed(stl.stem):
            continue
        st = mesh_stats(stl)
        dims = sorted(st["bbox_mm"])
        fits = all(d <= b for d, b in zip(dims, sorted(BED_MM)))
        material = "TPU" if stl.stem.startswith("foot_") else "PLA"
        parts[stl.stem] = {"material": material, "bbox_mm": st["bbox_mm"],
                           "solid_volume_cm3": round(st["volume_mm3"] / 1000, 2), "fits_bed": fits,
                           "triangles": st["triangles"]}
    wall = float(params.get("wall", 0))
    link = float(params.get("link_thickness", 0))
    return {
        "bed_mm": BED_MM,
        "parts": parts,
        "checks": {
            "all_parts_fit_bed": all(p["fits_bed"] for p in parts.values()),
            "wall_at_least_4_perimeters": wall >= 4 * NOZZLE_MM,
            "link_plate_at_least_5mm": link >= 5.0,
            "servo_pocket_clearance_mm": params.get("clearance"),
        },
        "assumptions": [
            "肉厚は設計パラメータの値で判定（形状全体の最小肉厚を計測したものではない）",
            "張り出し（オーバーハング）は未計測。印刷向き: シャーシは底面、カバーは上面を下、脚リンクは板面を下",
            "サーボ収納部のすきま 0.5 mm は一般的なFDMの目安。実機の試し刷りで要調整",
        ],
    }


def bom(inventory: list[dict], printing: dict, prices_path: Path, pla_density: float) -> dict:
    prices = json.loads(prices_path.read_text())["items"]
    purchased: dict[str, dict] = {}
    for row in inventory:
        stem = row["output"]
        if is_printed(stem):
            continue
        if row["family"]:
            key = f"{row['family']}/{row['sku']}"
        else:
            key = "standin/" + {"esp32": "esp32s3", "imu": "imu", "battery": "battery"}.get(stem, stem)
        entry = purchased.setdefault(key, {"qty": 0, "components": []})
        entry["qty"] += 1
        entry["components"].append(row["component"])
    n_joints = purchased.get("servo/mg90s", purchased.get("servo/sg90", {"qty": 0}))["qty"]
    purchased["extra/self_tap_m2x6"] = {"qty": 2 * n_joints, "components": []}
    purchased["extra/wiring"] = {"qty": 1, "components": []}
    lines = []
    total = 0.0
    for key, entry in sorted(purchased.items()):
        info = prices.get(key, {"name": key, "unit_price": None, "status": "不明"})
        cost = None if info.get("unit_price") is None else info["unit_price"] * entry["qty"]
        total += cost or 0.0
        lines.append({"item": key, "name": info["name"], "qty": entry["qty"], "unit_price_jpy": info.get("unit_price"),
                      "cost_jpy": cost, "price_status": info.get("status"), "components": entry["components"]})
    # フィラメント: 形状の体積 × 実効密度（充填率込み）
    pla_g = sum(p["solid_volume_cm3"] for p in printing["parts"].values() if p["material"] == "PLA") * pla_density / 1000
    tpu_g = sum(p["solid_volume_cm3"] for p in printing["parts"].values() if p["material"] == "TPU") * 1.2
    for key, grams in (("filament/pla", pla_g), ("filament/tpu", tpu_g)):
        info = prices[key]
        cost = grams * info["unit_price_per_g"]
        total += cost
        lines.append({"item": key, "name": info["name"], "qty": f"{grams:.1f} g", "unit_price_jpy": info["unit_price_per_g"],
                      "cost_jpy": round(cost), "price_status": info["status"], "components": []})
    printed = [{"part": k, "material": v["material"], "qty": 1} for k, v in printing["parts"].items()]
    return {"purchased": lines, "printed": printed, "estimated_total_jpy": round(total),
            "budget_jpy": 10000, "within_budget": total <= 10000,
            "note": "価格は推定。電池・充電・降圧電源は未計上（後回しの指定）"}
