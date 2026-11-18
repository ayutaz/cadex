"""機体バリアント（例: 基準 vs 脚20%増し）の比較。設計値・固定歩容・学習済み方策の評価を並べる。"""

from __future__ import annotations

from statistics import mean

from q8.physics import fewest_support

# Phase 7 で求められた評価項目と CadexEvaluation の指標の対応
KEY_METRICS = {
    "completed": "最後まで転倒・沈み込みなし（1=成功）",
    "falls": "転倒（1=転倒）",
    "mean_forward_speed_mm_s": "平均前進速度 [mm/s]",
    "speed_ratio": "速度比（実速度 / 指令）",
    "speed_error_rms_mm_s": "目標速度追従誤差 RMS [mm/s]",
    "max_tilt_deg": "最大傾斜 [deg]（姿勢安定性）",
    "max_heading_deg": "向きのずれの最大 [deg]（直進性。旋回指令は未学習）",
    "steps_min": "最少の脚の歩数（すり足でないこと）",
    "slip_share_max": "接地中の滑りの割合（最大の脚）",
    "effort_rms_rated_ratio_max": "モーター負荷: RMSトルク / 推定連続定格（最大の関節）",
    "effort_peak_ratio_max": "モーター負荷: ピークトルク / 失速トルク",
    "cost_of_transport": "移動コスト（機械仕事 / 重量 × 距離）",
    "command_rate_rms": "指令の変化率 RMS [range/s]（滑らかさ）",
}
DESIGN_KEYS = ["total_mass_g", "com_height_mm", "support_margin_mm", "trot_rated_ratio", "hold_rated_ratio",
               "thigh_mm", "shin_mm", "bom_jpy"]


def summarise_seeds(result: dict | None, keys) -> dict | None:
    if not result:
        return None
    seeds = result["seeds"]
    out = {"pass_rate": sum(1 for s in seeds if s["pass"]) / len(seeds) if seeds else None, "metrics": {}}
    for k in keys:
        vals = [s["metrics"].get(k) for s in seeds]
        vals = [float(v) for v in vals if v is not None]
        out["metrics"][k] = ({"mean": mean(vals), "min": min(vals), "max": max(vals), "n": len(vals)}
                             if vals else None)
    return out


def design_summary(report: dict) -> dict:
    ph = report["physics"]
    return {"total_mass_g": ph["total_mass_g"], "com_height_mm": ph["com_height_mm"],
            "support_margin_mm": ph["support_margin_mm"],
            "trot_rated_ratio": fewest_support(ph)["rated_ratio"],
            "hold_rated_ratio": ph["sim_hold"]["worst_rated_ratio"],
            "thigh_mm": ph["leg"]["thigh_mm"], "shin_mm": ph["leg"]["shin_mm"],
            "bom_jpy": report["bom"]["estimated_total_jpy"]}


def _change(a, b):
    if a is None or b is None or a == 0:
        return None
    return (b - a) / abs(a)


def _row(a, b, labels):
    return {labels[0]: a, labels[1]: b, "change": _change(a, b)}


def _section(a: dict | None, b: dict | None, labels) -> dict | None:
    if not a or not b:
        return None
    sa, sb = summarise_seeds(a, KEY_METRICS), summarise_seeds(b, KEY_METRICS)
    out = {"pass_rate": _row(sa["pass_rate"], sb["pass_rate"], labels)}
    for k in KEY_METRICS:
        ma, mb = sa["metrics"][k], sb["metrics"][k]
        out[k] = _row(None if ma is None else ma["mean"], None if mb is None else mb["mean"], labels)
    return out


def compare(base: dict, variant: dict, *, labels=("base", "variant")) -> dict:
    da, db = base.get("design") or {}, variant.get("design") or {}
    return {
        "labels": list(labels),
        "design": {k: _row(da.get(k), db.get(k), labels) for k in sorted(set(da) | set(db))},
        "fixed_gait": _section(base.get("fixed_gait"), variant.get("fixed_gait"), labels),
        "policy": _section(base.get("policy"), variant.get("policy"), labels),
    }


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def to_markdown(c: dict) -> str:
    a, b = c["labels"]
    lines = [f"# 比較: {a} と {b}", ""]
    for title, key in (("設計値", "design"), ("固定歩容（学習なし）", "fixed_gait"), ("学習済み方策", "policy")):
        lines.append(f"## {title}")
        sec = c.get(key)
        if not sec:
            lines += ["", "未評価（データなし）", ""]
            continue
        lines += ["", f"| 指標 | {a} | {b} | 変化 |", "|---|---|---|---|"]
        for name, row in sec.items():
            label = KEY_METRICS.get(name, name)
            ch = row.get("change")
            lines.append(f"| {label} | {_fmt(row.get(a))} | {_fmt(row.get(b))} | "
                         f"{'—' if ch is None else f'{ch * 100:+.1f}%'} |")
        lines.append("")
    return "\n".join(lines)
