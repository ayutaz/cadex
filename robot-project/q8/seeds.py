"""学習の乱数（学習シード）を変えた複数回の評価を、機体ごとにまとめて比べる。

入力は ``evaluation/evaluate_variant.py`` の ``summary.json``（1回の学習につき1つ）。各回の
「評価10シードの平均」を1つの値とみなし、学習シードの方向に平均・標準偏差・最小・最大をとる。
二機体の比較では、学習シードによる幅（最小〜最大）が重なるかを示す。重ならなければ、その差は
今回の回数で見た乱数の幅を超えている。
"""

from __future__ import annotations

from statistics import mean, stdev

from .compare import KEY_METRICS


def _spread(values: list[float]) -> dict:
    return {"values": values, "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values)}


def across(summaries: list[dict]) -> dict:
    """同じ機体の複数回の summary.json → 学習シード方向の集計。"""

    if not summaries:
        raise ValueError("summary が1つもありません")
    variants = {s.get("variant") for s in summaries}
    if len(variants) != 1:
        raise ValueError(f"機体が混ざっています: {sorted(map(str, variants))}")
    policies = [s["policy"] for s in summaries]
    keys = list(dict.fromkeys(k for p in policies for k in p.get("metrics", {})))
    metrics = {}
    for key in keys:
        values = [p["metrics"].get(key) for p in policies]
        values = [float(v["mean"]) for v in values if v]
        metrics[key] = _spread(values) if values else None
    return {"variant": variants.pop(), "n": len(summaries),
            "pass_rate": _spread([float(p["pass_rate"]) for p in policies]), "metrics": metrics}


def _row(a: dict | None, b: dict | None) -> dict | None:
    if not a or not b:
        return None
    return {"a": a, "b": b, "difference": b["mean"] - a["mean"],
            "separated": a["max"] < b["min"] or b["max"] < a["min"]}


def compare(a: dict, b: dict, *, labels=("base", "variant")) -> dict:
    keys = list(dict.fromkeys([*a["metrics"], *b["metrics"]]))
    return {"labels": list(labels), "n": [a["n"], b["n"]],
            "pass_rate": _row(a["pass_rate"], b["pass_rate"]),
            "metrics": {k: _row(a["metrics"].get(k), b["metrics"].get(k)) for k in keys}}


def _cell(s: dict) -> str:
    std = f" ± {s['std']:.2f}" if s["std"] is not None else ""
    return f"{s['mean']:.2f}{std}（{s['min']:.2f}〜{s['max']:.2f}）"


def _line(name: str, row: dict) -> str:
    verdict = "重ならない" if row["separated"] else "重なる"
    return f"| {name} | {_cell(row['a'])} | {_cell(row['b'])} | {row['difference']:+.2f} | {verdict} |"


def to_markdown(c: dict) -> str:
    a, b = c["labels"]
    na, nb = c["n"]
    n = f"学習シード {na} 回" if na == nb else f"学習シード {na} 回 / {nb} 回"
    lines = [
        f"# {a} と {b} の比較（{n}）",
        "",
        "各回の値は、評価10シードの平均。表は学習シード方向の 平均 ± 標準偏差（最小〜最大）。",
        "「幅」は二機体の最小〜最大が重なるかどうか。重ならなければ、差は今回の回数で見た乱数の幅を超えている。",
        "",
        f"| 項目 | {a} | {b} | 差（{b} − {a}） | 幅 |",
        "|---|---|---|---|---|",
    ]
    if c["pass_rate"]:
        lines.append(_line("合格率", c["pass_rate"]))
    for key, row in c["metrics"].items():
        if row:
            lines.append(_line(KEY_METRICS.get(key, key), row))
    return "\n".join(lines) + "\n"
