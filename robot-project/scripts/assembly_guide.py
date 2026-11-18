"""組立手順書（docs/assembly-guide.md）の図と印刷の計画を作る。

- 印刷部品ごとに、6つの向き（どの面をベッドに置くか）で張り出し（オーバーハング）の面積を STL から計算し、
  最も少ない向きを選ぶ。
- 部品を Bambu Lab A1 mini のベッド（180 x 180 mm）に並べ、プレートごとのフィラメント量を推定する。
- 組立の段ごとに、それまでの部品を灰色、その段で足す部品を橙色にして MuJoCo のオフスクリーン描画で描く。

STL はワールド座標の立位姿勢で出力されている（q8/visual.py と同じ前提）ので、そのまま並べれば組み上がった形になる。

使い方:
  uv run python scripts/assembly_guide.py --variant mini --out artifacts/mini/assembly
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from q8.stl import read_stl, write_binary_stl  # noqa: E402

LEGS = ("lf", "rf", "lh", "rh")
PRINTED_PARTS = (["chassis", "cover"] + [f"thigh_{g}" for g in LEGS] + [f"shin_{g}" for g in LEGS]
                 + [f"foot_{g}" for g in LEGS])
HIP_SERVOS = [f"hip_servo_{g}" for g in LEGS]
KNEE_SERVOS = [f"knee_servo_{g}" for g in LEGS]
HIP_HORNS = [f"hip_horn_{g}" for g in LEGS]
KNEE_HORNS = [f"knee_horn_{g}" for g in LEGS]
BOLTS = ["body_bolt_0", "body_bolt_1"]
ALL_PARTS = PRINTED_PARTS + HIP_SERVOS + KNEE_SERVOS + HIP_HORNS + KNEE_HORNS + BOLTS + ["esp32", "imu"]

BED_MM = (180.0, 180.0)        # Bambu Lab A1 mini の造形範囲（X, Y）
BED_HEIGHT_MM = 180.0
EFFECTIVE_DENSITY_G_CM3 = 0.8  # PLA、充填率約30%、外周4本の想定（docs/hardware-spec.md）
OVERHANG_DEG = 45.0            # 鉛直から 45° より寝た下向きの面を張り出しとみなす（FDM の一般的な目安）

# どの面をベッドに置くか（部品座標での「下」の向き）。6通りを試す。
AXES = ((0, 0, -1), (0, 0, 1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0))

# 1枚のプレートに載せる部品。失敗したときの損を小さくするため、種類ごとに分ける。
PLATES = [
    {"key": "plate1_body", "title": "胴体（chassis と cover）", "parts": ["chassis", "cover"]},
    {"key": "plate2_thighs", "title": "上腿 4本", "parts": [f"thigh_{g}" for g in LEGS]},
    {"key": "plate3_shins", "title": "下腿 4本", "parts": [f"shin_{g}" for g in LEGS]},
    # 足先の球はサポートが要るので、サポートなしで刷れる部品と分ける
    {"key": "plate4_feet", "title": "足先 4個（サポートあり）", "parts": [f"foot_{g}" for g in LEGS]},
]

# 組立の段。shown は描く部品、new はその段で足す部品（橙色）。detail は1脚だけを拡大した段。
# 視点: (方位角, 仰角)。MuJoCo の方位角はカメラが向く方向で、270 なら左（+Y）から、225 なら左前から見る。
_BODY = ["chassis"] + HIP_SERVOS
_CLOSED = _BODY + ["cover"] + BOLTS
_THIGHS = _CLOSED + HIP_HORNS + [f"thigh_{g}" for g in LEGS] + KNEE_SERVOS
_SHINS = _THIGHS + KNEE_HORNS + [f"shin_{g}" for g in LEGS]
_FEET = _SHINS + [f"foot_{g}" for g in LEGS]
STEPS = [
    {"key": "step1_hip_servos", "title": "股サーボ 4個を chassis に入れる",
     "shown": _BODY, "new": HIP_SERVOS, "view": (225.0, -35.0), "zoom": 1.05},
    {"key": "step2_cover", "title": "cover を被せて M2x10 を 2本締める",
     "shown": _CLOSED, "new": ["cover"] + BOLTS, "view": (225.0, -35.0), "zoom": 1.05},
    {"key": "step3_knee_servo", "title": "膝サーボを上腿に入れる（左前脚、外側から見た図）", "detail": True,
     "shown": ["thigh_lf", "knee_servo_lf"], "new": ["knee_servo_lf"], "view": (240.0, -20.0), "zoom": 1.15},
    # leg_rev 2 は膝サーボを内側（胴体の側）から差し込むので、差し込み口とホーンの凹みが見える内側からも描く
    {"key": "step3b_knee_servo_inside", "title": "膝サーボを上腿に入れる（左前脚、内側から見た図）", "detail": True,
     "shown": ["thigh_lf", "knee_servo_lf"], "new": ["knee_servo_lf"], "view": (120.0, -20.0), "zoom": 1.15},
    {"key": "step4_hip_horn", "title": "股ホーンを上腿に留め、股サーボの軸に差す（左前脚）", "detail": True,
     "shown": _CLOSED + ["hip_horn_lf", "thigh_lf", "knee_servo_lf"], "new": ["hip_horn_lf", "thigh_lf"],
     "view": (225.0, -15.0), "zoom": 0.9},
    {"key": "step5_thighs", "title": "4本の上腿を付けた状態",
     "shown": _THIGHS, "new": HIP_HORNS + [f"thigh_{g}" for g in LEGS] + KNEE_SERVOS,
     "view": (225.0, -25.0), "zoom": 1.2},
    {"key": "step6_shins", "title": "膝ホーンと下腿を付ける",
     "shown": _SHINS, "new": KNEE_HORNS + [f"shin_{g}" for g in LEGS], "view": (225.0, -25.0), "zoom": 1.2},
    {"key": "step7_feet", "title": "足先を付ける（滑り止めシートは描いていない）",
     "shown": _FEET, "new": [f"foot_{g}" for g in LEGS], "view": (225.0, -15.0), "zoom": 1.2},
    {"key": "step8_esp32", "title": "完成形（ESP32-S3 を cover 上面のポケットに載せた場合）",
     "shown": _FEET + ["esp32"], "new": ["esp32"], "view": (225.0, -30.0), "zoom": 1.2},
    {"key": "stand_side", "title": "立った姿勢を左から見た図（ホーンを付ける角度の目安）", "detail": True,
     "shown": _FEET + ["esp32"], "new": [f"thigh_{g}" for g in LEGS] + [f"shin_{g}" for g in LEGS],
     "view": (270.0, 0.0), "zoom": 1.0},
]

COLOUR_DONE = "0.80 0.80 0.78 1"      # 付け終わった印刷部品
COLOUR_BOUGHT = "0.22 0.23 0.26 1"    # 付け終わった購入品（サーボ・ホーン・ボルト・基板）
COLOUR_NEW = "0.96 0.52 0.12 1"       # この段で足す部品
COLOUR_PLATE = "0.93 0.62 0.20 1"     # プレートの図の部品


# ---------------------------------------------------------------------------
# 張り出しとプレート
# ---------------------------------------------------------------------------


def rotation_to_bed(down) -> np.ndarray:
    """部品座標の向き down（単位ベクトル）をベッドの下向き (0, 0, -1) に回す回転行列。"""
    a = np.asarray(down, float)
    a = a / np.linalg.norm(a)
    b = np.array([0.0, 0.0, -1.0])
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        return np.diag([1.0, -1.0, -1.0])  # X 軸まわりに 180°
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * (1.0 / (1.0 + c))


def _outward(tris: np.ndarray) -> np.ndarray:
    vol = np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum()
    return tris if vol >= 0 else tris[:, ::-1]


def place_on_bed(tris: np.ndarray, down) -> np.ndarray:
    """down の面を下にして、最も低い点を z=0、外形の隅を x=y=0 に置いた三角形を返す。"""
    t = _outward(tris) @ rotation_to_bed(down).T
    lo = t.reshape(-1, 3).min(axis=0)
    return t - lo


def overhang(tris: np.ndarray, down) -> dict:
    """down の面をベッドに置いたときの、ベッドに接する面積と張り出しの面積（mm²）。

    張り出し = 鉛直から OVERHANG_DEG より寝た下向きの面で、ベッドに接していないもの。
    そのうち水平な下向きの面を「橋渡し（ブリッジ）」として別に数える（両端が支えられていればサポート無しで刷れることが多い）。
    """
    t = place_on_bed(tris, down)
    cross = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2.0
    nz = np.divide(cross[:, 2], 2.0 * area, out=np.zeros_like(area), where=area > 0)
    down_facing = nz < -math.sin(math.radians(OVERHANG_DEG))
    flat = nz < -0.999
    on_bed = flat & (t[:, :, 2].max(axis=1) < 0.05)
    hang = down_facing & ~on_bed
    hi = t.reshape(-1, 3).max(axis=0)
    return {
        "down": [int(x) for x in down],
        "bed_contact_mm2": float(area[on_bed].sum()),
        "overhang_mm2": float(area[hang].sum()),
        "bridge_mm2": float(area[hang & flat].sum()),
        "height_mm": float(hi[2]),
        "footprint_mm": [float(hi[0]), float(hi[1])],
    }


def best_orientation(tris: np.ndarray) -> dict:
    """張り出しが最も少ない向き。同じなら、ベッドに接する面積が大きく、背の低い向き。"""
    reps = [overhang(tris, d) for d in AXES]
    best = min(reps, key=lambda r: (round(r["overhang_mm2"], 1), -round(r["bed_contact_mm2"], 1), r["height_mm"]))
    return {**best, "down": tuple(best["down"]), "all": reps}


def pack_plate(items, *, bed=BED_MM, margin=5.0, gap=5.0) -> list[dict]:
    """(名前, (幅, 奥行き)) を行ごとに左から詰める。収まらなければ ValueError。

    長い辺を X に向ける（必要なら 90° 回す: rotated=True）。
    """
    placed = []
    x = y = margin
    row_d = 0.0
    for name, (w, d) in items:
        rotated = d > w
        if rotated:
            w, d = d, w
        if w > bed[0] - 2 * margin or d > bed[1] - 2 * margin:
            raise ValueError(f"{name} ({w:.1f} x {d:.1f} mm) はベッドに収まらない")
        if x + w > bed[0] - margin:
            x = margin
            y += row_d + gap
            row_d = 0.0
        if y + d > bed[1] - margin:
            raise ValueError(f"{name} を置く場所がベッドに残っていない")
        placed.append({"name": name, "x": x, "y": y, "w": w, "d": d, "rotated": rotated})
        x += w + gap
        row_d = max(row_d, d)
    return placed


def _ray_hits(tris: np.ndarray, origin: np.ndarray, d: np.ndarray) -> np.ndarray:
    """直線 origin + s d と三角形の交点の s（昇順）。Möller–Trumbore。"""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    e1, e2 = b - a, c - a
    h = np.cross(d, e2)
    det = (e1 * h).sum(axis=1)
    ok = np.abs(det) > 1e-12
    f = 1.0 / np.where(ok, det, 1.0)
    s = origin - a
    u = f * (s * h).sum(axis=1)
    q = np.cross(s, e1)
    v = f * (q @ d)
    t = f * (e2 * q).sum(axis=1)
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
    return np.unique(np.round(t[hit], 6))


def blocked_fraction(part: np.ndarray, insert: np.ndarray, direction, *, samples: int = 30,
                     tol: float = 0.05) -> float:
    """insert を direction へまっすぐ動かして今の位置に入れるとき、part に当たる光線の割合（0 なら入る）。

    insert の投影を samples x samples の平行な光線で覆い、insert を通る光線ごとに、
    insert の奥の端より手前（来た側）に part の材料があれば「当たる」と数える。
    """
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    axis = int(np.argmax(np.abs(d)))
    others = [i for i in range(3) if i != axis]
    lo = insert.reshape(-1, 3).min(axis=0)
    hi = insert.reshape(-1, 3).max(axis=0)
    span = float(hi[axis] - lo[axis]) + 1000.0
    total = blocked = 0
    for u in np.linspace(lo[others[0]], hi[others[0]], samples + 2)[1:-1]:
        for w in np.linspace(lo[others[1]], hi[others[1]], samples + 2)[1:-1]:
            origin = np.zeros(3)
            origin[others[0]], origin[others[1]] = u, w
            origin[axis] = (lo[axis] + hi[axis]) / 2.0
            origin = origin - d * span  # 来た側の遠く
            s_ins = _ray_hits(insert, origin, d)
            if len(s_ins) < 2:
                continue
            total += 1
            far = s_ins.max()
            s_part = _ray_hits(part, origin, d)
            if any(s < far - tol for s in s_part[0::2]):
                blocked += 1
    return blocked / total if total else 0.0


def _voxelize(meshes: list[np.ndarray], lo: np.ndarray, h: float, shape) -> np.ndarray:
    """閉じたメッシュの内側を、Y 方向の光線の交点の偶奇で塗る（ボクセルの中心で判定）。"""
    occ = np.zeros(shape, bool)
    ys = lo[1] + np.arange(shape[1]) * h
    d = np.array([0.0, 1.0, 0.0])
    for tris in meshes:
        xmin, xmax = tris[:, :, 0].min(axis=1), tris[:, :, 0].max(axis=1)
        zmin, zmax = tris[:, :, 2].min(axis=1), tris[:, :, 2].max(axis=1)
        for i in range(shape[0]):
            x = lo[0] + i * h
            sel = (xmin <= x) & (xmax >= x)
            if not sel.any():
                continue
            t_x, z0, z1 = tris[sel], zmin[sel], zmax[sel]
            for k in range(shape[2]):
                z = lo[2] + k * h
                s = (z0 <= z) & (z1 >= z)
                if not s.any():
                    continue
                hits = _ray_hits(t_x[s], np.array([x, lo[1] - 50.0, z]), d) + lo[1] - 50.0
                for a, b in zip(hits[0::2], hits[1::2]):
                    occ[i, (ys >= a) & (ys <= b), k] = True
    return occ


def enclosed_void_mm3(part: np.ndarray, insert: np.ndarray, *, h: float = 0.35) -> float:
    """part に insert を入れた状態で、外につながらない空間の体積（mm³）。

    サーボのリード線の逃げなどが外へ抜けているかを見る。すきま（クリアランス 0.5 mm など、線が通れない幅）は
    塞いだものとみなすため、固体を1ボクセル（h）太らせてから外側から塗りつぶす。
    """
    from collections import deque

    pts = np.concatenate([part.reshape(-1, 3), insert.reshape(-1, 3)])
    lo = pts.min(axis=0) - 2.0
    hi = pts.max(axis=0) + 2.0
    shape = tuple(int(n) for n in ((hi - lo) / h).astype(int) + 1)
    solid = _voxelize([part, insert], lo, h, shape)
    grown = solid.copy()
    for ax in range(3):
        grown |= np.roll(solid, 1, ax) | np.roll(solid, -1, ax)
    reach = np.zeros(shape, bool)
    reach[0, 0, 0] = True
    queue = deque([(0, 0, 0)])
    while queue:
        i, j, k = queue.popleft()
        for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            a, b, c = i + di, j + dj, k + dk
            if 0 <= a < shape[0] and 0 <= b < shape[1] and 0 <= c < shape[2] and not grown[a, b, c] \
                    and not reach[a, b, c]:
                reach[a, b, c] = True
                queue.append((a, b, c))
    return float(((~grown) & (~reach)).sum() * h ** 3)


# リード線の逃げが外へ抜けているかを確かめる組（左前脚で代表）
VOID_CHECKS = [
    {"part": "chassis+cover", "insert": "hip_servo_lf", "label": "股サーボを挟んだ胴体（線は胴体の空洞から cover の窓へ）"},
    {"part": "thigh_lf", "insert": "knee_servo_lf", "label": "膝サーボを入れた上腿"},
]


# 組立で「まっすぐ差し込めるか」を確かめる組。direction は insert を動かす向き（左の脚で代表）。
FIT_CHECKS = [
    {"part": "chassis", "insert": "hip_servo_lf", "direction": (0, 0, -1), "label": "股サーボを chassis に上から入れる"},
    {"part": "cover", "insert": "hip_servo_lf", "direction": (0, 0, 1), "label": "cover を上から被せる（サーボから見て上へ）"},
    {"part": "thigh_lf", "insert": "knee_servo_lf", "direction": (0, -1, 0), "label": "膝サーボを上腿に外側から入れる"},
    {"part": "thigh_lf", "insert": "knee_servo_lf", "direction": (0, 1, 0), "label": "膝サーボを上腿に内側から入れる"},
    {"part": "thigh_lf", "insert": "hip_horn_lf", "direction": (0, 1, 0), "label": "股ホーンを上腿の凹みに内側の面から入れる"},
    {"part": "thigh_lf", "insert": "hip_horn_lf", "direction": (0, -1, 0), "label": "股ホーンを上腿の凹みに外側の面から入れる"},
    {"part": "shin_lf", "insert": "knee_horn_lf", "direction": (0, 1, 0), "label": "膝ホーンを下腿の凹みに内側の面から入れる"},
    {"part": "shin_lf", "insert": "knee_horn_lf", "direction": (0, -1, 0), "label": "膝ホーンを下腿の凹みに外側の面から入れる"},
]


def fit_checks(mesh_dir: Path) -> list[dict]:
    out = []
    for c in FIT_CHECKS:
        frac = blocked_fraction(read_stl(mesh_dir / f"{c['part']}.stl"), read_stl(mesh_dir / f"{c['insert']}.stl"),
                                c["direction"])
        out.append({**c, "direction": list(c["direction"]), "blocked_fraction": round(frac, 3)})
    return out


# 組めるかの判定。fit は (part, insert) を、どれか1つの向きから当たらずに入れられれば通る。
VOID_LIMIT_MM3 = 50.0  # これより小さい閉じた空間は、ネジ穴の底などの隅とみなす（線の出口の有無とは関係ない）
VERDICT_FITS = {
    "hip_servo_into_chassis": ("chassis", "hip_servo_lf"),
    "cover_closes": ("cover", "hip_servo_lf"),
    "knee_servo_into_thigh": ("thigh_lf", "knee_servo_lf"),
    "hip_horn_into_thigh": ("thigh_lf", "hip_horn_lf"),
    "knee_horn_into_shin": ("shin_lf", "knee_horn_lf"),
}
VERDICT_VOIDS = {
    "hip_lead_exits": ("chassis+cover", "hip_servo_lf"),
    "knee_lead_exits": ("thigh_lf", "knee_servo_lf"),
}


def assembly_verdict(plan: dict) -> dict:
    """print_plan の fit_checks と enclosed_voids から、組立の要所ごとに組めるか（True/False）。"""
    best: dict = {}
    for c in plan["fit_checks"]:
        key = (c["part"], c["insert"])
        best[key] = min(best.get(key, 1.0), c["blocked_fraction"])
    voids = {(v["part"], v["insert"]): v["enclosed_void_mm3"] for v in plan["enclosed_voids"]}
    out = {name: bool(best[key] <= 0.0) for name, key in VERDICT_FITS.items()}
    out.update({name: bool(voids[key] < VOID_LIMIT_MM3) for name, key in VERDICT_VOIDS.items()})
    return out


def print_plan(mesh_dir: Path) -> dict:
    """印刷部品ごとの向き・張り出し・体積と、プレートごとの配置・フィラメント量（推定）。"""
    parts = {}
    for name in PRINTED_PARTS:
        tris = read_stl(mesh_dir / f"{name}.stl")
        best = best_orientation(tris)
        vol = abs(float(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6.0)) / 1000.0
        parts[name] = {
            "down": list(best["down"]),
            "overhang_mm2": round(best["overhang_mm2"], 1),
            "bridge_mm2": round(best["bridge_mm2"], 1),
            "bed_contact_mm2": round(best["bed_contact_mm2"], 1),
            "height_mm": round(best["height_mm"], 1),
            "footprint_mm": [round(v, 1) for v in best["footprint_mm"]],
            "volume_cm3": round(vol, 3),
            "filament_g": round(vol * EFFECTIVE_DENSITY_G_CM3, 2),
            "orientations": [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()
                              if k in ("down", "overhang_mm2", "bridge_mm2", "bed_contact_mm2", "height_mm")}
                             for r in best["all"]],
        }
    plates = []
    for plate in PLATES:
        items = [(n, tuple(parts[n]["footprint_mm"])) for n in plate["parts"]]
        placed = pack_plate(items)
        plates.append({**plate, "placed": placed,
                       "filament_g": round(sum(parts[n]["volume_cm3"] for n in plate["parts"])
                                           * EFFECTIVE_DENSITY_G_CM3, 6)})
    plan = {
        "bed_mm": list(BED_MM),
        "effective_density_g_cm3": EFFECTIVE_DENSITY_G_CM3,
        "overhang_deg": OVERHANG_DEG,
        "note": "フィラメント量は 体積 x 実効密度 の推定。サポート・ブリム・パージは含まない。印刷時間はスライサーで確認する。",
        "parts": parts,
        "plates": plates,
        "fit_checks_note": "blocked_fraction は insert の投影を覆う平行な光線のうち、差し込む途中で part に当たる割合。0 なら入る。",
        "fit_checks": fit_checks(mesh_dir),
        "enclosed_voids_note": "部品にサーボを入れた状態で外につながらない空間（mm³、0.35 mm のボクセル、すきまは塞いだものとみなす）。"
                               "0 でなければ、リード線の逃げが外へ抜けていない。",
        "enclosed_voids": [{**c, "enclosed_void_mm3": round(enclosed_void_mm3(
            np.concatenate([read_stl(mesh_dir / f"{n}.stl") for n in c["part"].split("+")]),
            read_stl(mesh_dir / f"{c['insert']}.stl")), 0)} for c in VOID_CHECKS],
    }
    plan["assembly_verdict_note"] = ("組立の要所ごとの判定。fit はどれか1つの向きで当たる光線が 0、"
                                     f"リード線は閉じた空間が {VOID_LIMIT_MM3:g} mm³ 未満なら True。")
    plan["assembly_verdict"] = assembly_verdict(plan)
    return plan


# ---------------------------------------------------------------------------
# 描画（MuJoCo のオフスクリーン）
# ---------------------------------------------------------------------------


def colour_of(name: str, new: set[str]) -> str:
    if name in new:
        return COLOUR_NEW
    return COLOUR_DONE if name in PRINTED_PARTS else COLOUR_BOUGHT


def scene_xml(meshes: dict[str, Path], *, colours: dict[str, str], size=(800, 560), floor_z=None,
              plate=None) -> str:
    """部品の STL（mm、ワールド座標）を動かない見た目だけの形として並べた MJCF。"""
    assets = ['<texture name="sky" type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0.86 0.88 0.91" '
              'width="64" height="64"/>']
    geoms = []
    for i, (name, path) in enumerate(meshes.items()):
        assets.append(f'<mesh name="m{i}" file="{path}" scale="0.001 0.001 0.001"/>')
        geoms.append(f'<geom name="{name}" type="mesh" mesh="m{i}" rgba="{colours[name]}" '
                     'contype="0" conaffinity="0"/>')
    if floor_z is not None:
        geoms.append(f'<geom name="floor" type="plane" size="1 1 0.01" pos="0 0 {floor_z}" '
                     'rgba="0.93 0.94 0.95 1" contype="0" conaffinity="0"/>')
    if plate is not None:
        w, d = plate
        geoms.append(f'<geom name="bed" type="box" size="{w / 2000} {d / 2000} 0.001" '
                     f'pos="{w / 2000} {d / 2000} -0.001" rgba="0.25 0.26 0.29 1" contype="0" conaffinity="0"/>')
    w, h = size
    return (f'<mujoco model="assembly_guide"><compiler angle="radian"/>'
            f'<visual><global offwidth="{w}" offheight="{h}"/>'
            '<headlight ambient="0.42 0.42 0.42" diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1"/>'
            '<quality shadowsize="2048"/></visual>'
            f'<asset>{"".join(assets)}</asset>'
            '<worldbody><light pos="0.15 0.25 0.6" dir="-0.2 -0.35 -1" diffuse="0.45 0.45 0.45" castshadow="true"/>'
            f'{"".join(geoms)}</worldbody></mujoco>')


def _bounds(meshes: dict[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    pts = np.concatenate([read_stl(p).reshape(-1, 3) for p in meshes.values()])
    return pts.min(axis=0), pts.max(axis=0)


def render(meshes: dict[str, Path], colours: dict[str, str], *, view, zoom, out: Path, size=(800, 560),
           floor=True, plate=None, lookat=None) -> Path:
    import mujoco

    lo, hi = _bounds(meshes)
    if plate is not None:
        lo = np.minimum(lo, [0, 0, 0])
        hi = np.maximum(hi, [plate[0], plate[1], 0])
    xml = scene_xml(meshes, colours=colours, size=size, floor_z=(lo[2] / 1000.0 - 1e-4) if floor else None,
                    plate=plate)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation = view
    cam.lookat[:] = (np.asarray(lookat, float) if lookat is not None else (lo + hi) / 2.0) / 1000.0
    cam.distance = float(np.linalg.norm(hi - lo)) / 1000.0 * zoom
    r = mujoco.Renderer(m, size[1], size[0])
    r.update_scene(d, cam)
    rgb = r.render()
    r.close()
    save_png(rgb, out)
    return out


def step_images(mesh_dir: Path, out_dir: Path, size=(800, 560)) -> list[Path]:
    paths = []
    for step in STEPS:
        meshes = {n: mesh_dir / f"{n}.stl" for n in step["shown"]}
        new = set(step["new"])
        colours = {n: colour_of(n, new) for n in meshes}
        if step["key"] == "stand_side":
            colours = {n: colour_of(n, set()) for n in meshes}  # 立位の参考図は色分けしない
        lookat = None
        if step["key"] == "step4_hip_horn":
            lookat = (_bounds({n: mesh_dir / f"{n}.stl" for n in ("thigh_lf", "hip_horn_lf")})[0]
                      + _bounds({n: mesh_dir / f"{n}.stl" for n in ("thigh_lf", "hip_horn_lf")})[1]) / 2.0
        paths.append(render(meshes, colours, view=step["view"], zoom=step["zoom"],
                            out=out_dir / f"{step['key']}.png", size=size, lookat=lookat))
    return paths


def plate_images(mesh_dir: Path, plan: dict, out_dir: Path, size=(800, 560)) -> list[Path]:
    paths = []
    with tempfile.TemporaryDirectory() as tmp:
        for plate in plan["plates"]:
            meshes = {}
            for p in plate["placed"]:
                part = plan["parts"][p["name"]]
                t = place_on_bed(read_stl(mesh_dir / f"{p['name']}.stl"), part["down"])
                if p["rotated"]:
                    t = t[..., [1, 0, 2]] * np.array([-1.0, 1.0, 1.0])
                    t = t - t.reshape(-1, 3).min(axis=0)
                t = t + np.array([p["x"], p["y"], 0.0])
                path = Path(tmp) / f"{p['name']}.stl"
                write_binary_stl(t, path)
                meshes[p["name"]] = path
            colours = {n: COLOUR_PLATE for n in meshes}
            paths.append(render(meshes, colours, view=(90.0, -70.0), zoom=1.0, out=out_dir / f"{plate['key']}.png",
                                size=size, floor=False, plate=BED_MM,
                                lookat=(BED_MM[0] / 2.0, BED_MM[1] / 2.0, 0.0)))
    return paths


# ---------------------------------------------------------------------------
# PNG（標準ライブラリだけ。行ごとにフィルタを選んで小さくする）
# ---------------------------------------------------------------------------


def _paeth(a, b, c):
    a = a.astype(np.int16)
    b = b.astype(np.int16)
    c = c.astype(np.int16)
    p = a + b - c
    pa, pb, pc = np.abs(p - a), np.abs(p - b), np.abs(p - c)
    return np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c)).astype(np.uint8)


def save_png(rgb: np.ndarray, path: Path) -> None:
    """RGB 8bit の PNG を書く。行ごとに 5種のフィルタから、差の絶対値の和が最小のものを選ぶ。"""
    img = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = img.shape
    x = img.reshape(h, w * 3)
    left = np.zeros_like(x)
    left[:, 3:] = x[:, :-3]
    up = np.zeros_like(x)
    up[1:] = x[:-1]
    upleft = np.zeros_like(x)
    upleft[1:, 3:] = x[:-1, :-3]
    cands = [x, x - left, x - up, x - ((left.astype(np.uint16) + up) // 2).astype(np.uint8), x - _paeth(left, up, upleft)]
    cost = np.stack([np.abs(c.astype(np.int8).astype(np.int16)).sum(axis=1) for c in cands])
    pick = cost.argmin(axis=0)
    rows = np.stack(cands)[pick, np.arange(h)]
    raw = np.concatenate([pick.astype(np.uint8)[:, None], rows], axis=1).tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def unfilter_png(data: bytes, w: int, h: int) -> np.ndarray:
    """save_png の IDAT（伸張後）を画素に戻す（テスト用）。"""
    stride = w * 3
    out = np.zeros((h, stride), np.int32)
    buf = np.frombuffer(data, np.uint8).reshape(h, stride + 1)
    for y in range(h):
        f = buf[y, 0]
        row = buf[y, 1:].astype(np.int32)
        prev = out[y - 1] if y else np.zeros(stride, np.int32)
        cur = np.zeros(stride, np.int32)
        for i in range(stride):
            a = cur[i - 3] if i >= 3 else 0
            b = prev[i]
            c = prev[i - 3] if i >= 3 else 0
            if f == 0:
                pred = 0
            elif f == 1:
                pred = a
            elif f == 2:
                pred = b
            elif f == 3:
                pred = (a + b) // 2
            else:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
            cur[i] = (row[i] + pred) & 0xFF
        out[y] = cur
    return out.reshape(h, w, 3).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="mini")
    ap.add_argument("--out", type=Path, default=None, help="既定: artifacts/<variant>/assembly")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=560)
    ap.add_argument("--no-images", action="store_true", help="印刷の計画（JSON）だけを書く")
    args = ap.parse_args()
    mesh_dir = ROOT / "robot" / "meshes" / args.variant
    out = args.out or ROOT / "artifacts" / args.variant / "assembly"
    missing = [n for n in ALL_PARTS if n != "imu" and not (mesh_dir / f"{n}.stl").exists()]
    if missing:
        print(f"STL が無い: {missing}。先に scripts/build_robot.py --variant {args.variant} を実行する", file=sys.stderr)
        return 1
    plan = print_plan(mesh_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "print_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1) + "\n")
    print(out / "print_plan.json")
    if not args.no_images:
        size = (args.width, args.height)
        for p in plate_images(mesh_dir, plan, out, size) + step_images(mesh_dir, out, size):
            print(p, f"{p.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
