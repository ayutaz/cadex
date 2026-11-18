"""実機との差に強くする学習バンドル（段階A-2、docs/experiment-results.md §11）。

採用した機体 mini-r2-p7-eff のカットモデルのバンドル（``robot/mjcf/mini-r2-p7-eff/cut/``）に、
Cadex の学習器（``training/cadex_train.py``、変えない）がそのまま読める形でランダム化を足す。
学習器のランダム化は「モデルの配列の要素に、環境ごとに引いた係数を掛ける」だけ（``mode: scale``）なので、
掛け算で表せるようにモデルの側を少し変える。

足すもの（どれも環境ごとに1回引き、その回の学習のあいだ固定。学習器の規則）:

- **サーボの応答の遅れ**: 8つのアクチュエータを一次遅れ（``dyntype="filterexact"``、τ = 15 ms）にし、
  τ に共通の係数 ×0.1〜2.0 を掛ける（1.5〜30 ms）。MG90S の PWM は 20 ms ごとにしか指令を読まないので、
  指令が効くまでに 0〜20 ms の遅れがあり、それに推論と通信の遅れが乗る。純粋な遅延は学習器に無いので、
  一次遅れで近似する。
- **サーボの剛性（位置の制御の強さ）**: アクチュエータごとに ×0.7〜1.3。位置サーボの力は
  ``gain·ctrl + bias1·q`` なので、``gainprm[0]`` と ``biasprm[1]`` に同じ係数を掛ける（位置サーボのまま）。
- **重心のずれ**: 胴体（c_chassis）から 20 g を、胴体の最後の子の ``dr_payload``（関節なし）に移し、
  その重心の x・y に係数 ×−1〜1 を掛ける。payload の基準位置は胴体の重心の高さで x = +30 mm、y = +20 mm なので、
  機体全体（220.1 g）の重心は x に ±2.73 mm、y に ±1.82 mm ずれる（MuJoCo で計算。``dr.json``）。
  総質量は変わらない。元のタスクの胴体の質量 ×0.85〜1.15 は、残りの胴体（86.7 g）に掛かる。
- **押し**: 胴体の重心に水平の力（全周、0〜1.5 N、0.1 s）を2回（開始 1.0〜3.5 s と 4.0〜7.0 s）。
  1.5 N を 0.1 s は、220 g の機体の速度を約 0.68 m/s 変える。評価（``success``）には足さない。
  今の候補（s3）は 1.5 N まで全部戻り、2.0 N で半分が倒れた（q8/robustness.py の押しの評価）。

足の摩擦（0.35〜1.25）とサーボのトルク（×0.8〜1.1）は、学習器のフラグ（``--randomise-friction``、
``--randomise-drive-torque``）で学習設定に書く（``configs/train_full_mini-r2-p7-eff-dr.json``）。
ガタ（バックラッシュ）と制御周期のゆらぎは、学習器でもモデルでも表せないので、ここには無い（評価だけで扱う）。

既存の body・joint・sensor・actuator の番号は変えない（payload は body の末尾に付く）。元のタスクの
ランダム化・リセット・報酬・成功条件はそのまま。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import mujoco

from .portable import write_json

#: 一次遅れの時定数（公称）と、それに掛ける係数の範囲
SERVO_LAG_S = 0.015
SERVO_LAG_SCALE = (0.1, 2.0)
#: 位置サーボの剛性に掛ける係数の範囲
SERVO_KP_SCALE = (0.7, 1.3)
#: 胴体から移す質量と、その重心の基準位置（胴体の座標、胴体の重心の高さ）と係数の範囲
PAYLOAD_KG = 0.020
PAYLOAD_XY_M = (0.030, 0.020)
PAYLOAD_SCALE = (-1.0, 1.0)
PAYLOAD_NAME = "dr_payload"
#: 押し（胴体の重心に、世界座標の水平の力）
PUSH_NEWTONS = (0.0, 1.5)
PUSH_DURATION_S = 0.1
PUSH_WINDOWS_S = {"dr_push_early": (1.0, 3.5), "dr_push_late": (4.0, 7.0)}
CHASSIS = "c_chassis"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _g(value: float) -> str:
    return f"{value:.9g}"


def _floats(text: str) -> list[float]:
    return [float(v) for v in text.split()]


def dr_model_xml(source_xml: str) -> str:
    """元の cut.xml → 遅れのあるアクチュエータと payload を持つ cut.xml（文字列の置き換え）。"""

    xml = source_xml
    # 1) アクチュエータを一次遅れに
    generals = re.findall(r"<general [^>]*/>", xml)
    if len(generals) != 8:
        raise ValueError(f"<general> が 8 つではない（{len(generals)}）")
    for line in generals:
        if "dyntype=" in line:
            raise ValueError("元のモデルにすでに dyntype がある")
        xml = xml.replace(line, line[:-2].rstrip() + f' dyntype="filterexact" dynprm="{_g(SERVO_LAG_S)}" />', 1)

    # 2) 胴体の質量を payload の分だけ減らす（慣性は質量に比例して縮める）
    found = re.search(r'(<body name="c_chassis">\s*<inertial )([^>]*?)(/>)', xml)
    if not found:
        raise ValueError("c_chassis の inertial が見つからない")
    attrs = found.group(2)
    mass = float(re.search(r'mass="([^"]+)"', attrs).group(1))
    ipos = _floats(re.search(r'pos="([^"]+)"', attrs).group(1))
    inertia = _floats(re.search(r'diaginertia="([^"]+)"', attrs).group(1))
    kept = mass - PAYLOAD_KG
    if kept <= 0:
        raise ValueError("胴体が payload より軽い")
    ratio = kept / mass
    new_attrs = re.sub(r'mass="[^"]+"', f'mass="{_g(kept)}"', attrs)
    new_attrs = re.sub(r'diaginertia="[^"]+"', 'diaginertia="' + " ".join(_g(v * ratio) for v in inertia) + '"',
                       new_attrs)
    xml = xml[:found.start(2)] + new_attrs + xml[found.end(2):]

    # 3) payload を胴体の最後の子として足す（worldbody の直前の </body> が胴体の閉じ）
    close = xml.rfind("</body>", 0, xml.index("</worldbody>"))
    side = 0.04  # 一辺 40 mm の板に相当する小さな慣性（向きには効かない程度）
    inertia_payload = PAYLOAD_KG * side * side / 6.0
    payload = (f'  <body name="{PAYLOAD_NAME}">\n'
               f'        <inertial pos="{_g(PAYLOAD_XY_M[0])} {_g(PAYLOAD_XY_M[1])} {_g(ipos[2])}" '
               f'mass="{_g(PAYLOAD_KG)}" diaginertia="{_g(inertia_payload)} {_g(inertia_payload)} '
               f'{_g(inertia_payload)}" />\n'
               f'      </body>\n    ')
    xml = xml[:close] + payload + xml[close:]
    return xml


def _randomisation(model: mujoco.MjModel) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = [{
        "label": "dr_servo_lag", "subject": "servo", "target": "response_lag", "mode": "scale",
        "low": SERVO_LAG_SCALE[0], "high": SERVO_LAG_SCALE[1],
        "fields": [{"field": "actuator_dynprm", "index": a * mujoco.mjNDYN} for a in range(model.nu)],
    }]
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a).split("/")[0]
        entries.append({
            "label": f"dr_servo_kp_{name}", "subject": name, "target": "servo_stiffness", "mode": "scale",
            "low": SERVO_KP_SCALE[0], "high": SERVO_KP_SCALE[1],
            "fields": [{"field": "actuator_gainprm", "index": a * mujoco.mjNGAIN + 0},
                       {"field": "actuator_biasprm", "index": a * mujoco.mjNBIAS + 1}],
        })
    payload = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PAYLOAD_NAME)
    for axis, label in ((0, "dr_com_x"), (1, "dr_com_y")):
        entries.append({
            "label": label, "subject": PAYLOAD_NAME, "target": "centre_of_mass", "mode": "scale",
            "low": PAYLOAD_SCALE[0], "high": PAYLOAD_SCALE[1],
            "fields": [{"field": "body_ipos", "index": 3 * payload + axis}],
        })
    return entries


def _pushes(model: mujoco.MjModel) -> list[dict[str, Any]]:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CHASSIS)
    return [{
        "label": label, "body": CHASSIS, "body_id": int(body), "direction": "horizontal",
        "newtons_low": PUSH_NEWTONS[0], "newtons_high": PUSH_NEWTONS[1],
        "azimuth_low_rad": 0.0, "azimuth_high_rad": 2.0 * math.pi,
        "sustained": False, "at_low_s": low, "at_high_s": high, "duration_s": PUSH_DURATION_S,
        "applied_at": "centre_of_mass", "frame": "world",
    } for label, (low, high) in PUSH_WINDOWS_S.items()]


def _com_shift_mm(model: mujoco.MjModel) -> dict[str, list[float]]:
    """payload の係数を範囲の両端にしたときの、機体全体の重心のずれ（係数 0 からの差、mm）。"""
    payload = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PAYLOAD_NAME)

    def com(fx: float, fy: float) -> list[float]:
        m = copy.copy(model)
        m.body_ipos[payload, 0] *= fx
        m.body_ipos[payload, 1] *= fy
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        return [float(v) * 1000.0 for v in d.subtree_com[0]]

    centre = com(0.0, 0.0)
    return {"x": [round(com(f, 0.0)[0] - centre[0], 3) for f in PAYLOAD_SCALE],
            "y": [round(com(0.0, f)[1] - centre[1], 3) for f in PAYLOAD_SCALE]}


def build(source_dir: Path, out_dir: Path) -> dict[str, Any]:
    """``source_dir``（cut.xml と walk-task.json）から DR のバンドルを ``out_dir`` に書く。"""

    source_dir, out_dir = Path(source_dir), Path(out_dir)
    source_xml = (source_dir / "cut.xml").read_bytes()
    source_task_bytes = (source_dir / "walk-task.json").read_bytes()
    task = json.loads(source_task_bytes)
    if _sha256(source_xml) != task["model"]["sha256"]:
        raise ValueError("元の cut.xml が元のタスクの記録と違う")
    xml = dr_model_xml(source_xml.decode("utf-8")).encode("utf-8")
    model = mujoco.MjModel.from_xml_string(xml.decode("utf-8"))
    old = mujoco.MjModel.from_xml_string(source_xml.decode("utf-8"))
    for size in ("nq", "nv", "nu", "nsensor", "nsensordata", "njnt", "ngeom", "nsite"):
        if getattr(model, size) != getattr(old, size):
            raise ValueError(f"{size} が変わった")

    task = copy.deepcopy(task)
    task["model"] = {**task["model"], "bytes": len(xml), "sha256": _sha256(xml)}
    task["randomisation"] = list(task["randomisation"]) + _randomisation(model)
    task["disturbance"] = list(task.get("disturbance") or ()) + _pushes(model)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cut.xml").write_bytes(xml)
    task_text = json.dumps(task, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    (out_dir / "walk-task.json").write_text(task_text, encoding="utf-8")
    summary = {
        "schema": "q8-dr-bundle-v1",
        "source": {"task_sha256": _sha256(source_task_bytes), "model_sha256": _sha256(source_xml)},
        "model_sha256": _sha256(xml),
        "task_sha256": _sha256(task_text.encode("utf-8")),
        "servo_lag_ms": [round(SERVO_LAG_S * s * 1000.0, 6) for s in SERVO_LAG_SCALE],
        "servo_kp_scale": list(SERVO_KP_SCALE),
        "payload_g": PAYLOAD_KG * 1000.0,
        "total_mass_g": round(float(model.body_subtreemass[0]) * 1000.0, 3),
        "com_shift_mm": _com_shift_mm(model),
        "push": {"newtons": list(PUSH_NEWTONS), "duration_s": PUSH_DURATION_S,
                 "windows_s": {k: list(v) for k, v in PUSH_WINDOWS_S.items()},
                 "azimuth": "全周（世界座標）"},
        "trainer_flags": {"randomise_friction": [0.35, 1.25], "randomise_drive_torque": [0.8, 1.1]},
        "not_modelled": ["バックラッシュ（ガタ）", "制御周期のゆらぎ", "純粋な遅延（一次遅れで近似）"],
    }
    write_json(out_dir / "dr.json", summary, indent=1)
    return summary
