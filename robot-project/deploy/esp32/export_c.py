""".cxpolicy を ESP32 ファームウェア用の C ヘッダ ``policy_weights.h`` に書き出す。

ヘッダに入るもの:

* 層構成、活性化、float32 の重み（(出力, 入力) 行優先に転置）とバイアス
* int8 版の重み・int32 バイアス・float32 倍率・活性化 LUT（``--no-int8`` で省く）
* 観測の正規化（mean/std）、出力の写像（output_scale/output_bias）、行動の範囲
* 指令フィルタ（action_filter_alpha、command_slew_deg）と初期指令
* 観測チャネルの取り出し元（``policy_obs_kind/_index/_gain``）。ヘッダの
  ``observations`` の名前から解決する。名前で決められないチャネルがあれば
  止まるので、``--layout-template`` で雛形を出し、``--layout`` で渡す。
* サーボの角度→パルス幅の換算（``--servo`` の JSON、既定は MG90S の推定値）

使い方::

    python -m deploy.esp32.export_c artifacts/.../walk.cxpolicy \\
        --task robot/mjcf/base/walk-task.json \\
        --out deploy/esp32/firmware/main/policy_weights.h \\
        --samples rollout_obs.npy --servo servo.json

``--task`` に学習に使ったタスクバンドル（``walk-task.json``）を渡すと、観測行の種類・
``scale``・``frame``、ゴール、コマンド行から取り出し元を決める（名前の推定より確実）。
バンドルの sha256 がポリシーの ``task.sha256`` と違えば止まる。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from . import cxpolicy_io as cx
from . import quantize as qz

#: control.h の enum と同じ番号。
OBS_KINDS = {"quat": 0, "gyro": 1, "gravity": 2, "cmd": 3, "cmd_rate": 4, "goal": 5,
             "phase_sin": 6, "phase_cos": 7, "const": 8, "gyro_world": 9}

RAD_TO_DEG = 180.0 / math.pi
ACT_CODES = {"tanh": "POLICY_ACT_TANH", "relu": "POLICY_ACT_RELU", "swish": "POLICY_ACT_SWISH"}
F32 = np.float32


class LayoutError(ValueError):
    """観測チャネルの取り出し元を決められない。"""


@dataclass(frozen=True)
class ObsSource:
    name: str
    kind: str
    index: int = 0
    gain: float = 1.0


@dataclass(frozen=True)
class ServoConfig:
    """角度[度] → パルス幅[µs]。MG90S の既定値は推定（個体ごとに要較正）。

    ``pulse = center_us + sign * (angle + trim_deg) * us_per_deg`` を [min_us, max_us] に収める。
    500〜2400 µs で約 180° という一般的な値から us_per_deg = 1900/180 としている。
    """

    center_us: float = 1450.0
    us_per_deg: float = (2400.0 - 500.0) / 180.0
    sign: float = 1.0
    trim_deg: float = 0.0
    min_us: float = 500.0
    max_us: float = 2400.0


# ---------------------------------------------------------------------------
# 観測チャネルの取り出し元
# ---------------------------------------------------------------------------


def _guess(name: str, policy: cx.CxPolicy, goal_order: list[str]) -> ObsSource | None:
    actuators = [str(a.get("actuator")) for a in policy.actions]
    for i, suffix in enumerate(("_qw", "_qx", "_qy", "_qz")):
        if name.endswith(suffix):
            return ObsSource(name, "quat", i, 1.0)
    m = re.search(r"gyro_?([xyz])$", name)
    if m:
        # Cadex の角速度チャネルは deg/s、IMU ドライバは rad/s で渡す前提。
        return ObsSource(name, "gyro", "xyz".index(m.group(1)), RAD_TO_DEG)
    m = re.search(r"gravity_?([xyz])$", name)
    if m:
        return ObsSource(name, "gravity", "xyz".index(m.group(1)), 1.0)
    if name in ("phase_sin", "clock_sin"):
        return ObsSource(name, "phase_sin")
    if name in ("phase_cos", "clock_cos"):
        return ObsSource(name, "phase_cos")
    base = name[4:] if name.startswith("cmd_") else name
    if base.endswith("_rate") and base[:-5] in actuators:
        return ObsSource(name, "cmd_rate", actuators.index(base[:-5]), 1.0)
    if base in actuators:
        return ObsSource(name, "cmd", actuators.index(base), 1.0)
    if name.startswith("goal_"):
        goal_order.append(name)
        return ObsSource(name, "goal", len(goal_order) - 1, 1.0)
    return None


def _from_spec(name: str, spec: dict[str, Any]) -> ObsSource:
    kind = str(spec.get("source", ""))
    if kind not in OBS_KINDS:
        raise LayoutError(f"{name}: source {kind!r} は {sorted(OBS_KINDS)} のどれでもない")
    return ObsSource(name, kind, int(spec.get("index", 0)), float(spec.get("gain", 1.0)))


def _from_task(task: dict[str, Any]) -> tuple[dict[str, ObsSource], set[str]]:
    """タスクバンドルから、チャネル名 → 取り出し元。2 つ目は privileged のチャネル名。"""

    found: dict[str, ObsSource] = {}
    privileged: set[str] = set()
    for row in task.get("observations") or ():
        channels = [str(c) for c in row.get("channels") or ()]
        if str(row.get("role") or "policy") == "privileged":
            privileged.update(channels)
            continue
        kind, gain = str(row.get("kind")), float(row.get("scale", 1.0))
        for i, name in enumerate(channels):
            if kind == "component_orientation":
                found[name] = ObsSource(name, "quat", i, gain)
            elif kind == "component_angular_velocity":
                # frame が無ければ MuJoCo の FRAMEANGVEL は世界座標（エンジンは frame の
                # ときだけ reftype を立てる）。実機のジャイロは機体座標なので回して合わせる。
                found[name] = ObsSource(name, "gyro" if row.get("frame") else "gyro_world",
                                        i, gain)
    goal_index = 0
    for row in task.get("goal") or ():
        channels = [str(c) for c in row.get("channels") or ()]
        if str(row.get("kind")) == "phase" and len(channels) == 2:
            found[channels[0]] = ObsSource(channels[0], "phase_sin")
            found[channels[1]] = ObsSource(channels[1], "phase_cos")
            continue
        for name in channels:
            found[name] = ObsSource(name, "goal", goal_index, 1.0)
            goal_index += 1
    for row in task.get("commands") or ():
        channels = [str(c) for c in row.get("channels") or ()]
        action = int(row["action"])
        if channels:
            found[channels[0]] = ObsSource(channels[0], "cmd", action, 1.0)
        if len(channels) > 1:
            found[channels[1]] = ObsSource(channels[1], "cmd_rate", action, 1.0)
    return found, privileged


def clock_period_s(task: dict[str, Any] | None) -> float | None:
    """タスクの位相ゴール（歩行クロック）の周期[s]。無ければ None。"""

    for row in (task or {}).get("goal") or ():
        if str(row.get("kind")) == "phase" and row.get("period_s"):
            return float(row["period_s"])
    return None


def reset_command(task_path: str | Path) -> list[float]:
    """学習のリセットで効いている指令（``CadexDynamics.reset_command`` と同じ式）。

    タスクの ``episode.reset_keyframe`` のキーフレームの ``ctrl`` を、各行動の単位（``/ scale``）に直し、
    行動範囲で切り詰める。トレーナーは、両方の「前回の指令」をこの値から始める。モデルは
    タスクの ``model.path`` で、タスクの親の親か、タスクと同じディレクトリにある（学習器と同じ規則）。
    """

    import mujoco

    task_path = Path(task_path)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    relative = Path(str(task["model"]["path"]))
    model_path = task_path.resolve().parent.parent / relative
    if not model_path.exists():
        model_path = task_path.parent / relative.name
    model = mujoco.MjModel.from_xml_path(str(model_path))
    name = str((task.get("episode") or {}).get("reset_keyframe") or "")
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name) if name else 0
    if key < 0:
        raise LayoutError(f"モデルにキーフレーム {name!r} がありません")
    ctrl = model.key_ctrl[key]
    return [min(max(float(ctrl[int(a["index"])]) / float(a["scale"]), float(a["low"])), float(a["high"]))
            for a in task["actions"]]


def load_task(path: str | Path, policy: cx.CxPolicy | None = None) -> dict[str, Any]:
    import hashlib

    blob = Path(path).read_bytes()
    if policy is not None:
        want = str((policy.header.get("task") or {}).get("sha256") or "")
        got = hashlib.sha256(blob).hexdigest()
        if want and want != got:
            raise LayoutError(f"タスクバンドルの sha256 {got[:12]}… がポリシーの task.sha256 "
                              f"{want[:12]}… と違う（別のタスクで学習したポリシー）")
    return json.loads(blob.decode("utf-8"))


def resolve_layout(policy: cx.CxPolicy, layout: dict[str, Any] | str | Path | None = None,
                   *, task: dict[str, Any] | None = None) -> list[ObsSource]:
    """ヘッダの observations の各チャネルが、実機のどの値から来るか。

    優先順は ``layout``（明示）→ ``task``（タスクバンドル）→ 名前からの推定。決まらない
    チャネルや、タスクが privileged と宣言したチャネルが残れば :class:`LayoutError`
    （黙って 0 を入れたりしない）。
    """

    if isinstance(layout, (str, Path)):
        layout = json.loads(Path(layout).read_text(encoding="utf-8"))
    explicit = dict((layout or {}).get("channels") or {})
    from_task, privileged = _from_task(task) if task else ({}, set())
    leaked = [n for n in policy.observations if n in privileged and n not in explicit]
    if leaked:
        raise LayoutError(f"タスクが privileged（実機で測れない）とするチャネルをポリシーが読む: {leaked}")
    unknown = [n for n in explicit if n not in policy.observations]
    if unknown:
        raise LayoutError(f"layout にヘッダに無いチャネルがある: {unknown}")
    goals: list[str] = []
    resolved: list[ObsSource] = []
    missing: list[str] = []
    for name in policy.observations:
        if name in explicit:
            resolved.append(_from_spec(name, explicit[name]))
            continue
        if name in from_task:
            resolved.append(from_task[name])
            continue
        guess = _guess(name, policy, goals)
        if guess is None:
            missing.append(name)
        else:
            resolved.append(guess)
    if missing:
        raise LayoutError(
            f"取り出し元を名前から決められないチャネル: {missing}。"
            "--layout-template で雛形を出し、source/index/gain を埋めて --layout で渡すこと。")
    for src in resolved:
        limit = {"quat": 4, "gyro": 3, "gyro_world": 3, "gravity": 3, "goal": 4,
                 "cmd": len(policy.actions), "cmd_rate": len(policy.actions)}.get(src.kind)
        if limit is not None and not (0 <= src.index < limit):
            raise LayoutError(f"{src.name}: index {src.index} が {src.kind} の範囲外")
    return resolved


def layout_template(policy: cx.CxPolicy) -> dict[str, Any]:
    goals: list[str] = []
    channels = {}
    for name in policy.observations:
        guess = _guess(name, policy, goals)
        channels[name] = ({"source": guess.kind, "index": guess.index, "gain": guess.gain}
                          if guess else {"source": "TODO", "index": 0, "gain": 1.0})
    return {"_説明": "source は " + ", ".join(OBS_KINDS) + "。gyro は rad/s 入力に掛ける gain、"
            "const は値そのものを gain に書く。", "channels": channels}


def build_observation(layout: Sequence[ObsSource], *, quat: np.ndarray, gyro_rad_s: np.ndarray,
                      command: np.ndarray, command_before: np.ndarray, goal: np.ndarray,
                      phase_rad: float, dt_s: float) -> np.ndarray:
    """control.c の obs_build の Python 版（float64）。"""

    w, x, y, z = (float(v) for v in quat)
    gravity = [-2.0 * (x * z - w * y), -2.0 * (y * z + w * x), -(1.0 - 2.0 * (x * x + y * y))]
    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    gyro_world = rot @ np.asarray(gyro_rad_s, dtype=np.float64)
    out = []
    for src in layout:
        value = {
            "quat": lambda: quat[src.index],
            "gyro": lambda: gyro_rad_s[src.index],
            "gyro_world": lambda: gyro_world[src.index],
            "gravity": lambda: gravity[src.index],
            "cmd": lambda: command[src.index],
            "cmd_rate": lambda: (command[src.index] - command_before[src.index]) / dt_s,
            "goal": lambda: goal[src.index],
            "phase_sin": lambda: math.sin(phase_rad),
            "phase_cos": lambda: math.cos(phase_rad),
            "const": lambda: 1.0,
        }[src.kind]()
        out.append(src.gain * float(value))
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# サーボ
# ---------------------------------------------------------------------------


def servo_pulse_us(angle_deg: float, servo: ServoConfig = ServoConfig()) -> float:
    p = servo.center_us + servo.sign * (float(angle_deg) + servo.trim_deg) * servo.us_per_deg
    return float(min(max(p, servo.min_us), servo.max_us))


def ledc_duty(pulse_us: float, *, bits: int = 14, hz: int = 50) -> int:
    """control.c の servo_ledc_duty と同じ float32 演算。"""

    per_us = F32(1 << bits) / F32(F32(1.0e6) / F32(hz))
    d = int(np.rint(F32(pulse_us) * per_us))
    return min(max(d, 0), (1 << bits) - 1)


def _servo_keys(action: dict[str, Any]) -> set[str]:
    actuator = str(action.get("actuator"))
    keys = {actuator, actuator.split("/")[0]}
    if action.get("joint"):
        keys.add(str(action["joint"]))
    return keys


def load_servos(path: str | Path | None, policy: cx.CxPolicy) -> list[ServoConfig]:
    """``{"default": {...}, "joints": {"hip_lf": {...}}}`` 形式。無ければ全関節既定値。

    ``joints`` のキーはアクチュエータ名（``hip_lf/position``）でも関節名（``hip_lf``）でもよい。
    どの行動にも当たらないキーがあれば止まる（黙って既定値のままにしない）。
    """

    data = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    base = {**asdict(ServoConfig()), **(data.get("default") or {})}
    joints = data.get("joints") or {}
    keys = [_servo_keys(a) for a in policy.actions]
    unknown = [k for k in joints if not any(k in ks or k.split("/")[0] in ks for ks in keys)]
    if unknown:
        raise ValueError(f"サーボ換算のキー {unknown} が方策の行動に無い"
                         f"（行動: {[str(a.get('actuator')) for a in policy.actions]}）")
    out = []
    for ks in keys:
        spec: dict[str, Any] = {}
        for k, v in joints.items():
            if k in ks or k.split("/")[0] in ks:
                spec.update(v or {})
        out.append(ServoConfig(**{**base, **spec}))
    return out


# ---------------------------------------------------------------------------
# ヘッダの生成
# ---------------------------------------------------------------------------


def _f(v: float) -> str:
    v = float(F32(v))
    if not math.isfinite(v):
        raise ValueError(f"有限でない値 {v}")
    s = f"{v:.9g}"
    if "e" not in s and "." not in s:
        s += ".0"
    return s + "f"


def _array(ctype: str, name: str, values: Iterable[Any], fmt, *, attr: str = "",
           per_line: int = 8) -> str:
    vals = [fmt(v) for v in values]
    lines = [", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line)]
    body = ",\n    ".join(lines) if lines else "0"
    return f"{attr}const {ctype} {name}[{max(len(vals), 1)}] = {{\n    {body}\n}};\n"


def render_header(policy: cx.CxPolicy, qpolicy: qz.QuantPolicy | None, *,
                  layout: Sequence[ObsSource], servos: Sequence[ServoConfig] | None = None,
                  initial_command: Sequence[float] | None = None, source: str = "",
                  clock_period_s: float | None = None) -> str:
    cx.require_joint_count(policy)
    cx.require_joint_names(policy)
    sizes = policy.layer_sizes
    n_layers = len(policy.layers)
    servos = list(servos or [ServoConfig()] * len(policy.actions))
    initial = (list(initial_command) if initial_command is not None
               else [float(v) for v in policy.output_bias])
    if len(initial) != len(policy.actions):
        raise ValueError("初期指令の数が行動数と合わない")
    w_t = [w.T.astype(F32) for w, _ in policy.layers]
    w_off = np.cumsum([0] + [int(w.size) for w in w_t[:-1]]).tolist()
    b_off = np.cumsum([0] + [int(b.size) for _, b in policy.layers[:-1]]).tolist()
    i = lambda v: str(int(v))  # noqa: E731

    decl: list[str] = []
    defs: list[str] = []

    def emit(ctype: str, name: str, values, fmt, *, weight: bool = False) -> None:
        count = max(len(list(values)) if not isinstance(values, np.ndarray) else values.size, 1)
        decl.append(f"extern const {ctype} {name}[{count}];")
        defs.append(_array(ctype, name, values, fmt, attr="POLICY_WEIGHT_ATTR " if weight else ""))

    emit("int32_t", "policy_layer_in", [s for s in sizes[:-1]], i)
    emit("int32_t", "policy_layer_out", [s for s in sizes[1:]], i)
    emit("float", "policy_obs_mean", policy.mean, _f)
    emit("float", "policy_obs_std", policy.std, _f)
    emit("float", "policy_output_scale", policy.output_scale, _f)
    emit("float", "policy_output_bias", policy.output_bias, _f)
    emit("float", "policy_action_low", policy.action_low, _f)
    emit("float", "policy_action_high", policy.action_high, _f)
    emit("float", "policy_initial_command", initial, _f)
    emit("uint8_t", "policy_obs_kind", [OBS_KINDS[s.kind] for s in layout], i)
    emit("uint8_t", "policy_obs_index", [s.index for s in layout], i)
    emit("float", "policy_obs_gain", [s.gain for s in layout], _f)
    for field in ("center_us", "us_per_deg", "sign", "trim_deg", "min_us", "max_us"):
        emit("float", f"policy_servo_{field}", [getattr(s, field) for s in servos], _f)
    emit("int32_t", "policy_f32_w_off", w_off, i)
    emit("int32_t", "policy_f32_b_off", b_off, i)
    emit("float", "policy_f32_w", np.concatenate([w.reshape(-1) for w in w_t]), _f, weight=True)
    emit("float", "policy_f32_b", np.concatenate([b for _, b in policy.layers]), _f, weight=True)
    if qpolicy is not None:
        emit("int8_t", "policy_i8_w",
             np.concatenate([l.w_q.reshape(-1) for l in qpolicy.layers]), i, weight=True)
        emit("int32_t", "policy_i8_bias", np.concatenate([l.bias_q for l in qpolicy.layers]), i,
             weight=True)
        emit("float", "policy_i8_mult", np.concatenate([l.mult for l in qpolicy.layers]), _f,
             weight=True)
        luts = [l.lut for l in qpolicy.layers if l.lut is not None]
        emit("int8_t", "policy_i8_lut",
             np.concatenate(luts) if luts else np.zeros(1, np.int8), i, weight=True)

    channel_lines = [f" *   {k:2d} {s.name:<24} {s.kind}[{s.index}] x {s.gain:g}"
                     for k, s in enumerate(layout)]
    action_lines = [f" *   {k:2d} {a.get('actuator')}: [{a['low']:g}, {a['high']:g}] {a.get('unit')}"
                    for k, a in enumerate(policy.actions)]
    synthetic = bool(policy.header.get("synthetic"))
    head = [
        "/*",
        " * policy_weights.h — deploy/esp32/export_c.py が自動生成。手で編集しない。",
        f" * 元ファイル: {source}",
        f" * 層構成 {'-'.join(map(str, sizes))}、活性化 {policy.activation}、出力 tanh",
        f" * ラベル: {policy.header.get('label', '')}  task.sha256: "
        f"{(policy.header.get('task') or {}).get('sha256', '')}",
    ]
    if synthetic:
        head.append(" * 注意: 合成ポリシー（乱数重み）。実機のサーボを動かさないこと。")
    if qpolicy is not None:
        head.append(f" * int8: 重みスケール {'チャネルごと' if qpolicy.per_channel else '層ごと'}、"
                    f"較正標本 {qpolicy.calibration_count} 個、百分位 {qpolicy.percentile:g}")
    head += [" *", " * 観測チャネル（この順で obs を作る。gain は実機の値に掛ける倍率）:",
             *channel_lines, " *", " * 行動（関節指令）:", *action_lines, " */"]
    macros = [
        "#ifndef POLICY_WEIGHTS_H", "#define POLICY_WEIGHTS_H", "", "#include <stdint.h>", "",
        "#define POLICY_ACT_TANH 0", "#define POLICY_ACT_RELU 1", "#define POLICY_ACT_SWISH 2", "",
        f"#define POLICY_N_IN {sizes[0]}",
        f"#define POLICY_N_OUT {sizes[-1]}",
        f"#define POLICY_N_LAYERS {n_layers}",
        f"#define POLICY_MAX_WIDTH {max(sizes)}",
        f"#define POLICY_N_PARAMS {policy.weights.size}",
        f"#define POLICY_ACTIVATION {ACT_CODES[policy.activation]}",
        f"#define POLICY_HAS_INT8 {1 if qpolicy is not None else 0}",
        f"#define POLICY_SYNTHETIC {1 if synthetic else 0}",
        f"#define POLICY_FILTER_ALPHA {_f(policy.action_filter_alpha)}",
        f"#define POLICY_COMMAND_SLEW_DEG {_f(policy.command_slew_deg)}",
        "/* 歩行クロックの周期[s]（タスクの位相ゴール）。0 は不明。 */",
        f"#define POLICY_CLOCK_PERIOD_S {_f(clock_period_s or 0.0)}",
    ]
    if qpolicy is not None:
        macros.append(f"#define POLICY_I8_IN_INV_SCALE {_f(qpolicy.in_inv_scale)}")
    macros += [
        "",
        "/* 重み配列の置き場所。ESP-IDF で内部 RAM に置くなら DRAM_ATTR を定義してから include。 */",
        "#ifndef POLICY_WEIGHT_ATTR", "#define POLICY_WEIGHT_ATTR", "#endif", "",
        *decl, "",
        "#ifdef POLICY_WEIGHTS_DEFINE  /* policy.c だけが定義する */", "",
        *defs,
        "#endif /* POLICY_WEIGHTS_DEFINE */", "", "#endif /* POLICY_WEIGHTS_H */", "",
    ]
    return "\n".join(head + [""] + macros)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="export_c", description=".cxpolicy → policy_weights.h")
    parser.add_argument("policy", help=".cxpolicy のパス")
    parser.add_argument("--out", help="書き出す policy_weights.h")
    parser.add_argument("--samples", help="int8 較正用の観測（.npy / JSON、N×観測数）")
    parser.add_argument("--count", type=int, default=2048, help="--samples が無いときの合成標本数")
    parser.add_argument("--no-int8", action="store_true", help="int8 版を含めない")
    parser.add_argument("--per-layer", action="store_true", help="int8 重みスケールを層ごとに")
    parser.add_argument("--percentile", type=float, default=100.0)
    parser.add_argument("--layout", help="観測チャネルの取り出し元（JSON）。--task より優先")
    parser.add_argument("--task", help="学習に使ったタスクバンドル（walk-task.json）")
    parser.add_argument("--layout-template", action="store_true",
                        help="layout の雛形を標準出力に出して終わる")
    parser.add_argument("--servo", help="サーボ換算（JSON）")
    parser.add_argument("--initial-command", type=float, nargs="+",
                        help="起動時の関節指令[タスク単位]。既定は行動範囲の中央（要確認）")
    args = parser.parse_args(argv)
    try:
        policy = cx.load_policy(args.policy)
        task = load_task(args.task, policy) if args.task else None
        if args.layout_template:
            print(json.dumps(layout_template(policy), ensure_ascii=False, indent=2))
            return 0
        if not args.out:
            parser.error("--out が必要")
        layout = resolve_layout(policy, args.layout, task=task)
        qpolicy = None
        if not args.no_int8:
            samples = (qz.load_samples(args.samples, len(policy.observations)) if args.samples
                       else qz.calibration_samples(policy, count=args.count))
            qpolicy = qz.quantize_policy(policy, samples, per_channel=not args.per_layer,
                                         percentile=args.percentile)
        initial = args.initial_command
        if initial is None and args.task:
            initial = reset_command(args.task)
        text = render_header(policy, qpolicy, layout=layout,
                             servos=load_servos(args.servo, policy),
                             initial_command=initial, source=Path(args.policy).name,
                             clock_period_s=clock_period_s(task))
    except (OSError, ValueError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 2
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"書いた: {args.out}（{'-'.join(map(str, policy.layer_sizes))}、"
          f"int8 {'あり' if qpolicy else 'なし'}）")
    if args.initial_command:
        pass
    elif args.task:
        print(f"初期指令: 学習のリセットと同じ値（{args.task} のリセットのキーフレーム）")
    else:
        print("注意: 初期指令は行動範囲の中央。学習時のリセット姿勢（keyframe）に合わせること。"
              "--task を渡すと、学習のリセットと同じ値になる。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
