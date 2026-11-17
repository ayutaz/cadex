""".cxpolicy の読み書きと、numpy による参照順伝播。

形式は Cadex 本体の次の2か所を読んで確定させた（推測ではない）:

* 書き出し: ``training/cadex_train.py`` の ``encode_policy`` / ``flat_parameters`` /
  ``snapshot`` / ``policy_header``
* 読み込みと再生: ``src/Mod/cadex/CadexDynamics.py`` の ``decode_policy`` /
  ``policy_forward`` / ``policy_swish`` / ``recorded_command_filter`` と、
  エピソード再生中の ``_actions``（clamp → alpha → slew）

コンテナ::

    b"CXPOLICY1\\n" | u64 LE ヘッダ長 | 正準 JSON（キー整列・空白なし・ASCII） | float32 LE の平坦な重み

重みは層の順に「重み行列 (入力, 出力) を行優先」→「バイアス (出力)」。最終層は
Brax の出力（平均と分散の 2 倍幅）のうち決定的な半分（平均）だけが入っている。

順伝播（エンジンと同じ式）::

    x = (obs - normaliser.mean) / normaliser.std
    隠れ層:  x = act(x @ W + b)      act = swish | tanh | relu（network.activation）
    最終層:  z = x @ W + b            （活性化なし）
    action = tanh(z) * network.output_scale + network.output_bias   （タスクの単位。サーボは度）

その後、``training.action_filter_alpha`` と ``training.command_slew_deg`` を
この順に掛ける（:class:`CommandFilter`）。どちらもエンジンが再生時に掛けるもので、
実機でも同じものを掛けないと学習時と違う制御器になる。

このモジュールは numpy だけに依存する。
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

POLICY_MAGIC = b"CXPOLICY1\n"
POLICY_SCHEMA = "cadex-policy-v1"
ACTIVATIONS = ("tanh", "relu", "swish")

#: エンジンの CadexDynamics.POLICY_WITNESS_TOLERANCE（行動範囲に対する相対誤差）。
WITNESS_TOLERANCE = 1.0e-4

#: 合成ポリシーの既定の関節（8 自由度四足: 各脚 hip と knee）。
DEFAULT_JOINTS = ("hip_lf", "knee_lf", "hip_rf", "knee_rf",
                  "hip_lh", "knee_lh", "hip_rh", "knee_rh")

#: 合成ポリシーの既定の観測チャネル（実機で観測できるものだけ、計 20）。
#: 実物の .cxpolicy ではタスクが決めた名前になるので、配備側はこの名前を当てにせず
#: ヘッダの ``observations`` を読むこと（export_c.resolve_layout がそれをする）。
DEFAULT_OBSERVATIONS = (
    "body_qw", "body_qx", "body_qy", "body_qz",          # IMU 姿勢クォータニオン（wxyz）
    "body_gyro_x", "body_gyro_y", "body_gyro_z",        # 角速度（deg/s、機体座標）
    *(f"cmd_{j}" for j in DEFAULT_JOINTS),             # 前回の関節指令（deg）
    "goal_vx", "goal_vy", "goal_wz",                    # 目標速度（mm/s, mm/s, deg/s）
    "phase_sin", "phase_cos",                           # 歩行位相
)


class PolicyFormatError(ValueError):
    """.cxpolicy として読めない、またはエンジンが拒否する内容。"""


@dataclass(frozen=True)
class CxPolicy:
    """読み込んだ .cxpolicy。数値はエンジンと同じ値（JSON の数は float64、重みは float32）。"""

    header: dict[str, Any]
    weights: np.ndarray                                   # float32、平坦
    layers: list[tuple[np.ndarray, np.ndarray]]           # (W[in,out], b[out]) float32
    mean: np.ndarray                                      # float64
    std: np.ndarray                                       # float64
    activation: str
    output_scale: np.ndarray                              # float64
    output_bias: np.ndarray                               # float64
    observations: list[str]
    actions: list[dict[str, Any]]
    action_filter_alpha: float = 1.0
    command_slew_deg: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def layer_sizes(self) -> list[int]:
        return [int(self.layers[0][0].shape[0])] + [int(w.shape[1]) for w, _ in self.layers]

    @property
    def action_low(self) -> np.ndarray:
        return np.array([float(a["low"]) for a in self.actions], dtype=np.float64)

    @property
    def action_high(self) -> np.ndarray:
        return np.array([float(a["high"]) for a in self.actions], dtype=np.float64)

    @property
    def action_unit(self) -> str:
        units = {str(a.get("unit", "")) for a in self.actions}
        return units.pop() if len(units) == 1 else "mixed"


# ---------------------------------------------------------------------------
# コンテナ
# ---------------------------------------------------------------------------


def _canonical(header: dict[str, Any]) -> bytes:
    return json.dumps(header, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("ascii")


def encode_policy(header: dict[str, Any], weights: Sequence[float]) -> bytes:
    """トレーナー/エンジンの ``encode_policy`` と同じバイト列。"""

    values = np.asarray(weights, dtype="<f4")
    payload = _canonical(header)
    return b"".join((POLICY_MAGIC, len(payload).to_bytes(8, "little"), payload,
                     values.tobytes()))


def decode_policy(blob: bytes) -> tuple[dict[str, Any], np.ndarray]:
    """ヘッダ（dict）と float32 の重み配列。"""

    if not blob.startswith(POLICY_MAGIC):
        raise PolicyFormatError(f".cxpolicy ではない（先頭が {POLICY_MAGIC!r} でない）")
    start = len(POLICY_MAGIC)
    if len(blob) < start + 8:
        raise PolicyFormatError("ヘッダ長の途中で切れている")
    length = int.from_bytes(blob[start:start + 8], "little")
    body = start + 8
    if length > len(blob) - body:
        raise PolicyFormatError(f"ヘッダ長 {length} に対して残りが {len(blob) - body} バイト")
    try:
        header = json.loads(blob[body:body + length].decode("ascii"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise PolicyFormatError(f"ヘッダが JSON として読めない: {exc}") from exc
    if not isinstance(header, dict):
        raise PolicyFormatError("ヘッダが JSON オブジェクトでない")
    rest = blob[body + length:]
    if len(rest) % 4:
        raise PolicyFormatError(f"重みが {len(rest)} バイトで、float32 の整数個でない")
    return header, np.frombuffer(rest, dtype="<f4").astype(np.float32)


def parse_policy(blob: bytes) -> CxPolicy:
    """バイト列を検査して :class:`CxPolicy` にする。エンジンが拒否するものはここでも拒否する。"""

    header, weights = decode_policy(blob)
    network = header.get("network")
    if not isinstance(network, dict):
        raise PolicyFormatError("network が無い")
    if str(network.get("kind") or "") != "mlp":
        raise PolicyFormatError(f"network.kind {network.get('kind')!r} は mlp でない")
    activation = str(network.get("activation") or "")
    if activation not in ACTIVATIONS:
        raise PolicyFormatError(f"隠れ層の活性化 {activation!r} は未対応")
    if str(network.get("output") or "") != "tanh":
        raise PolicyFormatError(f"出力の活性化 {network.get('output')!r} は tanh でない")
    shapes = [(int(a), int(b)) for a, b in network.get("layers") or ()]
    if not shapes:
        raise PolicyFormatError("層が無い")
    for (_, out), (nxt, _) in zip(shapes[:-1], shapes[1:]):
        if out != nxt:
            raise PolicyFormatError(f"層の幅がつながらない: {shapes}")
    expected = sum(i * o + o for i, o in shapes)
    if weights.size != expected:
        raise PolicyFormatError(f"重みが {weights.size} 個、層構成 {shapes} には {expected} 個必要")
    observations = [str(n) for n in header.get("observations") or ()]
    if shapes[0][0] != len(observations):
        raise PolicyFormatError(
            f"入力幅 {shapes[0][0]} と観測チャネル数 {len(observations)} が合わない")
    normaliser = header.get("normaliser") or {}
    mean = np.array([float(v) for v in normaliser.get("mean") or ()], dtype=np.float64)
    std = np.array([float(v) for v in normaliser.get("std") or ()], dtype=np.float64)
    if mean.size != len(observations) or std.size != len(observations):
        raise PolicyFormatError("normaliser の mean/std の長さが観測チャネル数と合わない")
    if np.any(std == 0.0):
        raise PolicyFormatError("normaliser.std に 0 がある（エンジンも拒否する）")
    scale = np.array([float(v) for v in network.get("output_scale") or ()], dtype=np.float64)
    bias = np.array([float(v) for v in network.get("output_bias") or ()], dtype=np.float64)
    outputs = shapes[-1][1]
    if scale.size != outputs or bias.size != outputs:
        raise PolicyFormatError("output_scale/output_bias の長さが出力数と合わない")
    actions = [dict(a) for a in header.get("actions") or ()]
    if len(actions) != outputs:
        raise PolicyFormatError(f"actions が {len(actions)} 個、出力は {outputs}")

    layers: list[tuple[np.ndarray, np.ndarray]] = []
    cursor = 0
    for inputs, outs in shapes:
        w = weights[cursor:cursor + inputs * outs].reshape(inputs, outs)
        cursor += inputs * outs
        b = weights[cursor:cursor + outs]
        cursor += outs
        layers.append((w, b))

    training = header.get("training") or {}
    alpha = float(training.get("action_filter_alpha", 1.0))
    slew = float(training.get("command_slew_deg", 0.0) or 0.0)
    if not (0.0 < alpha <= 1.0):
        raise PolicyFormatError(f"action_filter_alpha {alpha} が (0, 1] の外")
    if slew < 0.0:
        raise PolicyFormatError(f"command_slew_deg {slew} が負")
    return CxPolicy(header=header, weights=weights, layers=layers, mean=mean, std=std,
                    activation=activation, output_scale=scale, output_bias=bias,
                    observations=observations, actions=actions,
                    action_filter_alpha=alpha, command_slew_deg=slew)


def load_policy(path: str | Path) -> CxPolicy:
    return parse_policy(Path(path).read_bytes())


# ---------------------------------------------------------------------------
# 順伝播
# ---------------------------------------------------------------------------


def swish(x: np.ndarray) -> np.ndarray:
    """``x * sigmoid(x)``。エンジンの ``policy_swish`` と同じく両側で exp があふれない形。"""

    x = np.asarray(x, dtype=np.float64)
    e = np.exp(-np.abs(x))
    return np.where(x >= 0.0, x / (1.0 + e), x * e / (1.0 + e))


def activate(name: str, x: np.ndarray) -> np.ndarray:
    if name == "swish":
        return swish(x)
    if name == "tanh":
        return np.tanh(x)
    return np.where(x > 0.0, x, 0.0)


def normalise(policy: CxPolicy, obs: np.ndarray) -> np.ndarray:
    return (np.asarray(obs, dtype=np.float64) - policy.mean) / policy.std


def forward_raw(policy: CxPolicy, obs: np.ndarray, *, act=None) -> np.ndarray:
    """出力 tanh の手前の値 z。``act`` で隠れ層の活性化を差し替えられる（近似の比較用）。"""

    act = act or (lambda v: activate(policy.activation, v))
    x = normalise(policy, obs)
    last = len(policy.layers) - 1
    for index, (w, b) in enumerate(policy.layers):
        x = x @ w.astype(np.float64) + b.astype(np.float64)
        if index < last:
            x = act(x)
    return x


def forward(policy: CxPolicy, obs: np.ndarray, *, act=None) -> np.ndarray:
    """観測（ヘッダの順・タスクの単位）→ 行動（タスクの単位）。float64。"""

    return np.tanh(forward_raw(policy, obs, act=act)) * policy.output_scale + policy.output_bias


def witness_error(policy: CxPolicy) -> tuple[float, int, int]:
    """ファイル内の witness（観測と、トレーナーが記録した行動）との最大相対誤差。

    エンジンの ``verify_policy`` と同じく、各行動の ``high - low`` に対する相対値。
    戻り値は (最大誤差, 標本番号, 行動番号)。
    """

    evaluation = policy.header.get("evaluation") or {}
    observed = np.asarray(evaluation.get("observations") or (), dtype=np.float64)
    recorded = np.asarray(evaluation.get("actions") or (), dtype=np.float64)
    if observed.size == 0:
        raise PolicyFormatError("evaluation（witness）が無い")
    produced = forward(policy, observed)
    ranges = np.maximum(policy.action_high - policy.action_low, 1.0e-12)
    error = np.abs(produced - recorded) / ranges
    sample, action = np.unravel_index(int(np.argmax(error)), error.shape)
    return float(error[sample, action]), int(sample), int(action)


class CommandFilter:
    """エンジンの再生と同じ指令フィルタ: clamp → 一次遅れ(alpha) → 変化量制限(slew)。

    最初の1回は clamp だけを通す（エンジンの ``_actions`` と同じ）。
    """

    def __init__(self, low: np.ndarray, high: np.ndarray, *, alpha: float = 1.0,
                 slew_deg: float = 0.0) -> None:
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.alpha = float(alpha)
        self.slew_deg = float(slew_deg)
        self.issued: np.ndarray | None = None

    @classmethod
    def from_policy(cls, policy: CxPolicy) -> "CommandFilter":
        return cls(policy.action_low, policy.action_high, alpha=policy.action_filter_alpha,
                   slew_deg=policy.command_slew_deg)

    def reset(self) -> None:
        self.issued = None

    def step(self, raw: np.ndarray) -> np.ndarray:
        command = np.clip(np.asarray(raw, dtype=np.float64), self.low, self.high)
        if self.issued is not None:
            if self.alpha < 1.0:
                command = self.alpha * command + (1.0 - self.alpha) * self.issued
            if self.slew_deg > 0.0:
                command = np.clip(command, self.issued - self.slew_deg,
                                  self.issued + self.slew_deg)
        self.issued = command.copy()
        return command


# ---------------------------------------------------------------------------
# 合成ポリシー（実物が無いあいだのテスト用。形式は実物と同じ）
# ---------------------------------------------------------------------------


def _default_actions(joints: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for index, joint in enumerate(joints):
        low, high = (-45.0, 45.0) if joint.startswith("hip") else (-80.0, 20.0)
        rows.append({
            "actuator": joint, "joint": f"{joint}_joint", "motion_type": "angular",
            "kind": "position", "index": index, "fallback": "0",
            "unit": "deg", "low": low, "high": high,
            "scale": math.radians(1.0), "source": "angle_limits_degrees",
        })
    return rows


def _default_normaliser(names: Sequence[str]) -> tuple[list[float], list[float]]:
    mean, std = [], []
    for name in names:
        if name.endswith("_qw"):
            m, s = 0.99, 0.01
        elif name.endswith(("_qx", "_qy", "_qz")):
            m, s = 0.0, 0.05
        elif "gyro" in name:
            m, s = 0.0, 30.0
        elif name.startswith("cmd_"):
            m, s = (0.0, 15.0) if "hip" in name else (-30.0, 20.0)
        elif name == "goal_vx":
            m, s = 100.0, 60.0
        elif name.startswith("goal_"):
            m, s = 0.0, 25.0
        else:
            m, s = 0.0, 0.7
        mean.append(float(np.float32(m)))
        std.append(float(np.float32(s)))
    return mean, std


def make_synthetic_policy(*, hidden: Sequence[int] = (64, 64), activation: str = "swish",
                          observations: Sequence[str] = DEFAULT_OBSERVATIONS,
                          joints: Sequence[str] = DEFAULT_JOINTS, seed: int = 0,
                          alpha: float = 1.0, slew_deg: float = 0.0,
                          witness_samples: int = 16) -> bytes:
    """``cadex_train.py`` が書くのと同じ構造のヘッダを持つ、乱数重みの .cxpolicy。

    witness はこの実装の float64 順伝播で記録するので、エンジンとトレーナーの
    witness 検査（相対 1e-4）を通る。ヘッダには ``synthetic: true`` を付ける。
    """

    if activation not in ACTIVATIONS:
        raise ValueError(activation)
    rng = np.random.default_rng(seed)
    names = list(observations)
    actions = _default_actions(joints)
    widths = [len(names), *[int(h) for h in hidden], len(actions)]
    shapes = list(zip(widths[:-1], widths[1:]))
    flat: list[np.ndarray] = []
    for inputs, outputs in shapes:
        w = rng.normal(size=(inputs, outputs)) / math.sqrt(inputs)
        b = rng.normal(size=outputs) * 0.1
        flat += [w.astype(np.float32).reshape(-1), b.astype(np.float32)]
    weights = np.concatenate(flat)
    mean, std = _default_normaliser(names)
    low = np.array([a["low"] for a in actions])
    high = np.array([a["high"] for a in actions])
    header: dict[str, Any] = {
        "schema": POLICY_SCHEMA,
        "label": "synthetic",
        "synthetic": True,
        "task": {"sha256": "0" * 64, "label": "synthetic"},
        "model": {"sha256": "0" * 64, "path": "outputs/synthetic-model.xml"},
        "observations": names,
        "actions": actions,
        "network": {
            "kind": "mlp", "layers": [[int(a), int(b)] for a, b in shapes],
            "activation": activation, "output": "tanh",
            "output_scale": [float(v) for v in (high - low) / 2.0],
            "output_bias": [float(v) for v in (high + low) / 2.0],
        },
        "normaliser": {"mean": mean, "std": std, "count": 0},
        "exploration": {"distribution": "gaussian", "log_std": [-1.0] * len(actions),
                        "space": "pre_activation", "state_dependent": True},
        "training": {"trainer_sha256": "synthetic", "seed": int(seed),
                     "action_filter_alpha": float(alpha), "command_slew_deg": float(slew_deg)},
        "evaluation": {"observations": [], "actions": []},
    }
    draft = parse_policy(encode_policy(header, weights))
    obs = (np.asarray(mean) + np.asarray(std) * rng.normal(size=(witness_samples, len(names))))
    obs = obs.astype(np.float32).astype(np.float64)
    header["evaluation"] = {
        "observations": [[float(v) for v in row] for row in obs],
        "actions": [[float(v) for v in row] for row in forward(draft, obs)],
    }
    return encode_policy(header, weights)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def describe(policy: CxPolicy) -> str:
    sizes = "-".join(str(n) for n in policy.layer_sizes)
    macs = sum(int(w.size) for w, _ in policy.layers)
    lines = [
        f"層構成      {sizes}（活性化 {policy.activation}、出力 tanh）",
        f"パラメータ  {policy.weights.size} 個（float32 で {policy.weights.size * 4} バイト）、MAC {macs}",
        f"観測        {len(policy.observations)} チャネル: {', '.join(policy.observations)}",
        f"行動        {len(policy.actions)} 個、単位 {policy.action_unit}",
        f"フィルタ    action_filter_alpha={policy.action_filter_alpha} "
        f"command_slew_deg={policy.command_slew_deg}",
    ]
    for a in policy.actions:
        lines.append(f"  - {a.get('actuator')}: [{a['low']:g}, {a['high']:g}] {a.get('unit')}")
    if policy.header.get("synthetic"):
        lines.append("注意        合成ポリシー（乱数重み）。実機で動かさないこと。")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cxpolicy_io", description=".cxpolicy の読み書きと検査")
    sub = parser.add_subparsers(dest="command", required=True)
    p_info = sub.add_parser("info", help="中身を表示する")
    p_info.add_argument("path")
    p_check = sub.add_parser("check", help="witness と参照実装の一致を検査する（不一致で終了コード 1）")
    p_check.add_argument("path")
    p_synth = sub.add_parser("synth", help="形式に忠実な合成ポリシーを書く")
    p_synth.add_argument("--out", required=True)
    p_synth.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p_synth.add_argument("--activation", choices=ACTIVATIONS, default="swish")
    p_synth.add_argument("--seed", type=int, default=0)
    p_run = sub.add_parser("run", help="観測ベクトル（JSON 配列）を1つ入れて行動を表示する")
    p_run.add_argument("path")
    p_run.add_argument("observation", help='例: "[0.99, 0, 0, ...]"')
    args = parser.parse_args(argv)

    try:
        if args.command == "synth":
            Path(args.out).write_bytes(make_synthetic_policy(
                hidden=args.hidden, activation=args.activation, seed=args.seed))
            print(f"書いた: {args.out}")
            return 0
        policy = load_policy(args.path)
    except (OSError, PolicyFormatError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 2
    if args.command == "info":
        print(describe(policy))
        return 0
    if args.command == "check":
        worst, sample, action = witness_error(policy)
        ok = worst <= WITNESS_TOLERANCE
        print(f"witness 最大相対誤差 {worst:.3e}（標本 {sample}、行動 {action}）"
              f" 許容 {WITNESS_TOLERANCE:.0e}: {'一致' if ok else '不一致'}")
        return 0 if ok else 1
    obs = np.asarray(json.loads(args.observation), dtype=np.float64)
    print(json.dumps([float(v) for v in forward(policy, obs)]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
