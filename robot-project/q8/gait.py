"""学習を使わない固定モーション（開ループ）。逆運動学で足先の軌道から関節指令を作る。

PWMサーボは角度指令しか受けないので、ここでの出力も8関節の目標角 [rad]（立位が0）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .robot import JOINTS, LEGS, TROT_PHASE, LegGeometry


@dataclass
class TrotParams:
    period_s: float = 0.4          # 1歩行周期
    stride_m: float = 0.030        # 立脚中に足先が後ろへ動く距離
    lift_m: float = 0.012          # 遊脚の足上げ高さ
    duty: float = 0.5              # 立脚の割合
    turn: float = 0.0              # -1..1。左右の歩幅差（正で左旋回）
    ramp_s: float = 1.0            # 歩幅を0から立ち上げる時間
    phase: dict = field(default_factory=lambda: dict(TROT_PHASE))

    @property
    def nominal_speed_m_s(self) -> float:
        return self.stride_m / (self.period_s * self.duty)


def foot_offset(phase: float, stride: float, lift: float, duty: float) -> tuple[float, float]:
    """立位の足先位置からのずれ (dx, dz)。phase は 0..1。"""
    phase %= 1.0
    if phase < duty:  # 立脚: 前 (+S/2) から後ろ (-S/2) へ一定速度
        s = phase / duty
        return stride * (0.5 - s), 0.0
    s = (phase - duty) / (1.0 - duty)  # 遊脚: 後ろから前へ、正弦で持ち上げる
    return stride * (-0.5 + 0.5 * (1 - math.cos(math.pi * s))), lift * math.sin(math.pi * s)


def trot_targets(t: float, geo: LegGeometry, p: TrotParams) -> dict[str, float]:
    ramp = min(1.0, t / p.ramp_s) if p.ramp_s > 0 else 1.0
    x0, z0 = geo.foot(0.0, 0.0)
    out = {}
    for leg in LEGS:
        side = 1.0 if leg[0] == "l" else -1.0
        stride = p.stride_m * ramp * (1.0 - side * p.turn)
        dx, dz = foot_offset(t / p.period_s + p.phase[leg], stride, p.lift_m * ramp, p.duty)
        qh, qk = geo.ik(x0 + dx, z0 + dz)
        out[f"hip_{leg}"] = qh
        out[f"knee_{leg}"] = qk
    return out


def sweep_targets(t: float, joint: str, amp_rad: float, period_s: float = 2.0) -> dict[str, float]:
    """1関節だけを正弦で動かす（方向確認用）。"""
    out = {name: 0.0 for name in JOINTS}
    out[joint] = amp_rad * math.sin(2 * math.pi * t / period_s)
    return out


def as_ctrl(targets: dict[str, float], ids: dict, nu: int, ranges: dict | None = None) -> np.ndarray:
    ctrl = np.zeros(nu)
    for name, q in targets.items():
        if ranges is not None:
            lo, hi = ranges[name]
            q = min(max(q, lo), hi)
        ctrl[ids[name]["actuator"]] = q
    return ctrl
