"""MJCF上のロボットの取り決め（関節名・脚・運動学）と、立位からの逆運動学。

関節角の0は立位姿勢（CAD を組んだ姿勢）。全関節の軸はワールドの +Y。
正の角度は、脚の先を後ろ（-X）へ振る向き。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

LEGS = ("lf", "rf", "lh", "rh")
JOINTS = [f"{kind}_{leg}" for leg in LEGS for kind in ("hip", "knee")]


def model_legs(model: mujoco.MjModel) -> list[str]:
    """モデルの関節名 ``hip_<leg>`` から脚の名前を、関節の並びの順に出す（しっぽは脚に入らない）。"""
    legs = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name and name.startswith("hip_"):
            legs.append(name[4:])
    return legs


def model_joints(model: mujoco.MjModel) -> list[str]:
    """アクチュエータの順の関節名（脚の hip・knee のあとに、しっぽなら tail_yaw・tail_pitch）。"""
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[a][0]))
            for a in range(model.nu)]


def model_feet(model: mujoco.MjModel) -> list[str]:
    """足の部品名 ``c_foot_<leg>`` を、脚の順に出す。"""
    return [f"c_foot_{leg}" for leg in model_legs(model)]


# トロットの対角ペア
TROT_PHASE = {"lf": 0.0, "rh": 0.0, "rf": 0.5, "lh": 0.5}


@dataclass
class LegGeometry:
    thigh: float          # m
    shin: float           # m
    stand_hip: float      # rad、立位での上腿の絶対角（下向きから後ろ向きが正）
    stand_shin: float     # rad、立位での下腿の絶対角

    @property
    def stand_rel(self) -> float:
        return self.stand_shin - self.stand_hip

    def foot(self, q_hip: float, q_knee: float) -> tuple[float, float]:
        """股軸から見た足先 (x, z) [m]。"""
        a1 = self.stand_hip + q_hip
        a2 = self.stand_shin + q_hip + q_knee
        return (-self.thigh * math.sin(a1) - self.shin * math.sin(a2),
                -self.thigh * math.cos(a1) - self.shin * math.cos(a2))

    def ik(self, x: float, z: float) -> tuple[float, float]:
        """股軸から見た足先 (x, z) を実現する (q_hip, q_knee) [rad]。膝の曲がる向きは立位と同じ。"""
        r2 = x * x + z * z
        c = (r2 - self.thigh ** 2 - self.shin ** 2) / (2 * self.thigh * self.shin)
        c = max(-1.0, min(1.0, c))
        beta = math.copysign(math.acos(c), self.stand_rel)
        gamma = math.atan2(-x, -z)
        a1 = gamma - math.atan2(self.shin * math.sin(beta), self.thigh + self.shin * math.cos(beta))
        return a1 - self.stand_hip, beta - self.stand_rel


def leg_geometry(model: mujoco.MjModel) -> LegGeometry:
    """MJCFの立位姿勢（keyframe）から、左前脚の寸法と立位角を読む。"""
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    hip = data.xanchor[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_lf")]
    knee = data.xanchor[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "knee_lf")]
    foot_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "c_foot_lf/collision0")
    foot = data.geom_xpos[foot_geom]
    th = hip - knee
    sh = knee - foot
    return LegGeometry(
        thigh=float(math.hypot(th[0], th[2])), shin=float(math.hypot(sh[0], sh[2])),
        stand_hip=float(math.atan2(-(knee[0] - hip[0]), -(knee[2] - hip[2]))),
        stand_shin=float(math.atan2(-(foot[0] - knee[0]), -(foot[2] - knee[2]))),
    )


def joint_ids(model: mujoco.MjModel) -> dict[str, dict[str, int]]:
    out = {}
    for name in model_joints(model):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        aid = next(a for a in range(model.nu) if model.actuator_trnid[a][0] == jid)
        out[name] = {"joint": jid, "qpos": int(model.jnt_qposadr[jid]), "dof": int(model.jnt_dofadr[jid]),
                     "actuator": aid}
    return out


def joint_ranges(model: mujoco.MjModel) -> dict[str, tuple[float, float]]:
    return {n: tuple(float(v) for v in model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in model_joints(model)}


def quat_tilt_deg(q: np.ndarray) -> float:
    """クォータニオン (w,x,y,z) の機体上方向と鉛直のなす角。"""
    up_z = 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)
    return float(math.degrees(math.acos(max(-1.0, min(1.0, up_z)))))


def quat_yaw_deg(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


class YawTracker:
    """±180度で折り返すヨー角を、連続した回転量（度）に積算する。"""

    def __init__(self) -> None:
        self.last: float | None = None
        self.total = 0.0

    def update(self, yaw_deg: float) -> float:
        if self.last is not None:
            self.total += (yaw_deg - self.last + 180.0) % 360.0 - 180.0
        self.last = yaw_deg
        return self.total
