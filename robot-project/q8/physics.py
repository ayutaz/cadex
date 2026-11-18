"""出力したMJCFに対する、学習前の物理チェック（質量・重心・支持・保持トルク）。

すべてMuJoCo上の計算で、実機での確認ではない。CADの質量はOCCTの厳密形状 × 宣言した密度。
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .robot import joint_ids, leg_geometry, model_joints, model_legs, quat_tilt_deg


def body_groups(model: mujoco.MjModel) -> dict[str, float]:
    groups: dict[str, float] = {}
    for b in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        key = name[2:] if name.startswith("c_") else name
        if "servo" in key:
            g = "servo"
        elif "horn" in key or "bolt" in key:
            g = "hardware"
        elif key in ("esp", "imu"):
            g = "electronics"
        elif key == "bat":
            g = "battery"
        elif key.startswith("foot"):
            g = "printed_tpu"
        else:
            g = "printed_pla"
        groups[g] = groups.get(g, 0.0) + float(model.body_mass[b])
    return {k: round(v * 1000, 2) for k, v in sorted(groups.items())}


def support_margin(com_xy: np.ndarray, feet_xy: list[np.ndarray]) -> float:
    """重心の投影から支持多角形（凸包）の辺までの最短距離 [m]。外なら負。"""
    pts = np.array(feet_xy)
    centre = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0]))
    pts = pts[order]
    margin = math.inf
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        edge = b - a
        normal = np.array([edge[1], -edge[0]]) / np.linalg.norm(edge)
        if np.dot(normal, centre - a) < 0:
            normal = -normal
        margin = min(margin, float(np.dot(normal, com_xy - a)))
    return margin


def fewest_support(physics: dict) -> dict:
    """static_report の static_estimate のうち、支える脚が最も少ない場合（4脚はトロット、3脚は1脚を上げた最悪）。

    static_estimate は支える脚の多い順に並ぶ（JSON を通しても順は変わらない）。
    """
    return list(physics["static_estimate"].values())[-1]


def static_report(model: mujoco.MjModel, stall_nm: float, rated_nm: float, settle_s: float = 2.0) -> dict:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    total = float(model.body_subtreemass[1])
    com = data.subtree_com[1].copy()
    legs = model_legs(model)
    feet = {leg: data.geom_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"c_foot_{leg}/collision0")].copy()
            for leg in legs}
    ids = joint_ids(model)
    geo = leg_geometry(model)
    hip_xyz = {leg: data.xanchor[ids[f"hip_{leg}"]["joint"]].copy() for leg in legs}
    knee_xyz = {leg: data.xanchor[ids[f"knee_{leg}"]["joint"]].copy() for leg in legs}
    g = 9.81
    # 静力学の見積り（脚の質量を無視し、足先に鉛直荷重 F）: 関節トルク = F × 水平距離
    def leg_torques(load_n: float) -> dict:
        out = {}
        for leg in legs:
            out[leg] = {"hip_nm": load_n * abs(hip_xyz[leg][0] - feet[leg][0]),
                        "knee_nm": load_n * abs(knee_xyz[leg][0] - feet[leg][0])}
        return out
    cases = {}
    n_legs = len(legs)
    if n_legs == 4:
        support_cases = (("4脚支持（立位）", 4), ("3脚支持（ウォーク）", 3), ("2脚支持（トロット）", 2))
    else:
        support_cases = ((f"{n_legs}脚支持（立位）", n_legs),
                         (f"{n_legs - 1}脚支持（1脚を上げた最悪）", n_legs - 1))
    for label, n_support in support_cases:
        tq = leg_torques(total * g / n_support)
        worst = max(max(v["hip_nm"], v["knee_nm"]) for v in tq.values())
        cases[label] = {"load_per_foot_n": round(total * g / n_support, 3),
                        "worst_joint_torque_nmm": round(worst * 1000, 1),
                        "stall_ratio": round(worst / stall_nm, 3), "rated_ratio": round(worst / rated_nm, 3)}
    # MuJoCo で立位を保持させる: 指令0（立位角）で settle_s 秒
    data.ctrl[:] = 0.0
    steps = int(settle_s / model.opt.timestep)
    forces = []
    z0 = float(data.qpos[2])
    for i in range(steps):
        mujoco.mj_step(model, data)
        if i >= steps * 3 // 4:
            forces.append(np.abs(data.actuator_force.copy()))
    forces = np.array(forces).mean(axis=0)
    hold = {name: round(float(forces[ids[name]["actuator"]]) * 1000, 2) for name in model_joints(model)}
    return {
        "total_mass_g": round(total * 1000, 1),
        "mass_by_group_g": body_groups(model),
        "com_m": com.round(5).tolist(),
        "com_height_mm": round(float(com[2]) * 1000, 1),
        "feet_xy_mm": {leg: (feet[leg][:2] * 1000).round(1).tolist() for leg in legs},
        "support_margin_mm": round(support_margin(com[:2], [f[:2] for f in feet.values()]) * 1000, 1),
        "leg": {"thigh_mm": round(geo.thigh * 1000, 2), "shin_mm": round(geo.shin * 1000, 2),
                "stand_hip_deg": round(math.degrees(geo.stand_hip), 2),
                "stand_shin_deg": round(math.degrees(geo.stand_shin), 2)},
        "static_estimate": cases,
        "sim_hold": {
            "seconds": settle_s,
            "body_drop_mm": round((z0 - float(data.qpos[2])) * 1000, 2),
            "tilt_deg": round(quat_tilt_deg(data.qpos[3:7]), 3),
            "mean_abs_torque_nmm": hold,
            "worst_stall_ratio": round(max(hold.values()) / (stall_nm * 1000), 3),
            "worst_rated_ratio": round(max(hold.values()) / (rated_nm * 1000), 3),
        },
        "note": "静力学の見積りは脚の質量と慣性力を無視した下限。MuJoCoの保持は位置サーボの比例帯（5度で失速トルク）を含む。いずれも実機で未検証。",
    }
