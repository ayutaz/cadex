"""Phase 4: 学習なしの固定モーションを MuJoCo で再生し、数値で確かめる。

順に実行する場面:
  1. joint_check  各関節を1つずつ ±15度 動かし、指令した向きに動くか確かめる（宙吊りではなく接地のまま）
  2. stand        指令0（立位）を保持
  3. trot         逆運動学のトロット（前進）
  4. turn         左右の歩幅差による旋回

出力: <out>/fixed_gait.json（判定つき）、<out>/<scene>.csv（時系列）、<out>/<scene>.mp4（--video 時）

使い方:
  uv run python simulation/fixed_gait.py robot/mjcf/quadruped8-visual.xml artifacts/sim/base --video
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from q8.gait import TrotParams, as_ctrl, sweep_targets, trot_targets  # noqa: E402
from q8.render import VideoWriter, camera  # noqa: E402
from q8.robot import JOINTS, YawTracker, joint_ids, joint_ranges, leg_geometry, quat_tilt_deg, quat_yaw_deg  # noqa: E402

FPS = 30


def robot_geoms(model: mujoco.MjModel) -> set[int]:
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "environment/floor")
    return {g for g in range(model.ngeom) if g != floor and model.geom_contype[g] + model.geom_conaffinity[g] > 0}


def run_scene(model, name, target_fn, seconds, out_dir: Path, video: bool, stall_nm: float) -> dict:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    ids = joint_ids(model)
    ranges = joint_ranges(model)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "environment/floor")
    feet = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"c_foot_{leg}/collision0") for leg in ("lf", "rf", "lh", "rh")}
    dt = model.opt.timestep
    steps = int(seconds / dt)
    frame_every = max(1, int(round(1.0 / (FPS * dt))))
    writer = renderer = None
    if video:
        renderer = mujoco.Renderer(model, 480, 768)
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        writer = VideoWriter(out_dir / f"{name}.mp4", 768, 480, FPS)
    start = data.qpos[:3].copy()
    z_start = float(data.qpos[2])
    rows = []
    worst_limit_violation = 0.0
    self_contacts: dict[str, float] = {}
    non_foot_floor: dict[str, float] = {}
    saturated = np.zeros(model.nu)
    peak_force = np.zeros(model.nu)
    max_tilt = 0.0
    min_height = math.inf
    tracking_err = []
    yaw = YawTracker()
    for i in range(steps):
        t = i * dt
        targets = target_fn(t)
        data.ctrl[:] = as_ctrl(targets, ids, model.nu, ranges)
        mujoco.mj_step(model, data)
        q = {n: float(data.qpos[ids[n]["qpos"]]) for n in JOINTS}
        for n in JOINTS:
            lo, hi = ranges[n]
            worst_limit_violation = max(worst_limit_violation, lo - q[n], q[n] - hi)
        f = np.abs(data.actuator_force)
        peak_force = np.maximum(peak_force, f)
        saturated += f >= 0.98 * stall_nm
        for c in data.contact[: data.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if floor in (g1, g2):
                other = g2 if g1 == floor else g1
                if other not in feet and c.dist < 0:
                    key = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
                    non_foot_floor[key] = min(non_foot_floor.get(key, 0.0), float(c.dist))
                continue
            key = " / ".join(sorted(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in (g1, g2)))
            self_contacts[key] = min(self_contacts.get(key, 0.0), float(c.dist))
        tilt = quat_tilt_deg(data.qpos[3:7])
        yaw.update(quat_yaw_deg(data.qpos[3:7]))
        max_tilt = max(max_tilt, tilt)
        min_height = min(min_height, float(data.qpos[2]) - z_start)
        if t > 0.5:
            tracking_err.append(max(abs(data.ctrl[ids[n]["actuator"]] - q[n]) for n in JOINTS))
        if i % 10 == 0:
            rows.append({"t": round(t, 4), "x": data.qpos[0], "y": data.qpos[1], "z": data.qpos[2],
                         "tilt_deg": tilt, "yaw_deg": yaw.total,
                         **{f"q_{n}": q[n] for n in JOINTS},
                         **{f"cmd_{n}": float(data.ctrl[ids[n]["actuator"]]) for n in JOINTS},
                         **{f"tau_{n}": float(data.actuator_force[ids[n]["actuator"]]) for n in JOINTS}})
        if writer is not None and i % frame_every == 0:
            renderer.update_scene(data, camera(150.0, -18.0, 0.45, data.subtree_com[1] + np.array([0, 0, -0.02])),
                                  scene_option=opt)
            writer.write(renderer.render())
    if writer is not None:
        writer.close()
        renderer.close()
    with (out_dir / f"{name}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    end = data.qpos[:3]
    travel = end - start
    return {
        "seconds": seconds,
        "travel_mm": (travel * 1000).round(1).tolist(),
        "mean_speed_x_mm_s": round(float(travel[0]) / seconds * 1000, 1),
        "final_yaw_deg": round(yaw.total, 2),
        "max_tilt_deg": round(max_tilt, 2),
        "max_base_drop_mm": round(-min(min_height, 0.0) * 1000, 1),
        "joint_limit_violation_deg": round(math.degrees(max(worst_limit_violation, 0.0)), 3),
        "self_contacts_min_dist_mm": {k: round(v * 1000, 2) for k, v in self_contacts.items()},
        "non_foot_floor_contacts_mm": {k: round(v * 1000, 2) for k, v in non_foot_floor.items()},
        "peak_torque_nmm": {n: round(float(peak_force[ids[n]["actuator"]]) * 1000, 1) for n in JOINTS},
        "saturation_share": {n: round(float(saturated[ids[n]["actuator"]]) / steps, 3) for n in JOINTS},
        "max_tracking_error_deg": round(math.degrees(max(tracking_err)), 2) if tracking_err else None,
        "video": str(out_dir / f"{name}.mp4") if video else None,
    }


def joint_direction_check(model) -> dict:
    """各関節に +10度 を指令し、足先が期待どおり後ろ（-X）へ動くかを見る（胴体は固定して確認）。"""
    out = {}
    m = model
    ids = joint_ids(m)
    for name in JOINTS:
        data = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, data, 0)
        mujoco.mj_forward(m, data)
        leg = name.split("_")[1]
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"c_foot_{leg}/collision0")
        before = data.geom_xpos[gid].copy()
        base = data.qpos[:7].copy()
        for _ in range(300):
            data.ctrl[:] = 0.0
            data.ctrl[ids[name]["actuator"]] = math.radians(10.0)
            data.qpos[:7] = base + np.array([0, 0, 0.05, 0, 0, 0, 0])  # 宙に浮かせて固定
            data.qvel[:6] = 0.0
            mujoco.mj_step(m, data)
        mujoco.mj_forward(m, data)
        moved = data.geom_xpos[gid] - before
        q = math.degrees(data.qpos[ids[name]["qpos"]])
        out[name] = {"commanded_deg": 10.0, "reached_deg": round(q, 2),
                     "foot_dx_mm": round(float(moved[0]) * 1000, 2),
                     "ok": abs(q - 10.0) < 1.5 and moved[0] < 0}
    return out


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("out_dir")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--period", type=float, default=0.5)
    ap.add_argument("--stride", type=float, default=0.040)
    ap.add_argument("--lift", type=float, default=0.008)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(args.model)
    stall = float(max(model.actuator_forcerange[:, 1]))
    geo = leg_geometry(model)
    trot = TrotParams(period_s=args.period, stride_m=args.stride, lift_m=args.lift)
    turn = TrotParams(period_s=args.period, stride_m=args.stride, lift_m=args.lift, turn=0.4)
    report = {
        "model": args.model,
        "leg_geometry_mm_deg": {"thigh": geo.thigh * 1000, "shin": geo.shin * 1000,
                                "stand_hip": math.degrees(geo.stand_hip), "stand_shin": math.degrees(geo.stand_shin)},
        "joint_direction": joint_direction_check(model),
        "scenes": {},
    }
    report["scenes"]["stand"] = run_scene(model, "stand", lambda t: {n: 0.0 for n in JOINTS}, 3.0, out, args.video, stall)
    report["scenes"]["joint_sweep_hip_lf"] = run_scene(
        model, "joint_sweep_hip_lf", lambda t: sweep_targets(t, "hip_lf", math.radians(15)), 4.0, out, args.video, stall)
    report["scenes"]["trot"] = run_scene(model, "trot", lambda t: trot_targets(t, geo, trot), args.seconds, out, args.video, stall)
    report["scenes"]["turn"] = run_scene(model, "turn", lambda t: trot_targets(t, geo, turn), args.seconds, out, args.video, stall)
    report["trot_params"] = {"period_s": trot.period_s, "stride_mm": trot.stride_m * 1000, "lift_mm": trot.lift_m * 1000,
                             "duty": trot.duty, "nominal_speed_mm_s": round(trot.nominal_speed_m_s * 1000, 1)}
    # 判定（固定モーションの成功条件）
    st, tr = report["scenes"]["stand"], report["scenes"]["trot"]
    report["checks"] = {
        "all_joints_move_in_commanded_direction": all(v["ok"] for v in report["joint_direction"].values()),
        "stand_holds": st["max_tilt_deg"] < 5 and st["max_base_drop_mm"] < 10,
        "no_joint_limit_violation": all(s["joint_limit_violation_deg"] < 1.0 for s in report["scenes"].values()),
        "no_body_floor_contact": all(not s["non_foot_floor_contacts_mm"] for s in report["scenes"].values()),
        "no_self_contact_penetration": all(all(v > -0.5 for v in s["self_contacts_min_dist_mm"].values())
                                           for s in report["scenes"].values()),
        "trot_moves_forward": tr["mean_speed_x_mm_s"] > 20 and tr["max_tilt_deg"] < 30,
    }
    (out / "fixed_gait.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    print(json.dumps(report["checks"], ensure_ascii=False, indent=2, default=_json_default))
    print(json.dumps({k: {kk: v[kk] for kk in ("mean_speed_x_mm_s", "final_yaw_deg", "max_tilt_deg", "max_base_drop_mm")}
                      for k, v in report["scenes"].items()}, ensure_ascii=False, indent=2))
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
