"""固定歩容（学習なし）を、学習済み方策と同じ物差しで評価する。

学習済み方策は `cadex evaluate`（エンジンが方策を再生し、CadexEvaluation で指標を計算）で評価する。
固定歩容はエンジンで再生できない（方策ファイルではない）ので、ここで MuJoCo 上で再生し、
エンジンと同じ形式のトレース（cadex-assembly-simulation-trace の frames）を書いて、
**エンジンと同じ** CadexEvaluation.measure / check に渡す。リグ（足・腰高などの尺度）も
エンジンと同じ CadexDynamics.evaluation_rig で作る。どちらも Cadex 本体のコードをそのまま読み込む（改変しない）。

違い（記録しておくこと）: リセット時の傾き・持ち上げは同じ範囲から引くが、乱数列はエンジンとは別。
質量・減衰などのランダム化は固定歩容では掛けない（公称値）。
"""

from __future__ import annotations

import math
import sys
from typing import Callable

import mujoco
import numpy as np

from .paths import CADEX_REPO
from .robot import JOINTS, joint_ids, joint_ranges, quat_tilt_deg

FEET = ["c_foot_lf", "c_foot_rf", "c_foot_lh", "c_foot_rh"]


def cadex_modules():
    """Cadex の CadexDynamics と CadexEvaluation（純Python + mujoco）を読み込む。"""
    mod_dir = str(CADEX_REPO / "src" / "Mod" / "cadex")
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
    import CadexDynamics  # noqa: PLC0415
    import CadexEvaluation  # noqa: PLC0415

    return CadexDynamics, CadexEvaluation


def success_spec(task: dict) -> dict:
    s = task["success"]
    return {"predicates": s["predicates"], "seeds": s.get("seeds") or [], "feet": s.get("feet") or FEET,
            "episode_seconds": (s.get("episode") or {}).get("episode_seconds", 8.0)}


def _channels(model: mujoco.MjModel, rated_fraction: float) -> list[dict]:
    ids = joint_ids(model)
    ranges = joint_ranges(model)
    out = []
    for name in JOINTS:
        a = ids[name]["actuator"]
        stall = float(model.actuator_forcerange[a][1]) * 1000.0
        lo, hi = (math.degrees(v) for v in ranges[name])
        out.append({"actuator": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a), "joint": name,
                    "kind": "position", "motion_type": "angular", "unit": "deg", "joint_unit": "deg",
                    "low": lo, "high": hi, "effort_low": -stall, "effort_high": stall,
                    "effort_rated": stall * rated_fraction, "effort_unit": "N·mm"})
    return out


def _placements(model, data) -> dict:
    out = {}
    for b in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        w, x, y, z = (float(v) for v in data.xquat[b])
        out[name] = {"position_mm": [float(v) * 1000.0 for v in data.xpos[b]], "rotation_xyzw": [x, y, z, w]}
    return out


def _apply_reset(model, data, rng: np.random.Generator, tilt_deg=(0.0, 3.0), height_mm=(4.0, 6.0)) -> None:
    tilt = math.radians(rng.uniform(*tilt_deg))
    azimuth = rng.uniform(0.0, 2.0 * math.pi)
    axis = np.array([-math.sin(azimuth), math.cos(azimuth), 0.0])
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, axis, tilt)
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q, data.qpos[3:7].copy())
    data.qpos[3:7] = out
    data.qpos[2] += rng.uniform(*height_mm) / 1000.0


def record_episode(model: mujoco.MjModel, target_fn: Callable[[float], dict], *, seconds: float,
                   control_hz: int, seed: int, command_mm_s: float, rated_fraction: float = 0.4,
                   clock_period_s: float = 0.5, reset: bool = True) -> dict:
    """target_fn(t) は関節名→目標角[rad]（省略した関節は0=立位）。Cadex形式のトレースを返す。"""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    if reset:
        _apply_reset(model, data, np.random.default_rng(seed))
    mujoco.mj_forward(model, data)
    ids = joint_ids(model)
    ranges = joint_ranges(model)
    substeps = max(1, int(round(1.0 / (control_hz * model.opt.timestep))))
    steps = int(round(seconds * control_hz))
    z0 = float(data.qpos[2])
    frames = [{"frame_index": 0, "frame_kind": "input", "nominal_time_s": None}]
    termination = ""
    commands = None

    def frame(k: int) -> dict:
        t = k / control_hz
        phase = 2.0 * math.pi * t / clock_period_s
        q = [math.degrees(float(data.qpos[ids[n]["qpos"]])) for n in JOINTS]
        v = [math.degrees(float(data.qvel[ids[n]["dof"]])) for n in JOINTS]
        eff = [float(data.actuator_force[ids[n]["actuator"]]) * 1000.0 for n in JOINTS]
        out = {"frame_index": k + 1, "frame_kind": "solver_output", "nominal_time_s": round(t, 6),
               "component_placements": _placements(model, data), "joint_positions": q, "joint_velocities": v,
               "goal": [command_mm_s, math.sin(phase), math.cos(phase)]}
        if commands is not None:  # リセット直後のフレームは、エンジンと同じく指令とトルクを持たない
            out["actuator_commands"] = commands
            out["actuator_efforts"] = eff
        return out

    frames.append(frame(0))
    done = 0
    for k in range(steps):
        t = k / control_hz
        targets = target_fn(t)
        cmd_rad = []
        for n in JOINTS:
            lo, hi = ranges[n]
            q = min(max(float(targets.get(n, 0.0)), lo), hi)
            data.ctrl[ids[n]["actuator"]] = q
            cmd_rad.append(q)
        commands = [math.degrees(v) for v in cmd_rad]
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        frames.append(frame(k + 1))
        done = k + 1
        up = 1.0 - 2.0 * (data.qpos[4] ** 2 + data.qpos[5] ** 2)
        if up < 0.7:
            termination = "fell"
            break
        if float(data.qpos[2]) - z0 < -0.030:
            termination = "sank"
            break
    trace = {
        "schema": "q8-fixed-gait-trace-v1 (cadex-assembly-simulation-trace の frames 互換)",
        "actuator_channels": _channels(model, rated_fraction),
        "goal_channels": [{"channel": "cmd", "goal": "cmd", "kind": "speed", "unit": "mm/s"},
                          {"channel": "clock_sin", "goal": "clock", "kind": "phase", "unit": "rad"},
                          {"channel": "clock_cos", "goal": "clock", "kind": "phase", "unit": "rad"}],
        "frames": frames,
        "dynamics": {"control_hz": control_hz, "steps_per_frame": 1, "solver_step_s": model.opt.timestep},
        "policy": {"seed": seed, "step_count": done, "termination": termination,
                   "truncated": not termination and done == steps},
    }
    return {"trace": trace, "max_tilt_deg": quat_tilt_deg(data.qpos[3:7])}


def evaluate_fixed_gait(model: mujoco.MjModel, task: dict, *, target_fn: Callable[[float], dict],
                        seeds: list[int], seconds: float, command_mm_s: float, feet: list[str] | None = None,
                        control_hz: int = 50, rated_fraction: float = 0.4) -> dict:
    dyn, ev = cadex_modules()
    feet = feet or success_spec(task)["feet"]
    rig = dyn.evaluation_rig(model, feet=feet)
    predicates = task["success"]["predicates"]
    results = []
    for seed in seeds:
        ep = record_episode(model, target_fn, seconds=seconds, control_hz=control_hz, seed=seed,
                            command_mm_s=command_mm_s, rated_fraction=rated_fraction)
        read = ev.read_trace(ep["trace"])
        solved = [f for f in ep["trace"]["frames"] if f["frame_kind"] == "solver_output"]
        measured = ev.measure(read["samples"], read["episode"], rig, command_mm_s=command_mm_s,
                              drives=read["drives"],
                              commands={"speed": [command_mm_s] * len(solved)})
        held = ev.check(predicates, measured["metrics"])
        results.append({"seed": seed, "pass": all(p["pass"] for p in held), "predicates": held,
                        "metrics": measured["metrics"]})
    return {"seeds": results, "passed": sum(r["pass"] for r in results), "of": len(results),
            "note": "固定歩容（開ループ）。指標は CadexEvaluation（エンジンと同じコード）。"}


def metrics_table(result: dict, *, label: str, keys: list[str]) -> list[dict]:
    return [{"label": label, "seed": s["seed"], "pass": s["pass"], **{k: s["metrics"].get(k) for k in keys}}
            for s in result["seeds"]]
