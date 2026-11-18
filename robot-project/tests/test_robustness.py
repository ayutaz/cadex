"""実機との差に対する強さの評価（q8/robustness.py）のテスト。

評価はエンジンの評価（CadexDynamics.evaluate_success）をそのまま使い、モデルの値の変更と、
2 ms ごとのサーボの遅れ・ガタ・制御周期のゆらぎ・押しを、mj_step を包んで入れる。
何も変えない条件では、エンジンの評価と同じ数字になることを最初に確かめる。
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import mujoco
import numpy as np
import pytest

from q8 import robustness as rb

ROOT = Path(__file__).resolve().parents[1]
VARIANT = "mini-r2-p7-eff"
MJCF = ROOT / "robot" / "mjcf" / VARIANT
POLICY = ROOT / "artifacts" / "training" / "full-mini-r2-p7-eff-s3" / "walk.cxpolicy"


@pytest.fixture(scope="module")
def engine_inputs():
    if not (MJCF / "model-model.xml").exists() or not POLICY.exists():
        pytest.skip("mini-r2-p7-eff のモデルか s3 の方策が無い")
    xml = (MJCF / "model-model.xml").read_bytes()
    task = json.loads((MJCF / "walk-task.json").read_text())
    return xml, task, POLICY.read_bytes()


@pytest.fixture(scope="module")
def model():
    if not (MJCF / "model-model.xml").exists():
        pytest.skip("mini-r2-p7-eff のモデルが無い")
    return mujoco.MjModel.from_xml_path(str(MJCF / "model-model.xml"))


# --- 何も変えなければ、エンジンの評価と同じ ----------------------------------

def test_nominal_conditions_reproduce_the_engine_evaluation(engine_inputs):
    xml, task, policy = engine_inputs
    dyn = rb.dynamics()
    container = dyn.decode_policy(policy)
    one = {**task, "success": {**task["success"], "seeds": [2101]}}
    compiled = dyn.load_model(xml)
    names = rb.component_names(compiled)
    engine = dyn.evaluate_success(xml, one, container, components=names)["seeds"][0]
    ours = rb.evaluate_case(xml, task, container, seed=2101, conditions=rb.Conditions())
    assert ours["pass"] == engine["pass"]
    for key, value in engine["metrics"].items():
        if isinstance(value, float):
            assert ours["metrics"][key] == pytest.approx(value, rel=1e-9, abs=1e-9), key
        else:
            assert ours["metrics"][key] == value, key


def test_the_engine_is_restored_after_a_case(engine_inputs):
    xml, task, policy = engine_inputs
    dyn = rb.dynamics()
    before = (dyn.load_model, dyn._mujoco_module)
    rb.evaluate_case(xml, task, dyn.decode_policy(policy), seed=2101,
                     conditions=rb.Conditions(latency_ms=20.0, backlash_deg=2.0))
    assert (dyn.load_model, dyn._mujoco_module) == before
    assert dyn._mujoco_module() is mujoco


def test_latency_and_backlash_change_the_episode_but_it_still_runs(engine_inputs):
    xml, task, policy = engine_inputs
    container = rb.dynamics().decode_policy(policy)
    nominal = rb.evaluate_case(xml, task, container, seed=2101, conditions=rb.Conditions())
    late = rb.evaluate_case(xml, task, container, seed=2101,
                            conditions=rb.Conditions(latency_ms=40.0, jitter_ms=4.0, backlash_deg=3.0))
    assert late["metrics"]["slip_share_max"] != nominal["metrics"]["slip_share_max"]
    assert late["metrics"]["duration_s"] > 0


# --- サーボへの指令の届き方（遅れ・ゆらぎ） ------------------------------------

def test_servo_bus_holds_the_old_command_until_the_latency_has_passed():
    bus = rb.ServoBus(initial=np.zeros(2), latency_s=0.030, jitter_s=0.0, dt=0.002, per_action=10,
                      rng=random.Random(0))
    seen = [bus.active(k, np.array([1.0, 2.0]) if k < 10 else np.array([5.0, 6.0]))[0] for k in range(40)]
    # 0 番目の小刻みで出した指令は 30 ms = 15 刻み後に届く。次の指令（10 刻み目）は 25 刻み目
    assert seen[:15] == [0.0] * 15
    assert seen[15:25] == [1.0] * 10
    assert seen[25:] == [5.0] * 15


def test_servo_bus_without_latency_passes_the_command_through():
    bus = rb.ServoBus(initial=np.zeros(1), latency_s=0.0, jitter_s=0.0, dt=0.002, per_action=10,
                      rng=random.Random(0))
    assert [bus.active(k, np.array([float(k // 10)]))[0] for k in range(30)] == [float(k // 10) for k in range(30)]


def test_servo_bus_jitter_stays_inside_its_window():
    bus = rb.ServoBus(initial=np.zeros(1), latency_s=0.010, jitter_s=0.004, dt=0.002, per_action=10,
                      rng=random.Random(3))
    arrivals = []
    last = 0.0
    for k in range(2000):
        value = bus.active(k, np.array([float(k // 10 + 1)]))[0]
        if value != last:
            arrivals.append(k - (int(value) - 1) * 10)
            last = value
    assert min(arrivals) >= 5 and max(arrivals) <= 7       # 10〜14 ms（2 ms 刻み）
    assert len(set(arrivals)) > 1                           # ゆらいでいる


# --- ガタ（不感帯） -----------------------------------------------------------

def test_backlash_deadband_gives_no_restoring_error_inside_the_play():
    assert rb.deadband(np.array([3.0, -0.5, -3.0, 0.9]), 1.0).tolist() == [2.0, 0.0, -2.0, 0.0]
    assert rb.deadband(np.array([0.3]), 0.0).tolist() == [0.3]


# --- モデルの値の変更 ---------------------------------------------------------

def test_friction_is_set_on_the_feet_and_the_floor(model):
    m = rb.perturb_model(mujoco.MjModel.from_xml_path(model_path(model)), rb.Conditions(friction=0.35))
    for name in [*(f"c_foot_{leg}/collision0" for leg in ("lf", "rf", "lh", "rh")), "environment/floor"]:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert m.geom_friction[g, 0] == pytest.approx(0.35), name


def test_servo_stiffness_scales_gain_and_bias_together_and_torque_scales_the_limit(model):
    m = rb.perturb_model(mujoco.MjModel.from_xml_path(model_path(model)),
                         rb.Conditions(kp_scale=0.7, torque_scale=0.8))
    for a in range(model.nu):
        assert m.actuator_gainprm[a, 0] == pytest.approx(0.7 * model.actuator_gainprm[a, 0])
        assert m.actuator_biasprm[a, 1] == pytest.approx(0.7 * model.actuator_biasprm[a, 1])
        assert m.actuator_gainprm[a, 0] == pytest.approx(-m.actuator_biasprm[a, 1])
        assert m.actuator_forcerange[a].tolist() == pytest.approx((0.8 * model.actuator_forcerange[a]).tolist())


def test_mass_and_centre_of_mass_offset_move_the_whole_robot_centre(model):
    base = mujoco.MjData(model)
    mujoco.mj_forward(model, base)
    m = rb.perturb_model(mujoco.MjModel.from_xml_path(model_path(model)),
                         rb.Conditions(com_offset_mm=(5.0, -3.0), mass_scale=1.0))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "c_chassis")
    share = model.body_mass[chassis] / model.body_subtreemass[0]
    shift = (d.subtree_com[0] - base.subtree_com[0]) * 1000.0
    assert shift[0] == pytest.approx(5.0 * share, rel=1e-6)
    assert shift[1] == pytest.approx(-3.0 * share, rel=1e-6)
    heavy = rb.perturb_model(mujoco.MjModel.from_xml_path(model_path(model)), rb.Conditions(mass_scale=1.2))
    assert heavy.body_mass[chassis] == pytest.approx(1.2 * model.body_mass[chassis])
    assert heavy.body_inertia[chassis].tolist() == pytest.approx((1.2 * model.body_inertia[chassis]).tolist())


def model_path(model) -> str:  # noqa: ARG001 - 同じファイルから毎回読み直す（その場で書き換えるため）
    return str(MJCF / "model-model.xml")


# --- 押し ---------------------------------------------------------------------

def test_push_is_sideways_to_the_heading():
    facing_x = np.array([1.0, 0.0, 0.0, 0.0])
    assert rb.push_force(facing_x, side=+1, newtons=0.5).tolist() == pytest.approx([0.0, 0.5, 0.0])
    yaw = math.radians(90.0)
    facing_y = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    assert rb.push_force(facing_y, side=+1, newtons=0.5).tolist() == pytest.approx([-0.5, 0.0, 0.0], abs=1e-12)
    assert rb.push_force(facing_y, side=-1, newtons=0.5).tolist() == pytest.approx([0.5, 0.0, 0.0], abs=1e-12)


def test_push_is_applied_to_the_chassis_only_inside_its_window(model):
    push = rb.Push(newtons=0.6, side=+1, at_s=0.02, duration_s=0.04)
    proxy = rb.EngineProxy(mujoco, rb.Conditions(push=push), per_action=10, rng=random.Random(0))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "c_chassis")
    applied = []
    for _ in range(50):
        proxy.mj_step(model, data)
        applied.append(float(np.linalg.norm(data.xfrc_applied[chassis, :3])))
    on = [i for i, f in enumerate(applied) if f > 0]
    assert on == list(range(10, 30))                      # 20〜60 ms、2 ms 刻みで 20 回
    assert applied[10] == pytest.approx(0.6)
    assert float(np.abs(data.xfrc_applied).sum()) == 0.0  # 窓が閉じたら消える


def _frames(tilts, speed_mm_s=60.0, cmd=75.0, dt=0.02):
    frames = []
    for i, tilt in enumerate(tilts):
        half = math.radians(tilt) / 2.0
        frames.append({"frame_kind": "solver_output", "nominal_time_s": i * dt, "goal": [cmd, 0.0, 1.0],
                       "component_placements": {"c_chassis": {
                           "position_mm": [speed_mm_s * i * dt, 0.0, 80.0],
                           "rotation_xyzw": [math.sin(half), 0.0, 0.0, math.cos(half)]}}})
    return frames


def test_push_recovery_reads_tilt_and_walking_after_the_push():
    push = rb.Push(newtons=0.5, side=+1, at_s=3.0, duration_s=0.1)
    calm = [5.0] * 401
    assert rb.push_recovery(_frames(calm), push, completed=True) == {
        "survived": True, "recovered": True, "recovery_s": pytest.approx(0.0, abs=1e-9),
        "walk_ratio_after": pytest.approx(0.8)}
    wobble = [5.0] * 155 + [20.0] * 30 + [5.0] * 216       # 3.1〜3.7 s は傾いたまま
    out = rb.push_recovery(_frames(wobble), push, completed=True)
    assert out["recovered"] and out["recovery_s"] == pytest.approx(0.6, abs=0.03)
    assert not rb.push_recovery(_frames([20.0] * 401), push, completed=True)["recovered"]
    assert not rb.push_recovery(_frames(calm, speed_mm_s=10.0), push, completed=True)["recovered"]
    fell = rb.push_recovery(_frames(calm), push, completed=False)
    assert fell["survived"] is False and fell["recovered"] is False


# --- 実機を想定したランダム化の引き方 -----------------------------------------

def test_sampled_conditions_stay_in_their_ranges_and_repeat_from_the_seed():
    draws = [rb.sample_conditions(random.Random(k)) for k in range(200)]
    for c in draws:
        for field, (low, high) in rb.REAL_RANGES.items():
            value = getattr(c, field)
            values = value if isinstance(value, tuple) else (value,)
            bounds = (low, high) if not isinstance(low, tuple) else None
            if bounds:
                assert all(low <= v <= high for v in values), field
            else:
                assert all(lo <= v <= hi for v, lo, hi in zip(values, low, high)), field
    assert rb.sample_conditions(random.Random(7)) == rb.sample_conditions(random.Random(7))
    assert len({c.friction for c in draws}) == len(draws)


def test_suites_cover_every_gap_the_real_robot_has():
    names = {name for name, _ in rb.SWEEP}
    for word in ("friction", "latency", "jitter", "backlash", "kp", "torque", "mass", "com"):
        assert any(name.startswith(word) for name in names), word
    assert all(level > 0 for level in rb.PUSH_NEWTONS)


# --- 学習シード方向のまとめと、2案の比較 ---------------------------------------

def _report(rate, recovered, slip):
    return {"summary": {
        "nominal/nominal": {"suite": "nominal", "name": "nominal", "n": 10, "pass_rate": rate,
                            "metrics": {"slip_share_max": slip}, "failing": {}},
        "push/push 2 N": {"suite": "push", "name": "push 2 N", "n": 20, "pass_rate": rate,
                          "survived_rate": recovered, "recovered_rate": recovered,
                          "metrics": {"slip_share_max": slip}, "failing": {}}}}


def test_reports_are_gathered_across_training_seeds_and_compared():
    a = rb.across([_report(1.0, 0.5, 0.3), _report(0.5, 0.5, 0.4)])
    assert a["n"] == 2
    assert a["groups"]["nominal/nominal"]["pass_rate"]["mean"] == pytest.approx(0.75)
    assert a["groups"]["push/push 2 N"]["recovered_rate"]["values"] == [0.5, 0.5]
    b = rb.across([_report(1.0, 0.9, 0.1), _report(1.0, 1.0, 0.2)])
    c = rb.compare_reports(a, b, labels=("orig", "dr"))
    row = c["groups"]["push/push 2 N"]["recovered_rate"]
    assert row["separated"] and row["difference"] == pytest.approx(0.45)
    md = rb.compare_markdown(c)
    assert "orig" in md and "dr" in md and "push 2 N" in md and "倒れず戻った" in md


def test_compare_robustness_evaluates_each_run_once_into_the_label_folder(tmp_path):
    sys_path = str(ROOT / "scripts")
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    import compare_robustness

    runs, calls = [], []
    for label, rate in (("orig", 0.5), ("dr", 1.0)):
        for s in range(2):
            d = tmp_path / "training" / f"{label}-s{s}"
            d.mkdir(parents=True)
            (d / "walk.cxpolicy").write_bytes(b"x")
            runs.append((label, d))

    def fake(variant, policy, out, processes=None):
        calls.append((variant, Path(out)))
        rate = 0.5 if "orig" in Path(policy).parent.name else 1.0
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "robustness.json").write_text(json.dumps(_report(rate, rate, 0.3)))

    for _ in range(2):
        result = compare_robustness.run(runs, variant="mini", artifacts=tmp_path / "a", evaluate=fake)
    assert len(calls) == 4 and {c[0] for c in calls} == {"mini"}
    assert calls[2][1] == tmp_path / "a" / "dr" / "robustness" / "dr-s0"
    assert result["groups"]["nominal/nominal"]["pass_rate"]["difference"] == pytest.approx(0.5)
    assert (tmp_path / "a" / "compare" / "robust_orig_vs_dr.md").exists()
