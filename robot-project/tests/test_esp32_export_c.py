"""deploy/esp32/export_c.py と firmware/ の純C推論のテスト。

生成した policy_weights.h と firmware/*.c をホストの clang でコンパイルし、
Python 参照実装と同じ入力で同じ出力になることを確かめる。
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from deploy.esp32 import cxpolicy_io as cx
from deploy.esp32 import export_c as ex
from deploy.esp32 import quantize as qz

ROOT = Path(__file__).resolve().parents[1]
ESP32 = ROOT / "deploy" / "esp32"
FIRMWARE = ESP32 / "firmware"
CC = shutil.which("clang") or shutil.which("cc")


@pytest.fixture(scope="module")
def policy() -> cx.CxPolicy:
    return cx.parse_policy(cx.make_synthetic_policy(hidden=(64, 64), seed=21,
                                                    alpha=0.6, slew_deg=4.0))


@pytest.fixture(scope="module")
def samples(policy) -> np.ndarray:
    return qz.calibration_samples(policy, count=600, seed=1)


@pytest.fixture(scope="module")
def qpolicy(policy, samples) -> qz.QuantPolicy:
    return qz.quantize_policy(policy, samples[:400], per_channel=True)


@pytest.fixture(scope="module")
def host_binary(tmp_path_factory, policy, qpolicy) -> Path:
    if CC is None:
        pytest.skip("C コンパイラが無い")
    return _build_host(tmp_path_factory.mktemp("host"), policy, qpolicy, ex.resolve_layout(policy))


def _build_host(work: Path, policy, qpolicy, layout, **header_options) -> Path:
    (work / "policy_weights.h").write_text(
        ex.render_header(policy, qpolicy, layout=layout, source="test", **header_options),
        encoding="utf-8")
    for name in ("policy.c", "policy.h", "control.c", "control.h"):
        shutil.copy(FIRMWARE / name, work / name)
    shutil.copy(FIRMWARE / "host" / "host_main.c", work / "host_main.c")
    binary = work / "host_policy"
    done = subprocess.run(
        [CC, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
         "-I", str(work), str(work / "policy.c"), str(work / "control.c"),
         str(work / "host_main.c"), "-lm", "-o", str(binary)],
        capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return binary


def _run(binary: Path, mode: str, lines: list[str], *args: str) -> list[list[str]]:
    done = subprocess.run([str(binary), mode, *args], input="\n".join(lines) + "\n",
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return [line.split() for line in done.stdout.strip().splitlines()]


def _fmt(row) -> str:
    return " ".join(f"{float(v):.9g}" for v in row)


def test_header_defines(policy, qpolicy):
    text = ex.render_header(policy, qpolicy, layout=ex.resolve_layout(policy), source="x.cxpolicy")
    for needle in ("#define POLICY_N_IN 20", "#define POLICY_N_OUT 8",
                   "#define POLICY_N_LAYERS 3", "#define POLICY_MAX_WIDTH 64",
                   "#define POLICY_ACTIVATION POLICY_ACT_SWISH", "#define POLICY_HAS_INT8 1",
                   "policy_f32_w", "policy_i8_w", "policy_i8_lut", "policy_obs_mean",
                   "policy_output_scale", "policy_action_low", "POLICY_FILTER_ALPHA",
                   "policy_obs_kind", "policy_servo_center_us"):
        assert needle in text, needle
    # 観測チャネル名をコメントで残す（実機側で順序を照合するため）
    assert "body_qw" in text


def test_header_without_int8(policy):
    text = ex.render_header(policy, None, layout=ex.resolve_layout(policy), source="x")
    assert "#define POLICY_HAS_INT8 0" in text
    assert "policy_i8_w" not in text


def test_c_float32_matches_python(host_binary, policy, samples):
    rows = samples[400:460]
    out = _run(host_binary, "forward", [_fmt(r) for r in rows])
    f32 = np.array([[float(v) for v in line[1:]] for line in out if line[0] == "f32"])
    ref = cx.forward(policy, rows)
    span = policy.action_high - policy.action_low
    assert f32.shape == ref.shape
    assert np.max(np.abs(f32 - ref) / span) < 1e-5


def test_c_int8_matches_python_int8(host_binary, qpolicy, samples):
    rows = samples[400:460]
    out = _run(host_binary, "forward", [_fmt(r) for r in rows])
    raw = np.array([[float(v) for v in line[1:]] for line in out if line[0] == "i8raw"],
                   dtype=np.float32)
    act = np.array([[float(v) for v in line[1:]] for line in out if line[0] == "i8"])
    expected_raw = np.stack([qz.forward_int8(qpolicy, r, trace=True)["raw"] for r in rows])
    # 整数経路と float32 の再量子化はビット一致する
    np.testing.assert_array_equal(raw, expected_raw)
    np.testing.assert_allclose(act, qz.forward_int8(qpolicy, rows), atol=1e-4)


def test_c_command_filter(host_binary, policy):
    rng = np.random.default_rng(3)
    raws = rng.uniform(-90, 90, size=(6, 8))
    out = _run(host_binary, "filter", [_fmt(r) for r in raws])
    got = np.array([[float(v) for v in line] for line in out])
    f = cx.CommandFilter.from_policy(policy)
    expected = np.stack([f.step(r) for r in raws])
    np.testing.assert_allclose(got, expected, atol=1e-4)


def test_c_observation_build(host_binary, policy):
    layout = ex.resolve_layout(policy)
    quat = np.array([0.98, 0.1, -0.05, 0.15]); quat /= np.linalg.norm(quat)
    gyro = np.array([0.3, -0.2, 1.1])          # rad/s
    cmd = np.linspace(-20, 20, 8)              # deg
    before = cmd - 1.5
    goal = np.array([150.0, 0.0, 10.0, 0.0])
    phase = 1.2
    expected = ex.build_observation(layout, quat=quat, gyro_rad_s=gyro, command=cmd,
                                    command_before=before, goal=goal, phase_rad=phase,
                                    dt_s=0.02)
    line = _fmt(np.concatenate([quat, gyro, cmd, before, goal, [phase, 0.02]]))
    got = np.array([float(v) for v in _run(host_binary, "obs", [line])[0]])
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-5)


def test_c_servo_pulse(host_binary):
    servo = ex.ServoConfig()
    for joint, angle in ((0, 0.0), (1, 30.0), (2, -45.0), (3, 120.0), (4, -120.0)):
        out = _run(host_binary, "pulse", [], str(joint), str(angle))
        pulse, duty = float(out[0][0]), int(out[0][1])
        assert pulse == pytest.approx(ex.servo_pulse_us(angle, servo), abs=1e-3)
        assert duty == ex.ledc_duty(pulse)
    assert ex.servo_pulse_us(500.0, servo) == servo.max_us
    assert ex.servo_pulse_us(-500.0, servo) == servo.min_us


def test_resolve_layout_rejects_unknown_channel(policy):
    header, weights = cx.decode_policy(cx.make_synthetic_policy(hidden=(8,), seed=0))
    header["observations"][0] = "mystery_channel"
    strange = cx.parse_policy(cx.encode_policy(header, weights))
    with pytest.raises(ex.LayoutError):
        ex.resolve_layout(strange)
    explicit = {"channels": {name: {"source": "const", "gain": 0.0} for name in header["observations"]}}
    assert len(ex.resolve_layout(strange, explicit)) == 20


def test_cli_writes_header(tmp_path):
    pol = tmp_path / "p.cxpolicy"
    pol.write_bytes(cx.make_synthetic_policy(hidden=(32, 16), seed=4))
    out = tmp_path / "policy_weights.h"
    done = subprocess.run(
        [sys.executable, "-m", "deploy.esp32.export_c", str(pol), "--out", str(out),
         "--count", "256"], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "#define POLICY_N_IN 20" in out.read_text()
    template = subprocess.run(
        [sys.executable, "-m", "deploy.esp32.export_c", str(pol), "--layout-template"],
        cwd=ROOT, capture_output=True, text=True)
    assert template.returncode == 0, template.stderr
    assert "body_gyro_x" in json.loads(template.stdout)["channels"]


def test_esp_idf_skeleton_present():
    main_c = (FIRMWARE / "main" / "main.c").read_text(encoding="utf-8")
    for needle in ("app_main", "vTaskDelayUntil", "ledc_set_duty", "policy_forward_f32",
                   "obs_build", "command_filter_step", "esp_cpu_get_cycle_count"):
        assert needle in main_c, needle
    assert (FIRMWARE / "CMakeLists.txt").exists()
    assert "policy.c" in (FIRMWARE / "main" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert (FIRMWARE / "main" / "policy_weights.h").exists()


# ---------------------------------------------------------------------------
# タスクバンドルからのレイアウト解決（実物の walk-task.json と同じ形）
# ---------------------------------------------------------------------------

REAL_NAMES = ("rot_qw", "rot_qx", "rot_qy", "rot_qz", "gyro_x", "gyro_y", "gyro_z", "cmd",
              "clock_sin", "clock_cos",
              *(n for j in cx.DEFAULT_JOINTS for n in (f"c_{j}", f"c_{j}_rate")))


def _mini_task(gyro_frame=None) -> dict:
    gyro = {"name": "gyro", "kind": "component_angular_velocity",
            "channels": ["gyro_x", "gyro_y", "gyro_z"], "unit": "deg/s",
            "scale": 57.29577951308232, "grounded_sensor": "imu"}
    if gyro_frame:
        gyro["frame"] = gyro_frame
    return {
        "observations": [
            {"name": "rot", "kind": "component_orientation",
             "channels": ["rot_qw", "rot_qx", "rot_qy", "rot_qz"], "unit": "quat", "scale": 1.0,
             "grounded_sensor": "imu"},
            gyro,
            {"name": "vel", "kind": "component_linear_velocity", "channels": ["vel_x"],
             "scale": 1000.0, "role": "privileged"},
        ],
        "goal": [{"channels": ["cmd"], "kind": "speed", "name": "cmd", "unit": "mm/s"},
                 {"channels": ["clock_sin", "clock_cos"], "kind": "phase", "name": "clock",
                  "period_s": 0.5, "radians_per_step": 0.25132741228718347}],
        "commands": [{"action": k, "channels": [f"c_{j}", f"c_{j}_rate"], "kind": "command",
                      "unit": "deg", "rate_unit": "deg/s"}
                     for k, j in enumerate(cx.DEFAULT_JOINTS)],
    }


@pytest.fixture(scope="module")
def real_like() -> cx.CxPolicy:
    return cx.parse_policy(cx.make_synthetic_policy(hidden=(16, 8), observations=REAL_NAMES,
                                                    seed=9))


def test_layout_from_task(real_like):
    layout = {s.name: s for s in ex.resolve_layout(real_like, task=_mini_task())}
    assert layout["rot_qz"].kind == "quat" and layout["rot_qz"].index == 3
    # frame の無い角速度は世界座標 → 機体座標のジャイロをクォータニオンで回す
    assert layout["gyro_y"].kind == "gyro_world" and layout["gyro_y"].index == 1
    assert layout["gyro_y"].gain == pytest.approx(57.29577951308232)
    assert layout["cmd"].kind == "goal" and layout["cmd"].index == 0
    assert layout["clock_sin"].kind == "phase_sin" and layout["clock_cos"].kind == "phase_cos"
    assert layout["c_knee_rf"].kind == "cmd" and layout["c_knee_rf"].index == 3
    assert layout["c_knee_rf_rate"].kind == "cmd_rate" and layout["c_knee_rf_rate"].index == 3
    framed = {s.name: s for s in ex.resolve_layout(real_like, task=_mini_task("c_chassis"))}
    assert framed["gyro_x"].kind == "gyro"
    assert ex.clock_period_s(_mini_task()) == pytest.approx(0.5)


def test_layout_from_task_refuses_privileged(real_like):
    header, weights = cx.decode_policy(cx.make_synthetic_policy(
        hidden=(8,), observations=("vel_x", *REAL_NAMES[1:]), seed=0))
    with pytest.raises(ex.LayoutError):
        ex.resolve_layout(cx.parse_policy(cx.encode_policy(header, weights)), task=_mini_task())


def test_gyro_world_rotates_body_rate():
    # 機体が z まわりに 90° 回っているとき、機体 x 軸まわりの回転は世界 y 軸まわり
    q = np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
    layout = [ex.ObsSource(f"g{k}", "gyro_world", k, 1.0) for k in range(3)]
    got = ex.build_observation(layout, quat=q, gyro_rad_s=np.array([1.0, 0.0, 0.0]),
                               command=np.zeros(8), command_before=np.zeros(8),
                               goal=np.zeros(4), phase_rad=0.0, dt_s=0.02)
    np.testing.assert_allclose(got, [0.0, 1.0, 0.0], atol=1e-12)


def test_c_matches_python_with_task_layout(tmp_path, real_like):
    if CC is None:
        pytest.skip("C コンパイラが無い")
    layout = ex.resolve_layout(real_like, task=_mini_task())
    binary = _build_host(tmp_path, real_like, None, layout, clock_period_s=0.5)
    quat = np.array([0.95, 0.12, -0.2, 0.2]); quat /= np.linalg.norm(quat)
    gyro = np.array([0.4, -0.7, 1.3])
    cmd = np.linspace(-30, 30, 8); before = cmd + 0.8
    goal = np.array([120.0, 0.0, 0.0, 0.0])
    expected = ex.build_observation(layout, quat=quat, gyro_rad_s=gyro, command=cmd,
                                    command_before=before, goal=goal, phase_rad=2.0, dt_s=0.02)
    line = _fmt(np.concatenate([quat, gyro, cmd, before, goal, [2.0, 0.02]]))
    got = np.array([float(v) for v in _run(binary, "obs", [line])[0]])
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-4)
    header = (tmp_path / "policy_weights.h").read_text(encoding="utf-8")
    assert "#define POLICY_CLOCK_PERIOD_S 0.5f" in header


def _real_cases() -> list[tuple[Path, Path]]:
    """実物の方策と、それを学習したタスクの組。

    手元の CPU 学習の1つ（Git の追跡外、無ければ飛ばす）と、``artifacts/training/full-*/`` に残した
    GPU 学習の最終の方策すべて。タスクは各回の run.json の記録（``<robot-project>/...``）から取る。
    """

    cases = [(ROOT / "artifacts" / "runs" / "cpu-a" / "walk.000050.cxpolicy",
              ROOT / "artifacts" / "tmp" / "export-base" / "walk-task.json")]
    for record in sorted((ROOT / "artifacts" / "training").glob("full-*/run.json")):
        task = json.loads(record.read_text(encoding="utf-8"))["task"].replace("<robot-project>", str(ROOT))
        cases.append((record.parent / "walk.cxpolicy", Path(task)))
    return cases


@pytest.mark.parametrize("real_policy,real_task", _real_cases(),
                         ids=lambda p: p.parent.name if p.suffix == ".cxpolicy" else "")
def test_real_policy_end_to_end(tmp_path, real_policy, real_task):
    if not (real_policy.exists() and real_task.exists()):
        pytest.skip("実物の .cxpolicy / walk-task.json が無い")
    if CC is None:
        pytest.skip("C コンパイラが無い")
    REAL_POLICY, REAL_TASK = real_policy, real_task
    policy = cx.load_policy(REAL_POLICY)
    assert cx.witness_error(policy)[0] < cx.WITNESS_TOLERANCE
    task = json.loads(REAL_TASK.read_text(encoding="utf-8"))
    layout = ex.resolve_layout(policy, task=task)
    samples = qz.calibration_samples(policy, count=600, seed=2)
    qp = qz.quantize_policy(policy, samples[:400])
    names = [str(a.get("actuator")).split("/")[0] for a in policy.actions]
    if len(policy.actions) != len(cx.DEFAULT_JOINTS):
        # しっぽ・3本足など関節が8個でない機体の方策は、配備側（8自由度の実機）では書き出さずに止める。
        with pytest.raises(ValueError, match=f"関節 {len(cx.DEFAULT_JOINTS)} 個"):
            ex.render_header(policy, qp, layout=layout, clock_period_s=ex.clock_period_s(task))
        return
    if names != list(cx.DEFAULT_JOINTS):
        # 関節が8個でも名前・順番が違う機体（tri3-r8 のロール付き3本足など）も、サーボの割り当てとずれるので止める。
        with pytest.raises(ValueError, match="関節の名前または順番が実機の割り当てと違う"):
            ex.render_header(policy, qp, layout=layout, clock_period_s=ex.clock_period_s(task))
        return
    binary = _build_host(tmp_path, policy, qp, layout, clock_period_s=ex.clock_period_s(task))
    rows = samples[400:]
    out = _run(binary, "forward", [_fmt(r) for r in rows])
    f32 = np.array([[float(v) for v in line[1:]] for line in out if line[0] == "f32"])
    raw = np.array([[float(v) for v in line[1:]] for line in out if line[0] == "i8raw"],
                   dtype=np.float32)
    span = policy.action_high - policy.action_low
    assert np.max(np.abs(f32 - cx.forward(policy, rows)) / span) < 1e-5
    np.testing.assert_array_equal(
        raw, np.stack([qz.forward_int8(qp, r, trace=True)["raw"] for r in rows]))


# ---------------------------------------------------------------------------
# 起動時の指令: 学習のリセットと同じ値（CadexDynamics.reset_command）
# ---------------------------------------------------------------------------

_KEYED_MODEL = """<mujoco>
  <worldbody><body name="b"><joint name="j0" type="hinge"/><joint name="j1" type="hinge"/>
    <geom size="0.01"/></body></worldbody>
  <actuator><position name="a0" joint="j0"/><position name="a1" joint="j1"/></actuator>
  <keyframe><key name="other" ctrl="0 0"/><key name="solved" ctrl="0.1 -2.0"/></keyframe>
</mujoco>
"""


def test_reset_command_is_the_reset_keyframe_ctrl_in_surface_units_held_in_box(tmp_path):
    import hashlib

    model = tmp_path / "outputs" / "m.xml"
    model.parent.mkdir()
    model.write_text(_KEYED_MODEL)
    task = {"model": {"path": "outputs/m.xml", "sha256": hashlib.sha256(model.read_bytes()).hexdigest()},
            "episode": {"reset_keyframe": "solved"},
            "actions": [{"actuator": "a0", "index": 0, "scale": 0.017453292519943295, "low": -45.0, "high": 20.0},
                        {"actuator": "a1", "index": 1, "scale": 0.017453292519943295, "low": -60.0, "high": 45.0}]}
    task_path = tmp_path / "outputs" / "walk-task.json"
    task_path.write_text(json.dumps(task))
    # 0.1 rad = 5.73°（範囲内）、-2.0 rad = -114.6° → 下限 -60 に切り詰め
    assert ex.reset_command(task_path) == pytest.approx([5.729577951308232, -60.0])


def test_reset_command_of_the_trained_robot_is_zero():
    # 基準機体の学習用カットモデル: キーフレーム solved に ctrl が無い → 全関節 0°
    task = ROOT / "robot" / "mjcf" / "base" / "cut" / "walk-task.json"
    assert ex.reset_command(task) == [0.0] * 8


def test_cli_uses_the_reset_command_when_the_task_is_given(tmp_path):
    from deploy.esp32 import cxpolicy_io

    task = ROOT / "robot" / "mjcf" / "base" / "cut" / "walk-task.json"
    import hashlib

    header, weights = cxpolicy_io.decode_policy(
        cxpolicy_io.make_synthetic_policy(hidden=(8,), observations=REAL_NAMES, seed=1))
    header["task"] = {**(header.get("task") or {}),
                      "sha256": hashlib.sha256(task.read_bytes()).hexdigest()}  # このタスクで学習したことにする
    blob = cxpolicy_io.encode_policy(header, weights)
    policy = tmp_path / "p.cxpolicy"
    policy.write_bytes(blob)
    out = tmp_path / "w.h"
    assert ex.main([str(policy), "--task", str(task), "--out", str(out), "--no-int8"]) == 0
    text = out.read_text()
    import re

    body = re.search(r"const float policy_initial_command\[8\] = \{(.*?)\};", text, re.S).group(1)
    values = [float(v.strip().rstrip("f")) for v in body.split(",") if v.strip()]
    assert values == [0.0] * 8


def test_rejects_same_count_wrong_names():
    """関節が8個でも名前と順番が違う方策（tri3-r8 など）は、書き出しで止める。"""
    names = ("roll_lf", "hip_lf", "knee_lf", "roll_rf", "hip_rf", "knee_rf", "hip_hc", "knee_hc")
    policy = cx.parse_policy(cx.make_synthetic_policy(hidden=(8,), seed=3, joints=names))
    assert len(policy.actions) == len(cx.DEFAULT_JOINTS)
    layout = ex.resolve_layout(cx.parse_policy(cx.make_synthetic_policy(hidden=(8,), seed=3)))
    with pytest.raises(ValueError, match="roll_lf"):
        ex.render_header(policy, None, layout=layout, source="x")
