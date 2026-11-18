"""firmware/servo_cal.c・servo_console.c（サーボの試験・較正モード）をホストの C で確かめる。

ハードに依存する部分（PWM・NVS・シリアル）は servo_hal.h の薄い層に分けてあり、ここでは
firmware/host/servo_host_main.c の偽の層（保存先はファイル）と組んで動かす。
ESP-IDF 用の層（main/servo_hal_esp.c）と main.c は、tests/esp_idf_stubs の仮のヘッダで構文と型だけを確かめる。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from deploy.esp32 import servo_calibration as sc

ROOT = Path(__file__).resolve().parents[1]
FIRMWARE = ROOT / "deploy" / "esp32" / "firmware"
STUBS = Path(__file__).resolve().parent / "esp_idf_stubs"
CC = shutil.which("clang") or shutil.which("cc")
UPD = 1900.0 / 180.0
RATE_30 = 30.0 * UPD / 50.0  # 既定の速さ 30°/s の 1 周期（20 ms）ぶん [µs]
CFLAGS = ["-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off"]


@pytest.fixture(scope="module")
def host(tmp_path_factory) -> Path:
    if CC is None:
        pytest.skip("C コンパイラが無い")
    work = tmp_path_factory.mktemp("servo")
    binary = work / "servo_host"
    srcs = [FIRMWARE / "servo_cal.c", FIRMWARE / "servo_console.c",
            FIRMWARE / "host" / "servo_host_main.c"]
    done = subprocess.run([CC, *CFLAGS, "-I", str(FIRMWARE), "-I", str(FIRMWARE / "main"),
                           *map(str, srcs), "-lm", "-o", str(binary)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return binary


class Session:
    def __init__(self, binary: Path, store: Path):
        self.binary, self.store = binary, store

    def run(self, *lines: str) -> list[str]:
        done = subprocess.run([str(self.binary), str(self.store)],
                              input="\n".join(lines) + "\n", capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        return done.stdout.splitlines()


@pytest.fixture
def s(host, tmp_path) -> Session:
    return Session(host, tmp_path / "nvs.bin")


def frames(out: list[str]) -> list[list[float | None]]:
    """``T`` の行（1 周期ごとの各チャネルのパルス幅、``-`` は出していない）。"""

    rows = []
    for line in out:
        if line.startswith("T "):
            rows.append([None if v == "-" else float(v) for v in line.split()[1:]])
    return rows


def last_dump(out: list[str]) -> sc.Calibration:
    return sc.parse_dump("\n".join(out), sc.DEFAULT_TASK)


# ---------------------------------------------------------------------------
# 起動・選択・停止
# ---------------------------------------------------------------------------


def test_boot_sends_no_pulse_and_never_runs_the_policy(s):
    out = s.run("#tick 5")
    assert all(v is None for row in frames(out) for v in row)
    text = "\n".join(out)
    assert "方策は動かさない" in text
    assert "既定値" in text  # 保存が無いので servo_map.h の値で始める


def test_select_starts_only_that_joint_at_its_center(s):
    out = s.run("sel knee_rf", "#tick 2")
    rows = frames(out)
    assert rows[-1] == [None, None, None, 1450.0, None, None, None, None]
    out = s.run("sel 3", "#tick 1")
    assert frames(out)[-1][3] == 1450.0


def test_off_stops_every_pulse(s):
    out = s.run("sel 0", "sel 1", "#tick 1", "off", "#tick 1")
    rows = frames(out)
    assert rows[0][:2] == [1450.0, 1450.0]
    assert rows[-1] == [None] * 8


def test_commands_that_need_a_joint_fail_without_one(s):
    out = s.run("+", "zero", "lo")
    assert sum(line.startswith("err") for line in out) == 3


def test_unknown_command_is_an_error(s):
    out = s.run("fly")
    assert any(line.startswith("err") for line in out)


# ---------------------------------------------------------------------------
# 安全: 速さの制限・角度の上限と下限
# ---------------------------------------------------------------------------


def test_motion_is_rate_limited(s):
    out = s.run("sel hip_lf", "#tick 1", "deg 20", "#tick 60")
    pulses = [row[0] for row in frames(out)]
    steps = [b - a for a, b in zip(pulses, pulses[1:])]
    assert max(steps) <= RATE_30 + 1e-3
    assert min(steps) >= 0.0
    assert pulses[-1] == pytest.approx(1450.0 + 20.0 * UPD, abs=0.01)
    assert len([d for d in steps if d > 0]) >= int(20.0 * UPD / RATE_30)


def test_speed_can_be_lowered_but_is_bounded(s):
    out = s.run("sel 0", "#tick 1", "speed 10", "deg 10", "#tick 5")
    pulses = [row[0] for row in frames(out)]
    assert pulses[1] - pulses[0] == pytest.approx(10.0 * UPD / 50.0, abs=1e-3)
    out = s.run("speed 1000")
    assert any(line.startswith("err") for line in out)


def test_targets_are_clamped_to_the_model_range_plus_margin(s):
    """未測定のあいだは、学習の範囲 ±10°（hip_lf は -55°〜+30°）の外へは出さない。"""

    out = s.run("sel hip_lf", "#tick 1", "us 2400", "#tick 200", "us 500", "#tick 300")
    pulses = [row[0] for row in frames(out)]
    assert max(pulses) == pytest.approx(1450.0 + 30.0 * UPD, abs=0.01)
    assert min(pulses) == pytest.approx(1450.0 - 55.0 * UPD, abs=0.01)
    assert any("範囲" in line for line in out)


def test_pulses_never_leave_the_hard_limits(s):
    # mark で us_per_deg を 16 にすると、knee_lf の -70° は 330 µs になる。それでも 500 µs で止まる
    out = s.run("sel 1", "us 1930", "#tick 100", "mark 30", "deg -500", "#tick 400",
                "deg 500", "#tick 400")
    pulses = [v for row in frames(out) for v in row if v is not None]
    assert min(pulses) == pytest.approx(500.0)
    assert max(pulses) <= 2400.0


# ---------------------------------------------------------------------------
# 較正: 中立・向き・範囲・倍率
# ---------------------------------------------------------------------------


def test_nudge_follows_the_joint_sign(s):
    # hip_lf は名目の符号 +1、hip_rf は -1。「+」は関節角を増やす（足先が後ろへ）
    out = s.run("sel hip_lf", "step 10", "+", "+", "#tick 10", "sel hip_rf", "step 10", "+",
                "#tick 10")
    rows = frames(out)
    assert rows[-1][0] == pytest.approx(1470.0)
    assert rows[-1][2] == pytest.approx(1440.0)


def test_step_is_bounded(s):
    out = s.run("sel 0", "step 0", "step 51", "step 5")
    errs = [line for line in out if line.startswith("err")]
    assert len(errs) == 2


def test_zero_flip_and_range_are_recorded(s):
    out = s.run("sel hip_lf", "us 1480", "#tick 10", "zero",
                "deg -46", "#tick 100", "lo",
                "deg 21", "#tick 150", "hi",
                "dump")
    cal = last_dump(out)
    j = cal.joints[0]
    assert j.center_us == pytest.approx(1480.0)
    assert j.lo_us == pytest.approx(1480.0 - 46.0 * UPD, abs=0.05)
    assert j.hi_us == pytest.approx(1480.0 + 21.0 * UPD, abs=0.05)
    assert cal.saved is False  # まだ保存していない
    out = s.run("sel hip_lf", "flip", "dump")
    assert last_dump(out).joints[0].sign == -1.0


def test_recorded_range_clamps_later_moves(s):
    out = s.run("sel knee_lf", "deg -20", "#tick 100", "lo", "deg 20", "#tick 200", "hi",
                "deg 40", "#tick 200", "deg -40", "#tick 200")
    pulses = [row[1] for row in frames(out)]
    assert max(pulses) == pytest.approx(1450.0 + 20.0 * UPD, abs=0.01)
    assert min(pulses) == pytest.approx(1450.0 - 20.0 * UPD, abs=0.01)


def test_range_ends_that_disagree_with_the_sign_are_flagged(s):
    # 名目の符号 +1 で、足先が前の端（lo）を大きいパルスで記録した = 向きが逆
    out = s.run("sel hip_lf", "deg 15", "#tick 100", "lo", "deg -15", "#tick 200", "hi")
    assert any("flip" in line for line in out)


def test_clear_forgets_the_range(s):
    out = s.run("sel 0", "deg 10", "#tick 100", "hi", "clear", "dump")
    assert last_dump(out).joints[0].hi_us == 0.0


def test_mark_sets_scale_and_sign_from_a_known_angle(s):
    out = s.run("sel hip_lh", "us 1750", "#tick 100", "mark 30", "dump")
    j = last_dump(out).joints[4]
    assert j.us_per_deg == pytest.approx(10.0, abs=1e-3)
    assert j.sign == 1.0
    out = s.run("sel hip_lh", "us 1750", "#tick 100", "mark -30", "dump")
    assert last_dump(out).joints[4].sign == -1.0
    out = s.run("sel hip_lh", "us 1460", "#tick 100", "mark 5", "mark 30")
    assert sum(line.startswith("err") for line in out) == 2  # 角度が小さい、倍率が外れ


def test_reset_restores_the_defaults_of_the_selected_joint(s):
    out = s.run("sel 2", "us 1500", "#tick 50", "zero", "reset", "dump")
    assert last_dump(out).joints[2].center_us == 1450.0


# ---------------------------------------------------------------------------
# 保存（NVS の代わりのファイル）
# ---------------------------------------------------------------------------


def test_save_survives_a_reboot(s):
    s.run("sel hip_rh", "us 1430", "#tick 20", "zero", "deg -36", "#tick 200", "lo",
          "deg 46", "#tick 300", "hi", "save")
    out = s.run("dump")
    assert any("保存した較正" in line for line in out)
    cal = last_dump(out)
    j = cal.joints[6]
    assert cal.saved is True
    assert j.center_us == pytest.approx(1430.0)
    assert j.lo_us == pytest.approx(1430.0 - 36.0 * -1.0 * UPD, abs=0.05)
    assert j.hi_us == pytest.approx(1430.0 + 46.0 * -1.0 * UPD, abs=0.05)


def test_a_corrupted_store_falls_back_to_defaults(s):
    s.run("sel 0", "us 1500", "#tick 50", "zero", "save")
    out = s.run("#corrupt 20", "#reboot", "dump")
    text = "\n".join(out)
    assert "壊れて" in text
    assert last_dump(out).joints[0].center_us == 1450.0


def test_load_discards_unsaved_changes(s):
    out = s.run("sel 0", "us 1500", "#tick 50", "zero", "load", "dump")
    assert last_dump(out).joints[0].center_us == 1450.0


def test_a_full_session_passes_the_python_check(s, tmp_path):
    """C の dump を Python が読み、検査を通り、export_c の servo.json になる（結合）。"""

    import json

    from deploy.esp32 import cxpolicy_io as cx
    from deploy.esp32 import export_c as ex

    lines = []
    for k, j in enumerate(sc.task_joints(sc.DEFAULT_TASK)):
        lines += [f"sel {k}", f"deg {j.model_low_deg - 2.0}", "#tick 400", "lo",
                  f"deg {j.model_high_deg + 2.0}", "#tick 500", "hi"]
    out = s.run(*lines, "save", "dump")
    assert not [line for line in out if line.startswith(("err", "warn"))]
    cal = last_dump(out)
    assert sc.check(cal, sc.DEFAULT_TASK) == ([], [])
    path = tmp_path / "servo.json"
    path.write_text(json.dumps(sc.servo_json(cal, task=sc.DEFAULT_TASK)), encoding="utf-8")
    policy = cx.parse_policy(cx.make_synthetic_policy(hidden=(8,), seed=1))
    for servo, j in zip(ex.load_servos(path, policy), cal.joints):
        assert ex.servo_pulse_us(j.model_low_deg, servo) == pytest.approx(j.angle_to_us(j.model_low_deg))
        assert ex.servo_pulse_us(j.model_low_deg - 5.0, servo) == pytest.approx(
            min(j.lo_us, j.hi_us) if j.sign > 0 else max(j.lo_us, j.hi_us))


# ---------------------------------------------------------------------------
# 1個ずつの動作確認
# ---------------------------------------------------------------------------


def test_wiggle_visits_plus_and_minus_ten_degrees_slowly(s):
    out = s.run("sel hip_lf", "#tick 1", "test", "#tick 300")
    pulses = [row[0] for row in frames(out)]
    assert max(pulses) == pytest.approx(1450.0 + 10.0 * UPD, abs=0.01)
    assert min(pulses) == pytest.approx(1450.0 - 10.0 * UPD, abs=0.01)
    assert pulses[-1] == pytest.approx(1450.0)
    steps = [abs(b - a) for a, b in zip(pulses, pulses[1:])]
    assert max(steps) <= RATE_30 + 1e-3


# ---------------------------------------------------------------------------
# ESP-IDF 用の層（仮のヘッダで構文と型だけ）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("run_policy", [0, 1])
def test_esp_layer_compiles_against_stub_headers(tmp_path, run_policy):
    if CC is None:
        pytest.skip("C コンパイラが無い")
    for src in (FIRMWARE / "main" / "main.c", FIRMWARE / "main" / "servo_hal_esp.c"):
        done = subprocess.run(
            [CC, *CFLAGS, "-fsyntax-only", f"-DQ8_RUN_POLICY={run_policy}", "-I", str(STUBS),
             "-I", str(FIRMWARE), "-I", str(FIRMWARE / "main"), str(src)],
            capture_output=True, text=True)
        assert done.returncode == 0, f"{src.name}: {done.stderr}"


def test_firmware_defaults_to_the_servo_console():
    main_c = (FIRMWARE / "main" / "main.c").read_text(encoding="utf-8")
    assert "#define Q8_RUN_POLICY 0" in main_c
    cmake = (FIRMWARE / "main" / "CMakeLists.txt").read_text(encoding="utf-8")
    for name in ("servo_hal_esp.c", "../servo_cal.c", "../servo_console.c", "nvs_flash",
                 "Q8_RUN_POLICY=1"):
        assert name in cmake, name
