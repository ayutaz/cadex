"""deploy/esp32/servo_calibration.py: サーボの較正結果の受け取り・JSON・C ヘッダ・servo.json。

関節の順序と符号の取り決めが、学習のモデル（mini-r2-p7-eff の MJCF とタスク）と一致することを固定する。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from deploy.esp32 import cxpolicy_io as cx
from deploy.esp32 import export_c as ex
from deploy.esp32 import servo_calibration as sc

ROOT = Path(__file__).resolve().parents[1]
FIRMWARE = ROOT / "deploy" / "esp32" / "firmware"
TASK = ROOT / "robot" / "mjcf" / "mini-r2-p7-eff" / "cut" / "walk-task.json"
MODEL = ROOT / "robot" / "mjcf" / "mini-r2-p7-eff" / "cut" / "cut.xml"
EXPORTED = ROOT / "deploy" / "esp32" / "exports" / "full-mini-r2-p7-eff-s3" / "policy_weights.h"
GUIDE = ROOT / "docs" / "assembly-guide.md"

ORDER = ("hip_lf", "knee_lf", "hip_rf", "knee_rf", "hip_lh", "knee_lh", "hip_rh", "knee_rh")


def _need(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} が無い")
    return path


# ---------------------------------------------------------------------------
# 関節の順序・符号・範囲が学習のモデルと一致する
# ---------------------------------------------------------------------------


def test_default_task_is_the_final_robot():
    assert sc.DEFAULT_TASK == TASK


def test_joint_order_matches_the_task_actions():
    joints = sc.task_joints(_need(TASK))
    assert tuple(j.name for j in joints) == ORDER
    assert [j.actuator for j in joints] == [f"{n}/position" for n in ORDER]
    task = json.loads(TASK.read_text(encoding="utf-8"))
    for j, a in zip(joints, task["actions"]):
        assert (j.model_low_deg, j.model_high_deg) == (a["low"], a["high"])


def test_joint_order_matches_the_mjcf_actuators():
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(_need(MODEL)))
    assert [m.actuator(i).name for i in range(m.nu)] == [f"{n}/position" for n in ORDER]
    # 学習の可動域（タスクの行動の範囲）は MJCF の関節の範囲と同じ
    for j in sc.task_joints(TASK):
        lo, hi = np.degrees(m.joint(j.name).range)
        assert (lo, hi) == pytest.approx((j.model_low_deg, j.model_high_deg), abs=1e-3)


def test_joint_order_matches_the_exported_policy():
    text = _need(EXPORTED).read_text(encoding="utf-8")
    names = re.findall(r"^ \*\s+\d+ (\w+)/position: \[", text, flags=re.M)
    assert tuple(names) == ORDER


def test_shipped_servo_map_is_the_nominal_header():
    shipped = (FIRMWARE / "main" / "servo_map.h").read_text(encoding="utf-8")
    assert shipped == sc.render_servo_map_h(sc.nominal(TASK))


def test_gpio_matches_the_wiring_table_in_the_assembly_guide():
    rows = dict(re.findall(r"^\| `(\w+)` \| [^|]+ \| GPIO (\d+) \|", GUIDE.read_text(encoding="utf-8"),
                           flags=re.M))
    assert {n: int(g) for n, g in rows.items()} == dict(zip(ORDER, sc.DEFAULT_GPIO))
    assert [j.gpio for j in sc.nominal(TASK).joints] == list(sc.DEFAULT_GPIO)


def test_positive_angle_moves_every_foot_backward():
    """手順書の「＋で足先が後ろへ」は MJCF の運動学のとおり（股・膝とも）。膝の＋は脚を伸ばす。"""

    import mujoco

    m = mujoco.MjModel.from_xml_path(str(_need(MODEL)))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for name in ORDER:
        leg = name.split("_")[1]
        foot = m.body(f"c_foot_{leg}").id
        jid = m.joint(name).id
        # 関節軸はワールドの +Y（左右どちらの脚も）
        np.testing.assert_allclose(d.xaxis[jid], [0.0, 1.0, 0.0], atol=1e-9)
        before = d.xpos[foot].copy()
        reach0 = np.linalg.norm(d.xpos[foot] - d.xanchor[m.joint(f"hip_{leg}").id])
        d.qpos[m.jnt_qposadr[jid]] = np.radians(10.0)
        mujoco.mj_forward(m, d)
        moved = d.xpos[foot] - before
        assert moved[0] < -0.005, (name, moved)
        assert sc.POSITIVE_MOTION[name.split("_")[0]] == "足先が後ろへ"
        if name.startswith("knee"):
            reach1 = np.linalg.norm(d.xpos[foot] - d.xanchor[m.joint(f"hip_{leg}").id])
            assert reach1 > reach0, name  # 膝の＋は伸ばす向き
        d.qpos[:] = m.qpos0
        mujoco.mj_forward(m, d)


def test_nominal_sign_mirrors_left_and_right():
    """股・膝のサーボは出力軸が外向き（左は +Y、右は −Y）なので、同じ向きに回すには左右で符号が逆。"""

    cal = sc.nominal(TASK)
    signs = {j.name: j.sign for j in cal.joints}
    for n in ORDER:
        expected = 1.0 if n.endswith(("lf", "lh")) else -1.0
        assert signs[n] == expected * sc.PULSE_UP_TURNS_CCW
    assert sc.PULSE_UP_TURNS_CCW in (1.0, -1.0)


# ---------------------------------------------------------------------------
# 名目値・換算
# ---------------------------------------------------------------------------


def test_nominal_values():
    cal = sc.nominal(TASK)
    assert cal.source == "nominal"
    for j in cal.joints:
        assert j.center_us == sc.NOMINAL_CENTER_US == 1450.0
        assert j.us_per_deg == pytest.approx(1900.0 / 180.0)
        assert (j.lo_us, j.hi_us) == (0.0, 0.0)  # 0 = 未測定


def test_angle_and_pulse_conversion_round_trip():
    j = sc.JointCal(name="hip_rf", actuator="hip_rf/position", gpio=6, center_us=1470.0,
                    sign=-1.0, us_per_deg=10.0, lo_us=1920.0, hi_us=1270.0,
                    model_low_deg=-45.0, model_high_deg=20.0)
    assert j.angle_to_us(0.0) == 1470.0
    assert j.angle_to_us(20.0) == 1270.0
    assert j.us_to_angle(1920.0) == pytest.approx(-45.0)
    assert j.measured_range_deg() == pytest.approx((-45.0, 20.0))


# ---------------------------------------------------------------------------
# ファームウェアの dump の読み取り
# ---------------------------------------------------------------------------


def _dump(rows=None, saved=1) -> str:
    rows = rows or {}
    lines = ["I (312) q8: 起動", "> dump", f"CAL begin v=1 n=8 saved={saved}"]
    for k, n in enumerate(ORDER):
        r = {"center": 1450.0, "sign": 1 if n.endswith(("lf", "lh")) else -1, "upd": 10.5556,
             "lo": 0.0, "hi": 0.0, **rows.get(n, {})}
        lines.append(f"CAL {k} {n} gpio={sc.DEFAULT_GPIO[k]} center={r['center']:.1f} "
                     f"sign={r['sign']} upd={r['upd']:.4f} lo={r['lo']:.1f} hi={r['hi']:.1f}")
    lines.append("CAL end")
    return "\n".join(lines) + "\n"


def _measured_rows():
    """学習の範囲をすべて少し越えて測れた、という較正。"""

    rows = {}
    for j in sc.task_joints(TASK):
        sign = 1 if j.name.endswith(("lf", "lh")) else -1
        center = 1440.0 + 5.0 * ORDER.index(j.name)
        upd = 10.0
        rows[j.name] = {"center": center, "sign": sign, "upd": upd,
                        "lo": center + sign * (j.model_low_deg - 2.0) * upd,
                        "hi": center + sign * (j.model_high_deg + 2.0) * upd}
    return rows


def test_parse_dump_reads_every_joint():
    rows = _measured_rows()
    cal = sc.parse_dump(_dump(rows), TASK)
    assert cal.source == "measured"
    assert cal.saved is True
    assert [j.name for j in cal.joints] == list(ORDER)
    j = cal.joints[2]
    assert j.name == "hip_rf" and j.sign == -1.0 and j.gpio == 6
    assert j.center_us == rows["hip_rf"]["center"]
    assert j.lo_us == pytest.approx(rows["hip_rf"]["lo"], abs=0.05)
    assert (j.model_low_deg, j.model_high_deg) == (-45.0, 20.0)


def test_parse_dump_rejects_missing_or_reordered_joints():
    text = _dump()
    with pytest.raises(ValueError, match="CAL end"):
        sc.parse_dump(text.replace("CAL end\n", ""), TASK)
    swapped = text.replace(" hip_lf ", " tmp ").replace(" knee_lf ", " hip_lf ").replace(" tmp ", " knee_lf ")
    with pytest.raises(ValueError, match="順"):
        sc.parse_dump(swapped, TASK)


def test_parse_dump_uses_the_last_block():
    text = _dump() + _dump(_measured_rows())
    cal = sc.parse_dump(text, TASK)
    assert cal.joints[0].lo_us != 0.0


def test_json_round_trip(tmp_path):
    cal = sc.parse_dump(_dump(_measured_rows()), TASK)
    path = tmp_path / "cal.json"
    sc.save_json(cal, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema"] == sc.SCHEMA
    assert [j["name"] for j in data["joints"]] == list(ORDER)
    assert sc.load_json(path) == cal


# ---------------------------------------------------------------------------
# 検査
# ---------------------------------------------------------------------------


def test_check_passes_a_full_calibration():
    errors, warnings = sc.check(sc.parse_dump(_dump(_measured_rows()), TASK), TASK)
    assert errors == [] and warnings == []


def test_check_reports_unmeasured_ranges():
    errors, _ = sc.check(sc.nominal(TASK), TASK)
    assert any("hip_lf" in e and "未測定" in e for e in errors)


def test_check_reports_a_range_that_does_not_cover_the_policy():
    rows = _measured_rows()
    r = rows["knee_rh"]
    r["hi"] = r["center"] + r["sign"] * 30.0 * r["upd"]  # 学習は +45° まで使う
    errors, _ = sc.check(sc.parse_dump(_dump(rows), TASK), TASK)
    assert any("knee_rh" in e and "45" in e for e in errors)


def test_check_reports_ends_that_disagree_with_the_sign():
    rows = _measured_rows()
    rows["hip_lh"]["sign"] = -rows["hip_lh"]["sign"]
    errors, _ = sc.check(sc.parse_dump(_dump(rows), TASK), TASK)
    assert any("hip_lh" in e and "向き" in e for e in errors)


def test_check_warns_about_a_far_center_and_odd_scale():
    rows = _measured_rows()
    rows["hip_lf"]["center"] = 1700.0
    rows["hip_lf"]["lo"] = 1700.0 + (-47.0) * 10.0
    rows["hip_lf"]["hi"] = 1700.0 + 22.0 * 10.0
    _, warnings = sc.check(sc.parse_dump(_dump(rows), TASK), TASK)
    assert any("hip_lf" in w and "ホーン" in w for w in warnings)


def test_check_reports_a_wrong_order():
    cal = sc.parse_dump(_dump(_measured_rows()), TASK)
    cal.joints[0], cal.joints[1] = cal.joints[1], cal.joints[0]
    errors, _ = sc.check(cal, TASK)
    assert any("順" in e for e in errors)


# ---------------------------------------------------------------------------
# 出力: servo.json（export_c --servo）と servo_map.h
# ---------------------------------------------------------------------------


def test_servo_json_feeds_export_c(tmp_path):
    cal = sc.parse_dump(_dump(_measured_rows()), TASK)
    path = tmp_path / "servo.json"
    path.write_text(json.dumps(sc.servo_json(cal)), encoding="utf-8")
    policy = cx.parse_policy(cx.make_synthetic_policy(hidden=(8,), seed=3))
    # 合成ポリシーの actuator 名は関節名そのもの、実物は "hip_lf/position"。どちらにも当たる
    servos = ex.load_servos(path, policy)
    for s, j in zip(servos, cal.joints):
        assert s.center_us == j.center_us and s.sign == j.sign and s.us_per_deg == j.us_per_deg
        assert (s.min_us, s.max_us) == (min(j.lo_us, j.hi_us), max(j.lo_us, j.hi_us))
        assert s.trim_deg == 0.0
        assert ex.servo_pulse_us(0.0, s) == j.center_us


def test_servo_json_matches_the_real_policy_actuators(tmp_path):
    real = ROOT / "artifacts" / "training" / "full-mini-r2-p7-eff-s3" / "walk.cxpolicy"
    policy = cx.load_policy(_need(real))
    cal = sc.parse_dump(_dump(_measured_rows()), TASK)
    path = tmp_path / "servo.json"
    path.write_text(json.dumps(sc.servo_json(cal)), encoding="utf-8")
    servos = ex.load_servos(path, policy)
    assert [s.center_us for s in servos] == [j.center_us for j in cal.joints]


def test_load_servos_accepts_joint_names_and_rejects_unknown_keys(tmp_path):
    policy = cx.load_policy(_need(ROOT / "artifacts" / "training" / "full-mini-r2-p7-eff-s3"
                                  / "walk.cxpolicy"))
    path = tmp_path / "servo.json"
    path.write_text(json.dumps({"joints": {"hip_lf": {"trim_deg": 3.0, "sign": -1}}}),
                    encoding="utf-8")
    servos = ex.load_servos(path, policy)
    assert (servos[0].trim_deg, servos[0].sign) == (3.0, -1)
    path.write_text(json.dumps({"joints": {"hip_xx": {"sign": -1}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="hip_xx"):
        ex.load_servos(path, policy)


def test_servo_json_refuses_a_calibration_with_errors():
    with pytest.raises(ValueError, match="未測定"):
        sc.servo_json(sc.nominal(TASK), task=TASK)


def test_servo_map_header_lists_joints_in_policy_order():
    cal = sc.parse_dump(_dump(_measured_rows()), TASK)
    text = sc.render_servo_map_h(cal)
    assert "#define SERVO_MAP_N 8" in text
    assert "#define SERVO_MAP_CALIBRATED 1" in text
    names = re.search(r"#define SERVO_MAP_NAMES \{(.*)\}", text).group(1)
    assert tuple(re.findall(r'"(\w+)"', names)) == ORDER
    gpio = re.search(r"#define SERVO_MAP_GPIO \{(.*)\}", text).group(1)
    assert [int(g) for g in gpio.split(",")] == list(sc.DEFAULT_GPIO)
    assert "#define SERVO_MAP_CALIBRATED 0" in sc.render_servo_map_h(sc.nominal(TASK))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_parse_check_and_outputs(tmp_path, capsys):
    log = tmp_path / "monitor.log"
    log.write_text(_dump(_measured_rows()), encoding="utf-8")
    out = tmp_path / "cal.json"
    assert sc.main(["parse", str(log), "--out", str(out)]) == 0
    assert sc.main(["check", str(out)]) == 0
    header = tmp_path / "servo_map.h"
    servo = tmp_path / "servo.json"
    assert sc.main(["header", str(out), "--out", str(header)]) == 0
    assert sc.main(["servo-json", str(out), "--out", str(servo)]) == 0
    assert "SERVO_MAP_CALIBRATED 1" in header.read_text(encoding="utf-8")
    assert set(json.loads(servo.read_text(encoding="utf-8"))["joints"]) == {
        f"{n}/position" for n in ORDER}


def test_cli_check_fails_on_an_unmeasured_calibration(tmp_path, capsys):
    out = tmp_path / "cal.json"
    assert sc.main(["nominal", "--out", str(out)]) == 0
    assert sc.main(["check", str(out)]) == 1
    assert "未測定" in capsys.readouterr().out
    assert sc.main(["servo-json", str(out), "--out", str(tmp_path / "s.json")]) == 2


def test_read_serial_sends_dump_and_returns_the_block(tmp_path):
    """標準ライブラリだけのシリアル読み取りを、擬似端末（pty）の相手で確かめる。"""

    import os
    import pty
    import threading

    master, slave = pty.openpty()
    name = os.ttyname(slave)
    got = bytearray()

    def board():
        while b"dump" not in got:
            got.extend(os.read(master, 1024))
        os.write(master, ("I (10) q8: 起動\r\n" + _dump(_measured_rows())).encode("utf-8"))

    t = threading.Thread(target=board, daemon=True)
    t.start()
    try:
        text = sc.read_serial(name, wait_s=0.2, timeout_s=5.0)
    finally:
        t.join(timeout=5.0)
        os.close(master)
        os.close(slave)
    assert b"dump\n" in bytes(got)
    cal = sc.parse_dump(text, TASK)
    assert cal.joints[7].name == "knee_rh" and cal.joints[7].measured()


def test_cli_creates_the_output_folder(tmp_path):
    out = tmp_path / "calibration" / "mini-r2.json"
    assert sc.main(["nominal", "--out", str(out)]) == 0
    assert out.exists()
