"""3本足の脚を横へ開く関節（ロール、leg_roll）: tri3-r8（前の2本）と tri3-r9（3本とも）。

ロールのサーボは胴体に付き、軸は前後（ワールドの x）。その出力の台（c_rollbr_<脚>）が股のサーボを持ち、
連なりは 胴体 →roll→ 台 →hip→ 上腿 →knee→ 下腿。正の角度は脚を外へ開く向き。

前半は偽物の API（tests/test_design_battery.py の run_design）で宣言だけを見る。後半は実ビルドの出力
（robot/mjcf/tri3-r8 など。無ければ確かめ用の別名 tri3-r8-chk）を MuJoCo で読み、無ければ飛ばす。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from test_design_battery import run_design
from test_design_tail import DATA, MINI_R2_P7_EFF, actuators, connector_z_axis, joint, of_kind, signature

ROOT = Path(__file__).resolve().parents[1]
TRI_R2 = dict(MINI_R2_P7_EFF, rear_legs=1.0, gait_mode=0.0)
TRI_R2_WAVE = dict(MINI_R2_P7_EFF, rear_legs=1.0, gait_mode=1.0)
TRI_R2_DYN = dict(TRI_R2_WAVE, hc_foot_center=1.0, vel_source=1.0, wave_swing=0.8)
TRI3 = dict(TRI_R2_WAVE, hc_foot_center=1.0, vel_source=1.0)
TRI3_R8 = dict(TRI3, leg_roll=1.0)
TRI3_R9 = dict(TRI3, leg_roll=2.0)
NEW_DEFAULTS = dict(leg_roll=0.0, roll_min=-20.0, roll_max=20.0)
ORDER_R8 = ["roll_lf", "hip_lf", "knee_lf", "roll_rf", "hip_rf", "knee_rf", "hip_hc", "knee_hc"]
ORDER_R9 = ["roll_lf", "hip_lf", "knee_lf", "roll_rf", "hip_rf", "knee_rf", "roll_hc", "hip_hc", "knee_hc"]
W2 = 31.0   # mini-r2-p7-eff の body_width / 2


# --- 偽物の API ---

def test_leg_roll_rejects_bad_values():
    with pytest.raises(ValueError, match="leg_roll"):
        run_design(**TRI3, leg_roll=3.0)
    with pytest.raises(ValueError, match="leg_roll"):
        run_design(**dict(MINI_R2_P7_EFF, rear_legs=2.0), leg_roll=1.0)
    with pytest.raises(ValueError, match="leg_roll"):
        run_design(**TRI3, leg_roll=0.5)
    with pytest.raises(ValueError, match="roll"):
        run_design(**TRI3_R8, roll_min=5.0)


@pytest.mark.parametrize("name, overrides", [
    ("default", {}), ("mini-r2-p7-eff", MINI_R2_P7_EFF), ("tri-r2", TRI_R2), ("tri-r2-wave", TRI_R2_WAVE),
    ("tri-r2-dyn", TRI_R2_DYN)])
def test_default_unchanged(name, overrides):
    baseline = json.loads((DATA / f"design_log_{name}.json").read_text())
    _, log = run_design(**overrides)
    assert signature(log) == baseline
    _, log0 = run_design(**overrides, **NEW_DEFAULTS)
    assert signature(log0) == baseline


def test_joint_order_r8():
    _, log = run_design(**TRI3_R8)
    assert actuators(log) == [j.split("_")[0] + "_drive_" + j.split("_")[1] for j in ORDER_R8]


def test_joint_order_r9():
    _, log = run_design(**TRI3_R9)
    assert actuators(log) == [j.split("_")[0] + "_drive_" + j.split("_")[1] for j in ORDER_R9]


def _roll_legs(log):
    return [t.label[len("roll_"):] for t in of_kind(log, "assembly.joint")
            if t.args[0] == "revolute" and t.label.startswith("roll_")]


@pytest.mark.parametrize("params, legs", [(TRI3_R8, ["lf", "rf"]), (TRI3_R9, ["lf", "rf", "hc"])])
def test_roll_axis_is_forward(params, legs):
    result, log = run_design(**params)
    assert _roll_legs(log) == legs
    for leg in legs:
        j = joint(log, "roll_" + leg)
        assert j.args[0] == "revolute"
        a, b = j.args[1], j.args[2]
        assert a.args[0] is result["c_chassis"] and b.args[0] is result["c_rollbr_" + leg]
        za, zb = connector_z_axis(a), connector_z_axis(b)
        assert za == zb
        assert abs(za[0]) >= 0.99
        # 外へ開くのが正: 左（足先 y > 0）は +x まわり、右と hc（足先 y < 0）は -x まわり
        assert za[0] == pytest.approx(1.0 if leg == "lf" else -1.0)


@pytest.mark.parametrize("params, legs", [(TRI3_R8, ["lf", "rf"]), (TRI3_R9, ["lf", "rf", "hc"])])
def test_roll_chain_and_parts(params, legs):
    result, log = run_design(**params)
    for leg in legs:
        # 股の関節は台から上腿へ、股のサーボは台に溶接、ロールのサーボは胴体に溶接
        hip = joint(log, "hip_" + leg)
        assert hip.args[1].args[0] is result["c_rollbr_" + leg]
        assert result["weld_hip_servo_" + leg].args[1].args[0] is result["c_rollbr_" + leg]
        assert result["weld_roll_servo_" + leg].args[1].args[0] is result["c_chassis"]
        assert result["weld_roll_horn_" + leg].args[1].args[0] is result["c_rollbr_" + leg]
        # 台には箱の衝突形状（板とハウジングの2つ）
        (body,) = [t for t in of_kind(log, "assembly.body") if t.args[0] is result["c_rollbr_" + leg]]
        assert [cl.args[0] for cl in body.kwargs["collision"]] == ["box", "box"]
    for leg in {"lf", "rf", "hc"} - set(legs):
        assert "c_rollbr_" + leg not in result
        assert result["weld_hip_servo_" + leg].args[1].args[0] is result["c_chassis"]
    # サーボの数（BOM の数）: 脚 2 × 3 + ロール
    servos = [t.label for t in of_kind(log, "lib.servo") if t.label]
    assert len(servos) == 6 + len(legs)
    assert sorted(s for s in servos if s.startswith("roll_servo_")) == sorted("roll_servo_" + g for g in legs)
    # 観測・ランダム化・報酬の平均も関節の数ぶん
    obs = {t.kwargs["name"] for t in of_kind(log, "assembly.observation")}
    for leg in legs:
        assert {"q_roll_" + leg, "c_roll_" + leg, "tau_roll_" + leg} <= obs
    rand = [(t.args[0].label, t.args[1]) for t in of_kind(log, "assembly.randomise")]
    for leg in legs:
        assert ("roll_" + leg, "damping") in rand and ("roll_" + leg, "friction_loss") in rand
    eff = [t for t in of_kind(log, "assembly.reward") if t.kwargs["label"] == "effort"][0].args[0]
    assert eff.endswith(") / %d" % (6 + len(legs)))


@pytest.mark.parametrize("params, legs", [(TRI3_R8, ["lf", "rf"]), (TRI3_R9, ["lf", "rf", "hc"])])
def test_roll_limits(params, legs):
    _, log = run_design(**params)
    acts = {t.kwargs["label"]: t for t in of_kind(log, "assembly.actuator")}
    for leg in legs:
        assert joint(log, "roll_" + leg).kwargs["angle_limits_degrees"] == [-20.0, 20.0]
        assert acts["roll_drive_" + leg].kwargs["command_limits_degrees"] == [-20.0, 20.0]
    _, log = run_design(**params, roll_min=-30.0, roll_max=25.0)
    for leg in legs:
        assert joint(log, "roll_" + leg).kwargs["angle_limits_degrees"] == [-30.0, 25.0]


def _box(c):
    pos, size = c.kwargs["offset"]["position"], c.kwargs["size_mm"]
    return [(p - s / 2.0, p + s / 2.0) for p, s in zip(pos, size)]


def _overlap(a, b):
    """2つの軸に平行な箱の重なりの厚み（重なった箱の、いちばん薄い向きの寸法。離れていれば負）。"""
    return min(min(a1, b1) - max(a0, b0) for (a0, a1), (b0, b1) in zip(a, b))


@pytest.mark.parametrize("params", [TRI3_R8, TRI3_R9])
def test_roll_mount_overlap_declared(params):
    # 前の脚のロールのサーボは胴体の側面の外の台に入る。台は胴体の箱に 4 mm 以上重なる
    _, log = run_design(**params)
    coll = {t.label: t for t in of_kind(log, "assembly.collision")}
    body = _box(coll["body_box"])
    for leg in ("lf", "rf"):
        assert _overlap(_box(coll["roll_mount_box_" + leg]), body) >= 4.0
    assert "roll_mount_box_hc" not in coll


@pytest.mark.parametrize("params", [TRI3_R8, TRI3_R9])
def test_roll_の箱の寸法はすべて正(params):
    # 実ビルドで、hc のロールのサーボを囲む塊の前の端が四足の空洞の中に入り、長さが負の箱で落ちた
    _, log = run_design(**params)
    bad = [(t.args, t.kwargs.get("origin")) for t in of_kind(log, "part.box") if min(t.args[:3]) <= 0.0]
    assert not bad, bad


def test_r9_の_hc_のロールのサーボは胴体の中():
    # hc のロールのサーボは胴体の後ろの中（ケース上面が後ろの面、軸は後ろ向き）。外に台を足さない
    _, log = run_design(**TRI3_R9)
    (sv,) = [t for t in of_kind(log, "lib.servo") if t.label == "roll_servo_hc"]
    x, y, z = sv.origin
    assert x == pytest.approx(-96.0 / 2.0)
    assert abs(y) < W2 - 2.0 - 12.2 / 2.0
    coll = {t.label for t in of_kind(log, "assembly.collision")}
    assert "hc_mount_box" not in coll


def test_r8_の前脚の足先は_W2_を変えずに今の幅以下():
    _, log0 = run_design(**TRI_R2_DYN)
    _, log = run_design(**TRI3_R8)
    feet0 = {t.label: t.kwargs["center"] for t in of_kind(log0, "part.sphere")}
    feet = {t.label: t.kwargs["center"] for t in of_kind(log, "part.sphere")}
    for leg, s in (("lf", 1.0), ("rf", -1.0)):
        y0, y = s * feet0["foot_" + leg][1], s * feet["foot_" + leg][1]
        assert W2 < y <= y0 + 0.01
        assert feet["foot_" + leg][0] == pytest.approx(feet0["foot_" + leg][0])
        assert feet["foot_" + leg][2] == pytest.approx(feet0["foot_" + leg][2])
    (hull,) = [t for t in of_kind(log, "part.box") if t.args[:2] == (96.0, 62.0)]
    assert hull.args[1] == 2.0 * W2


# --- 実ビルド（エンジンで作った出力。無ければ飛ばす） ---

def _built(variant):
    for name in (variant, variant + "-chk"):
        xml = ROOT / "robot" / "mjcf" / name / "model-model.xml"
        if xml.exists():
            return name, xml
    pytest.skip(f"robot/mjcf/{variant} の MJCF が無い（scripts/build_robot.py --variant {variant} で作る）")


def _model(variant):
    mujoco = pytest.importorskip("mujoco")
    name, xml = _built(variant)
    return name, mujoco, mujoco.MjModel.from_xml_path(str(xml))


@pytest.mark.parametrize("variant, order", [("tri3-r8", ORDER_R8), ("tri3-r9", ORDER_R9)])
def test_built_joint_order(variant, order):
    from q8.robot import model_joints
    _, _, model = _model(variant)
    assert model_joints(model) == order


@pytest.mark.parametrize("variant, order", [("tri3-r8", ORDER_R8), ("tri3-r9", ORDER_R9)])
def test_built_roll_axis_and_limits(variant, order):
    _, mujoco, model = _model(variant)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    for j in [j for j in order if j.startswith("roll_")]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        assert abs(float(data.xaxis[jid][0])) >= 0.99
        assert [round(math.degrees(v), 3) for v in model.jnt_range[jid]] == [-20.0, 20.0]


@pytest.mark.parametrize("variant, order", [("tri3-r8", ORDER_R8), ("tri3-r9", ORDER_R9)])
def test_roll_positive_opens_outward(variant, order):
    _, mujoco, model = _model(variant)
    data = mujoco.MjData(model)

    def foot_y(leg, deg):
        mujoco.mj_resetDataKeyframe(model, data, 0)
        if deg:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "roll_" + leg)
            data.qpos[model.jnt_qposadr[jid]] += math.radians(deg)
        mujoco.mj_kinematics(model, data)
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"c_foot_{leg}/collision0")
        return float(data.geom_xpos[gid][1]) * 1000.0

    for leg in [j[len("roll_"):] for j in order if j.startswith("roll_")]:
        assert abs(foot_y(leg, 10.0)) - abs(foot_y(leg, 0.0)) >= 3.0, leg
        assert abs(foot_y(leg, -10.0)) < abs(foot_y(leg, 0.0)), leg


@pytest.mark.parametrize("variant", ["tri3-r8", "tri3-r9"])
def test_roll_build_verdicts(variant):
    name, _ = _built(variant)
    report = ROOT / "artifacts" / name / "design_report.json"
    plan = ROOT / "artifacts" / name / "assembly" / "print_plan.json"
    if not report.exists() or not plan.exists():
        pytest.skip(f"artifacts/{name} の design_report.json か assembly/print_plan.json が無い")
    verdict = json.loads(report.read_text())["verdict"]
    assert all(verdict.values()), verdict
    av = json.loads(plan.read_text())["assembly_verdict"]
    assert all(av.values()), av
    # ロールを知らない印刷計画（前脚の股サーボを胴体で確かめる、中身なしの組）では通さない
    assert {"roll_servo_into_chassis", "hip_servo_into_rollbr", "roll_horn_into_rollbr", "roll_lead_exits"} <= set(av)
    assert "hip_servo_into_chassis" not in av
    if variant == "tri3-r9":
        assert {"hc_roll_servo_into_chassis", "hc_hip_servo_into_rollbr", "hc_roll_lead_exits"} <= set(av)


@pytest.mark.parametrize("variant, order", [("tri3-r8", ORDER_R8), ("tri3-r9", ORDER_R9)])
def test_roll_mount_overlap(variant, order):
    # 胴体（c_chassis）の衝突の箱: collision0 が胴体の箱、hc にロールが無ければ collision1 が hc の股サーボの台、
    # そのあとに前のロールのサーボの台が lf, rf の順に並ぶ。台は胴体の箱に 4 mm 以上重なる（tri-r2-dyn の台は 2.25 mm だった）
    _, mujoco, model = _model(variant)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    def box(i):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"c_chassis/collision{i}")
        assert gid >= 0, i
        assert int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_BOX)
        c, h = data.geom_xpos[gid] * 1000.0, model.geom_size[gid] * 1000.0
        return [(float(c[k] - h[k]), float(c[k] + h[k])) for k in range(3)]

    # hc のロールのサーボは胴体の中なので、外の台は前の2本だけ
    first = 1 if "roll_hc" in order else 2
    body = box(0)
    for i, j in enumerate(("roll_lf", "roll_rf")):
        assert _overlap(box(first + i), body) >= 4.0, j


# --- ロールの部品を、干渉の判定・部品表・組立の手順が知っていること ---

def _scripts():
    import sys
    path = str(ROOT / "scripts")
    if path not in sys.path:
        sys.path.insert(0, path)


def test_printed_は_rollbr_を印刷部品にする():
    from q8.manufacturing import is_printed
    assert is_printed("rollbr_lf") and is_printed("rollbr_hc")
    assert not is_printed("roll_servo_lf") and not is_printed("roll_horn_lf")


@pytest.mark.parametrize("variant, legs", [("tri3-r8", ["lf", "rf"]), ("tri3-r9", ["lf", "rf", "hc"])])
def test_intended_contacts_は_roll_のサーボとホーンを含む(variant, legs):
    _scripts()
    from build_robot import intended_contacts
    _, _, model = _model(variant)
    pairs = intended_contacts(model)
    for leg in legs:
        assert frozenset((f"c_roll_servo_{leg}", f"c_roll_horn_{leg}")) in pairs
    for leg in ("lf", "rf", "hc"):
        assert frozenset((f"c_hip_servo_{leg}", f"c_hip_horn_{leg}")) in pairs
    assert len(pairs) == 2 * 3 + len(legs)


def test_intended_contacts_は_roll_の無い機体では変わらない(base_model):
    _scripts()
    from build_robot import intended_contacts
    pairs = intended_contacts(base_model)
    assert pairs == {frozenset((f"c_{j}_servo_{g}", f"c_{j}_horn_{g}")) for j in ("hip", "knee")
                     for g in ("lf", "rf", "lh", "rh")}


def _check_layout(lay):
    from test_assembly_guide import _check_layout as check
    check(lay)


def test_layout_は前のロールの部品と差し込みを並べる():
    _scripts()
    import assembly_guide as ag
    lay = ag.layout(["lf", "rf", "hc"], [], ["lf", "rf"])
    _check_layout(lay)
    assert lay["roll"] == ["lf", "rf"]
    assert {"rollbr_lf", "rollbr_rf"} <= set(lay["printed"]) and "rollbr_hc" not in lay["printed"]
    assert {"roll_servo_lf", "roll_horn_lf", "roll_servo_rf", "roll_horn_rf"} <= set(lay["all"])
    assert "台" in " ".join(p["title"] for p in lay["plates"])
    # 前脚の股サーボはロールの台に入る（胴体には入らない）。胴体に入るのはロールのサーボ
    fits = lay["verdict_fits"]
    assert "hip_servo_into_chassis" not in fits
    assert fits["hip_servo_into_rollbr"] == ("rollbr_lf", "hip_servo_lf")
    assert fits["roll_servo_into_chassis"] == ("chassis", "roll_servo_lf")
    assert fits["cover_closes"] == ("cover", "roll_servo_lf")
    assert fits["roll_horn_into_rollbr"] == ("rollbr_lf", "roll_horn_lf")
    assert lay["verdict_voids"]["hip_lead_exits"] == ("rollbr_lf", "hip_servo_lf")
    assert lay["verdict_voids"]["roll_lead_exits"] == ("chassis+cover", "roll_servo_lf")
    # 中身なしの組（股サーボ lf と胴体）を、どの確かめにも残さない
    pairs = {(c["part"], c["insert"]) for c in lay["fit_checks"]} | {(c["part"], c["insert"]) for c in lay["void_checks"]}
    assert not {p for p in pairs if p[1] == "hip_servo_lf" and p[0] in ("chassis", "cover", "chassis+cover")}
    # hc にロールが無ければ、hc の股サーボは今のとおり胴体の台に入る
    assert fits["hc_hip_servo_into_chassis"] == ("chassis", "hip_servo_hc")


def test_layout_は_hc_のロールも並べる():
    _scripts()
    import assembly_guide as ag
    lay = ag.layout(["lf", "rf", "hc"], [], ["lf", "rf", "hc"])
    _check_layout(lay)
    fits, voids = lay["verdict_fits"], lay["verdict_voids"]
    assert "hc_hip_servo_into_chassis" not in fits
    assert fits["hc_hip_servo_into_rollbr"] == ("rollbr_hc", "hip_servo_hc")
    assert fits["hc_roll_servo_into_chassis"] == ("chassis", "roll_servo_hc")
    assert fits["hc_cover_closes"] == ("cover", "roll_servo_hc")
    assert fits["hc_roll_horn_into_rollbr"] == ("rollbr_hc", "roll_horn_hc")
    assert voids["hc_hip_lead_exits"] == ("rollbr_hc", "hip_servo_hc")
    assert voids["hc_roll_lead_exits"] == ("chassis+cover", "roll_servo_hc")


def test_layout_の既定と3本足は_roll_を渡しても変わらない():
    _scripts()
    import assembly_guide as ag
    assert ag.layout(roll=()) == ag.layout()
    assert ag.layout(["lf", "rf", "hc"], [], []) == ag.layout(["lf", "rf", "hc"])
    assert ag.layout()["roll"] == []


@pytest.mark.parametrize("variant, roll", [("tri3-r8", ["lf", "rf"]), ("tri3-r9", ["lf", "rf", "hc"])])
def test_read_layout_はロールの脚を読む(variant, roll):
    _scripts()
    import assembly_guide as ag
    name, _ = _built(variant)
    assert ag.read_layout(name)["roll"] == roll


def test_print_plan_は中身の無い差し込みの確かめを断る():
    # 股サーボが胴体の外にある機体を、ロールを知らない layout で確かめると、光線が胴体に当たらず「入る」になっていた
    _scripts()
    import assembly_guide as ag
    name, _ = _built("tri3-r8")
    mesh_dir = ROOT / "robot" / "meshes" / name
    with pytest.raises(ValueError, match="hip_servo_lf"):
        ag.fit_checks(mesh_dir, [{"part": "chassis", "insert": "hip_servo_lf", "direction": (0, 0, -1), "label": "x"}])
    with pytest.raises(ValueError, match="hip_servo_lf"):
        ag.enclosed_voids(mesh_dir, [{"part": "chassis+cover", "insert": "hip_servo_lf", "label": "x"}])
