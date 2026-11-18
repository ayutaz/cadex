"""脚と関節の数をモデルから読む（q8.robot.model_*）と、それに従う physics・dr_bundle・書き出しの検査。"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import mujoco
import pytest

from q8 import dr_bundle
from q8.physics import static_report
from q8.robot import JOINTS, LEGS, model_feet, model_joints, model_legs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.esp32 import cxpolicy_io as cx  # noqa: E402
from deploy.esp32 import export_c  # noqa: E402

LEG_X = {"lf": (0.05, 0.03), "rf": (0.05, -0.03), "lh": (-0.05, 0.03), "rh": (-0.05, -0.03),
         "hc": (-0.05, 0.0)}


def _leg(leg: str) -> str:
    x, y = LEG_X[leg]
    return f"""
    <body name="c_thigh_{leg}" pos="{x} {y} 0.05">
      <joint name="hip_{leg}" axis="0 1 0" range="-1 1"/>
      <geom size="0.003" pos="0 0 -0.02"/>
      <body name="c_shin_{leg}" pos="0 0 -0.03">
        <joint name="knee_{leg}" axis="0 1 0" range="-1 1"/>
        <geom size="0.003" pos="0 0 -0.02"/>
        <body name="c_foot_{leg}" pos="0 0 -0.03">
          <geom name="c_foot_{leg}/collision0" type="sphere" size="0.004"/>
        </body>
      </body>
    </body>"""


def _xml(legs: list[str], tail: bool = False, com_x: float = 0.0) -> str:
    bodies = "".join(_leg(leg) for leg in legs)
    tail_xml = ""
    acts = [f'<general name="{k}_{leg}/position" joint="{k}_{leg}" forcerange="-0.2 0.2" '
            f'biastype="affine" gainprm="2" biasprm="0 -2 -0.1"/>' for leg in legs for k in ("hip", "knee")]
    if tail:
        tail_xml = """
    <body name="c_tail" pos="-0.07 0 0.05">
      <joint name="tail_yaw" axis="0 0 1" range="-1 1"/>
      <joint name="tail_pitch" axis="0 1 0" range="-1 1"/>
      <geom size="0.003"/>
    </body>"""
        acts += [f'<general name="{j}_drive" joint="{j}" forcerange="-0.1 0.1" biastype="affine" '
                 f'gainprm="1" biasprm="0 -1 -0.05"/>' for j in ("tail_yaw", "tail_pitch")]
    return f"""<mujoco>
  <worldbody>
    <body name="c_chassis" pos="0 0 0.08">
      <inertial pos="{com_x} 0 0" mass="0.3" diaginertia="1e-4 1e-4 1e-4"/>
      <freejoint/>{bodies}{tail_xml}
    </body>
  </worldbody>
  <actuator>
    {chr(10).join("    " + a for a in acts)}
  </actuator>
</mujoco>"""


def _model(legs, tail=False, com_x=0.0) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(_xml(legs, tail, com_x))


def test_model_legs_は関節名から脚を出す():
    assert model_legs(_model(["lf", "rf", "hc"])) == ["lf", "rf", "hc"]
    m = _model(["lf", "rf", "lh", "rh"], tail=True)
    assert model_legs(m) == ["lf", "rf", "lh", "rh"]
    assert model_joints(m)[-2:] == ["tail_yaw", "tail_pitch"]
    assert len(model_joints(m)) == 10
    assert model_feet(m)[0] == "c_foot_lf"


def test_static_report_は3脚で支持多角形を三角形にする():
    # 重心を前へずらす: 三角形（lf, rf, hc）の外ではなく中なら正、外へ出せば負
    inside = static_report(_model(["lf", "rf", "hc"], com_x=0.0), 0.2, 0.1, settle_s=0.01)
    assert inside["support_margin_mm"] > 0
    outside = static_report(_model(["lf", "rf", "hc"], com_x=-0.09), 0.2, 0.1, settle_s=0.01)
    assert outside["support_margin_mm"] < 0
    assert set(outside["feet_xy_mm"]) == {"lf", "rf", "hc"}
    labels = list(outside["static_estimate"])
    assert any("1脚を上げた最悪" in label for label in labels)


def test_dr_bundle_は8個でないアクチュエータにも遅れを入れる():
    xml = dr_bundle.dr_model_xml(_xml(["lf", "rf", "lh", "rh"], tail=True).replace(
        '<body name="c_chassis" pos="0 0 0.08">', '<body name="c_chassis">'))
    assert xml.count('dyntype="filterexact"') == 10


def test_関節の数が違う方策は書き出しで止まる():
    blob = cx.make_synthetic_policy(joints=[f"hip_x{i}" if i % 2 == 0 else f"knee_x{i}" for i in range(10)])
    policy = cx.parse_policy(blob)
    with pytest.raises(ValueError) as e:
        cx.require_joint_count(policy)
    assert "8" in str(e.value) and "10" in str(e.value)
    with pytest.raises(ValueError, match="10"):
        export_c.render_header(policy, None, layout=[])


def test_今の機体では今と同じ():
    path = ROOT / "robot" / "mjcf" / "mini-r2-p7-eff" / "model-model.xml"
    m = mujoco.MjModel.from_xml_path(str(path))
    assert model_legs(m) == list(LEGS)
    assert model_joints(m) == JOINTS
    report = json.loads((ROOT / "artifacts" / "mini-r2-p7-eff" / "design_report.json").read_text())["physics"]
    now = json.loads(json.dumps(static_report(m, 0.17652, 0.17652 * 0.4)))
    for key in ("total_mass_g", "com_height_mm", "support_margin_mm"):
        assert now[key] == pytest.approx(report[key], abs=0.2), key
    assert now["feet_xy_mm"] == report["feet_xy_mm"]
    assert now["static_estimate"] == report["static_estimate"]
    assert now["sim_hold"]["mean_abs_torque_nmm"] == pytest.approx(report["sim_hold"]["mean_abs_torque_nmm"], abs=0.1)
