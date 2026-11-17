"""MJCF → URDF 変換（q8/urdf.py）のテスト。

URDF の正しさは、生成した URDF を MuJoCo で読み戻し、元の MJCF と
同じ姿勢で関節のワールド位置・軸、各リンクの重心と慣性、全体の重心が
一致することで確かめる（リンクのフレームを置き直しても、物体そのものは
1 mm も動いてはならない）。
"""

from __future__ import annotations

import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from q8 import urdf as U  # noqa: E402

REAL_MJCF = PROJECT / "robot" / "mjcf" / "base" / "model-model.xml"
REAL_MESHES = PROJECT / "robot" / "meshes" / "base"
TOL = 1e-4  # 0.1 mm

# ---------------------------------------------------------------- 合成MJCF

TOY = """
<mujoco model="toy">
  <compiler angle="radian"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.1"/>
    <body name="c_base">
      <freejoint name="root"/>
      <inertial pos="0 0 0.1" quat="0 0.707107 0 0.707107" mass="0.5" diaginertia="3e-4 2e-4 1e-4"/>
      <geom name="c_base/collision0" type="box" size="0.05 0.03 0.01" pos="0 0 0.1"/>
      <body name="c_plate">
        <inertial pos="0 0 0.12" mass="0.05" diaginertia="1e-6 1e-6 1e-6"/>
      </body>
      <body name="c_leg">
        <inertial pos="0.04 0.02 0.06" quat="0.965926 0 0.258819 0" mass="0.02" diaginertia="2e-6 1.5e-6 0.6e-6"/>
        <joint name="hip" pos="0.04 0 0.09" axis="0 1 0" limited="true" range="-0.5 0.3" damping="0.01"/>
        <geom name="c_leg/collision0" type="capsule" size="0.005 0.02" pos="0.04 0.02 0.06" quat="0.965926 0 0.258819 0"/>
        <body name="c_foot">
          <inertial pos="0.05 0.02 0.03" mass="0.001" diaginertia="1e-8 1e-8 1e-8"/>
          <geom name="c_foot/collision0" type="sphere" size="0.006" pos="0.05 0.02 0.03"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <general name="hip/position" joint="hip" forcelimited="true" forcerange="-0.2 0.2"
             biastype="affine" gainprm="2" biasprm="0 -2 0"/>
  </actuator>
</mujoco>
"""

# 各ボディが回転・平行移動していて、関節軸も Y 以外の場合
ROTATED = """
<mujoco model="rot">
  <compiler angle="radian"/>
  <worldbody>
    <body name="c_base" pos="0.01 -0.02 0.2" quat="0.92388 0 0 0.382683">
      <freejoint/>
      <inertial pos="0.01 0 0" quat="0.9 0.1 0.3 0.2" mass="0.3" diaginertia="2e-4 1.5e-4 1e-4"/>
      <body name="c_arm" pos="0.05 0.01 0" quat="0.965926 0.258819 0 0">
        <joint name="j1" pos="0.01 0 0.005" axis="1 0 0" limited="true" range="-1 1"/>
        <inertial pos="0.03 0.01 -0.01" quat="0.7 0.1 0.7 0.1" mass="0.05" diaginertia="3e-6 2e-6 1.2e-6"/>
        <geom name="c_arm/collision0" type="capsule" fromto="0.01 0 0 0.06 0.01 -0.02" size="0.004"/>
        <body name="c_bracket" pos="0.06 0 0" quat="0.8 0 0.6 0">
          <inertial pos="0.0 0.01 0" mass="0.01" diaginertia="1e-7 2e-7 1.5e-7"/>
          <geom name="c_bracket/collision0" type="box" size="0.01 0.005 0.002" pos="0 0.01 0" euler="0.2 0.1 0.3"/>
          <body name="c_hand" pos="0.02 0 0.01" euler="0.3 -0.2 1.1">
            <joint name="j2" pos="0 0.004 0" axis="0.3 0.4 0.866025"/>
            <inertial pos="0.01 0 0" mass="0.02" diaginertia="2e-7 3e-7 4e-7"/>
            <geom name="c_hand/collision0" type="cylinder" size="0.003 0.01" pos="0.01 0 0"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def _convert(tmp_path: Path, mjcf_text: str, **kw) -> tuple[Path, Path]:
    src = _write(tmp_path, "model.xml", mjcf_text)
    out = tmp_path / "urdf" / "robot.urdf"
    U.write_urdf(src, out, **kw)
    return src, out


def _links(root: ET.Element) -> dict[str, ET.Element]:
    return {e.get("name"): e for e in root.findall("link")}


def _joints(root: ET.Element) -> dict[str, ET.Element]:
    return {e.get("name"): e for e in root.findall("joint")}


def _vec(s: str | None) -> np.ndarray:
    return np.array([float(x) for x in (s or "0 0 0").split()])


def _inertia_tensor(inertial: ET.Element) -> np.ndarray:
    i = inertial.find("inertia").attrib
    f = {k: float(v) for k, v in i.items()}
    return np.array([[f["ixx"], f["ixy"], f["ixz"]],
                     [f["ixy"], f["iyy"], f["iyz"]],
                     [f["ixz"], f["iyz"], f["izz"]]])


def _quat_mat(q) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


# ---------------------------------------------------------------- 比較の道具

def _pose(m: mujoco.MjModel, d: mujoco.MjData, root_body: int):
    """ルートボディ座標で表した、関節・リンク・重心の幾何。"""
    p0 = d.xpos[root_body].copy()
    r0 = d.xmat[root_body].reshape(3, 3).copy()

    def to_root(p):
        return r0.T @ (p - p0)

    joints = {}
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        joints[m.joint(j).name] = (to_root(d.xanchor[j]), r0.T @ d.xaxis[j])
    bodies = {}
    for b in range(1, m.nbody):
        rot = r0.T @ d.ximat[b].reshape(3, 3)
        inertia = rot @ np.diag(m.body_inertia[b]) @ rot.T
        bodies[m.body(b).name] = (m.body_mass[b], to_root(d.xipos[b]), inertia)
    com = to_root(d.subtree_com[0])
    return joints, bodies, com


def _set_hinges(m: mujoco.MjModel, d: mujoco.MjData, q: dict[str, float]) -> None:
    for name, v in q.items():
        d.qpos[m.jnt_qposadr[m.joint(name).id]] = v


def _compare(src: Path, out: Path, q: dict[str, float] | None = None, root_name: str = "base_link",
             use_key: bool = False):
    """元のMJCFと読み戻したURDFを同じ関節角で比べ、最大誤差を返す。"""
    ms = mujoco.MjModel.from_xml_path(str(src))
    ds = mujoco.MjData(ms)
    if use_key and ms.nkey:
        mujoco.mj_resetDataKeyframe(ms, ds, 0)
    mu = mujoco.MjModel.from_xml_path(str(out))
    du = mujoco.MjData(mu)
    hinge = {ms.joint(j).name: float(ds.qpos[ms.jnt_qposadr[j]])
             for j in range(ms.njnt) if ms.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE}
    hinge.update(q or {})
    _set_hinges(ms, ds, hinge)
    _set_hinges(mu, du, hinge)
    mujoco.mj_forward(ms, ds)
    mujoco.mj_forward(mu, du)
    root_src = ms.body_rootid[ms.jnt_bodyid[0]] if ms.njnt else 1
    js, bs, cs = _pose(ms, ds, int(root_src))
    ju, bu, cu = _pose(mu, du, mu.body(root_name).id)
    src_root_name = ms.body(int(root_src)).name
    assert set(js) == set(ju)
    rename = {src_root_name: root_name}
    assert {rename.get(k, k) for k in bs} == set(bu)
    err = {"joint_pos": 0.0, "joint_axis": 0.0, "body_com": 0.0, "body_mass": 0.0, "body_inertia": 0.0}
    for name, (p, a) in js.items():
        err["joint_pos"] = max(err["joint_pos"], np.linalg.norm(p - ju[name][0]))
        err["joint_axis"] = max(err["joint_axis"], np.linalg.norm(a - ju[name][1]))
    for name, (mass, c, inertia) in bs.items():
        mu_, cu_, iu = bu[rename.get(name, name)]
        err["body_mass"] = max(err["body_mass"], abs(mass - mu_))
        err["body_com"] = max(err["body_com"], np.linalg.norm(c - cu_))
        scale = max(np.abs(inertia).max(), 1e-30)
        err["body_inertia"] = max(err["body_inertia"], np.abs(inertia - iu).max() / scale)
    err["com"] = float(np.linalg.norm(cs - cu))
    err["mass_total"] = abs(ms.body_mass.sum() - mu.body_mass.sum())
    return err


# ---------------------------------------------------------------- 合成MJCFでの単体テスト

def test_toy_structure(tmp_path):
    _, out = _convert(tmp_path, TOY, velocity=10.0)
    root = ET.parse(out).getroot()
    links, joints = _links(root), _joints(root)
    assert set(links) == {"base_link", "c_plate", "c_leg", "c_foot"}
    kinds = sorted(j.get("type") for j in joints.values())
    assert kinds == ["fixed", "fixed", "revolute"]
    hip = joints["hip"]
    assert hip.find("parent").get("link") == "base_link"
    assert hip.find("child").get("link") == "c_leg"
    # 関節原点 = 関節の pos（親 base_link のフレームはワールド原点）
    np.testing.assert_allclose(_vec(hip.find("origin").get("xyz")), [0.04, 0, 0.09], atol=1e-12)
    np.testing.assert_allclose(_vec(hip.find("axis").get("xyz")), [0, 1, 0], atol=1e-12)
    lim = hip.find("limit")
    assert float(lim.get("lower")) == pytest.approx(-0.5)
    assert float(lim.get("upper")) == pytest.approx(0.3)
    assert float(lim.get("effort")) == pytest.approx(0.2)
    assert float(lim.get("velocity")) == pytest.approx(10.0)
    assert float(hip.find("dynamics").get("damping")) == pytest.approx(0.01)
    # 子リンクの中身は関節位置の分だけずれる
    leg_inertial = links["c_leg"].find("inertial")
    np.testing.assert_allclose(_vec(leg_inertial.find("origin").get("xyz")), [0, 0.02, -0.03], atol=1e-9)
    # 足は固定関節で c_leg に付き、c_leg のフレーム（関節位置）から見た元のボディ原点に置かれる
    foot_j = next(j for j in joints.values() if j.find("child").get("link") == "c_foot")
    assert foot_j.get("type") == "fixed"
    np.testing.assert_allclose(_vec(foot_j.find("origin").get("xyz")), [-0.04, 0, -0.09], atol=1e-12)


def test_toy_inertia_is_rotated_principal_inertia(tmp_path):
    _, out = _convert(tmp_path, TOY)
    links = _links(ET.parse(out).getroot())
    tensor = _inertia_tensor(links["c_leg"].find("inertial"))
    r = _quat_mat(np.array([0.965926, 0, 0.258819, 0]) / np.linalg.norm([0.965926, 0, 0.258819, 0]))
    expected = r @ np.diag([2e-6, 1.5e-6, 0.6e-6]) @ r.T
    np.testing.assert_allclose(tensor, expected, rtol=0, atol=1e-6 * 2e-6)
    assert float(links["c_leg"].find("inertial/mass").get("value")) == pytest.approx(0.02)


def test_toy_capsule_becomes_cylinder_and_two_spheres(tmp_path):
    _, out = _convert(tmp_path, TOY)
    leg = _links(ET.parse(out).getroot())["c_leg"]
    cols = leg.findall("collision")
    shapes = [c.find("geometry")[0] for c in cols]
    assert [s.tag for s in shapes] == ["cylinder", "sphere", "sphere"]
    assert float(shapes[0].get("radius")) == pytest.approx(0.005)
    assert float(shapes[0].get("length")) == pytest.approx(0.04)
    assert all(float(s.get("radius")) == pytest.approx(0.005) for s in shapes[1:])
    # 球はカプセルの軸の両端（リンク座標）
    centre = np.array([0.04, 0.02, 0.06]) - np.array([0.04, 0, 0.09])
    axis = _quat_mat(np.array([0.965926, 0, 0.258819, 0]) / np.linalg.norm([0.965926, 0, 0.258819, 0]))[:, 2]
    ends = sorted([_vec(c.find("origin").get("xyz")) for c in cols[1:]], key=lambda v: v @ axis)
    np.testing.assert_allclose(ends[0], centre - 0.02 * axis, atol=1e-6)
    np.testing.assert_allclose(ends[1], centre + 0.02 * axis, atol=1e-6)
    # 箱は半幅→全幅、球はそのまま
    base = _links(ET.parse(out).getroot())["base_link"]
    box = base.find("collision/geometry/box")
    np.testing.assert_allclose(_vec(box.get("size")), [0.1, 0.06, 0.02])


def test_toy_has_no_floor_and_single_root(tmp_path):
    _, out = _convert(tmp_path, TOY)
    root = ET.parse(out).getroot()
    names = [c.get("name") for c in root.iter("collision")]
    assert "floor" not in names
    children = {j.find("child").get("link") for j in root.findall("joint")}
    roots = set(_links(root)) - children
    assert roots == {"base_link"}


def test_toy_visual_mesh_is_relative_and_scaled(tmp_path):
    meshes = tmp_path / "meshes" / "toy"
    meshes.mkdir(parents=True)
    for stem in ("base", "leg"):
        (meshes / f"{stem}.stl").write_bytes(_tetra_stl())
    _, out = _convert(tmp_path, TOY, mesh_dir=meshes)
    links = _links(ET.parse(out).getroot())
    vis = links["c_leg"].find("visual")
    mesh = vis.find("geometry/mesh")
    assert mesh.get("filename") == "../meshes/toy/leg.stl"
    assert mesh.get("scale") == "0.001 0.001 0.001"
    # STL はボディ座標（=ワールド）で書かれているので、関節位置の分だけ戻す
    np.testing.assert_allclose(_vec(vis.find("origin").get("xyz")), [-0.04, 0, -0.09], atol=1e-12)
    assert links["c_plate"].find("visual") is None  # STL が無い部品は見た目なし
    m = mujoco.MjModel.from_xml_path(str(out))
    assert m.nmesh == 2


def test_toy_roundtrip_in_mujoco(tmp_path):
    src, out = _convert(tmp_path, TOY)
    for q in ({}, {"hip": 0.25}, {"hip": -0.4}):
        err = _compare(src, out, q)
        assert max(err.values()) < 1e-7, (q, err)


def test_rotated_frames_roundtrip_in_mujoco(tmp_path):
    src, out = _convert(tmp_path, ROTATED)
    root = ET.parse(out).getroot()
    assert sorted(j.get("type") for j in root.findall("joint")) == ["continuous", "fixed", "revolute"]
    for q in ({}, {"j1": 0.7, "j2": -1.3}, {"j1": -0.9, "j2": 2.5}):
        err = _compare(src, out, q)
        assert max(err.values()) < 1e-7, (q, err)


def test_rpy_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        r = _quat_mat(q)
        np.testing.assert_allclose(U.rpy_to_matrix(U.matrix_to_rpy(r)), r, atol=1e-12)
    # ジンバルロック（pitch = ±90°）
    for pitch in (math.pi / 2, -math.pi / 2):
        r = U.rpy_to_matrix(np.array([0.3, pitch, -0.2]))
        np.testing.assert_allclose(U.rpy_to_matrix(U.matrix_to_rpy(r)), r, atol=1e-12)


def test_unsupported_joint_is_rejected(tmp_path):
    text = """<mujoco><worldbody><body name="a"><freejoint/><geom size="0.1"/>
      <body name="b"><joint name="ball" type="ball"/><geom size="0.1"/></body>
    </body></worldbody></mujoco>"""
    with pytest.raises(NotImplementedError):
        _convert(tmp_path, text)


# ---------------------------------------------------------------- 実機のMJCF

needs_real = pytest.mark.skipif(not REAL_MJCF.exists(), reason="robot/mjcf/base/model-model.xml がない")


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    out = tmp_path_factory.mktemp("urdf") / "urdf" / "base" / "quadruped8.urdf"
    U.write_urdf(REAL_MJCF, out, mesh_dir=REAL_MESHES if REAL_MESHES.exists() else None,
                 velocity=U.MG90S_VELOCITY, robot_name="quadruped8")
    return out


@needs_real
def test_real_counts(real):
    m = mujoco.MjModel.from_xml_path(str(REAL_MJCF))
    root = ET.parse(real).getroot()
    nbody = m.nbody - 1
    jointless = sum(1 for b in range(1, m.nbody) if m.body_jntnum[b] == 0)
    joints = root.findall("joint")
    assert len(root.findall("link")) == nbody
    assert sum(j.get("type") == "revolute" for j in joints) == 8
    assert sum(j.get("type") == "fixed" for j in joints) == jointless
    assert len(joints) == nbody - 1


@needs_real
def test_real_total_mass(real):
    m = mujoco.MjModel.from_xml_path(str(REAL_MJCF))
    root = ET.parse(real).getroot()
    total = sum(float(e.get("value")) for e in root.iter("mass"))
    assert total == pytest.approx(m.body_mass.sum(), rel=1e-9, abs=1e-12)


@needs_real
def test_real_limits_match_mjcf(real):
    m = mujoco.MjModel.from_xml_path(str(REAL_MJCF))
    joints = _joints(ET.parse(real).getroot())
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        e = joints[m.joint(j).name]
        assert e.get("type") == "revolute"
        lim = e.find("limit")
        assert float(lim.get("lower")) == pytest.approx(m.jnt_range[j][0], abs=1e-9)
        assert float(lim.get("upper")) == pytest.approx(m.jnt_range[j][1], abs=1e-9)
        act = [a for a in range(m.nu) if m.actuator_trnid[a][0] == j]
        assert len(act) == 1
        assert float(lim.get("effort")) == pytest.approx(np.abs(m.actuator_forcerange[act[0]]).max(), abs=1e-9)
        assert float(lim.get("effort")) > 0
        # MG90S: 4.8 V で 0.10 s/60°
        assert float(lim.get("velocity")) == pytest.approx(math.radians(60) / 0.10, rel=1e-6)
        np.testing.assert_allclose(_vec(e.find("axis").get("xyz")), [0, 1, 0], atol=1e-12)


@needs_real
def test_real_inertia_symmetric_positive_definite(real):
    root = ET.parse(real).getroot()
    inertials = list(root.iter("inertial"))
    assert len(inertials) == mujoco.MjModel.from_xml_path(str(REAL_MJCF)).nbody - 1
    for e in inertials:
        t = _inertia_tensor(e)
        np.testing.assert_array_equal(t, t.T)
        ev = np.linalg.eigvalsh(t)
        assert ev.min() > 0, ev
        # 主慣性の三角不等式（物理的にありうる慣性）
        a, b, c = sorted(ev)
        assert a + b >= c * (1 - 1e-6)


@needs_real
def test_real_roundtrip_standing_pose(real):
    err = _compare(REAL_MJCF, real, use_key=True)
    print("立位での最大誤差:", err)
    assert err["joint_pos"] < TOL
    assert err["com"] < TOL
    assert err["body_com"] < TOL
    assert err["joint_axis"] < 1e-9
    assert err["body_mass"] < 1e-12
    assert err["body_inertia"] < 1e-6


@needs_real
def test_real_roundtrip_bent_pose(real):
    m = mujoco.MjModel.from_xml_path(str(REAL_MJCF))
    rng = np.random.default_rng(1)
    for _ in range(3):
        q = {m.joint(j).name: float(rng.uniform(*m.jnt_range[j]))
             for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE}
        err = _compare(REAL_MJCF, real, q)
        assert err["joint_pos"] < TOL and err["com"] < TOL and err["body_com"] < TOL, (q, err)


@needs_real
@pytest.mark.skipif(not (REAL_MESHES / "chassis.stl").exists(), reason="STL 未生成（scripts/build_robot.py）")
def test_real_visual_meshes_load(real):
    root = ET.parse(real).getroot()
    meshes = [e for e in root.iter("mesh")]
    assert len(meshes) == mujoco.MjModel.from_xml_path(str(REAL_MJCF)).nbody - 1
    # 相対パスで、URDF の場所から実在する STL を指す
    assert all(not Path(e.get("filename")).is_absolute() for e in meshes)
    assert all((real.parent / e.get("filename")).resolve().parent == REAL_MESHES.resolve() for e in meshes)
    names = {Path(e.get("filename")).name for e in meshes}
    assert {"esp32.stl", "battery.stl", "chassis.stl"} <= names
    m = mujoco.MjModel.from_xml_path(str(real))
    assert m.nmesh == len(meshes)
    # mm → m の縮尺が効いている（機体は 30 cm に収まる）
    assert np.abs(m.mesh_vert).max() < 0.3


def test_cli_writes_urdf(tmp_path):
    if not REAL_MJCF.exists():
        pytest.skip("robot/mjcf/base/model-model.xml がない")
    out = tmp_path / "q.urdf"
    r = subprocess.run([sys.executable, str(PROJECT / "scripts" / "mjcf_to_urdf.py"), "--variant", "base",
                        "--out", str(out)], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    root = ET.parse(out).getroot()
    assert root.tag == "robot" and root.get("name") == "quadruped8"
    mujoco.MjModel.from_xml_path(str(out))


def _tetra_stl() -> bytes:
    """MuJoCo が読める最小のバイナリSTL（四面体、mm）。"""
    import struct
    v = [(0, 0, 0), (10, 0, 0), (0, 10, 0), (0, 0, 10)]
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    data = bytearray(b"\0" * 80) + struct.pack("<I", len(faces))
    for f in faces:
        data += struct.pack("<3f", 0, 0, 0)
        for i in f:
            data += struct.pack("<3f", *v[i])
        data += b"\0\0"
    return bytes(data)
