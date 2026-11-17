"""Cadexが書いたMJCFから URDF を作る（Cadex には URDF 出力がない）。

MJCFを MuJoCo で一度コンパイルし、そのモデル（ボディの姿勢・慣性・関節・衝突形状）
から URDF を組み立てる。XMLを直接読まないので、既定値・fromto・euler・
ジオメトリから推定された慣性なども MuJoCo と同じ解釈になる。

フレームの置き方:
- URDF のリンク = MJCF のボディ。ワールド直下の唯一のボディがルート（既定名 base_link）。
  free 関節は URDF に入らない（URDF のルートは浮遊ベースとして扱われる前提）。
- 関節を持つボディのリンクフレームは「ボディの向きのまま、原点を関節の pos へ移したもの」。
  URDF では関節の原点が子リンクの原点になるため。関節を持たないボディ（溶接された部品）
  のリンクフレームはボディのフレームそのもので、fixed 関節で親リンクに付く。
- リンク内の inertial / visual / collision の原点は、リンクフレームから見た位置に直す。
  慣性は主慣性（diaginertia と iquat）を回してリンクの軸で表したテンソル（ixx..izz）。
- 見た目は STL（mm）を scale 0.001 で参照する。STL はボディ座標で書かれている前提
  （このロボットではボディ座標 = ワールド座標の立位姿勢）。

URDF に入らないもの: アクチュエータ（力の上限だけ limit の effort に残す）、センサ、
接触パラメータ（solref・friction・contype など）、armature、キーフレーム。
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from q8.visual import COLOURS, role_of, stl_for_body

# MG90S の無負荷速度: 4.8 V で 0.10 s/60° → 約 10.47 rad/s
MG90S_VELOCITY = math.radians(60.0) / 0.10

_PRIMITIVES = {
    mujoco.mjtGeom.mjGEOM_BOX: "box",
    mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
}


# ---------------------------------------------------------------- 回転と剛体変換

def quat_to_matrix(q) -> np.ndarray:
    """MuJoCo の四元数（w, x, y, z）を回転行列に。"""
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def rpy_to_matrix(rpy) -> np.ndarray:
    """URDF の rpy（固定軸 X→Y→Z、R = Rz(y)·Ry(p)·Rx(r)）を回転行列に。"""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def matrix_to_rpy(r: np.ndarray) -> np.ndarray:
    """回転行列を URDF の rpy に。ジンバルロック（pitch = ±90°）では roll を 0 に寄せる。"""
    pitch = math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0]))
    if math.hypot(r[0, 0], r[1, 0]) > 1e-9:
        roll = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(r[1, 0], r[0, 0])
    else:
        roll = 0.0
        # r[0,1] = sin(p)·sin(r)·cy - sy·cr, r[1,1] = sin(p)·sin(r)·sy + cy·cr（r = 0 で -sy, cy）
        yaw = math.atan2(-r[0, 1], r[1, 1])
    return np.array([roll, pitch, yaw])


@dataclass(frozen=True)
class Frame:
    """剛体変換 x ↦ rot·x + pos。"""

    rot: np.ndarray
    pos: np.ndarray

    @staticmethod
    def of(pos, quat=(1.0, 0.0, 0.0, 0.0)) -> "Frame":
        return Frame(quat_to_matrix(quat), np.asarray(pos, dtype=float).copy())

    def __mul__(self, other: "Frame") -> "Frame":
        return Frame(self.rot @ other.rot, self.rot @ other.pos + self.pos)

    def inv(self) -> "Frame":
        return Frame(self.rot.T, -self.rot.T @ self.pos)


IDENTITY = Frame(np.eye(3), np.zeros(3))


# ---------------------------------------------------------------- 書き出しの道具

def _num(v: float, zero: float = 1e-12) -> str:
    v = float(v)
    if abs(v) < zero:
        return "0"
    return f"{v:.10g}"


def _vec(v, zero: float = 1e-12) -> str:
    return " ".join(_num(x, zero) for x in v)


def _origin(parent: ET.Element, f: Frame) -> None:
    attrs = {"xyz": _vec(f.pos)}
    rpy = matrix_to_rpy(f.rot)
    if np.abs(rpy).max() > 1e-12:
        attrs["rpy"] = _vec(rpy)
    ET.SubElement(parent, "origin", attrs)


def _name(m: mujoco.MjModel, objtype, i: int) -> str:
    return mujoco.mj_id2name(m, objtype, i) or ""


# ---------------------------------------------------------------- 変換

@dataclass
class _Link:
    body: int
    name: str
    frame: Frame            # ワールド（モデルの基準姿勢）から見たリンクフレーム
    body_frame: Frame       # ワールドから見た MJCF ボディのフレーム
    joint: int | None       # このボディの hinge（なければ None = fixed）


def _effort(m: mujoco.MjModel, j: int) -> float | None:
    """関節に掛かりうる最大トルク [N·m]。アクチュエータの forcerange×gear と actuatorfrcrange の小さい方。"""
    limits = []
    for a in range(m.nu):
        if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT or m.actuator_trnid[a][0] != j:
            continue
        if m.actuator_forcelimited[a]:
            limits.append(np.abs(m.actuator_forcerange[a]).max() * abs(m.actuator_gear[a][0]))
    if m.jnt_actfrclimited[j]:
        limits.append(np.abs(m.jnt_actfrcrange[j]).max())
    return min(limits) if limits else None


def mjcf_to_urdf(
    mjcf: str | Path,
    *,
    mesh_dir: str | Path | None = None,
    mesh_ref: str | None = None,
    velocity: float | Mapping[str, float] = MG90S_VELOCITY,
    root_name: str = "base_link",
    robot_name: str | None = None,
    mujoco_tag: bool = True,
) -> ET.ElementTree:
    """MJCF を URDF の要素木に変換する。

    mesh_dir: 見た目用 STL のあるディレクトリ。ボディ c_<key> に <key>.stl があれば visual を付ける。
    mesh_ref: URDF から見た mesh_dir の参照（相対パス）。省略時は mesh_dir の絶対パス。
    velocity: revolute の速度上限 [rad/s]（MJCF に無い）。関節名→値の辞書も可。
    mujoco_tag: MuJoCo 用の拡張要素 <mujoco><compiler .../></mujoco> を入れる。MuJoCo は URDF を読むとき
        既定で固定リンクを親に統合し見た目を捨てるので、それを止めるため。他の URDF パーサは無視する。
    """
    mjcf = Path(mjcf)
    m = mujoco.MjModel.from_xml_path(str(mjcf))

    tops = [b for b in range(1, m.nbody) if m.body_parentid[b] == 0]
    if len(tops) != 1:
        raise ValueError(f"ワールド直下のボディはちょうど1つである必要がある（{len(tops)} 個）")
    root = tops[0]

    # ボディの基準姿勢（qpos とは無関係のモデル上の配置）をワールドまで合成する
    world: dict[int, Frame] = {0: IDENTITY}
    for b in range(1, m.nbody):  # MuJoCo のボディ番号は親が先
        world[b] = world[int(m.body_parentid[b])] * Frame.of(m.body_pos[b], m.body_quat[b])

    links: dict[int, _Link] = {}
    for b in range(1, m.nbody):
        name = root_name if b == root else (_name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}")
        jnts = list(range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]))
        joint = None
        frame = world[b]
        if len(jnts) > 1:
            raise NotImplementedError(f"ボディ {name} に関節が {len(jnts)} 個ある（URDF は1リンク1関節）")
        if jnts:
            j = jnts[0]
            jt = m.jnt_type[j]
            if jt == mujoco.mjtJoint.mjJNT_FREE:
                if b != root:
                    raise NotImplementedError(f"free 関節はルート以外に置けない（{name}）")
            elif jt == mujoco.mjtJoint.mjJNT_HINGE:
                if b == root:
                    raise NotImplementedError("ルートボディが hinge でワールドに付いている")
                if abs(m.qpos0[m.jnt_qposadr[j]]) > 1e-12:
                    raise NotImplementedError(f"関節 {_name(m, mujoco.mjtObj.mjOBJ_JOINT, j)} の ref が 0 でない")
                joint = j
                frame = world[b] * Frame(np.eye(3), m.jnt_pos[j].copy())
            else:
                kind = {mujoco.mjtJoint.mjJNT_BALL: "ball", mujoco.mjtJoint.mjJNT_SLIDE: "slide"}[int(jt)]
                raise NotImplementedError(f"{kind} 関節は未対応（ボディ {name}）")
        links[b] = _Link(b, name, frame, world[b], joint)

    robot = ET.Element("robot", name=robot_name or m.names[: m.names.index(b"\0")].decode() or "robot")
    robot.append(ET.Comment(
        f" {mjcf.name} から q8/urdf.py が生成。手で編集しない（scripts/mjcf_to_urdf.py で作り直す）。"
        " capsule は cylinder + 両端の sphere で近似。アクチュエータ・センサ・接触パラメータ・armature は入らない。 "))
    if mujoco_tag:
        mj = ET.SubElement(robot, "mujoco")
        ET.SubElement(mj, "compiler", discardvisual="false", fusestatic="false", strippath="false")

    # 見た目の材質
    mesh_dir = Path(mesh_dir) if mesh_dir is not None else None
    stls: dict[int, str] = {}
    if mesh_dir is not None:
        for b, link in links.items():
            stem = stl_for_body(_name(m, mujoco.mjtObj.mjOBJ_BODY, b))
            if stem and (mesh_dir / f"{stem}.stl").exists():
                stls[b] = stem
        for role in sorted({role_of(s) for s in stls.values()}):
            mat = ET.SubElement(robot, "material", name="vis_" + role)
            ET.SubElement(mat, "color", rgba=COLOURS[role])
    ref = (mesh_ref if mesh_ref is not None else (str(mesh_dir.resolve()) if mesh_dir else "")).rstrip("/")

    geoms_of: dict[int, list[int]] = {}
    for g in range(m.ngeom):
        geoms_of.setdefault(int(m.geom_bodyid[g]), []).append(g)

    for b, link in links.items():
        el = ET.SubElement(robot, "link", name=link.name)
        to_link = link.frame.inv()
        if b == root:
            el.append(ET.Comment(f" MJCF のボディ {_name(m, mujoco.mjtObj.mjOBJ_BODY, b)}（浮遊ベース） "))

        # 慣性: 主慣性を回してリンクの軸で表す
        mass = float(m.body_mass[b])
        if mass > 0:
            f = to_link * link.body_frame * Frame.of(m.body_ipos[b], m.body_iquat[b])
            tensor = f.rot @ np.diag(m.body_inertia[b]) @ f.rot.T
            tensor = 0.5 * (tensor + tensor.T)
            zero = 1e-12 * float(np.abs(np.diag(tensor)).max())
            inertial = ET.SubElement(el, "inertial")
            ET.SubElement(inertial, "origin", xyz=_vec(f.pos))
            ET.SubElement(inertial, "mass", value=_num(mass, 0.0))
            ET.SubElement(inertial, "inertia", {
                k: _num(tensor[i, j], zero)
                for k, (i, j) in {"ixx": (0, 0), "ixy": (0, 1), "ixz": (0, 2),
                                  "iyy": (1, 1), "iyz": (1, 2), "izz": (2, 2)}.items()})

        # 見た目: STL はボディ座標（mm）
        if b in stls:
            vis = ET.SubElement(el, "visual", name="vis/" + stls[b])
            _origin(vis, to_link * link.body_frame)
            geo = ET.SubElement(vis, "geometry")
            ET.SubElement(geo, "mesh", filename=f"{ref}/{stls[b]}.stl" if ref else f"{stls[b]}.stl",
                          scale="0.001 0.001 0.001")
            ET.SubElement(vis, "material", name="vis_" + role_of(stls[b]))

        # 衝突: 接触しうる primitive だけ
        for g in geoms_of.get(b, []):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            gname = _name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"{link.name}/geom{g}"
            kind = _PRIMITIVES.get(int(m.geom_type[g]))
            if kind is None:
                el.append(ET.Comment(f" 衝突形状 {gname} は URDF の primitive にできないので省いた "))
                continue
            f = to_link * link.body_frame * Frame.of(m.geom_pos[g], m.geom_quat[g])
            size = m.geom_size[g]
            if kind == "capsule":
                # URDF 標準に capsule は無い: 同じ半径・長さの cylinder と、両端の sphere で近似
                el.append(ET.Comment(f" {gname}: capsule（半径 {_num(size[0])}, 半長 {_num(size[1])}）を "
                                     "cylinder + 両端の sphere で近似 "))
                _collision(el, gname, f, "cylinder", radius=_num(size[0]), length=_num(2 * size[1]))
                for tag, sign in (("a", -1.0), ("b", 1.0)):
                    end = f * Frame(np.eye(3), np.array([0.0, 0.0, sign * size[1]]))
                    _collision(el, f"{gname}/cap_{tag}", Frame(np.eye(3), end.pos), "sphere", radius=_num(size[0]))
            elif kind == "box":
                _collision(el, gname, f, "box", size=_vec(2 * size[:3]))
            elif kind == "sphere":
                _collision(el, gname, Frame(np.eye(3), f.pos), "sphere", radius=_num(size[0]))
            else:  # cylinder
                _collision(el, gname, f, "cylinder", radius=_num(size[0]), length=_num(2 * size[1]))

    # 関節: 親リンクのフレームから見た子リンクのフレーム
    for b, link in links.items():
        if b == root:
            continue
        parent = links[int(m.body_parentid[b])]
        origin = parent.frame.inv() * link.frame
        if link.joint is None:
            jel = ET.SubElement(robot, "joint", name=f"{link.name}_fixed", type="fixed")
            _origin(jel, origin)
            ET.SubElement(jel, "parent", link=parent.name)
            ET.SubElement(jel, "child", link=link.name)
            continue
        j = link.joint
        jname = _name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"{link.name}_joint"
        limited = bool(m.jnt_limited[j])
        jel = ET.SubElement(robot, "joint", name=jname, type="revolute" if limited else "continuous")
        _origin(jel, origin)
        ET.SubElement(jel, "parent", link=parent.name)
        ET.SubElement(jel, "child", link=link.name)
        ET.SubElement(jel, "axis", xyz=_vec(m.jnt_axis[j]))  # ボディ座標 = リンクの軸
        effort = _effort(m, j)
        vel = velocity[jname] if isinstance(velocity, Mapping) else velocity
        lim = {}
        if limited:
            lim["lower"], lim["upper"] = _num(m.jnt_range[j][0]), _num(m.jnt_range[j][1])
        lim["effort"] = _num(effort if effort is not None else 0.0)
        lim["velocity"] = _num(vel)
        if effort is None:
            jel.append(ET.Comment(" MJCF に力の上限が無いので effort=0（上限なしの意味ではない） "))
        ET.SubElement(jel, "limit", lim)
        dof = m.jnt_dofadr[j]
        dyn = {"damping": _num(m.dof_damping[dof])}
        if m.dof_frictionloss[dof] > 0:
            dyn["friction"] = _num(m.dof_frictionloss[dof])
        ET.SubElement(jel, "dynamics", dyn)

    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ")
    return tree


def _collision(link: ET.Element, name: str, f: Frame, shape: str, **attrs: str) -> None:
    col = ET.SubElement(link, "collision", name=name)
    _origin(col, f)
    geo = ET.SubElement(col, "geometry")
    ET.SubElement(geo, shape, attrs)


def write_urdf(mjcf: str | Path, out: str | Path, *, mesh_dir: str | Path | None = None, **kw) -> Path:
    """MJCF を URDF にして out に書く。STL は out から見た相対パスで参照する。"""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if mesh_dir is not None and "mesh_ref" not in kw:
        kw["mesh_ref"] = Path(os.path.relpath(Path(mesh_dir).resolve(), out.parent.resolve())).as_posix()
    tree = mjcf_to_urdf(mjcf, mesh_dir=mesh_dir, **kw)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out
