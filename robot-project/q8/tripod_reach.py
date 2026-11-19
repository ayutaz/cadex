"""3本足で1本の脚を上げたとき、重心を「残る2本の足を結ぶ線」の上に置けるか（静的な余裕）を計算する。

定義は調査（``.superpowers/sdd/2026-11-18-tripod-and-tail/diag-tripod-report.md`` の「仮説2」と
``static_best.py``）に合わせる。

- **余裕** ``lift_margin``: 上げる足以外の2本の足先（足の衝突形状の中心）を結ぶ線から、機体全体の重心
  （``subtree_com[0]``）までの距離を床へ写して測る [mm]。重心が、線に対して上げる足と反対の側にあれば正
  （静的に立てる）。3本で立っているとき重心は三角形の中にあるので、立位の値はかならず負になる。
- **最良の余裕** ``best_lift_margin``: 関節角と、支えの2本を結ぶ線のまわりの胴体の傾き（``phi``）を選んで
  余裕を最大にする。条件は次のとおり。
  - 支えの2本は床に着く（線を水平に置くので、足先の z は足先の半径に一致する）。
  - 上げる足の一番低い点が ``clear_mm`` 以上浮く。
  - 胴体の傾き（胴体の z 軸と鉛直の角）が ``tilt_max_deg`` 以下（調査の表は 10°）。
  - 床と当たる胴体などの衝突形状（足以外）が床より上にある。
  - 関節は MJCF の範囲の中。ロールの関節（``roll_<脚>``）は、``roll_range_deg`` を渡すと ±その値にする。

解き方: 関節ごとに 5° 刻みの格子を作る。格子の点の数が ``samples`` 以下なら全部、多ければ乱数で
``samples`` 点を選ぶ（6〜9 関節の格子全部は数千万点になるため）。最良は関節の端に張り付きやすいので、
関節ごとの {下限, 0, 上限} の組み合わせも全部足す。点ごとに ``phi`` を 0.5° 刻みで調べて
最良を取り、互いに離れた上位10点から ``scipy.optimize.minimize``（Nelder-Mead、条件は罰で入れる）で詰める。
Nelder-Mead では ``phi`` を内側で 0.05° 刻みに選び、関節角だけを動かす。そのあと、座標ごとの探索
（2°〜0.05° 刻み）で仕上げる。scipy が無いときは、1° 刻みの座標ごとの探索だけで詰める。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .robot import model_feet, model_legs

#: 関節の名前の頭（脚の中の順。ロールの関節が無い機体もある）
JOINT_PREFIXES = ("roll_", "hip_", "knee_")
GRID_STEP_DEG = 5.0
PHI_STEP_DEG = 0.5
PHI_FINE_DEG = 0.05
REFINE_STEP_DEG = 1.0
#: Nelder-Mead のあとに、座標ごとの探索で詰める刻み [deg]（大きい順）
POLISH_STEPS_DEG = (2.0, 1.0, 0.5, 0.25, 0.1, 0.05)
TOP_K = 10
#: 詰める出発点どうしを、どれかの関節で少なくともこれだけ離す（同じ谷ばかりから出発しないように）
DIVERSE_DEG = 20.0
DEFAULT_SAMPLES = 20000
_PENALTY = 1.0e3


def bundle_xml(bundle_dir: Path | str) -> Path:
    """DR バンドルのフォルダ（``artifacts/<機体>-dr`` か、その ``cut``）から ``cut.xml`` を探す。"""
    path = Path(bundle_dir)
    if path.is_file():
        return path
    for candidate in (path / "cut" / "cut.xml", path / "cut.xml"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{path} に cut.xml が無い（<bundle>/cut/cut.xml か <bundle>/cut.xml）")


def _floor_geom(model: mujoco.MjModel) -> int:
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0 and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            return g
    raise ValueError("床（world の plane）が無い")


def _touches(model: mujoco.MjModel, g: int, floor: int) -> bool:
    return bool((model.geom_contype[g] & model.geom_conaffinity[floor])
                or (model.geom_contype[floor] & model.geom_conaffinity[g]))


def _foot_geoms(model: mujoco.MjModel) -> dict[str, int]:
    """脚ごとに、足の部品 ``c_foot_<脚>`` の、床と当たる衝突形状（最初の1つ）。"""
    floor = _floor_geom(model)
    out: dict[str, int] = {}
    for leg, body_name in zip(model_legs(model), model_feet(model)):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body and _touches(model, g, floor)]
        if body < 0 or not geoms:
            raise ValueError(f"{body_name} に床と当たる衝突形状が無い")
        out[leg] = geoms[0]
    return out


def _split(model: mujoco.MjModel, lift: str) -> tuple[dict[str, int], list[str]]:
    feet = _foot_geoms(model)
    if len(feet) != 3:
        raise ValueError(f"3本足の機体ではない（脚 {list(feet)}）")
    if lift not in feet:
        raise ValueError(f"脚 {lift!r} が無い（脚 {list(feet)}）")
    return feet, [leg for leg in feet if leg != lift]


def lift_margin(model: mujoco.MjModel, data: mujoco.MjData, lift: str) -> float:
    """今の姿勢（``data`` は ``mj_forward`` か ``mj_kinematics`` + ``mj_comPos`` のあと）での余裕 [mm]。

    ``lift`` 以外の2本の足先を結ぶ線から、重心（``subtree_com[0]``）までの、床に写した距離。
    重心が、線に対して上げる足と反対の側にあれば正。
    """
    feet, support = _split(model, lift)
    a = data.geom_xpos[feet[support[0]]][:2]
    b = data.geom_xpos[feet[support[1]]][:2]
    up = data.geom_xpos[feet[lift]][:2]
    com = data.subtree_com[0][:2]
    ab = b - a
    normal = np.array([-ab[1], ab[0]]) / np.linalg.norm(ab)
    if normal @ (up - a) < 0:
        normal = -normal
    return float(-(normal @ (com - a)) * 1000.0)


def _lowest_below(model: mujoco.MjModel, g: int, normal: np.ndarray, mats: np.ndarray) -> np.ndarray:
    """衝突形状 ``g`` の中心から、``normal`` の向きで一番低い点までの深さ [m]（形ごとの支持関数）。

    ``normal`` は (..., 3)、``mats`` は形の回転 (..., 3, 3)。
    """
    size = model.geom_size[g]
    kind = model.geom_type[g]
    along = np.einsum("...i,...ij->...j", normal, mats)  # 形の軸ごとの normal の成分
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.full(along.shape[:-1], size[0])
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return np.abs(along) @ size
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return np.abs(along[..., 2]) * size[1] + size[0]
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        cz = np.abs(along[..., 2])
        return cz * size[1] + np.sqrt(np.clip(1.0 - cz * cz, 0.0, 1.0)) * size[0]
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return np.sqrt((along * along) @ (size * size))
    return np.full(along.shape[:-1], model.geom_rbound[g])


class _Problem:
    """1本を上げる問題。関節角の並び ``x``（rad）と ``phi``（rad）から、余裕と条件の破れを計算する。"""

    def __init__(self, model: mujoco.MjModel, lift: str, clear_mm: float,
                 roll_range_deg: float | None, tilt_max_deg: float):
        self.model = model
        self.data = mujoco.MjData(model)
        self.lift = lift
        self.feet, self.support = _split(model, lift)
        self.clear = clear_mm / 1000.0
        self.tilt_max = math.radians(tilt_max_deg)
        self.joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
        self.joints = [n for n in self.joints if n and n.startswith(JOINT_PREFIXES)]
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joints]
        self.qadr = np.array([model.jnt_qposadr[j] for j in ids])
        lo = np.array([model.jnt_range[j][0] if model.jnt_limited[j] else -math.pi for j in ids])
        hi = np.array([model.jnt_range[j][1] if model.jnt_limited[j] else math.pi for j in ids])
        if roll_range_deg is not None:
            for i, name in enumerate(self.joints):
                if name.startswith("roll_"):
                    lo[i], hi[i] = -math.radians(roll_range_deg), math.radians(roll_range_deg)
        self.lo, self.hi = lo, hi
        free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if not free:
            raise ValueError("胴体の自由関節が無い")
        self.base = int(model.jnt_qposadr[free[0]])
        floor = _floor_geom(model)
        foot_ids = set(self.feet.values())
        self.body_geoms = [g for g in range(model.ngeom)
                           if model.geom_bodyid[g] != 0 and g not in foot_ids and _touches(model, g, floor)]
        self.r_support = float(model.geom_size[self.feet[self.support[0]]][0])
        self.r_lift = float(model.geom_size[self.feet[lift]][0])
        self.phis_fine = np.radians(np.arange(-tilt_max_deg, tilt_max_deg + 1e-9, PHI_FINE_DEG))

    # 胴体を原点・回転なしに置いたときの、点の位置（胴体の座標）
    def _body_frame(self, x: np.ndarray) -> dict[str, np.ndarray]:
        d = self.data
        d.qpos[:] = self.model.qpos0
        d.qpos[self.base:self.base + 3] = 0.0
        d.qpos[self.base + 3:self.base + 7] = (1.0, 0.0, 0.0, 0.0)
        d.qpos[self.qadr] = x
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        return {
            "a": d.geom_xpos[self.feet[self.support[0]]].copy(),
            "b": d.geom_xpos[self.feet[self.support[1]]].copy(),
            "up": d.geom_xpos[self.feet[self.lift]].copy(),
            "com": d.subtree_com[0].copy(),
            "gpos": d.geom_xpos[self.body_geoms].copy(),
            "gmat": d.geom_xmat[self.body_geoms].reshape(-1, 3, 3).copy(),
        }

    def evaluate(self, f: dict[str, np.ndarray], phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``phi``（rad の配列）ごとに、余裕 [mm]、条件の破れの合計 [mm]、傾き [rad]。"""
        a, b, up, com = f["a"], f["b"], f["up"], f["com"]
        u = (b - a) / np.linalg.norm(b - a)
        z = np.array([0.0, 0.0, 1.0])
        e1 = z - u * (u @ z)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        n = np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2     # 床の法線（胴体の座標）
        tilt = np.arccos(np.clip(n @ z, -1.0, 1.0))
        w = np.cross(n, u)
        w *= np.where(w @ (up - a) < 0, -1.0, 1.0)[:, None]
        margin = -(w @ (com - a)) * 1000.0
        clear = n @ (up - a) + self.r_support - self.r_lift            # 上げる足の一番低い点の高さ
        bad = np.maximum(0.0, self.clear - clear) + np.maximum(0.0, tilt - self.tilt_max)
        for i, g in enumerate(self.body_geoms):
            low = n @ (f["gpos"][i] - a) + self.r_support - _lowest_below(
                self.model, g, n, np.broadcast_to(f["gmat"][i], (len(phi), 3, 3)))
            bad = bad + np.maximum(0.0, -low)
        return margin, bad * 1000.0, tilt

    def best_phi(self, x: np.ndarray) -> tuple[float, float, float]:
        """関節角 ``x`` のまま、``phi`` を ``PHI_FINE_DEG`` 刻みで動かした最良。(余裕 [mm], phi [rad], 破れ [mm])。

        条件を満たす ``phi`` が無ければ、余裕は -inf、破れは一番小さい破れ。
        """
        margin, bad, _ = self.evaluate(self._body_frame(x), self.phis_fine)
        ok = bad <= 1e-9
        if ok.any():
            i = int(np.argmax(np.where(ok, margin, -np.inf)))
            return float(margin[i]), float(self.phis_fine[i]), 0.0
        i = int(np.argmin(bad))
        return -math.inf, float(self.phis_fine[i]), float(bad[i])

    def objective(self, x: np.ndarray) -> float:
        """関節角だけの目的関数（``phi`` は内側で最良を取る）。範囲の外と条件の破れは罰にする。"""
        clipped = np.clip(x, self.lo, self.hi)
        out = math.degrees(float(np.abs(x - clipped).sum()))
        margin, _, bad = self.best_phi(clipped)
        if bad > 0.0:
            return float(_PENALTY * (1.0 + bad + out))
        return float(-margin + _PENALTY * out)


def _grid(lo: np.ndarray, hi: np.ndarray, step: float) -> list[np.ndarray]:
    return [np.linspace(l, h, max(2, int(math.ceil((h - l) / step - 1e-9)) + 1)) for l, h in zip(lo, hi)]


def _samples(problem: _Problem, samples: int, seed: int) -> np.ndarray:
    axes = _grid(problem.lo, problem.hi, math.radians(GRID_STEP_DEG))
    total = math.prod(len(a) for a in axes)
    if total <= samples:
        mesh = np.meshgrid(*axes, indexing="ij")
        points = np.stack([m.ravel() for m in mesh], axis=1)
    else:
        rng = np.random.default_rng(seed)
        points = np.stack([rng.choice(a, size=samples) for a in axes], axis=1)
    stand = np.clip(np.zeros(len(axes)), problem.lo, problem.hi)
    # 最良は関節の端に張り付きやすいので、関節ごとに {下限, 0, 上限} の組み合わせを全部足す（9 関節で 19683 点）
    levels = [np.unique([l, min(max(0.0, l), h), h]) for l, h in zip(problem.lo, problem.hi)]
    corners = np.stack([m.ravel() for m in np.meshgrid(*levels, indexing="ij")], axis=1)
    return np.vstack([stand[None, :], corners, points])


def _diverse_starts(scored: list[tuple[float, np.ndarray]], k: int, apart: float) -> list[np.ndarray]:
    """余裕の大きい順に、すでに選んだ点のどれからも（関節のどれかで）``apart`` 以上離れた点を ``k`` 個選ぶ。"""
    chosen: list[np.ndarray] = []
    for _, v in scored:
        if all(np.max(np.abs(v[:-1] - c[:-1])) >= apart for c in chosen):
            chosen.append(v)
            if len(chosen) == k:
                break
    return chosen


def _pattern_search(problem: _Problem, x: np.ndarray, steps_deg: tuple[float, ...]) -> np.ndarray:
    """関節ごとに ±刻み を試し、良くなる間は続け、良くならなければ刻みを小さくする。"""
    best = problem.objective(x)
    for step in np.radians(steps_deg):
        improved = True
        while improved:
            improved = False
            for i in range(len(x)):
                for sign in (1.0, -1.0):
                    trial = x.copy()
                    trial[i] = np.clip(trial[i] + sign * step, problem.lo[i], problem.hi[i])
                    value = problem.objective(trial)
                    if value < best - 1e-9:
                        x, best, improved = trial, value, True
    return x


def place(model: mujoco.MjModel, data: mujoco.MjData, lift: str, joint_deg: dict[str, float],
          phi_deg: float) -> dict[str, dict[str, float]]:
    """``best_lift_margin`` の答え（関節角と ``phi``）を、床の上の本当の姿勢として ``data`` に置く。

    支えの2本を結ぶ線を世界の x 軸に平行に、床の法線を鉛直に向け、支えの足先の z を足先の半径にする。
    返り値は脚ごとの足先の ``z_mm`` と ``radius_mm``。``data`` は ``lift_margin`` にそのまま渡せる。
    """
    problem = _Problem(model, lift, 0.0, None, 180.0)
    x = np.array([math.radians(joint_deg[n]) for n in problem.joints])
    f = problem._body_frame(x)
    a, b = f["a"], f["b"]
    u = (b - a) / np.linalg.norm(b - a)
    z = np.array([0.0, 0.0, 1.0])
    e1 = z - u * (u @ z)
    e1 /= np.linalg.norm(e1)
    phi = math.radians(phi_deg)
    n = math.cos(phi) * e1 + math.sin(phi) * np.cross(u, e1)
    rot = np.vstack([u, np.cross(n, u), n])           # 胴体の座標 → 世界の座標
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.ravel())
    data.qpos[:] = model.qpos0
    data.qpos[problem.qadr] = x
    data.qpos[problem.base:problem.base + 3] = (0.0, 0.0, problem.r_support - n @ a)
    data.qpos[problem.base + 3:problem.base + 7] = quat
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    return {leg: {"z_mm": float(data.geom_xpos[g][2] * 1000.0), "radius_mm": float(model.geom_size[g][0] * 1000.0)}
            for leg, g in problem.feet.items()}


def best_lift_margin(bundle_dir: Path | str, lift: str, clear_mm: float = 10.0,
                     roll_range_deg: float | None = None, tilt_max_deg: float = 10.0,
                     samples: int = DEFAULT_SAMPLES, seed: int = 0) -> dict[str, Any]:
    """1本 ``lift`` を ``clear_mm`` 以上上げ、残りの2本を床に着けたまま、余裕を最大にする。

    返り値: ``margin_mm``、``joint_deg``（関節名 → 角度）、``feasible``（条件を満たす点が見つかったか）、
    ``phi_deg``（支えの線のまわりの傾き。``place`` に渡す）、``tilt_deg``（胴体の傾き）。
    """
    model = mujoco.MjModel.from_xml_path(str(bundle_xml(bundle_dir)))
    problem = _Problem(model, lift, clear_mm, roll_range_deg, tilt_max_deg)
    phis = np.radians(np.arange(-tilt_max_deg, tilt_max_deg + 1e-9, PHI_STEP_DEG))
    scored: list[tuple[float, np.ndarray]] = []
    for x in _samples(problem, samples, seed):
        margin, bad, _ = problem.evaluate(problem._body_frame(x), phis)
        ok = bad <= 1e-9
        if ok.any():
            i = int(np.argmax(np.where(ok, margin, -np.inf)))
            scored.append((float(margin[i]), np.append(x, phis[i])))
    if not scored:
        x = np.zeros(len(problem.joints))
        margin, _, _ = problem.evaluate(problem._body_frame(x), np.array([0.0]))
        return {"margin_mm": float(margin[0]), "joint_deg": dict(zip(problem.joints, map(float, x))),
                "feasible": False, "phi_deg": 0.0, "tilt_deg": 0.0}
    scored.sort(key=lambda item: -item[0])
    best_value, best_v = scored[0]
    try:
        from scipy.optimize import minimize
    except ImportError:  # pragma: no cover - venv には scipy がある
        minimize = None
    for v0 in _diverse_starts(scored, TOP_K, math.radians(DIVERSE_DEG)):
        x = v0[:-1]
        if minimize is not None:
            simplex = np.vstack([x] + [x + np.eye(len(x))[i] * math.radians(2.5) for i in range(len(x))])
            result = minimize(problem.objective, x, method="Nelder-Mead",
                              options={"initial_simplex": simplex, "xatol": 1e-5, "fatol": 1e-6,
                                       "maxiter": 300 * len(x), "maxfev": 300 * len(x)})
            x = np.clip(result.x, problem.lo, problem.hi)
            x = _pattern_search(problem, x, POLISH_STEPS_DEG)
        else:
            x = _pattern_search(problem, x, (REFINE_STEP_DEG,))
        value, phi, bad = problem.best_phi(x)
        if bad <= 0.0 and value > best_value:
            best_value, best_v = value, np.append(x, phi)
    _, _, tilt = problem.evaluate(problem._body_frame(best_v[:-1]), np.array([best_v[-1]]))
    return {
        "margin_mm": float(best_value),
        "joint_deg": {n: math.degrees(q) for n, q in zip(problem.joints, best_v[:-1])},
        "feasible": True,
        "phi_deg": math.degrees(best_v[-1]),
        "tilt_deg": math.degrees(float(tilt[0])),
    }
