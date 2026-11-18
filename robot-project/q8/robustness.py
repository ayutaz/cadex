"""実機との差に対する強さを測る（段階A-2）。

評価そのものは **エンジンの評価**（``CadexDynamics.evaluate_success``。``cadex evaluate`` と同じ関数）を
そのまま使う。成功条件・評価シード・指標の計算・タスクのランダム化（胴体の質量、関節の粘性と摩擦、
初期の傾き）は通常の評価と同じで、そこに実機との差を重ねる。

- モデルの値（``perturb_model``）: 足と床の摩擦、サーボの剛性とトルク、胴体の質量と重心。
  エンジンがモデルを読む ``CadexDynamics.load_model`` を包んで、読んだ直後に書き換える。
- 時間の差（``EngineProxy``）: エンジンの1制御（20 ms）は ``mj_step`` を10回（2 ms ずつ）呼ぶ。
  その ``mj_step`` を包み、2 ms ごとに次を入れる。
  - サーボの遅れ（``latency_ms``）: 方策の指令が、遅れてサーボに届く（純粋な遅れ）。
  - 制御周期のゆらぎ（``jitter_ms``）: 指令ごとに 0〜jitter の遅れが足される。
  - ガタ（``backlash_deg``、全幅）: 指令と関節角の差が ±幅/2 の内側ではサーボが押し返さない（不感帯）。
    ギアのガタと、サーボ自身の不感帯をまとめた近似で、ギアの衝突は表さない。
  - 押し（``Push``）: 胴体の重心に、向きに対して真横の力を、決めた時刻に決めた長さだけ加える。

何も変えない条件（``Conditions()``）では、エンジンの評価と同じ数字になる（tests/test_robustness.py）。
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import random
import sys
import zlib
from dataclasses import dataclass
from statistics import mean
from typing import Any

import numpy as np

from .paths import CADEX_REPO

CHASSIS = "c_chassis"
FLOOR = "environment/floor"
FOOT_PREFIX = "c_foot_"


def dynamics():
    """Cadex の CadexDynamics（改変しない）。"""
    mod_dir = str(CADEX_REPO / "src" / "Mod" / "cadex")
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
    import CadexDynamics  # noqa: PLC0415

    return CadexDynamics


# --- 条件 ---------------------------------------------------------------------

@dataclass(frozen=True)
class Push:
    """胴体への横からの押し。``side`` は +1 が左（進む向きに対して）、-1 が右。"""

    newtons: float
    side: int = +1
    at_s: float = 3.0
    duration_s: float = 0.1


@dataclass(frozen=True)
class Conditions:
    """実機との差。既定値はどれも「変えない」。"""

    friction: float | None = None           # 足と床の摩擦係数 μ（None はモデルのまま。床 1.0、足 0.9 で接触は大きい方の 1.0）
    kp_scale: float = 1.0                   # サーボの剛性（位置の比例ゲイン）
    torque_scale: float = 1.0               # サーボの最大トルク
    mass_scale: float = 1.0                 # 胴体の質量と慣性（タスクのランダム化に重ねる）
    com_offset_mm: tuple[float, float] = (0.0, 0.0)  # 胴体の重心のずれ（x 前、y 左）
    latency_ms: float = 0.0                 # 指令がサーボに届くまでの遅れ
    jitter_ms: float = 0.0                  # 指令ごとに 0〜jitter_ms の遅れが足される
    backlash_deg: float = 0.0               # ガタの全幅（不感帯 ±幅/2）
    push: Push | None = None


#: 実機を想定した幅（``sample_conditions`` が一様に引く）。根拠は docs/experiment-results.md §11。
REAL_RANGES: dict[str, tuple] = {
    "friction": (0.3, 1.2),                 # PLA の足（約 0.3）〜ゴム足（約 1.0 以上）
    "kp_scale": (0.7, 1.3),                 # MG90S の個体差（剛性は推定値）
    "torque_scale": (0.8, 1.1),             # 個体差と電圧（4.8〜5 V、電圧降下）
    "mass_scale": (0.9, 1.1),               # 胴体の印刷の充填率・配線
    "com_offset_mm": ((-5.0, -3.0), (5.0, 3.0)),
    "latency_ms": (0.0, 30.0),              # IMU の読み出し＋推論＋PWM の周期（50 Hz で最大 20 ms）
    "jitter_ms": (0.0, 4.0),                # 制御周期のゆらぎ
    "backlash_deg": (0.0, 3.0),             # ギアのガタと不感帯
}

#: 1つずつ変える条件（他は公称）。
SWEEP: list[tuple[str, Conditions]] = [
    ("friction 0.3", Conditions(friction=0.3)),
    ("friction 0.5", Conditions(friction=0.5)),
    ("friction 0.7", Conditions(friction=0.7)),
    ("latency 20 ms", Conditions(latency_ms=20.0)),
    ("latency 40 ms", Conditions(latency_ms=40.0)),
    ("jitter 4 ms", Conditions(jitter_ms=4.0)),
    ("jitter 20 ms", Conditions(jitter_ms=20.0)),
    ("backlash 2 deg", Conditions(backlash_deg=2.0)),
    ("backlash 4 deg", Conditions(backlash_deg=4.0)),
    ("kp 0.6", Conditions(kp_scale=0.6)),
    ("torque 0.8", Conditions(torque_scale=0.8)),
    ("mass 1.2", Conditions(mass_scale=1.2)),
    ("com x +5 mm", Conditions(com_offset_mm=(5.0, 0.0))),
    ("com x -5 mm", Conditions(com_offset_mm=(-5.0, 0.0))),
    ("com y +3 mm", Conditions(com_offset_mm=(0.0, 3.0))),
]

#: 押しの強さ [N]（0.1 秒、3.0 秒目、左右）。機体の重さは約 2.16 N。
PUSH_NEWTONS: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5)
PUSH_AT_S = 3.0
PUSH_DURATION_S = 0.1

#: 「倒れずに戻った」の判定。押しの終わりから RECOVER_WITHIN_S 以内に、傾きが RECOVER_TILT_DEG 以下の
#: 状態が RECOVER_HOLD_S 続き、押しの終わりの WALK_WINDOW_S 後の区間で指令の WALK_MIN_RATIO 以上で前へ進む。
RECOVER_TILT_DEG = 10.0
RECOVER_HOLD_S = 0.5
RECOVER_WITHIN_S = 1.5
WALK_WINDOW_S = (1.0, 3.0)
WALK_MIN_RATIO = 0.5


def sample_conditions(rng: random.Random, ranges: dict[str, tuple] = REAL_RANGES) -> Conditions:
    """``ranges`` から一様に1組引く（キーの順に引くので、同じ乱数なら同じ組）。"""
    values: dict[str, Any] = {}
    for name, (low, high) in ranges.items():
        if isinstance(low, tuple):
            values[name] = tuple(rng.uniform(lo, hi) for lo, hi in zip(low, high))
        else:
            values[name] = rng.uniform(low, high)
    return Conditions(**values)


# --- モデルの値 ---------------------------------------------------------------

def _foot_and_floor_geoms(model) -> list[int]:
    import mujoco  # noqa: PLC0415

    out = []
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
        colliding = int(model.geom_contype[g]) or int(model.geom_conaffinity[g])
        if name == FLOOR or (body.startswith(FOOT_PREFIX) and colliding):
            out.append(g)
    return out


def perturb_model(model, conditions: Conditions):
    """``model`` をその場で書き換えて返す。"""
    import mujoco  # noqa: PLC0415

    c = conditions
    if c.friction is not None:
        for g in _foot_and_floor_geoms(model):
            model.geom_friction[g, 0] = float(c.friction)
    if c.kp_scale != 1.0:
        for a in range(model.nu):
            model.actuator_gainprm[a, 0] *= c.kp_scale
            model.actuator_biasprm[a, 1] *= c.kp_scale
    if c.torque_scale != 1.0:
        model.actuator_forcerange[:] *= c.torque_scale
    chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CHASSIS)
    if c.mass_scale != 1.0:
        model.body_mass[chassis] *= c.mass_scale
        model.body_inertia[chassis] *= c.mass_scale
    if any(c.com_offset_mm):
        model.body_ipos[chassis, 0] += c.com_offset_mm[0] / 1000.0
        model.body_ipos[chassis, 1] += c.com_offset_mm[1] / 1000.0
    if c.mass_scale != 1.0 or any(c.com_offset_mm):
        mujoco.mj_setConst(model, mujoco.MjData(model))
    return model


# --- 時間の差: サーボへの指令の届き方、ガタ、押し ------------------------------

def deadband(error: np.ndarray, half_width: float) -> np.ndarray:
    """±half_width の内側は 0、外側は幅の分だけ縮めた差。"""
    if half_width <= 0.0:
        return error
    return np.sign(error) * np.maximum(np.abs(error) - half_width, 0.0)


class ServoBus:
    """方策の指令が、遅れてサーボに届く。

    ``active(k, loop_ctrl)`` は k 番目の小刻み（2 ms）でサーボが従う指令を返す。制御の切れ目
    （k が ``per_action`` の倍数）で ``loop_ctrl`` を新しい指令として受け取り、
    ``latency_s`` ＋ 一様(0, ``jitter_s``) 後（小刻みに丸める）に届いたものとして扱う。
    届くまでは前の指令（最初は ``initial``）を保つ。
    """

    def __init__(self, *, initial: np.ndarray, latency_s: float, jitter_s: float, dt: float,
                 per_action: int, rng: random.Random):
        self.current = np.array(initial, dtype=float)
        self.latency_s, self.jitter_s, self.dt, self.per_action = latency_s, jitter_s, dt, per_action
        self.rng = rng
        self.queue: list[tuple[int, np.ndarray]] = []

    def active(self, k: int, loop_ctrl: np.ndarray) -> np.ndarray:
        if k % self.per_action == 0:
            delay = self.latency_s + (self.rng.uniform(0.0, self.jitter_s) if self.jitter_s > 0 else 0.0)
            self.queue.append((k + int(round(delay / self.dt)), np.array(loop_ctrl, dtype=float)))
        while self.queue and self.queue[0][0] <= k:
            self.current = self.queue.pop(0)[1]
        return self.current


def push_force(quat_wxyz: np.ndarray, *, side: int, newtons: float) -> np.ndarray:
    """胴体の向き（ヨー）に対して真横の、水平な力（ワールド座標）。"""
    w, x, y, z = (float(v) for v in quat_wxyz)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([-math.sin(yaw), math.cos(yaw), 0.0]) * float(side) * float(newtons)


class EngineProxy:
    """エンジンに渡す mujoco の代わり。``mj_step`` だけを包み、ほかは本物に渡す。"""

    def __init__(self, real, conditions: Conditions, *, per_action: int, rng: random.Random):
        self._real, self._c, self._per_action, self._rng = real, conditions, per_action, rng
        self._data = None

    def __getattr__(self, name):
        return getattr(self._real, name)

    def _start(self, model, data) -> None:
        mujoco = self._real
        self._data, self._k = data, 0
        initial = np.array(model.key_ctrl[0]) if model.nkey else np.zeros(model.nu)
        self._bus = ServoBus(initial=initial, latency_s=self._c.latency_ms / 1000.0,
                             jitter_s=self._c.jitter_ms / 1000.0, dt=float(model.opt.timestep),
                             per_action=self._per_action, rng=self._rng)
        self._qpos = np.array([model.jnt_qposadr[model.actuator_trnid[a, 0]] for a in range(model.nu)])
        self._half = math.radians(self._c.backlash_deg) / 2.0
        self._chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CHASSIS)
        self._force = None
        self._timed = self._c.latency_ms > 0 or self._c.jitter_ms > 0 or self._half > 0

    def mj_step(self, model, data, *args):
        if data is not self._data:
            self._start(model, data)
        t = self._k * float(model.opt.timestep)
        push = self._c.push
        if push is not None:
            if push.at_s - 1e-9 <= t < push.at_s + push.duration_s - 1e-9:
                if self._force is None:
                    self._force = push_force(data.xquat[self._chassis], side=push.side, newtons=push.newtons)
                data.xfrc_applied[self._chassis, :3] = self._force
            else:
                data.xfrc_applied[self._chassis, :3] = 0.0
        if not self._timed:
            self._k += 1
            return self._real.mj_step(model, data, *args)
        loop_ctrl = np.array(data.ctrl)
        target = self._bus.active(self._k, loop_ctrl)
        q = np.array(data.qpos[self._qpos])
        data.ctrl[:] = q + deadband(target - q, self._half)
        try:
            return self._real.mj_step(model, data, *args)
        finally:
            data.ctrl[:] = loop_ctrl     # エンジンのループが書いた指令（記録に残るもの）に戻す
            self._k += 1


@contextlib.contextmanager
def patched_engine(conditions: Conditions, *, per_action: int, rng: random.Random):
    """CadexDynamics のモデルの読み込みと mujoco を、この条件のものに差し替える。"""
    dyn = dynamics()
    real_mujoco = dyn._mujoco_module()
    load_model, mujoco_module = dyn.load_model, dyn._mujoco_module
    proxy = EngineProxy(real_mujoco, conditions, per_action=per_action, rng=rng)
    dyn.load_model = lambda xml: perturb_model(real_mujoco.MjModel.from_xml_string(bytes(xml).decode("utf-8")),
                                               conditions)
    dyn._mujoco_module = lambda: proxy
    try:
        yield proxy
    finally:
        dyn.load_model, dyn._mujoco_module = load_model, mujoco_module


# --- 押しからの回復 -----------------------------------------------------------

def _tilt_deg(xyzw) -> float:
    x, y = float(xyzw[0]), float(xyzw[1])
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


def _yaw(xyzw) -> float:
    x, y, z, w = (float(v) for v in xyzw)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def push_recovery(frames: list[dict], push: Push, *, completed: bool, base: str = CHASSIS) -> dict:
    """押しのあと、倒れずに戻ったか。``frames`` はエンジンのトレースの solver_output。"""
    solved = [f for f in frames if f.get("frame_kind") == "solver_output"]
    if not completed:
        return {"survived": False, "recovered": False, "recovery_s": None, "walk_ratio_after": None}
    time = [float(f["nominal_time_s"]) for f in solved]
    poses = [f["component_placements"][base] for f in solved]
    tilt = [_tilt_deg(p["rotation_xyzw"]) for p in poses]
    end = push.at_s + push.duration_s
    dt = time[1] - time[0]
    hold = int(round(RECOVER_HOLD_S / dt))
    recovery = None
    for i, moment in enumerate(time):
        if moment < end - 1e-9:
            continue
        if moment > end + RECOVER_WITHIN_S + 1e-9 or i + hold >= len(time):
            break
        if all(v <= RECOVER_TILT_DEG for v in tilt[i:i + hold + 1]):
            recovery = moment - end
            break
    onset = min(range(len(time)), key=lambda i: abs(time[i] - push.at_s))
    heading = _yaw(poses[onset]["rotation_xyzw"])
    lo, hi = (min(range(len(time)), key=lambda i, w=w: abs(time[i] - (end + w))) for w in WALK_WINDOW_S)
    moved = np.array(poses[hi]["position_mm"][:2]) - np.array(poses[lo]["position_mm"][:2])
    speed = float(moved @ np.array([math.cos(heading), math.sin(heading)])) / (time[hi] - time[lo])
    cmd = mean(float(f["goal"][0]) for f in solved[lo:hi + 1])
    ratio = speed / cmd if cmd else None
    walking = ratio is not None and ratio >= WALK_MIN_RATIO
    return {"survived": True, "recovered": recovery is not None and walking, "recovery_s": recovery,
            "walk_ratio_after": ratio}


# --- 1エピソード ----------------------------------------------------------------

def component_names(model) -> list[str]:
    import mujoco  # noqa: PLC0415

    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(1, model.nbody)]


def _case_rng(seed: int, conditions: Conditions) -> random.Random:
    return random.Random(int(seed) * 1_000_003 + zlib.crc32(repr(conditions).encode("utf-8")))


def evaluate_case(xml: bytes, task: dict, container: dict, *, seed: int, conditions: Conditions) -> dict:
    """1つの評価シードを、1つの条件で、エンジンの評価にかける。"""
    dyn = dynamics()
    one = {**task, "success": {**task["success"], "seeds": [int(seed)]}}
    names = component_names(dyn.load_model(xml))
    frames: list[dict] = []
    with patched_engine(conditions, per_action=int(task["episode"]["solver_steps_per_action"]),
                        rng=_case_rng(seed, conditions)):
        result = dyn.evaluate_success(xml, one, container, components=names,
                                      on_trace=lambda _seed, doc: frames.extend(doc["frames"]))
    row = result["seeds"][0]
    out = {"seed": int(seed), "pass": bool(row["pass"]), "failing": row.get("failing"),
           "metrics": row["metrics"], "conditions": dataclasses.asdict(conditions)}
    if conditions.push is not None:
        out["push"] = push_recovery(frames, conditions.push, completed=row["metrics"].get("completed") == 1.0)
    return out


# --- 組（suite） -----------------------------------------------------------------

def cases(seeds: list[int], *, random_draws: int = 5, draw_seed: int = 0) -> list[tuple[str, str, int, Conditions]]:
    """(組, 条件の名前, 評価シード, 条件) の並び。"""
    out: list[tuple[str, str, int, Conditions]] = []
    out += [("nominal", "nominal", s, Conditions()) for s in seeds]
    out += [("sweep", name, s, c) for name, c in SWEEP for s in seeds]
    rng = random.Random(draw_seed)
    for _ in range(random_draws):
        for s in seeds:
            out.append(("random", "real ranges", s, sample_conditions(rng)))
    for newtons in PUSH_NEWTONS:
        for side in (+1, -1):
            push = Push(newtons=newtons, side=side, at_s=PUSH_AT_S, duration_s=PUSH_DURATION_S)
            out += [("push", f"push {newtons:g} N", s, Conditions(push=push)) for s in seeds]
    return out


_WORKER: dict[str, Any] = {}


def _init_worker(xml: bytes, task_text: str, policy: bytes) -> None:
    _WORKER.update(xml=xml, task=json.loads(task_text), container=dynamics().decode_policy(policy))


def _run_one(case: tuple[str, str, int, Conditions]) -> dict:
    suite, name, seed, conditions = case
    row = evaluate_case(_WORKER["xml"], _WORKER["task"], _WORKER["container"], seed=seed, conditions=conditions)
    return {"suite": suite, "name": name, **row}


def run(xml: bytes, task: dict, policy: bytes, *, random_draws: int = 5, processes: int | None = None,
        suites: tuple[str, ...] = ("nominal", "sweep", "random", "push")) -> list[dict]:
    """全部の組を並列に回して、行を返す（順番は ``cases`` と同じ）。"""
    import multiprocessing as mp  # noqa: PLC0415

    todo = [c for c in cases(list(task["success"]["seeds"]), random_draws=random_draws) if c[0] in suites]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=processes, initializer=_init_worker,
                  initargs=(xml, json.dumps(task), policy)) as pool:
        return pool.map(_run_one, todo, chunksize=1)


SUMMARY_METRICS = ("completed", "max_tilt_deg", "slip_share_max", "effort_rms_rated_ratio_max", "speed_ratio",
                   "steps_min")


def summarise(rows: list[dict]) -> dict:
    """(組, 条件の名前) ごとに、合格率・完走率・主な指標の平均、押しなら倒れなかった割合と戻った割合。"""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["suite"], r["name"]), []).append(r)
    out: dict[str, dict] = {}
    for (suite, name), rs in groups.items():
        entry = {"suite": suite, "name": name, "n": len(rs),
                 "pass_rate": sum(r["pass"] for r in rs) / len(rs),
                 "metrics": {}}
        for key in SUMMARY_METRICS:
            values = [float(r["metrics"][key]) for r in rs if r["metrics"].get(key) is not None]
            entry["metrics"][key] = mean(values) if values else None
        failing: dict[str, int] = {}
        for r in rs:
            for f in r.get("failing") or ():
                failing[str(f)] = failing.get(str(f), 0) + 1
        entry["failing"] = failing
        if suite == "push":
            entry["survived_rate"] = sum(r["push"]["survived"] for r in rs) / len(rs)
            entry["recovered_rate"] = sum(r["push"]["recovered"] for r in rs) / len(rs)
        out[f"{suite}/{name}"] = entry
    pushes = [r for r in rows if r["suite"] == "push"]
    if pushes:
        out["push/all"] = {"suite": "push", "name": "all", "n": len(pushes),
                           "pass_rate": sum(r["pass"] for r in pushes) / len(pushes),
                           "survived_rate": sum(r["push"]["survived"] for r in pushes) / len(pushes),
                           "recovered_rate": sum(r["push"]["recovered"] for r in pushes) / len(pushes),
                           "metrics": {}, "failing": {}}
    return out


def settings() -> dict:
    """結果の JSON に残す、評価の決まりごと。"""
    return {"real_ranges": {k: list(v) for k, v in REAL_RANGES.items()},
            "sweep": [{"name": n, "conditions": dataclasses.asdict(c)} for n, c in SWEEP],
            "push": {"newtons": list(PUSH_NEWTONS), "at_s": PUSH_AT_S, "duration_s": PUSH_DURATION_S,
                     "sides": ["left", "right"]},
            "recovered": {"tilt_deg_max": RECOVER_TILT_DEG, "hold_s": RECOVER_HOLD_S,
                          "within_s": RECOVER_WITHIN_S, "walk_window_after_push_end_s": list(WALK_WINDOW_S),
                          "walk_ratio_min": WALK_MIN_RATIO}}



# --- 学習シード方向のまとめと、2案の比較 ---------------------------------------

RATES = ("pass_rate", "survived_rate", "recovered_rate")
COMPARE_METRICS = ("slip_share_max", "effort_rms_rated_ratio_max", "max_tilt_deg", "speed_ratio")
RATE_LABELS = {"pass_rate": "合格率", "survived_rate": "倒れなかった", "recovered_rate": "倒れず戻った"}
METRIC_LABELS = {"slip_share_max": "滑り", "effort_rms_rated_ratio_max": "トルク比", "max_tilt_deg": "最大傾斜",
                 "speed_ratio": "速度比"}


def across(reports: list[dict]) -> dict:
    """同じ案の複数回（学習シード違い）の robustness.json → 組ごとに 平均・標準偏差・最小・最大。"""
    from .seeds import _spread  # noqa: PLC0415

    keys = list(dict.fromkeys(k for r in reports for k in r["summary"]))
    groups: dict[str, dict] = {}
    for key in keys:
        entries = [r["summary"][key] for r in reports if key in r["summary"]]
        group: dict[str, Any] = {"suite": entries[0]["suite"], "name": entries[0]["name"], "runs": len(entries)}
        for rate in RATES:
            values = [float(e[rate]) for e in entries if e.get(rate) is not None]
            if values:
                group[rate] = _spread(values)
        group["metrics"] = {}
        for metric in COMPARE_METRICS:
            values = [float(e["metrics"][metric]) for e in entries if (e.get("metrics") or {}).get(metric) is not None]
            if values:
                group["metrics"][metric] = _spread(values)
        groups[key] = group
    return {"n": len(reports), "groups": groups}


def compare_reports(a: dict, b: dict, *, labels: tuple[str, str]) -> dict:
    from .seeds import _row  # noqa: PLC0415

    out: dict[str, dict] = {}
    for key in dict.fromkeys([*a["groups"], *b["groups"]]):
        ga, gb = a["groups"].get(key), b["groups"].get(key)
        if not ga or not gb:
            continue
        row: dict[str, Any] = {"suite": ga["suite"], "name": ga["name"]}
        for rate in RATES:
            if ga.get(rate) and gb.get(rate):
                row[rate] = _row(ga[rate], gb[rate])
        row["metrics"] = {m: _row(ga["metrics"].get(m), gb["metrics"].get(m)) for m in COMPARE_METRICS
                          if ga["metrics"].get(m) and gb["metrics"].get(m)}
        out[key] = row
    return {"labels": list(labels), "n": [a["n"], b["n"]], "groups": out}


def compare_markdown(c: dict) -> str:
    from .seeds import _cell  # noqa: PLC0415

    a, b = c["labels"]
    lines = [f"# 実機との差の評価: {a} と {b}（学習シード {c['n'][0]} 回 / {c['n'][1]} 回）", "",
             "各回の値は、その組のエピソード（評価シード10個。random は 50、push は左右で 20）の平均。",
             "表は学習シード方向の 平均 ± 標準偏差（最小〜最大）。「幅」は2案の最小〜最大が重なるかどうか。", "",
             f"| 組 | 条件 | 項目 | {a} | {b} | 差（{b} − {a}） | 幅 |", "|---|---|---|---|---|---|---|"]
    for row in c["groups"].values():
        items = [(RATE_LABELS[r], row[r]) for r in RATES if row.get(r)]
        if row["suite"] in ("nominal", "random"):
            items += [(METRIC_LABELS[m], v) for m, v in row["metrics"].items() if v]
        elif row["suite"] == "sweep":
            items += [(METRIC_LABELS[m], v) for m, v in row["metrics"].items() if v and m == "slip_share_max"]
        for label, v in items:
            verdict = "重ならない" if v["separated"] else "重なる"
            lines.append(f"| {row['suite']} | {row['name']} | {label} | {_cell(v['a'])} | {_cell(v['b'])} | "
                         f"{v['difference']:+.2f} | {verdict} |")
    return "\n".join(lines) + "\n"
