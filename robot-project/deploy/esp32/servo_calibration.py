"""サーボの較正の結果を受け取り、JSON に保存し、ファームウェアと書き出しの道具に渡す形にする。

ファームウェアの試験・較正モード（``firmware/servo_cal.c``）で各サーボの中立・向き・範囲を測り、
``dump`` コマンドの出力（``CAL ...`` の行）をここで読む。そこから次を作る。

* 較正の JSON（``q8-servo-calibration/1``）: 測った値の保存先
* ``servo.json``: ``export_c --servo`` に渡す。方策の関節ごとのパルス幅の換算になる
* ``servo_map.h``: ファームウェアの関節の表（方策の行動の順、GPIO、向き、中立、範囲）

関節の順序は学習のタスク（``walk-task.json`` の ``actions``）の順で、方策の出力の順と同じ。
角度とパルス幅の関係は ``export_c.ServoConfig`` と同じ::

    pulse_us = center_us + sign * angle_deg * us_per_deg

関節角 0 は組み立てた立ち姿勢、正の角度は足先が後ろへ動く向き（股・膝とも。膝の正は脚を伸ばす向き）。

使い方::

    python -m deploy.esp32.servo_calibration nominal --out cal.json        # 名目値（未較正）
    python -m deploy.esp32.servo_calibration read --port /dev/cu.usbmodem1101 --out cal.json
    python -m deploy.esp32.servo_calibration parse monitor.log --out cal.json  # 貼り付けた出力から
    python -m deploy.esp32.servo_calibration check cal.json
    python -m deploy.esp32.servo_calibration servo-json cal.json --out servo.json
    python -m deploy.esp32.servo_calibration header cal.json --out servo_map.h
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
#: 作る機体（mini-r2-p7-eff）の学習タスク。関節の順序と範囲はここから取る。
DEFAULT_TASK = ROOT / "robot" / "mjcf" / "mini-r2-p7-eff" / "cut" / "walk-task.json"
SCHEMA = "q8-servo-calibration/1"

#: 信号線の GPIO（仮）。docs/assembly-guide.md §5 の表と同じ。方策の行動の順。
DEFAULT_GPIO = (4, 5, 6, 7, 15, 16, 17, 18)

HARD_MIN_US = 500.0
HARD_MAX_US = 2400.0
NOMINAL_CENTER_US = 1450.0
NOMINAL_US_PER_DEG = (2400.0 - 500.0) / 180.0
#: 未測定のあいだ、ファームウェアが学習の範囲の外へ許す余裕 [度]（可動域は接触の 10° 手前に置いてある）
EXPLORE_MARGIN_DEG = 10.0

#: 正の関節角で起きる動き（MJCF の運動学。tests/test_servo_calibration.py が固定する）。
POSITIVE_MOTION = {"hip": "足先が後ろへ", "knee": "足先が後ろへ"}
#: 出力軸（ホーン側）のワールド Y 方向。股・膝とも外向き（docs/assembly-guide.md §4.2、§4.4）。
SHAFT_OUTWARD_Y = {"l": 1.0, "r": -1.0}
#: 仮定: パルスを増やすと、ホーン側から見て反時計回り（出力軸まわりの右ねじの向き）に回る。
#: MG90S で確かめていない。逆なら全関節の名目の符号が反対になるだけで、較正（flip / mark）で直る。
PULSE_UP_TURNS_CCW = 1.0

#: 中立がこれ以上ずれていたら、ホーンを 1 歯差し直すほうがよい [µs]（1 歯 ≈ 17°）。
CENTER_WARN_US = 120.0
UPD_RANGE = (6.0, 16.0)


@dataclass
class JointCal:
    name: str
    actuator: str
    gpio: int
    center_us: float
    sign: float
    us_per_deg: float
    lo_us: float  # 関節角の下限の端（足先が前の端）のパルス幅。0 = 未測定
    hi_us: float  # 関節角の上限の端（足先が後ろの端）のパルス幅。0 = 未測定
    model_low_deg: float
    model_high_deg: float

    def angle_to_us(self, deg: float) -> float:
        return self.center_us + self.sign * deg * self.us_per_deg

    def us_to_angle(self, us: float) -> float:
        return (us - self.center_us) / (self.sign * self.us_per_deg)

    def measured(self) -> bool:
        return self.lo_us > 0.0 and self.hi_us > 0.0

    def measured_range_deg(self) -> tuple[float, float]:
        return self.us_to_angle(self.lo_us), self.us_to_angle(self.hi_us)


@dataclass
class Calibration:
    joints: list[JointCal]
    source: str = "measured"  # "nominal"（名目値）か "measured"（ファームウェアから）
    saved: bool | None = None  # ファームウェアで保存済みだったか（dump の saved=）
    task: str = ""
    task_sha256: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TaskJoint:
    name: str
    actuator: str
    model_low_deg: float
    model_high_deg: float


# ---------------------------------------------------------------------------
# タスク・名目値
# ---------------------------------------------------------------------------


def _task(task: str | Path) -> dict[str, Any]:
    return json.loads(Path(task).read_text(encoding="utf-8"))


def _rel(path: str | Path) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def task_joints(task: str | Path = DEFAULT_TASK) -> list[TaskJoint]:
    """学習タスクの行動の順に、関節名・アクチュエータ名・行動の範囲[度]。"""

    out = []
    for a in _task(task)["actions"]:
        if a.get("unit") != "deg":
            raise ValueError(f"{a.get('actuator')}: 単位が deg でない（{a.get('unit')}）")
        joint = str(a.get("joint") or str(a["actuator"]).split("/")[0])
        out.append(TaskJoint(joint, str(a["actuator"]), float(a["low"]), float(a["high"])))
    return out


def nominal_sign(joint: str) -> float:
    side = joint.split("_")[-1][0]
    return SHAFT_OUTWARD_Y[side] * PULSE_UP_TURNS_CCW


def nominal(task: str | Path = DEFAULT_TASK) -> Calibration:
    joints = task_joints(task)
    if len(joints) != len(DEFAULT_GPIO):
        raise ValueError(f"関節が {len(joints)} 個（GPIO の表は {len(DEFAULT_GPIO)} 個）")
    return Calibration(
        joints=[JointCal(j.name, j.actuator, gpio, NOMINAL_CENTER_US, nominal_sign(j.name),
                         NOMINAL_US_PER_DEG, 0.0, 0.0, j.model_low_deg, j.model_high_deg)
                for j, gpio in zip(joints, DEFAULT_GPIO)],
        source="nominal", task=_rel(task), task_sha256=_sha256(task))


def _sha256(path: str | Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# ファームウェアの dump を読む
# ---------------------------------------------------------------------------

_BEGIN = re.compile(r"CAL begin v=(\d+) n=(\d+) saved=([01])")
_ROW = re.compile(r"CAL (\d+) (\w+) gpio=(\d+) center=([-\d.]+) sign=([-+]?\d+) "
                  r"upd=([-\d.]+) lo=([-\d.]+) hi=([-\d.]+)")


def parse_dump(text: str, task: str | Path = DEFAULT_TASK) -> Calibration:
    """``dump`` の出力（シリアルのログ全体でよい）の最後の ``CAL begin``〜``CAL end`` を読む。"""

    lines = [line.strip() for line in text.splitlines()]
    starts = [k for k, line in enumerate(lines) if _BEGIN.search(line)]
    if not starts:
        raise ValueError("CAL begin の行が無い（ファームウェアで dump を打つ）")
    start = starts[-1]
    m = _BEGIN.search(lines[start])
    version, count, saved = int(m.group(1)), int(m.group(2)), m.group(3) == "1"
    if version != 1:
        raise ValueError(f"dump の版 {version} は読めない")
    rows = []
    for line in lines[start + 1:]:
        if line.endswith("CAL end"):
            break
        r = _ROW.search(line)
        if r:
            rows.append(r)
    else:
        raise ValueError("CAL end の行が無い（出力が途中で切れている）")
    joints = task_joints(task)
    if len(rows) != count or count != len(joints):
        raise ValueError(f"関節が {len(rows)} 行（dump は n={count}、タスクは {len(joints)} 関節）")
    out = []
    for k, (r, tj) in enumerate(zip(rows, joints)):
        index, name = int(r.group(1)), r.group(2)
        if index != k or name != tj.name:
            raise ValueError(f"{k} 番目が {name}（タスクの順では {tj.name}）。関節の順が学習と違う")
        out.append(JointCal(name, tj.actuator, int(r.group(3)), float(r.group(4)),
                            float(r.group(5)), float(r.group(6)), float(r.group(7)),
                            float(r.group(8)), tj.model_low_deg, tj.model_high_deg))
    return Calibration(joints=out, source="measured", saved=saved, task=_rel(task),
                       task_sha256=_sha256(task))


def read_serial(port: str, *, baud: int = 115200, timeout_s: float = 15.0,
                wait_s: float = 2.0) -> str:
    """シリアルに ``dump`` を送り、``CAL end`` までの出力を返す（macOS・Linux。標準ライブラリだけ）。

    ポートを開くとボードが再起動することがある（USB シリアル変換の自動リセット）。保存していない較正は
    消えるので、先にファームウェアで ``save`` しておく。
    """

    import os
    import termios

    fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        attrs = termios.tcgetattr(fd)
        speed = getattr(termios, f"B{baud}")
        attrs[0] = 0                                   # iflag
        attrs[1] = 0                                   # oflag
        attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        attrs[3] = 0                                   # lflag（エコーなし、行単位にしない）
        attrs[4] = attrs[5] = speed
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        buf = bytearray()
        end = time.monotonic() + wait_s                # 起動の表示を読み捨てる
        while time.monotonic() < end:
            buf += _read_some(fd)
        os.write(fd, b"\ndump\n")
        end = time.monotonic() + timeout_s
        text = ""
        while time.monotonic() < end:
            buf += _read_some(fd)
            text = buf.decode("utf-8", errors="replace")
            tail = text[text.rfind("CAL begin"):] if "CAL begin" in text else ""
            if "CAL end" in tail:
                return text
        raise TimeoutError(f"{timeout_s} 秒待っても CAL end が来ない（ポートと baud を確かめる）")
    finally:
        os.close(fd)


def _read_some(fd: int) -> bytes:
    import os
    import select

    ready, _, _ = select.select([fd], [], [], 0.05)
    if not ready:
        return b""
    try:
        return os.read(fd, 4096)
    except BlockingIOError:
        return b""


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def to_dict(cal: Calibration) -> dict[str, Any]:
    return {"schema": SCHEMA, "source": cal.source, "saved": cal.saved, "task": cal.task,
            "task_sha256": cal.task_sha256, "notes": list(cal.notes),
            "convention": "pulse_us = center_us + sign * angle_deg * us_per_deg; "
                          "angle 0 = standing pose; positive angle moves the foot backward; "
                          "lo_us/hi_us = pulse at the low/high angle end (0 = not measured)",
            "joints": [asdict(j) for j in cal.joints]}


def from_dict(data: dict[str, Any]) -> Calibration:
    if data.get("schema") != SCHEMA:
        raise ValueError(f"schema が {data.get('schema')!r}（{SCHEMA} を期待）")
    return Calibration(joints=[JointCal(**j) for j in data["joints"]], source=data["source"],
                       saved=data.get("saved"), task=data.get("task", ""),
                       task_sha256=data.get("task_sha256", ""), notes=list(data.get("notes", [])))


def _write(path: str | Path, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def save_json(cal: Calibration, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(to_dict(cal), ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


def load_json(path: str | Path) -> Calibration:
    return from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# 検査
# ---------------------------------------------------------------------------


def check(cal: Calibration, task: str | Path = DEFAULT_TASK,
          *, tolerance_deg: float = 0.5) -> tuple[list[str], list[str]]:
    """(エラー, 注意)。エラーがあれば servo.json を作らない。"""

    errors: list[str] = []
    warnings: list[str] = []
    expected = task_joints(task)
    if [j.name for j in cal.joints] != [t.name for t in expected]:
        errors.append("関節の順が学習のタスクと違う: "
                      f"{[j.name for j in cal.joints]} / 学習 {[t.name for t in expected]}")
        return errors, warnings
    gpios = [j.gpio for j in cal.joints]
    if len(set(gpios)) != len(gpios):
        errors.append(f"GPIO が重なっている: {gpios}")
    for j, t in zip(cal.joints, expected):
        n = j.name
        if j.sign not in (1.0, -1.0):
            errors.append(f"{n}: 向き（sign）が {j.sign}（+1 か -1）")
            continue
        if not UPD_RANGE[0] <= j.us_per_deg <= UPD_RANGE[1]:
            errors.append(f"{n}: us_per_deg {j.us_per_deg:.2f} が {UPD_RANGE[0]}〜{UPD_RANGE[1]} の外")
        if not HARD_MIN_US <= j.center_us <= HARD_MAX_US:
            errors.append(f"{n}: 中立 {j.center_us:.0f} µs が {HARD_MIN_US:.0f}〜{HARD_MAX_US:.0f} の外")
        elif abs(j.center_us - NOMINAL_CENTER_US) > CENTER_WARN_US:
            warnings.append(f"{n}: 中立が {j.center_us:.0f} µs（名目 {NOMINAL_CENTER_US:.0f} から "
                            f"{j.center_us - NOMINAL_CENTER_US:+.0f} µs）。ホーンを1歯差し直すとよい")
        if not j.measured():
            errors.append(f"{n}: 範囲が未測定（lo と hi を記録する）")
            continue
        for end, us in (("lo", j.lo_us), ("hi", j.hi_us)):
            if not HARD_MIN_US <= us <= HARD_MAX_US:
                errors.append(f"{n}: {end} {us:.0f} µs が {HARD_MIN_US:.0f}〜{HARD_MAX_US:.0f} の外")
        lo_deg, hi_deg = j.measured_range_deg()
        if not lo_deg < 0.0 < hi_deg:
            errors.append(f"{n}: 範囲の端の向きが sign と合わない（lo {lo_deg:+.1f}°, hi {hi_deg:+.1f}°）。"
                          "flip して測り直す")
            continue
        if lo_deg > t.model_low_deg + tolerance_deg:
            errors.append(f"{n}: 下限 {lo_deg:+.1f}° まで（学習は {t.model_low_deg:+.0f}° まで使う）")
        if hi_deg < t.model_high_deg - tolerance_deg:
            errors.append(f"{n}: 上限 {hi_deg:+.1f}° まで（学習は {t.model_high_deg:+.0f}° まで使う）")
    if cal.source == "measured" and cal.saved is False:
        warnings.append("ファームウェアで save していない値（再起動すると消える）")
    return errors, warnings


# ---------------------------------------------------------------------------
# 出力
# ---------------------------------------------------------------------------


def servo_json(cal: Calibration, *, task: str | Path | None = None,
               force: bool = False) -> dict[str, Any]:
    """``export_c --servo`` の形。キーはアクチュエータ名（``hip_lf/position``）。"""

    if task is not None and not force:
        errors, _ = check(cal, task)
        if errors:
            raise ValueError("較正にエラーがある: " + "; ".join(errors))
    joints = {}
    for j in cal.joints:
        lo, hi = (sorted((j.lo_us, j.hi_us)) if j.measured() else (HARD_MIN_US, HARD_MAX_US))
        joints[j.actuator] = {"center_us": j.center_us, "sign": j.sign,
                              "us_per_deg": j.us_per_deg, "trim_deg": 0.0,
                              "min_us": lo, "max_us": hi}
    return {"source": f"{SCHEMA} ({cal.source})", "task": cal.task, "joints": joints}


def _cf(v: float) -> str:
    s = f"{float(v):.6f}".rstrip("0")
    return (s + "0" if s.endswith(".") else s) + "f"


def render_servo_map_h(cal: Calibration) -> str:
    calibrated = cal.source == "measured" and not check(cal, ROOT / cal.task
                                                         if cal.task else DEFAULT_TASK)[0]
    lines = [
        "/*",
        " * servo_map.h — deploy/esp32/servo_calibration.py が生成。手で編集しない。",
        f" * 元: {cal.source}{'（名目値。未較正）' if cal.source == 'nominal' else ''}"
        f"  タスク: {cal.task}",
        " *",
        " * 方策の行動の順。角度[度] → パルス幅: center_us + sign * angle * us_per_deg",
        " * 関節角 0 は立った姿勢、正の角度で足先が後ろへ動く。lo/hi は範囲の端のパルス幅（0 = 未測定）。",
        " *",
        " *   #  関節      GPIO  向き  中立[µs]  µs/度    lo[µs]   hi[µs]  学習の範囲[度]",
    ]
    for k, j in enumerate(cal.joints):
        lines.append(f" *  {k:2d}  {j.name:<9} {j.gpio:4d}  {int(j.sign):+d}   {j.center_us:8.1f}  "
                     f"{j.us_per_deg:6.3f}  {j.lo_us:7.1f}  {j.hi_us:7.1f}  "
                     f"{j.model_low_deg:+.0f}〜{j.model_high_deg:+.0f}")
    lines += [" */", "", "#ifndef SERVO_MAP_H", "#define SERVO_MAP_H", "",
              f"#define SERVO_MAP_N {len(cal.joints)}",
              f"#define SERVO_MAP_CALIBRATED {1 if calibrated else 0}",
              f"#define SERVO_MAP_HARD_MIN_US {_cf(HARD_MIN_US)}",
              f"#define SERVO_MAP_HARD_MAX_US {_cf(HARD_MAX_US)}",
              f"#define SERVO_MAP_EXPLORE_MARGIN_DEG {_cf(EXPLORE_MARGIN_DEG)}", ""]

    def row(name: str, values: Sequence[str]) -> None:
        lines.append(f"#define SERVO_MAP_{name} {{{', '.join(values)}}}")

    row("NAMES", [f'"{j.name}"' for j in cal.joints])
    row("GPIO", [str(j.gpio) for j in cal.joints])
    row("SIGN", [_cf(j.sign) for j in cal.joints])
    row("CENTER_US", [_cf(j.center_us) for j in cal.joints])
    row("US_PER_DEG", [_cf(j.us_per_deg) for j in cal.joints])
    row("LO_US", [_cf(j.lo_us) for j in cal.joints])
    row("HI_US", [_cf(j.hi_us) for j in cal.joints])
    row("MODEL_LOW_DEG", [_cf(j.model_low_deg) for j in cal.joints])
    row("MODEL_HIGH_DEG", [_cf(j.model_high_deg) for j in cal.joints])
    lines += ["", "#endif /* SERVO_MAP_H */", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_table(cal: Calibration) -> None:
    print(f"{'関節':<9} {'GPIO':>4} {'向き':>3} {'中立[µs]':>8} {'µs/度':>6} "
          f"{'範囲[度]':>15} {'学習[度]':>11}")
    for j in cal.joints:
        rng = ("未測定" if not j.measured() else
               "{:+.1f}〜{:+.1f}".format(*j.measured_range_deg()))
        print(f"{j.name:<9} {j.gpio:4d} {int(j.sign):+3d} {j.center_us:8.1f} {j.us_per_deg:6.2f} "
              f"{rng:>15} {j.model_low_deg:+4.0f}〜{j.model_high_deg:+.0f}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="servo_calibration",
                                     description="サーボの較正の結果（JSON・servo.json・servo_map.h）")
    parser.add_argument("--task", default=str(DEFAULT_TASK), help="学習タスク（関節の順と範囲）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("nominal", help="名目値（未較正）の JSON を作る")
    p.add_argument("--out", required=True)
    p = sub.add_parser("read", help="シリアルから dump を読んで JSON にする")
    p.add_argument("--port", required=True, help="例: /dev/cu.usbmodem1101、/dev/cu.usbserial-0001")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument("--log", help="受け取った出力も保存する")
    p.add_argument("--out", required=True)
    p = sub.add_parser("parse", help="保存したシリアルのログ（dump の出力）から JSON にする")
    p.add_argument("log")
    p.add_argument("--out", required=True)
    p = sub.add_parser("check", help="較正を検査する（エラーがあれば終了コード 1）")
    p.add_argument("cal")
    p = sub.add_parser("servo-json", help="export_c --servo に渡す servo.json を作る")
    p.add_argument("cal")
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true", help="エラーがあっても作る")
    p = sub.add_parser("header", help="ファームウェアの servo_map.h を作る")
    p.add_argument("cal")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.cmd == "nominal":
            cal = nominal(args.task)
            save_json(cal, args.out)
            print(f"書いた: {args.out}（名目値。未較正）")
            return 0
        if args.cmd in ("read", "parse"):
            if args.cmd == "read":
                text = read_serial(args.port, baud=args.baud, timeout_s=args.timeout)
                if args.log:
                    _write(args.log, text)
            else:
                text = Path(args.log).read_text(encoding="utf-8", errors="replace")
            cal = parse_dump(text, args.task)
            save_json(cal, args.out)
            print(f"書いた: {args.out}")
            _print_table(cal)
            return 0
        cal = load_json(args.cal)
        if args.cmd == "check":
            _print_table(cal)
            errors, warnings = check(cal, args.task)
            for w in warnings:
                print(f"注意: {w}")
            for e in errors:
                print(f"エラー: {e}")
            print("合格" if not errors else f"不合格（エラー {len(errors)} 件）")
            return 1 if errors else 0
        if args.cmd == "servo-json":
            data = servo_json(cal, task=args.task, force=args.force)
            _write(args.out, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
            print(f"書いた: {args.out}（export_c --servo に渡す）")
            return 0
        if args.cmd == "header":
            _write(args.out, render_servo_map_h(cal))
            print(f"書いた: {args.out}")
            return 0
    except (OSError, ValueError, TimeoutError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
