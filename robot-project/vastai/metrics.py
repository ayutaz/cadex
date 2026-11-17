"""cadex_train.py の progress.json を CSV と TensorBoard のイベントファイルに直す。

progress.json（``cadex-training-progress-v1``）の曲線は ``*_curve`` に
``[[iteration, value], ...]`` で入っている。cadex_train は各曲線を
``CURVE_POINTS_CAP`` 点に間引くので、長い学習では全反復が残るわけではない
（最初と最後の点は必ず残る）。曲線ごとに間引かれる位置が揃うとは限らないので、
欠けた点は CSV では空欄、TensorBoard では書かない。

TensorBoard 形式は依存を増やさないため、イベントファイル（TFRecord に包んだ
``Event`` protobuf）を標準ライブラリだけで書く。tensorboard を入れれば
そのまま ``tensorboard --logdir`` で読める。
"""

from __future__ import annotations

import csv
import json
import struct
import time
from pathlib import Path
from typing import Any, Iterable

#: progress.json のキー → 列名。先頭の 4 つは cadex_train が必ず書く曲線。
CURVE_KEYS = {
    "curve": "reward_per_step",
    "loss_curve": "loss",
    "episode_steps_curve": "episode_steps",
    "action_std_curve": "action_std",
}


def load_progress(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _column(key: str) -> str:
    return CURVE_KEYS.get(key, key.removesuffix("_curve"))


def progress_rows(progress: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """(列名, 行) を返す。行は ``iteration`` 昇順で、欠けた値は None。"""

    keys = [key for key in CURVE_KEYS]
    keys += sorted(key for key in progress
                   if key.endswith("_curve") and key not in CURVE_KEYS
                   and isinstance(progress[key], list))
    columns = ["iteration"] + [_column(key) for key in keys]
    table: dict[int, dict[str, Any]] = {}
    for key in keys:
        for point in progress.get(key) or ():
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            iteration, value = int(point[0]), point[1]
            if isinstance(value, list):
                value = [float(v) for v in value]
            elif value is not None:
                value = float(value)
            table.setdefault(iteration, {})[_column(key)] = value
    rows = []
    for iteration in sorted(table):
        row = {column: None for column in columns}
        row.update(table[iteration])
        row["iteration"] = iteration
        rows.append(row)
    return columns, rows


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ";".join(repr(float(v)) for v in value)
    return repr(value) if isinstance(value, float) else str(value)


def write_csv(progress: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns, rows = progress_rows(progress)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(row[column]) for column in columns])
    temporary.replace(path)
    return path


# --- TensorBoard（依存なし） -------------------------------------------------

def _crc32c_table() -> list[int]:
    table = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0x82F63B78 if crc & 1 else crc >> 1
        table.append(crc)
    return table


_CRC32C = _crc32c_table()


def crc32c(data: bytes) -> int:
    """CRC-32C（Castagnoli）。TFRecord の検査に使う。"""

    crc = 0xFFFFFFFF
    for byte in data:
        crc = _CRC32C[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _masked_crc(data: bytes) -> int:
    crc = crc32c(data)
    return ((((crc >> 15) | (crc << 17)) & 0xFFFFFFFF) + 0xA282EAD8) & 0xFFFFFFFF


def _varint(value: int) -> bytes:
    out = bytearray()
    value &= (1 << 64) - 1
    while True:
        bits = value & 0x7F
        value >>= 7
        if value:
            out.append(bits | 0x80)
        else:
            out.append(bits)
            return bytes(out)


def _field_bytes(number: int, payload: bytes) -> bytes:
    return _varint((number << 3) | 2) + _varint(len(payload)) + payload


def _event(wall_time: float, step: int, *, file_version: str | None = None,
           tag: str | None = None, value: float | None = None) -> bytes:
    data = _varint((1 << 3) | 1) + struct.pack("<d", wall_time)   # wall_time: double
    data += _varint((2 << 3) | 0) + _varint(step)                    # step: int64
    if file_version is not None:
        data += _field_bytes(3, file_version.encode())               # file_version
    if tag is not None:
        summary_value = _field_bytes(1, tag.encode())                # Value.tag
        summary_value += _varint((2 << 3) | 5) + struct.pack("<f", value)  # simple_value
        data += _field_bytes(5, _field_bytes(1, summary_value))      # summary.value
    return data


def _record(data: bytes) -> bytes:
    length = struct.pack("<Q", len(data))
    return (length + struct.pack("<I", _masked_crc(length))
            + data + struct.pack("<I", _masked_crc(data)))


def write_tensorboard(progress: dict[str, Any], logdir: str | Path,
                      prefix: str = "train/") -> Path:
    """スカラーの曲線を ``logdir/events.out.tfevents.<時刻>.vastai`` に書く。"""

    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    path = logdir / f"events.out.tfevents.{int(now)}.vastai"
    columns, rows = progress_rows(progress)
    started = float(progress.get("started_at") or now)
    wall = float(progress.get("wall_time_s") or 0.0)
    last = rows[-1]["iteration"] if rows else 0
    with path.open("wb") as handle:
        handle.write(_record(_event(now, 0, file_version="brain.Event:2")))
        for row in rows:
            # 各反復の時刻は wall_time_s を反復数で割った直線で近似する
            stamp = started + (wall * row["iteration"] / last if last else 0.0)
            for column in columns[1:]:
                value = row[column]
                if value is None or isinstance(value, list):
                    continue
                handle.write(_record(_event(stamp, row["iteration"],
                                            tag=prefix + column, value=value)))
    return path


# --- 読み戻し（テストと確認用） --------------------------------------------

def _read_varint(data: bytes, index: int) -> tuple[int, int]:
    shift = result = 0
    while True:
        byte = data[index]
        index += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, index
        shift += 7


def _fields(data: bytes) -> Iterable[tuple[int, int, Any]]:
    index = 0
    while index < len(data):
        key, index = _read_varint(data, index)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, index = _read_varint(data, index)
        elif wire == 1:
            value, index = data[index:index + 8], index + 8
        elif wire == 5:
            value, index = data[index:index + 4], index + 4
        elif wire == 2:
            length, index = _read_varint(data, index)
            value, index = data[index:index + length], index + length
        else:
            raise ValueError(f"未対応の wire type {wire}")
        yield number, wire, value


def read_scalar_events(path: str | Path) -> list[tuple[str, int, float]]:
    """イベントファイルから (tag, step, value) を読む。CRC も確かめる。"""

    blob = Path(path).read_bytes()
    index, out = 0, []
    while index < len(blob):
        header = blob[index:index + 8]
        (length,) = struct.unpack("<Q", header)
        (header_crc,) = struct.unpack("<I", blob[index + 8:index + 12])
        if header_crc != _masked_crc(header):
            raise ValueError("TFRecord の長さの CRC が合いません")
        data = blob[index + 12:index + 12 + length]
        (data_crc,) = struct.unpack("<I", blob[index + 12 + length:index + 16 + length])
        if data_crc != _masked_crc(data):
            raise ValueError("TFRecord の本体の CRC が合いません")
        index += 16 + length
        step = 0
        for number, _, value in _fields(data):
            if number == 2:
                step = value
            elif number == 5:
                for summary_number, _, summary_value in _fields(value):
                    if summary_number != 1:
                        continue
                    tag, scalar = "", None
                    for value_number, _, item in _fields(summary_value):
                        if value_number == 1:
                            tag = item.decode()
                        elif value_number == 2:
                            (scalar,) = struct.unpack("<f", item)
                    if scalar is not None:
                        out.append((tag, step, scalar))
    return out
