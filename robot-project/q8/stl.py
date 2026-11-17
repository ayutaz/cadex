"""STLの読み書き（ASCII/バイナリ）。CadexはASCII STLを書き、MuJoCoはバイナリしか読めない。"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def read_stl(path: Path) -> np.ndarray:
    """三角形の頂点配列 (n, 3, 3) を返す（単位はファイルのまま）。"""
    raw = path.read_bytes()
    if raw[:5].lower() == b"solid" and b"facet" in raw[:512]:
        verts = [line.split()[1:4] for line in raw.decode("ascii", "replace").splitlines()
                 if line.strip().startswith("vertex")]
        arr = np.asarray(verts, dtype=np.float64)
        return arr.reshape(-1, 3, 3)
    n = struct.unpack("<I", raw[80:84])[0]
    rec = np.frombuffer(raw[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return rec["v"].astype(np.float64)


def write_binary_stl(tris: np.ndarray, path: Path) -> None:
    v = tris.astype(np.float32)
    normals = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, norm, out=np.zeros_like(normals), where=norm > 0)
    rec = np.zeros(len(v), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"] = normals
    rec["v"] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"binary STL written by q8".ljust(80, b" "))
        f.write(struct.pack("<I", len(v)))
        f.write(rec.tobytes())


def to_binary_dir(src_dir: Path, dst_dir: Path) -> list[Path]:
    out = []
    for stl in sorted(src_dir.glob("*.stl")):
        dst = dst_dir / stl.name
        write_binary_stl(read_stl(stl), dst)
        out.append(dst)
    return out


def mesh_stats(path: Path) -> dict:
    """体積 (mm^3)、外形寸法 (mm)、三角形数。閉じたメッシュを前提にした符号付き体積。"""
    t = read_stl(path)
    vol = float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)
    lo = t.reshape(-1, 3).min(axis=0)
    hi = t.reshape(-1, 3).max(axis=0)
    return {"triangles": int(len(t)), "volume_mm3": abs(vol), "bbox_mm": (hi - lo).round(3).tolist(),
            "bbox_min_mm": lo.round(3).tolist(), "bbox_max_mm": hi.round(3).tolist()}
