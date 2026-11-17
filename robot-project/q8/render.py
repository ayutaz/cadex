"""MuJoCoのオフスクリーン描画（静止画と動画）。動画は ffmpeg へフレームを流す。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np

VIEWS = {
    # 名前: (azimuth, elevation, distance[m])
    "iso": (135.0, -25.0, 0.42),
    "front": (180.0, -5.0, 0.38),
    "side": (90.0, -5.0, 0.40),
    "top": (90.0, -89.0, 0.40),
    "rear_iso": (-45.0, -25.0, 0.42),
}


def camera(azimuth: float, elevation: float, distance: float, lookat) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.distance = distance
    cam.lookat[:] = lookat
    return cam


def save_png(rgb: np.ndarray, path: Path) -> None:
    """標準ライブラリだけでPNGを書く（Pillowに依存しない）。"""
    import struct
    import zlib

    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def render_views(model: mujoco.MjModel, data: mujoco.MjData, out_dir: Path, *,
                 width: int = 960, height: int = 640, show_collision: bool = False) -> list[Path]:
    renderer = mujoco.Renderer(model, height, width)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 1 if show_collision else 0, 0, 0]
    lookat = data.subtree_com[1].copy()
    paths = []
    for name, (az, el, dist) in VIEWS.items():
        renderer.update_scene(data, camera(az, el, dist, lookat), scene_option=opt)
        path = out_dir / f"view_{name}{'_collision' if show_collision else ''}.png"
        save_png(renderer.render(), path)
        paths.append(path)
    renderer.close()
    return paths


class VideoWriter:
    """RGBフレームを ffmpeg に渡して mp4 (H.264) を書く。"""

    def __init__(self, path: Path, width: int, height: int, fps: int):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg が見つからない")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
             "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(path)],
            stdin=subprocess.PIPE)
        self.path = path

    def write(self, rgb: np.ndarray) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(rgb).tobytes())

    def close(self) -> Path:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg が失敗した")
        return self.path
