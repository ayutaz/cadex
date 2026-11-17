"""deploy/esp32/budget.py のテスト: MAC 数、重みサイズ、推論時間の見積り。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from deploy.esp32 import budget as bd

ROOT = Path(__file__).resolve().parents[1]


def test_macs_and_params():
    sizes = [20, 64, 64, 8]
    assert bd.macs(sizes) == 20 * 64 + 64 * 64 + 64 * 8
    assert bd.parameters(sizes) == bd.macs(sizes) + 64 + 64 + 8


def test_weight_bytes():
    sizes = [20, 512, 256, 128, 8]
    p = bd.parameters(sizes)
    assert bd.weight_bytes(sizes, "f32") == 4 * p
    # int8: 重み 1 バイト、バイアス int32、出力ごとの倍率 float32、隠れ層ごとの LUT 256 バイト
    weights = bd.macs(sizes)
    outputs = sum(sizes[1:])
    assert bd.weight_bytes(sizes, "i8") == weights + 8 * outputs + 256 * (len(sizes) - 2)


def test_estimate_monotonic_and_has_basis():
    small = bd.estimate([20, 64, 64, 8])
    large = bd.estimate([20, 512, 256, 128, 8])
    for mode in ("f32_c", "i8_c", "i8_simd"):
        lo, hi = small["time_us"][mode]
        assert 0 < lo <= hi
        assert large["time_us"][mode][0] > small["time_us"][mode][0]
    assert small["fits_sram"]["f32"] is True
    assert large["fits_sram"]["f32"] is False
    assert bd.ASSUMPTIONS["clock_mhz"] == 240
    assert all("根拠" in row for row in bd.ASSUMPTIONS["cycles_per_mac"].values())


@pytest.mark.parametrize("sizes", [[20, 64, 64, 8], [20, 128, 64, 8]])
def test_small_networks_fit_50hz_budget(sizes):
    est = bd.estimate(sizes)
    # 20 ms 周期の 5% 未満（悲観側の値で）
    assert est["time_us"]["f32_c"][1] < 1000


def test_cli_table():
    done = subprocess.run([sys.executable, "-m", "deploy.esp32.budget"], cwd=ROOT,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "20-512-256-128-8" in done.stdout
    assert "20-64-64-8" in done.stdout
    assert "MAC" in done.stdout
