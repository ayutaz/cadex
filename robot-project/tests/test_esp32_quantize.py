"""deploy/esp32/quantize.py のテスト: int8 量子化、較正、誤差評価、swish 近似。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from deploy.esp32 import cxpolicy_io as cx
from deploy.esp32 import quantize as qz

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def policy() -> cx.CxPolicy:
    return cx.parse_policy(cx.make_synthetic_policy(hidden=(64, 64), seed=11))


@pytest.fixture(scope="module")
def samples(policy) -> np.ndarray:
    return qz.calibration_samples(policy, count=1024, seed=0)


def test_symmetric_per_channel():
    rng = np.random.default_rng(0)
    # 列（出力チャネル）ごとに桁の違う重み
    w = rng.normal(size=(30, 6)) * np.array([0.01, 0.1, 1.0, 3.0, 0.5, 0.05])
    q, scale = qz.quantize_symmetric(w, per_channel=True)
    assert q.dtype == np.int8 and q.shape == (6, 30)  # (出力, 入力) に転置して持つ
    assert scale.shape == (6,)
    assert int(np.abs(q.astype(int)).max()) <= 127
    back = q.astype(np.float64) * scale[:, None]
    assert np.all(np.abs(back - w.T) <= scale[:, None] / 2 + 1e-12)


def test_per_channel_beats_per_layer():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(40, 8)) * np.geomspace(0.01, 2.0, 8)
    qc, sc = qz.quantize_symmetric(w, per_channel=True)
    ql, sl = qz.quantize_symmetric(w, per_channel=False)
    assert sl.shape == (8,) and np.all(sl == sl[0])
    err_c = np.abs(qc * sc[:, None] - w.T).mean()
    err_l = np.abs(ql * sl[:, None] - w.T).mean()
    assert err_c < err_l


def test_calibration_samples_include_witness(policy, samples):
    assert samples.shape == (1024, 20)
    witness = np.asarray(policy.header["evaluation"]["observations"])
    np.testing.assert_array_equal(samples[: len(witness)], witness)


def test_quantized_policy_structure(policy, samples):
    qp = qz.quantize_policy(policy, samples, per_channel=True)
    assert len(qp.layers) == 3
    for index, layer in enumerate(qp.layers):
        assert layer.w_q.dtype == np.int8
        assert layer.bias_q.dtype == np.int32
        assert layer.mult.dtype == np.float32
        if index < 2:
            assert layer.lut is not None and layer.lut.shape == (256,)
            assert layer.lut.dtype == np.int8
        else:
            assert layer.lut is None


def test_int8_forward_close_to_float(policy, samples):
    qp = qz.quantize_policy(policy, samples[:512], per_channel=True)
    ref = cx.forward(policy, samples[512:])
    got = qz.forward_int8(qp, samples[512:])
    span = policy.action_high - policy.action_low
    # スケールの取り違えなどの大きな誤りを捕まえる閾値。合成ポリシーでの実測は
    # 平均 0.4%、最大 5% 程度（範囲比）。実物の値は quantize の CLI で見る。
    rel = np.abs(got - ref) / span
    assert rel.mean() < 0.01
    assert rel.max() < 0.10


def test_weight_only_more_accurate_than_full_int8(policy, samples):
    qp = qz.quantize_policy(policy, samples[:512], per_channel=True)
    ref = cx.forward(policy, samples[512:])
    w8 = qz.forward_weights_int8(qp, samples[512:])
    a8 = qz.forward_int8(qp, samples[512:])
    assert np.abs(w8 - ref).max() < np.abs(a8 - ref).max()
    assert np.abs(w8 - ref).max() > 0  # 量子化しているなら誤差はゼロではない


def test_int8_forward_is_integer_exact(policy, samples):
    """C 実装が真似る中間値（int8 入力、int32 累積）を返せる。"""

    qp = qz.quantize_policy(policy, samples[:256])
    trace = qz.forward_int8(qp, samples[0], trace=True)
    assert trace["q_in"].dtype == np.int8
    assert all(a.dtype == np.int32 for a in trace["acc"])
    assert trace["raw"].dtype == np.float32


def test_swish_approximations():
    x = np.linspace(-10, 10, 4001)
    exact = cx.swish(x)
    hard = qz.hard_swish(x)
    pwl = qz.swish_pwl(x, segments=32, limit=8.0)
    # 折れ線補間の誤差は h^2/8 * max|swish''| = 0.5^2/8 * 0.5 ≈ 0.0156
    assert np.abs(pwl - exact).max() < 0.016
    assert np.abs(pwl - exact).max() < np.abs(hard - exact).max()
    # 区間外は 0 と x に漸近
    assert abs(qz.swish_pwl(np.array([-50.0]))[0]) < 1e-12
    assert qz.swish_pwl(np.array([50.0]))[0] == pytest.approx(50.0)


def test_lut_matches_activation_within_one_step():
    pre, out = 0.06, 0.05
    lut = qz.build_lut("swish", pre, out)
    p = np.arange(-128, 128)
    expected = cx.swish(p * pre)
    assert np.all(np.abs(lut.astype(float) * out - np.clip(expected, -127 * out, 127 * out))
                  <= out / 2 + 1e-9)


def test_evaluate_report(policy, samples):
    report = qz.evaluate(policy, samples, per_channel=True)
    modes = report["modes"]
    for name in ("float32", "w8", "w8a8_lut", "float_hard_swish", "float_pwl_swish"):
        assert name in modes, name
        row = modes[name]
        assert {"max_abs_deg", "mean_abs_deg", "max_rel_range"} <= set(row)
    assert modes["float32"]["max_abs_deg"] < 1e-3
    assert report["unit"] == "deg"
    assert report["layer_sizes"] == [20, 64, 64, 8]


def test_cli_prints_table_and_json(tmp_path):
    path = tmp_path / "p.cxpolicy"
    path.write_bytes(cx.make_synthetic_policy(hidden=(32, 32), seed=2))
    out = tmp_path / "r.json"
    done = subprocess.run(
        [sys.executable, "-m", "deploy.esp32.quantize", str(path), "--count", "512",
         "--json", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "w8a8_lut" in done.stdout
    assert "最大誤差" in done.stdout
    report = json.loads(out.read_text())
    assert report["modes"]["w8"]["max_abs_deg"] >= 0


def test_cli_accepts_samples_file(tmp_path, policy):
    path = tmp_path / "p.cxpolicy"
    path.write_bytes(cx.make_synthetic_policy(hidden=(64, 64), seed=11))
    npy = tmp_path / "obs.npy"
    np.save(npy, qz.calibration_samples(policy, count=300, seed=4))
    done = subprocess.run(
        [sys.executable, "-m", "deploy.esp32.quantize", str(path), "--samples", str(npy)],
        cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "300" in done.stdout
