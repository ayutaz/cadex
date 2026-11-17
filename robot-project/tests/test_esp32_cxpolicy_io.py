"""deploy/esp32/cxpolicy_io.py のテスト。

.cxpolicy の読み書きと numpy 参照順伝播が、Cadex 本体（エンジンの
CadexDynamics とトレーナーの cadex_train）と同じ数式・同じバイト列になることを確かめる。
本体はここから読むだけで、変更しない。
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from deploy.esp32 import cxpolicy_io as cx

ROOT = Path(__file__).resolve().parents[1]
CADEX = ROOT.parent


def _engine():
    """エンジンの CadexDynamics。読めない環境ではテストを飛ばす。"""

    path = str(CADEX / "src" / "Mod" / "cadex")
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        import CadexDynamics  # type: ignore
    except Exception as exc:  # pragma: no cover - 環境依存
        pytest.skip(f"CadexDynamics を読めない: {exc}")
    return CadexDynamics


def _trainer():
    path = str(CADEX / "training")
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        import cadex_train  # type: ignore
    except Exception as exc:  # pragma: no cover - 環境依存
        pytest.skip(f"cadex_train を読めない: {exc}")
    return cadex_train


@pytest.fixture(scope="module")
def blob() -> bytes:
    return cx.make_synthetic_policy(hidden=(64, 64), seed=3)


@pytest.fixture(scope="module")
def policy(blob) -> cx.CxPolicy:
    return cx.parse_policy(blob)


def test_container_layout(blob):
    assert blob.startswith(b"CXPOLICY1\n")
    length = int.from_bytes(blob[10:18], "little")
    header = json.loads(blob[18:18 + length].decode("ascii"))
    assert header["schema"] == "cadex-policy-v1"
    assert (len(blob) - 18 - length) % 4 == 0
    # 正準 JSON: 読んで書き直すと同じバイト列になる
    decoded_header, weights = cx.decode_policy(blob)
    assert cx.encode_policy(decoded_header, weights) == blob


def test_parsed_fields(policy):
    assert policy.layer_sizes == [20, 64, 64, 8]
    assert policy.activation == "swish"
    assert len(policy.observations) == 20
    assert len(policy.actions) == 8
    assert all(a["unit"] == "deg" for a in policy.actions)
    assert policy.layers[0][0].shape == (20, 64)
    assert policy.layers[-1][1].shape == (8,)
    assert policy.weights.dtype == np.float32
    assert policy.mean.shape == (20,) and policy.std.shape == (20,)
    assert policy.output_scale.shape == (8,)


def test_engine_decodes_the_same_numbers(blob, policy):
    engine = _engine()
    decoded = engine.decode_policy(blob)
    assert decoded["header"] == policy.header
    np.testing.assert_array_equal(np.asarray(decoded["weights"], np.float32), policy.weights)


def test_forward_matches_engine_policy_forward(blob, policy):
    engine = _engine()
    decoded = engine.decode_policy(blob)
    rng = np.random.default_rng(0)
    for _ in range(10):
        obs = policy.mean + policy.std * rng.normal(size=policy.mean.shape) * 2.0
        expected = engine.policy_forward(decoded["header"], decoded["weights"], obs.tolist())
        got = cx.forward(policy, obs)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-9)


def test_synthetic_file_passes_trainer_witness_check(blob):
    trainer = _trainer()
    header, weights = trainer.decode_policy(blob)["header"], trainer.decode_policy(blob)["weights"]
    worst, _sample, _action = trainer.witness_disagreement(header, weights)
    assert worst < 1e-4  # エンジンの POLICY_WITNESS_TOLERANCE


def test_witness_error_is_small_and_detects_tampering(policy, blob):
    assert cx.witness_error(policy)[0] < 1e-9
    header, weights = cx.decode_policy(blob)
    header["evaluation"]["actions"][0][0] += 10.0
    tampered = cx.parse_policy(cx.encode_policy(header, weights))
    assert cx.witness_error(tampered)[0] > 1e-3


def test_batch_forward_equals_single(policy):
    rng = np.random.default_rng(1)
    batch = policy.mean + policy.std * rng.normal(size=(5, 20))
    stacked = np.stack([cx.forward(policy, row) for row in batch])
    np.testing.assert_allclose(cx.forward(policy, batch), stacked, atol=1e-12)


def test_swish_is_stable():
    x = np.array([-1000.0, -40.0, -1.0, 0.0, 1.0, 40.0, 1000.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y = cx.swish(x)
    expected = [v / (1.0 + math.exp(-v)) if v >= 0 else v * math.exp(v) / (1.0 + math.exp(v))
                for v in x]
    np.testing.assert_allclose(y, expected, atol=1e-12)


@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_other_activations_match_engine(activation):
    engine = _engine()
    blob = cx.make_synthetic_policy(hidden=(16,), activation=activation, seed=5)
    policy = cx.parse_policy(blob)
    decoded = engine.decode_policy(blob)
    obs = policy.mean + 0.5 * policy.std
    np.testing.assert_allclose(
        cx.forward(policy, obs),
        engine.policy_forward(decoded["header"], decoded["weights"], obs.tolist()),
        atol=1e-9)


def test_rejects_bad_files(blob):
    with pytest.raises(cx.PolicyFormatError):
        cx.parse_policy(b"NOTAPOLICY" + blob[10:])
    with pytest.raises(cx.PolicyFormatError):
        cx.parse_policy(blob[:-3])
    header, weights = cx.decode_policy(blob)
    header["normaliser"]["std"][0] = 0.0
    with pytest.raises(cx.PolicyFormatError):
        cx.parse_policy(cx.encode_policy(header, weights))
    header, weights = cx.decode_policy(blob)
    with pytest.raises(cx.PolicyFormatError):
        cx.parse_policy(cx.encode_policy(header, weights[:-1]))


def test_command_filter_matches_engine_order():
    low = np.array([-10.0, -10.0])
    high = np.array([10.0, 10.0])
    f = cx.CommandFilter(low, high, alpha=0.5, slew_deg=2.0)
    # 1回目は clamp だけ
    np.testing.assert_allclose(f.step(np.array([20.0, -3.0])), [10.0, -3.0])
    # 2回目: clamp → alpha → slew
    # now=[-10, 5] → alpha: [0.5*-10+0.5*10, 0.5*5+0.5*-3] = [0, 1] → slew ±2: [8, -1]
    np.testing.assert_allclose(f.step(np.array([-50.0, 5.0])), [8.0, -1.0])
    f.reset()
    np.testing.assert_allclose(f.step(np.array([1.0, 1.0])), [1.0, 1.0])


def test_command_filter_reads_header(policy):
    f = cx.CommandFilter.from_policy(policy)
    assert f.alpha == policy.action_filter_alpha
    assert f.slew_deg == policy.command_slew_deg


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "deploy.esp32.cxpolicy_io", *args],
                          cwd=ROOT, capture_output=True, text=True)


def test_cli_synth_info_check(tmp_path):
    out = tmp_path / "s.cxpolicy"
    done = _cli("synth", "--out", str(out), "--hidden", "32", "32", "--seed", "1")
    assert done.returncode == 0, done.stderr
    info = _cli("info", str(out))
    assert info.returncode == 0, info.stderr
    assert "20-32-32-8" in info.stdout
    assert "swish" in info.stdout
    check = _cli("check", str(out))
    assert check.returncode == 0, check.stderr

    header, weights = cx.decode_policy(out.read_bytes())
    header["evaluation"]["actions"][0][0] += 10.0
    bad = tmp_path / "bad.cxpolicy"
    bad.write_bytes(cx.encode_policy(header, weights))
    assert _cli("check", str(bad)).returncode == 1
