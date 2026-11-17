"""ポリシーの int8 量子化と、float 参照に対する誤差評価。

方式（ESP-NN / TFLite Micro の int8 と同じ型の対称量子化）:

* **重み**: 対称 int8（-127..127）。出力チャネルごと（既定）か層ごとのスケール。
  C 側の内積を連続アクセスにするため、行列は (出力, 入力) に転置して持つ。
* **活性値**: 対称 int8。スケールは較正用の観測サンプルを float で流したときの
  最大絶対値（または百分位点）/ 127。
* **バイアス**: int32、スケールは ``入力スケール × 重みスケール``。
* **隠れ層の活性化**: int32 累積 → float32 の倍率で前活性化 int8 に再量子化 →
  256 要素の LUT（``lut[p + 128]``）で活性化後 int8 へ。LUT は swish/tanh/relu の
  どれでも同じ仕組みで、MCU 上で exp を一度も呼ばない。
* **最終層**: int32 累積 → float32 倍率で実数 z に戻し、float で
  ``tanh(z) * output_scale + output_bias``（出力は 8 個だけなので float で十分）。

:func:`forward_int8` は C の ``policy_forward_i8`` と同じ順序・同じ float32 演算で
書いてあり、ホストでコンパイルした C と前 tanh 値がビット一致する（テストで確認）。

ここでの較正サンプルは既定では「ファイル内の witness 観測 + 正規化統計に沿った
正規乱数」であり、実機の分布ではない。実際のロールアウトで記録した観測を
``--samples`` で渡すこと（.npy で N×観測数、または JSON の2次元配列）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import cxpolicy_io as cx

F32 = np.float32
INT32_LIMIT = 2**31 - 1


# ---------------------------------------------------------------------------
# 量子化の部品
# ---------------------------------------------------------------------------


def quantize_symmetric(w: np.ndarray, *, per_channel: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """W[入力, 出力] → (q[出力, 入力] int8, scale[出力] float32)。``q * scale ≈ W.T``。"""

    wt = np.asarray(w, dtype=np.float64).T
    if per_channel:
        peak = np.abs(wt).max(axis=1)
    else:
        peak = np.full(wt.shape[0], np.abs(wt).max())
    scale = np.where(peak > 0.0, peak / 127.0, 1.0).astype(F32)
    q = np.clip(np.rint(wt / scale.astype(np.float64)[:, None]), -127, 127).astype(np.int8)
    return q, scale


def _peak(values: np.ndarray, percentile: float) -> float:
    a = np.abs(np.asarray(values, dtype=np.float64)).reshape(-1)
    if a.size == 0:
        return 1.0
    p = float(a.max()) if percentile >= 100.0 else float(np.percentile(a, percentile))
    return p if p > 0.0 else 1.0


def build_lut(activation: str, pre_scale: float, out_scale: float) -> np.ndarray:
    """前活性化 int8 p（スケール pre_scale）→ 活性化後 int8（スケール out_scale）の表。"""

    p = np.arange(-128, 128, dtype=np.float64)
    y = cx.activate(activation, p * float(pre_scale))
    return np.clip(np.rint(y / float(out_scale)), -127, 127).astype(np.int8)


def hard_swish(x: np.ndarray) -> np.ndarray:
    """MobileNetV3 の hard-swish ``x * relu6(x + 3) / 6``（区分 2 次の近似）。"""

    x = np.asarray(x, dtype=np.float64)
    return x * np.clip(x + 3.0, 0.0, 6.0) / 6.0


def swish_pwl(x: np.ndarray, *, segments: int = 32, limit: float = 8.0) -> np.ndarray:
    """[-limit, limit] を等間隔の折れ線で近似した swish。外側は 0 と x。"""

    x = np.asarray(x, dtype=np.float64)
    knots = np.linspace(-limit, limit, segments + 1)
    y = np.interp(x, knots, cx.swish(knots))
    return np.where(x < -limit, 0.0, np.where(x > limit, x, y))


# ---------------------------------------------------------------------------
# 量子化済みポリシー
# ---------------------------------------------------------------------------


@dataclass
class QuantLayer:
    w_q: np.ndarray            # int8 [出力, 入力]
    w_scale: np.ndarray        # float32 [出力]
    bias_q: np.ndarray         # int32 [出力]
    in_scale: float
    mult: np.ndarray           # float32 [出力]: 隠れ層は in*w/pre、最終層は in*w
    pre_scale: float | None = None
    out_scale: float | None = None
    lut: np.ndarray | None = None   # int8 [256]（隠れ層のみ）


@dataclass
class QuantPolicy:
    policy: cx.CxPolicy
    layers: list[QuantLayer]
    in_inv_scale: np.float32
    per_channel: bool
    percentile: float
    calibration_count: int

    @property
    def mean32(self) -> np.ndarray:
        return self.policy.mean.astype(F32)

    @property
    def std32(self) -> np.ndarray:
        return self.policy.std.astype(F32)


def calibration_samples(policy: cx.CxPolicy, *, count: int = 2048, seed: int = 0) -> np.ndarray:
    """witness 観測を先頭に置き、残りを N(mean, std) で埋めた float32 値の観測。"""

    witness = np.asarray((policy.header.get("evaluation") or {}).get("observations") or (),
                         dtype=np.float64).reshape(-1, len(policy.observations))
    witness = witness[:count]
    rng = np.random.default_rng(seed)
    extra = policy.mean + policy.std * rng.normal(size=(count - len(witness), policy.mean.size))
    return np.concatenate([witness, extra]).astype(F32).astype(np.float64)


def quantize_policy(policy: cx.CxPolicy, samples: np.ndarray, *, per_channel: bool = True,
                    percentile: float = 100.0) -> QuantPolicy:
    """較正サンプルで活性値スケールを決めて量子化する。"""

    samples = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    x = cx.normalise(policy, samples)
    in_scale = _peak(x, percentile) / 127.0
    first_in_scale = in_scale
    layers: list[QuantLayer] = []
    last = len(policy.layers) - 1
    for index, (w, b) in enumerate(policy.layers):
        q, w_scale = quantize_symmetric(w, per_channel=per_channel)
        acc_scale = float(F32(in_scale)) * w_scale.astype(np.float64)
        bias_q = np.clip(np.rint(b.astype(np.float64) / acc_scale),
                         -(2**30), 2**30).astype(np.int32)
        z = x @ w.astype(np.float64) + b.astype(np.float64)
        if index < last:
            pre_scale = _peak(z, percentile) / 127.0
            y = cx.activate(policy.activation, z)
            out_scale = _peak(y, percentile) / 127.0
            layers.append(QuantLayer(
                w_q=q, w_scale=w_scale, bias_q=bias_q, in_scale=float(F32(in_scale)),
                mult=(acc_scale / pre_scale).astype(F32), pre_scale=pre_scale,
                out_scale=out_scale, lut=build_lut(policy.activation, pre_scale, out_scale)))
            x, in_scale = y, out_scale
        else:
            layers.append(QuantLayer(
                w_q=q, w_scale=w_scale, bias_q=bias_q, in_scale=float(F32(in_scale)),
                mult=acc_scale.astype(F32)))
    return QuantPolicy(policy=policy, layers=layers, in_inv_scale=F32(1.0 / first_in_scale),
                       per_channel=per_channel, percentile=percentile,
                       calibration_count=int(samples.shape[0]))


def forward_int8(qp: QuantPolicy, obs: np.ndarray, *, trace: bool = False):
    """C の ``policy_forward_i8`` と同じ演算。``trace`` で中間値も返す。"""

    obs = np.asarray(obs)
    single = obs.ndim == 1
    o = np.atleast_2d(obs).astype(F32)
    xn = (o - qp.mean32) / qp.std32                       # float32 の減算と除算
    q = np.clip(np.rint(xn * qp.in_inv_scale), -127, 127).astype(np.int8)
    q_in = q.copy()
    accs: list[np.ndarray] = []
    raw = None
    last = len(qp.layers) - 1
    for index, layer in enumerate(qp.layers):
        acc64 = q.astype(np.int64) @ layer.w_q.T.astype(np.int64) + layer.bias_q.astype(np.int64)
        if np.abs(acc64).max(initial=0) > INT32_LIMIT:
            raise OverflowError("int32 累積があふれた")
        acc = acc64.astype(np.int32)
        accs.append(acc)
        scaled = acc.astype(F32) * layer.mult             # float32 の乗算
        if index < last:
            p = np.clip(np.rint(scaled), -128, 127).astype(np.int64)
            q = layer.lut[p + 128]
        else:
            raw = scaled
    policy = qp.policy
    actions = np.tanh(raw.astype(np.float64)) * policy.output_scale + policy.output_bias
    if single:
        actions, raw, q_in = actions[0], raw[0], q_in[0]
        accs = [a[0] for a in accs]
    if trace:
        return {"q_in": q_in, "acc": accs, "raw": raw, "actions": actions}
    return actions


def forward_weights_int8(qp: QuantPolicy, obs: np.ndarray) -> np.ndarray:
    """重みだけ int8（展開して float64 で計算）。活性値は float のまま。"""

    policy = qp.policy
    x = cx.normalise(policy, obs)
    last = len(qp.layers) - 1
    for index, ((_, b), layer) in enumerate(zip(policy.layers, qp.layers)):
        w = (layer.w_q.astype(np.float64) * layer.w_scale.astype(np.float64)[:, None]).T
        x = x @ w + b.astype(np.float64)
        if index < last:
            x = cx.activate(policy.activation, x)
    return np.tanh(x) * policy.output_scale + policy.output_bias


def forward_float32(policy: cx.CxPolicy, obs: np.ndarray) -> np.ndarray:
    """全部 float32 で計算（C の float32 版に相当）。"""

    x = ((np.atleast_2d(obs).astype(F32) - policy.mean.astype(F32)) / policy.std.astype(F32))
    last = len(policy.layers) - 1
    for index, (w, b) in enumerate(policy.layers):
        x = (x @ w + b).astype(F32)
        if index < last:
            x = cx.activate(policy.activation, x).astype(F32)
    out = (np.tanh(x) * policy.output_scale.astype(F32) + policy.output_bias.astype(F32))
    return out.astype(np.float64).reshape(np.shape(obs)[:-1] + (out.shape[-1],))


# ---------------------------------------------------------------------------
# 評価
# ---------------------------------------------------------------------------


def _to_degrees_factor(unit: str) -> float:
    if unit in ("deg", "degrees"):
        return 1.0
    if unit in ("rad", "radians"):
        return 180.0 / math.pi
    return 1.0


def _metrics(got: np.ndarray, ref: np.ndarray, policy: cx.CxPolicy) -> dict[str, float]:
    err = np.abs(np.asarray(got) - np.asarray(ref))
    factor = _to_degrees_factor(policy.action_unit)
    span = np.maximum(policy.action_high - policy.action_low, 1e-12)
    return {
        "max_abs_deg": float(err.max() * factor),
        "mean_abs_deg": float(err.mean() * factor),
        "p99_abs_deg": float(np.percentile(err, 99) * factor),
        "max_rel_range": float((err / span).max()),
    }


def evaluate(policy: cx.CxPolicy, samples: np.ndarray, *, per_channel: bool = True,
             percentile: float = 100.0) -> dict[str, Any]:
    """前半で較正し、後半（保留）で float64 参照との誤差を測る。"""

    samples = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    half = samples.shape[0] // 2 if samples.shape[0] >= 4 else samples.shape[0]
    calib, held = samples[:half], samples[half:] if samples.shape[0] >= 4 else samples
    ref = cx.forward(policy, held)
    qp = quantize_policy(policy, calib, per_channel=per_channel, percentile=percentile)
    modes: dict[str, Callable[[], np.ndarray]] = {
        "float32": lambda: forward_float32(policy, held),
        "w8": lambda: forward_weights_int8(qp, held),
        "w8a8_lut": lambda: forward_int8(qp, held),
    }
    other = quantize_policy(policy, calib, per_channel=not per_channel, percentile=percentile)
    modes["w8a8_lut_" + ("per_layer" if per_channel else "per_channel")] = \
        lambda: forward_int8(other, held)
    if policy.activation == "swish":
        modes["float_hard_swish"] = lambda: cx.forward(policy, held, act=hard_swish)
        modes["float_pwl_swish"] = lambda: cx.forward(policy, held, act=swish_pwl)
        modes["float_pwl8_swish"] = lambda: cx.forward(
            policy, held, act=lambda v: swish_pwl(v, segments=8, limit=6.0))
    return {
        "layer_sizes": policy.layer_sizes,
        "activation": policy.activation,
        "unit": policy.action_unit,
        "calibration_count": int(calib.shape[0]),
        "evaluation_count": int(held.shape[0]),
        "per_channel": per_channel,
        "percentile": percentile,
        "modes": {name: _metrics(fn(), ref, policy) for name, fn in modes.items()},
    }


MODE_LABELS = {
    "float32": "float32（C の float 版に相当）",
    "w8": "重みのみ int8・活性値 float",
    "w8a8_lut": "重み int8・活性値 int8・LUT 活性化（C の int8 版）",
    "w8a8_lut_per_layer": "同上、重みスケールを層ごとに",
    "w8a8_lut_per_channel": "同上、重みスケールをチャネルごとに",
    "float_hard_swish": "float、swish を hard-swish に置換",
    "float_pwl_swish": "float、swish を 32 区間折れ線に置換",
    "float_pwl8_swish": "float、swish を 8 区間折れ線に置換",
}


def format_report(report: dict[str, Any]) -> str:
    sizes = "-".join(str(n) for n in report["layer_sizes"])
    unit = "度" if report["unit"] in ("deg", "degrees", "rad", "radians") else report["unit"]
    lines = [
        f"層構成 {sizes}（{report['activation']}）、標本 "
        f"{report['calibration_count'] + report['evaluation_count']} 個"
        f"（較正 {report['calibration_count']}、評価 {report['evaluation_count']}）",
        f"誤差は float64 参照（エンジンと同じ式）との差。単位: {unit}",
        f"{'方式':<22} {'最大誤差':>10} {'平均誤差':>10} {'99%点':>10} {'範囲比 最大':>12}",
    ]
    for name, row in report["modes"].items():
        lines.append(f"{name:<22} {row['max_abs_deg']:>10.4f} {row['mean_abs_deg']:>10.4f} "
                     f"{row['p99_abs_deg']:>10.4f} {row['max_rel_range']:>12.2e}")
    lines.append("")
    for name in report["modes"]:
        lines.append(f"  {name}: {MODE_LABELS.get(name, '')}")
    return "\n".join(lines)


def load_samples(path: str | Path, width: int) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        data = np.load(path)
    else:
        data = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float64)
    data = np.atleast_2d(np.asarray(data, dtype=np.float64))
    if data.shape[1] != width:
        raise ValueError(f"標本の列数 {data.shape[1]} が観測チャネル数 {width} と合わない")
    return data


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quantize", description="int8 量子化と誤差評価")
    parser.add_argument("policy", help=".cxpolicy のパス")
    parser.add_argument("--samples", help="較正・評価用の観測（.npy または JSON、N×観測数）")
    parser.add_argument("--count", type=int, default=2048, help="--samples が無いときの合成標本数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-layer", action="store_true", help="重みスケールを層ごとにする")
    parser.add_argument("--percentile", type=float, default=100.0,
                        help="活性値スケールに使う絶対値の百分位点（100 は最大値）")
    parser.add_argument("--json", help="評価結果を JSON で書く")
    args = parser.parse_args(argv)
    try:
        policy = cx.load_policy(args.policy)
        samples = (load_samples(args.samples, len(policy.observations)) if args.samples
                   else calibration_samples(policy, count=args.count, seed=args.seed))
    except (OSError, ValueError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 2
    report = evaluate(policy, samples, per_channel=not args.per_layer, percentile=args.percentile)
    report["samples"] = "file" if args.samples else "synthetic_normal"
    print(format_report(report))
    if not args.samples:
        print("\n注意: 標本は正規化統計からの合成。実機の分布での誤差は --samples で確かめること。")
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
