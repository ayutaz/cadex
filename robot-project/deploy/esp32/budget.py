"""層構成ごとの計算量・メモリ・ESP32-S3 での推論時間の見積り。

**時間はすべて推定値**（実機で測っていない）。前提は :data:`ASSUMPTIONS` に根拠つきで
書いてある。firmware/main/main.c は推論サイクル数を毎秒ログに出すので、実機に
書き込めば実測値で置き換えられる。

使い方::

    python -m deploy.esp32.budget                  # 既定の候補を表で出す
    python -m deploy.esp32.budget 20-128-64-8      # 任意の層構成
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Sequence

#: 推定の前提。値は (楽観, 悲観) の幅。
ASSUMPTIONS: dict[str, Any] = {
    "clock_mhz": 240,
    "cycles_per_mac": {
        "f32_c": {
            "range": (2.0, 5.0),
            "根拠": "Xtensa LX7 の単精度 FPU は madd.s を 1 命令で持つが、手書き C の内積は"
                    "毎回 重み と 活性値 の 2 ロード + ループ制御が付く。-O2 で 1 MAC あたり"
                    " 2〜5 サイクルと見る（ESP-DSP のアセンブリ版 dotprod ならさらに速い）。",
        },
        "i8_c": {
            "range": (1.5, 4.0),
            "根拠": "int8 のロード 2 回と 32bit 乗算加算（mull + add）。SIMD を使わない"
                    "スカラー C では float と大差ない。メモリ量が 1/4 になるのが主な利点。",
        },
        "i8_simd": {
            "range": (0.15, 0.5),
            "根拠": "ESP32-S3 の PIE（128bit SIMD）は int8 を 16 レーン同時に積和できる"
                    "（理論 16 MAC/サイクル）。ESP-NN / ESP-DL の最適化カーネルを想定し、"
                    "ロード・再量子化・端数処理で理論値の 1/2〜1/8 と見る。",
        },
    },
    "activation_cycles": {
        "f32_c": {"range": (100.0, 300.0),
                  "根拠": "swish は expf と除算を 1 回ずつ。newlib の expf はソフト実装で数百サイクル。"},
        "i8_c": {"range": (10.0, 25.0),
                 "根拠": "float32 乗算 1 回 + 丸め + LUT 参照。"},
        "i8_simd": {"range": (10.0, 25.0),
                    "根拠": "再量子化と LUT はスカラーのまま。"},
    },
    "output_cycles": (100.0, 300.0),   # 出力 1 個あたり tanhf + 積和
    "input_cycles": (20.0, 40.0),      # 入力 1 個あたり (x-mean)/std（int8 は丸めも）
    "sram_budget_bytes": 256 * 1024,
    "sram_根拠": "内部 SRAM 512 KiB のうち、ESP-IDF・FreeRTOS・スタック・ドライバを除いて"
                 "重みに回せる量の控えめな見積り（Wi-Fi/BLE を使うならさらに減る）。",
    "flash_mb_s": (20.0, 40.0),
    "flash_根拠": "SRAM に載らない重みは flash からキャッシュ経由で読む。QSPI 80 MHz の理論 40 MB/s、"
                  "キャッシュミスを含めた実効 20 MB/s 程度と見る（毎周期すべての重みを読み直す前提）。",
}

CANDIDATES = (
    [20, 512, 256, 128, 8],
    [20, 256, 128, 8],
    [20, 128, 128, 8],
    [20, 128, 64, 8],
    [20, 64, 64, 8],
    [20, 32, 32, 8],
)


def macs(sizes: Sequence[int]) -> int:
    return sum(a * b for a, b in zip(sizes[:-1], sizes[1:]))


def parameters(sizes: Sequence[int]) -> int:
    return macs(sizes) + sum(sizes[1:])


def weight_bytes(sizes: Sequence[int], kind: str) -> int:
    """f32: 全パラメータ 4 バイト。i8: 重み 1 バイト + 出力ごとに int32 バイアスと float 倍率 + LUT。"""

    if kind == "f32":
        return 4 * parameters(sizes)
    hidden_layers = len(sizes) - 2
    return macs(sizes) + 8 * sum(sizes[1:]) + 256 * hidden_layers


def estimate(sizes: Sequence[int]) -> dict[str, Any]:
    sizes = [int(s) for s in sizes]
    clock = ASSUMPTIONS["clock_mhz"]
    m = macs(sizes)
    hidden = sum(sizes[1:-1])
    budget = ASSUMPTIONS["sram_budget_bytes"]
    fits = {"f32": weight_bytes(sizes, "f32") <= budget, "i8": weight_bytes(sizes, "i8") <= budget}
    times: dict[str, tuple[float, float]] = {}
    for mode in ("f32_c", "i8_c", "i8_simd"):
        kind = "f32" if mode.startswith("f32") else "i8"
        cpm = ASSUMPTIONS["cycles_per_mac"][mode]["range"]
        act = ASSUMPTIONS["activation_cycles"][mode]["range"]
        out = ASSUMPTIONS["output_cycles"]
        inp = ASSUMPTIONS["input_cycles"]
        bounds = []
        for k in (0, 1):
            cycles = m * cpm[k] + hidden * act[k] + sizes[-1] * out[k] + sizes[0] * inp[k]
            us = cycles / clock
            if not fits[kind]:
                # flash から毎回読み直す時間（速い側の帯域で楽観、遅い側で悲観）
                bandwidth = ASSUMPTIONS["flash_mb_s"][1 - k]
                us += weight_bytes(sizes, kind) / bandwidth  # bytes / (MB/s) = µs
            bounds.append(us)
        times[mode] = (bounds[0], bounds[1])
    return {"sizes": sizes, "macs": m, "parameters": parameters(sizes),
            "bytes": {"f32": weight_bytes(sizes, "f32"), "i8": weight_bytes(sizes, "i8")},
            "fits_sram": fits, "time_us": times}


def format_table(rows: Sequence[dict[str, Any]], period_ms: float = 20.0) -> str:
    head = (f"{'層構成':<20} {'MAC':>8} {'重み f32':>10} {'重み int8':>10} "
            f"{'f32 手書きC [µs]':>18} {'int8 手書きC [µs]':>19} {'int8 SIMD [µs]':>16}")
    lines = [head]
    for r in rows:
        name = "-".join(str(s) for s in r["sizes"])
        def kb(n: int, fits: bool) -> str:
            return f"{n / 1024:.1f}K{'' if fits else '*'}"
        def span(t: tuple[float, float]) -> str:
            return f"{t[0]:.0f}〜{t[1]:.0f}"
        lines.append(f"{name:<20} {r['macs']:>8} {kb(r['bytes']['f32'], r['fits_sram']['f32']):>10} "
                     f"{kb(r['bytes']['i8'], r['fits_sram']['i8']):>10} "
                     f"{span(r['time_us']['f32_c']):>18} {span(r['time_us']['i8_c']):>19} "
                     f"{span(r['time_us']['i8_simd']):>16}")
    lines += [
        "",
        f"* は内部 SRAM の重み予算 {ASSUMPTIONS['sram_budget_bytes'] // 1024} KiB に載らない"
        "（flash から読む時間を加算済み）。",
        f"制御周期 {period_ms:g} ms（50 Hz）に対する推論時間の比で判断する。値はすべて推定。",
        "前提: " + ", ".join(f"{k} {v['range'][0]}〜{v['range'][1]} サイクル/MAC"
                             for k, v in ASSUMPTIONS["cycles_per_mac"].items())
        + f"、{ASSUMPTIONS['clock_mhz']} MHz。",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="budget", description="推論の計算量・メモリ・時間の見積り")
    parser.add_argument("sizes", nargs="*", help="層構成（例: 20-128-64-8）。省略で既定の候補")
    args = parser.parse_args(argv)
    try:
        candidates = ([[int(v) for v in s.split("-")] for s in args.sizes]
                      if args.sizes else [list(c) for c in CANDIDATES])
    except ValueError:
        print("エラー: 層構成は 20-64-64-8 の形で書く", file=sys.stderr)
        return 2
    print(format_table([estimate(c) for c in candidates]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
