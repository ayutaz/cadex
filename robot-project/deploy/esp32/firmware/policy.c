/*
 * policy.c — MLP 推論の本体。policy.h の説明を参照。
 *
 * 重みは (出力, 入力) の行優先で持つので、各出力は連続メモリ上の内積になる。
 * 累積の順序は「バイアスから始めて入力 0, 1, 2, ... の順に足す」で、エンジンと同じ。
 * int8 版の演算は deploy/esp32/quantize.py の forward_int8 と 1 対 1 に対応しており、
 * ホストでコンパイルすると tanh 手前の値がビット一致する（tests/test_esp32_export_c.py）。
 */
#define POLICY_WEIGHTS_DEFINE
#include "policy.h"

#include <math.h>
#include <stdint.h>

#define POLICY_BUF POLICY_MAX_WIDTH

static float policy_swishf(float x)
{
    /* x * sigmoid(x)。両側で expf があふれない形（エンジンの policy_swish と同じ）。 */
    if (x >= 0.0f) {
        return x / (1.0f + expf(-x));
    }
    const float e = expf(x);
    return x * e / (1.0f + e);
}

static float policy_activate(float x)
{
#if POLICY_ACTIVATION == POLICY_ACT_SWISH
    return policy_swishf(x);
#elif POLICY_ACTIVATION == POLICY_ACT_TANH
    return tanhf(x);
#else
    return x > 0.0f ? x : 0.0f;
#endif
}

int policy_forward_f32(const float *obs, float *action_out)
{
    float buf_a[POLICY_BUF];
    float buf_b[POLICY_BUF];
    float *x = buf_a;
    float *y = buf_b;

    for (int i = 0; i < POLICY_N_IN; ++i) {
        x[i] = (obs[i] - policy_obs_mean[i]) / policy_obs_std[i];
    }
    for (int l = 0; l < POLICY_N_LAYERS; ++l) {
        const int n_in = policy_layer_in[l];
        const int n_out = policy_layer_out[l];
        const float *w = policy_f32_w + policy_f32_w_off[l];
        const float *b = policy_f32_b + policy_f32_b_off[l];
        const int hidden = l < POLICY_N_LAYERS - 1;
        for (int c = 0; c < n_out; ++c) {
            const float *row = w + (int32_t)c * n_in;
            float t = b[c];
            for (int i = 0; i < n_in; ++i) {
                t += x[i] * row[i];
            }
            y[c] = hidden ? policy_activate(t) : t;
        }
        float *swap = x;
        x = y;
        y = swap;
    }
    for (int c = 0; c < POLICY_N_OUT; ++c) {
        action_out[c] = tanhf(x[c]) * policy_output_scale[c] + policy_output_bias[c];
    }
    return 0;
}

#if POLICY_HAS_INT8

static int32_t policy_round_clamp(float v, float lo, float hi)
{
    /* 先に範囲へ収めてから最近接偶数丸め（numpy の rint → clip と同じ結果）。 */
    if (v < lo) {
        v = lo;
    }
    if (v > hi) {
        v = hi;
    }
    return (int32_t)lrintf(v);
}

int policy_forward_i8(const float *obs, float *action_out, float *raw_out)
{
    int8_t buf_a[POLICY_BUF];
    int8_t buf_b[POLICY_BUF];
    int8_t *x = buf_a;
    int8_t *y = buf_b;
    float z[POLICY_N_OUT];

    for (int i = 0; i < POLICY_N_IN; ++i) {
        const float xn = (obs[i] - policy_obs_mean[i]) / policy_obs_std[i];
        x[i] = (int8_t)policy_round_clamp(xn * POLICY_I8_IN_INV_SCALE, -127.0f, 127.0f);
    }
    int32_t w_off = 0;
    int32_t c_off = 0;
    for (int l = 0; l < POLICY_N_LAYERS; ++l) {
        const int n_in = policy_layer_in[l];
        const int n_out = policy_layer_out[l];
        const int hidden = l < POLICY_N_LAYERS - 1;
        for (int c = 0; c < n_out; ++c) {
            const int8_t *row = policy_i8_w + w_off + (int32_t)c * n_in;
            int32_t acc = policy_i8_bias[c_off + c];
            for (int i = 0; i < n_in; ++i) {
                acc += (int32_t)x[i] * (int32_t)row[i];
            }
            const float scaled = (float)acc * policy_i8_mult[c_off + c];
            if (hidden) {
                const int32_t p = policy_round_clamp(scaled, -128.0f, 127.0f);
                y[c] = policy_i8_lut[l * 256 + p + 128];
            } else {
                z[c] = scaled;
            }
        }
        w_off += (int32_t)n_in * n_out;
        c_off += n_out;
        int8_t *swap = x;
        x = y;
        y = swap;
    }
    for (int c = 0; c < POLICY_N_OUT; ++c) {
        if (raw_out) {
            raw_out[c] = z[c];
        }
        action_out[c] = tanhf(z[c]) * policy_output_scale[c] + policy_output_bias[c];
    }
    return 0;
}

#endif /* POLICY_HAS_INT8 */
