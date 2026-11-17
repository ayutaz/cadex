/*
 * policy.h — .cxpolicy から書き出した MLP の推論（依存なしの純 C、C11）。
 *
 * 数式は Cadex エンジンの CadexDynamics.policy_forward と同じ:
 *   x = (obs - mean) / std
 *   隠れ層: x = act(W x + b)       act は POLICY_ACTIVATION（swish / tanh / relu）
 *   最終層: z = W x + b
 *   action = tanh(z) * output_scale + output_bias      （タスクの単位。サーボは度）
 *
 * 重み・正規化・スケールは export_c.py が生成する policy_weights.h にある。
 * obs はヘッダの observations の順・タスクの単位で渡す（control.c の obs_build が作る）。
 * 戻り値の action は clamp 前。指令フィルタ（command_filter_step）を必ず通すこと。
 */
#ifndef POLICY_H
#define POLICY_H

#include "policy_weights.h"

#ifdef __cplusplus
extern "C" {
#endif

/* float32 版。action_out は POLICY_N_OUT 個。0 を返す。 */
int policy_forward_f32(const float *obs, float *action_out);

#if POLICY_HAS_INT8
/*
 * int8 版（重み int8・活性値 int8・int32 累積・LUT 活性化、最終層の出力だけ float）。
 * raw_out が NULL でなければ tanh の手前の値 z を POLICY_N_OUT 個書く（検証用）。
 */
int policy_forward_i8(const float *obs, float *action_out, float *raw_out);
#endif

#ifdef __cplusplus
}
#endif

#endif /* POLICY_H */
