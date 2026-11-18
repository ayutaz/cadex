/*
 * servo_cal.h — サーボの試験・較正モードの中身（ハードウェアに依存しない純 C、C11）。
 *
 * シリアルの 1 行コマンドで関節を 1 つ選び、中立へ動かし、少しずつ動かして、
 * 中立（関節角 0 = 立った姿勢）のパルス幅・向き・範囲の端を決める。結果はバイト列にして保存する。
 *
 *   角度[度] → パルス幅[µs]:  pulse = center_us + sign * angle * us_per_deg
 *   （deploy/esp32/export_c.py の ServoConfig と同じ。trim_deg は center_us に含めて 0 にする）
 *
 * 安全のための決まり:
 *   - 起動直後はどのチャネルもパルスを出さない（sel で選んだ関節だけ出す）。
 *   - パルス幅は必ず SERVO_MAP_HARD_MIN_US〜SERVO_MAP_HARD_MAX_US に収める。
 *   - 範囲の端（lo/hi）を記録するまでは、学習の範囲 ± SERVO_MAP_EXPLORE_MARGIN_DEG の外へ出さない。
 *     記録した後は、記録した端の外へ出さない。
 *   - 目標へは servo_cal_tick（20 ms ごと）で speed_deg_s を超えない速さで近づく。
 *   - 方策（policy.c）は呼ばない。
 *
 * ハードウェア（PWM・保存・シリアル）は呼び出し側が受け持つ（servo_console.c と servo_hal.h）。
 * テスト: tests/test_servo_cal_c.py（ホストの clang でビルドする）。
 */
#ifndef SERVO_CAL_H
#define SERVO_CAL_H

#include <stddef.h>
#include <stdint.h>

#include "servo_map.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SERVO_CAL_N SERVO_MAP_N
#define SERVO_CAL_TICK_HZ 50
#define SERVO_CAL_BLOB_VERSION 1
/* "Q8SC" + 版(2) + 個数(2) + 関節ごとに float 5 個 + CRC32(4) */
#define SERVO_CAL_BLOB_SIZE (4 + 2 + 2 + SERVO_CAL_N * 5 * 4 + 4)

#define SERVO_CAL_SPEED_DEFAULT 30.0f /* [度/s] */
#define SERVO_CAL_SPEED_MIN 5.0f
#define SERVO_CAL_SPEED_MAX 90.0f
#define SERVO_CAL_STEP_DEFAULT 5.0f /* [µs]（約 0.5°） */
#define SERVO_CAL_STEP_MAX 50.0f
#define SERVO_CAL_UPD_MIN 6.0f /* us_per_deg の許す範囲 */
#define SERVO_CAL_UPD_MAX 16.0f
#define SERVO_CAL_MARK_MIN_DEG 15.0f
#define SERVO_CAL_WIGGLE_DEG 10.0f
#define SERVO_CAL_WIGGLE_HOLD_TICKS 25

typedef struct {
    float center_us;  /* 関節角 0（立った姿勢）のパルス幅 */
    float sign;       /* +1: パルスを増やすと関節角が増える（足先が後ろへ）。-1: その逆 */
    float us_per_deg; /* 1 度あたりのパルス幅 */
    float lo_us;      /* 関節角の下限の端（足先が前）のパルス幅。0 = 未測定 */
    float hi_us;      /* 関節角の上限の端（足先が後ろ）のパルス幅。0 = 未測定 */
} servo_joint_cal_t;

/* 応答の 1 行を出す。保存と読み出しは 0 で成功、読み出しの 1 は「保存が無い」。 */
typedef struct {
    void (*emit)(void *ctx, const char *line);
    int (*save)(void *ctx, const uint8_t *blob, size_t len);
    int (*load)(void *ctx, uint8_t *blob, size_t cap, size_t *len);
    void *ctx;
} servo_cal_io_t;

typedef struct {
    servo_joint_cal_t cal[SERVO_CAL_N];
    float target_us[SERVO_CAL_N];
    float current_us[SERVO_CAL_N];
    uint8_t enabled[SERVO_CAL_N];
    int selected; /* -1 = なし */
    float step_us;
    float speed_deg_s;
    int dirty;    /* 保存していない変更がある */
    int in_store; /* 保存先に有効な較正がある */
    /* test コマンド（中立 → +10° → -10° → 中立） */
    int wiggle_joint; /* -1 = していない */
    int wiggle_stage;
    int wiggle_hold;
} servo_cal_t;

extern const char *const servo_cal_names[SERVO_CAL_N];
extern const int servo_cal_gpio[SERVO_CAL_N];

/* servo_map.h の値。 */
void servo_cal_defaults(servo_joint_cal_t cal[SERVO_CAL_N]);

/* 状態を初期化する。保存から読み出し、無いか壊れていれば既定値。結果を io->emit で知らせる。 */
void servo_cal_init(servo_cal_t *s, const servo_cal_io_t *io);

/* 1 行のコマンドを処理する。0 = 成功、-1 = エラー（"err ..." を出す）。 */
int servo_cal_command(servo_cal_t *s, const char *line, const servo_cal_io_t *io);

/* 1 周期（1 / SERVO_CAL_TICK_HZ 秒）進める。出すパルス幅と、出すかどうか（0 = 止める）。 */
void servo_cal_tick(servo_cal_t *s, float out_us[SERVO_CAL_N], uint8_t out_on[SERVO_CAL_N]);

float servo_cal_angle_to_us(const servo_joint_cal_t *c, float deg);
float servo_cal_us_to_angle(const servo_joint_cal_t *c, float us);
/* 今の関節 j に許すパルス幅の範囲。 */
void servo_cal_window(const servo_cal_t *s, int j, float *lo_us, float *hi_us);

/* 保存用のバイト列。書いたバイト数（足りなければ 0）。 */
size_t servo_cal_serialize(const servo_joint_cal_t cal[SERVO_CAL_N], uint8_t *buf, size_t cap);
/* 0 = 成功、-1 = 長さ・印・版・個数の違い、-2 = CRC の違い、-3 = 値がおかしい。 */
int servo_cal_deserialize(const uint8_t *buf, size_t len, servo_joint_cal_t cal[SERVO_CAL_N]);

/* パルス幅[µs] → LEDC のデューティ（50 Hz、14 bit）。control.c の servo_ledc_duty と同じ式。 */
uint32_t servo_cal_duty(float pulse_us);

#ifdef __cplusplus
}
#endif

#endif /* SERVO_CAL_H */
