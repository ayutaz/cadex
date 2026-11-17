/*
 * control.h — 制御ループの純 C 部品（ハードウェアに依存しない）。
 *
 *  - obs_build:           センサ値と前回指令から、ポリシーの観測ベクトルを作る
 *  - command_filter_step: エンジンと同じ指令フィルタ（clamp → alpha → slew）
 *  - servo_pulse_us:      関節角[度] → サーボのパルス幅[µs]
 *  - servo_ledc_duty:     パルス幅[µs] → LEDC のデューティ値
 *
 * 観測の並びと単位換算は policy_weights.h の policy_obs_kind / _index / _gain が決める
 * （export_c.py がヘッダの observations から解決したもの）。
 */
#ifndef CONTROL_H
#define CONTROL_H

#include <stdint.h>

#include "policy_weights.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 観測チャネルの取り出し元。deploy/esp32/export_c.py の OBS_KINDS と同じ番号。 */
enum {
    OBS_QUAT = 0,      /* 姿勢クォータニオン wxyz の index 番目（無次元） */
    OBS_GYRO = 1,      /* 機体座標の角速度 [rad/s] の index 番目 × gain（Cadex は deg/s） */
    OBS_GRAVITY = 2,   /* 機体座標の重力方向単位ベクトル（クォータニオンから計算） */
    OBS_CMD = 3,       /* 現在有効な関節指令（前回出した指令）[タスク単位] */
    OBS_CMD_RATE = 4,  /* 指令の変化量 / 制御周期 [タスク単位/s] */
    OBS_GOAL = 5,      /* ゴール（目標速度など）の index 番目 */
    OBS_PHASE_SIN = 6, /* sin(歩行位相) */
    OBS_PHASE_COS = 7, /* cos(歩行位相) */
    OBS_CONST = 8,     /* 定数（値は gain） */
    OBS_GYRO_WORLD = 9 /* 世界座標の角速度: 機体座標のジャイロ [rad/s] を姿勢で回したもの × gain */
};

#define CONTROL_MAX_GOAL 4
#define SERVO_PWM_HZ 50
#define SERVO_LEDC_BITS 14

typedef struct {
    float quat_wxyz[4]; /* 機体→世界の回転。w が先頭 */
    float gyro_rad_s[3];
} imu_sample_t;

typedef struct {
    float issued[POLICY_N_OUT];
    int primed;
} command_filter_t;

void obs_build(const imu_sample_t *imu, const float *cmd, const float *cmd_before,
               const float *goal, float phase_rad, float dt_s, float *obs_out);

void command_filter_reset(command_filter_t *f);
void command_filter_step(command_filter_t *f, const float *raw, float *cmd_out);

float servo_pulse_us(int joint, float angle_deg);
uint32_t servo_ledc_duty(float pulse_us);

#ifdef __cplusplus
}
#endif

#endif /* CONTROL_H */
