/*
 * control.c — control.h の実装。Python 側の対応物は
 * deploy/esp32/export_c.py（build_observation, servo_pulse_us, ledc_duty）と
 * deploy/esp32/cxpolicy_io.py（CommandFilter）。
 */
#include "control.h"

#include <math.h>

static float obs_source(int kind, int index, const imu_sample_t *imu, const float *cmd,
                        const float *cmd_before, const float *goal, float phase_rad, float dt_s)
{
    const float *q = imu->quat_wxyz;
    switch (kind) {
    case OBS_QUAT:
        return q[index];
    case OBS_GYRO:
        return imu->gyro_rad_s[index];
    case OBS_GYRO_WORLD: {
        /* 世界座標の角速度 = R(q) × 機体座標のジャイロ（frame の無い Cadex の角速度観測） */
        const float w = q[0], x = q[1], y = q[2], z = q[3];
        const float *g = imu->gyro_rad_s;
        if (index == 0) {
            return (1.0f - 2.0f * (y * y + z * z)) * g[0] + 2.0f * (x * y - w * z) * g[1] +
                   2.0f * (x * z + w * y) * g[2];
        }
        if (index == 1) {
            return 2.0f * (x * y + w * z) * g[0] + (1.0f - 2.0f * (x * x + z * z)) * g[1] +
                   2.0f * (y * z - w * x) * g[2];
        }
        return 2.0f * (x * z - w * y) * g[0] + 2.0f * (y * z + w * x) * g[1] +
               (1.0f - 2.0f * (x * x + y * y)) * g[2];
    }
    case OBS_GRAVITY: {
        /* g_body = R(q)^T (0, 0, -1) */
        const float w = q[0], x = q[1], y = q[2], z = q[3];
        if (index == 0) {
            return -2.0f * (x * z - w * y);
        }
        if (index == 1) {
            return -2.0f * (y * z + w * x);
        }
        return -(1.0f - 2.0f * (x * x + y * y));
    }
    case OBS_CMD:
        return cmd[index];
    case OBS_CMD_RATE:
        return (cmd[index] - cmd_before[index]) / dt_s;
    case OBS_GOAL:
        return goal[index];
    case OBS_PHASE_SIN:
        return sinf(phase_rad);
    case OBS_PHASE_COS:
        return cosf(phase_rad);
    default:
        return 1.0f;
    }
}

void obs_build(const imu_sample_t *imu, const float *cmd, const float *cmd_before,
               const float *goal, float phase_rad, float dt_s, float *obs_out)
{
    for (int i = 0; i < POLICY_N_IN; ++i) {
        const float v = obs_source(policy_obs_kind[i], policy_obs_index[i], imu, cmd,
                                   cmd_before, goal, phase_rad, dt_s);
        obs_out[i] = policy_obs_gain[i] * v;
    }
}

void command_filter_reset(command_filter_t *f)
{
    f->primed = 0;
    for (int j = 0; j < POLICY_N_OUT; ++j) {
        f->issued[j] = 0.0f;
    }
}

void command_filter_step(command_filter_t *f, const float *raw, float *cmd_out)
{
    for (int j = 0; j < POLICY_N_OUT; ++j) {
        float c = raw[j];
        if (c < policy_action_low[j]) {
            c = policy_action_low[j];
        }
        if (c > policy_action_high[j]) {
            c = policy_action_high[j];
        }
        if (f->primed) {
            const float then = f->issued[j];
            if (POLICY_FILTER_ALPHA < 1.0f) {
                c = POLICY_FILTER_ALPHA * c + (1.0f - POLICY_FILTER_ALPHA) * then;
            }
            if (POLICY_COMMAND_SLEW_DEG > 0.0f) {
                if (c < then - POLICY_COMMAND_SLEW_DEG) {
                    c = then - POLICY_COMMAND_SLEW_DEG;
                }
                if (c > then + POLICY_COMMAND_SLEW_DEG) {
                    c = then + POLICY_COMMAND_SLEW_DEG;
                }
            }
        }
        f->issued[j] = c;
        cmd_out[j] = c;
    }
    f->primed = 1;
}

float servo_pulse_us(int joint, float angle_deg)
{
    float p = policy_servo_center_us[joint] +
              policy_servo_sign[joint] * (angle_deg + policy_servo_trim_deg[joint]) *
                  policy_servo_us_per_deg[joint];
    if (p < policy_servo_min_us[joint]) {
        p = policy_servo_min_us[joint];
    }
    if (p > policy_servo_max_us[joint]) {
        p = policy_servo_max_us[joint];
    }
    return p;
}

uint32_t servo_ledc_duty(float pulse_us)
{
    /* duty = pulse / 周期 × 2^bits。周期 20000 µs、14 bit なら 0.8192 / µs。 */
    const float per_us = (float)(1u << SERVO_LEDC_BITS) / (1.0e6f / (float)SERVO_PWM_HZ);
    long d = lrintf(pulse_us * per_us);
    const long top = (1L << SERVO_LEDC_BITS) - 1;
    if (d < 0) {
        d = 0;
    }
    if (d > top) {
        d = top;
    }
    return (uint32_t)d;
}
