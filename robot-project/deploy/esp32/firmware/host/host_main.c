/*
 * host_main.c — ホスト（macOS/Linux）で policy.c / control.c を動かす検証用ハーネス。
 * tests/test_esp32_export_c.py が clang でコンパイルして使う。ESP-IDF のビルドには入らない。
 *
 *   host_policy forward   標準入力: 1 行 POLICY_N_IN 個の観測
 *                         出力: "f32 ..."、int8 版があれば "i8raw ..."（tanh 手前）と "i8 ..."
 *   host_policy filter    標準入力: 1 行 POLICY_N_OUT 個の生の行動 → フィルタ後の指令
 *   host_policy obs       標準入力: quat(4) gyro(3) cmd(N_OUT) before(N_OUT) goal(4) phase dt
 *                         出力: 観測ベクトル
 *   host_policy pulse J A 関節 J、角度 A[度] → "パルス幅[µs] デューティ"
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "control.h"
#include "policy.h"

static int read_floats(float *dst, int n)
{
    for (int i = 0; i < n; ++i) {
        if (scanf("%f", &dst[i]) != 1) {
            return 0;
        }
    }
    return 1;
}

static void print_floats(const char *tag, const float *v, int n)
{
    if (tag) {
        printf("%s", tag);
    }
    for (int i = 0; i < n; ++i) {
        printf("%s%.9g", (tag || i) ? " " : "", (double)v[i]);
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "使い方: %s forward|filter|obs|pulse\n", argv[0]);
        return 2;
    }
    const char *mode = argv[1];
    if (strcmp(mode, "forward") == 0) {
        float obs[POLICY_N_IN];
        float act[POLICY_N_OUT];
        while (read_floats(obs, POLICY_N_IN)) {
            policy_forward_f32(obs, act);
            print_floats("f32", act, POLICY_N_OUT);
#if POLICY_HAS_INT8
            float raw[POLICY_N_OUT];
            policy_forward_i8(obs, act, raw);
            print_floats("i8raw", raw, POLICY_N_OUT);
            print_floats("i8", act, POLICY_N_OUT);
#endif
        }
        return 0;
    }
    if (strcmp(mode, "filter") == 0) {
        command_filter_t f;
        command_filter_reset(&f);
        float raw[POLICY_N_OUT];
        float cmd[POLICY_N_OUT];
        while (read_floats(raw, POLICY_N_OUT)) {
            command_filter_step(&f, raw, cmd);
            print_floats(NULL, cmd, POLICY_N_OUT);
        }
        return 0;
    }
    if (strcmp(mode, "obs") == 0) {
        imu_sample_t imu;
        float cmd[POLICY_N_OUT], before[POLICY_N_OUT], goal[CONTROL_MAX_GOAL], tail[2];
        float obs[POLICY_N_IN];
        while (read_floats(imu.quat_wxyz, 4) && read_floats(imu.gyro_rad_s, 3) &&
               read_floats(cmd, POLICY_N_OUT) && read_floats(before, POLICY_N_OUT) &&
               read_floats(goal, CONTROL_MAX_GOAL) && read_floats(tail, 2)) {
            obs_build(&imu, cmd, before, goal, tail[0], tail[1], obs);
            print_floats(NULL, obs, POLICY_N_IN);
        }
        return 0;
    }
    if (strcmp(mode, "pulse") == 0 && argc >= 4) {
        const int joint = atoi(argv[2]);
        const float angle = strtof(argv[3], NULL);
        const float pulse = servo_pulse_us(joint, angle);
        printf("%.6f %u\n", (double)pulse, (unsigned)servo_ledc_duty(pulse));
        return 0;
    }
    fprintf(stderr, "不明なモード: %s\n", mode);
    return 2;
}
