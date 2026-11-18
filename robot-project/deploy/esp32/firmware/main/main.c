/*
 * main.c — ESP32-S3 のファームウェア（ESP-IDF v5 系）。2 つのモードがある。
 *
 *   既定（Q8_RUN_POLICY == 0）: サーボの試験・較正モード（servo_console.c）。
 *       方策は動かさない。シリアルのコマンドで関節を1つずつ動かし、中立・向き・範囲を測って NVS に保存する。
 *       手順は docs/servo-calibration.md。
 *
 *   Q8_RUN_POLICY == 1: 50 Hz の制御ループ
 *       IMU 読み取り → 観測生成 → 推論 → 指令フィルタ → LEDC PWM 出力
 *       実機 I/O（IMU、ゴール入力）はスタブ。TODO の箇所を実装すること。
 *       推論にかかったサイクル数を 1 秒ごとにログへ出すので、書き込めばそのまま実測になる。
 *       policy_weights.h が合成ポリシー（POLICY_SYNTHETIC == 1）のときはサーボへ出力しない
 *       （乱数重みで関節を動かさないため）。
 *
 * サーボの GPIO と関節の順は servo_map.h（deploy/esp32/servo_calibration.py が生成）にある。
 */
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "servo_map.h"

/* 1 にすると方策の制御ループを動かす。既定は 0（方策を動かさず、試験・較正モード）。 */
#ifndef Q8_RUN_POLICY
#define Q8_RUN_POLICY 0
#endif

static const char *TAG = "q8";

#if !Q8_RUN_POLICY

#include "servo_console.h"
#include "servo_hal.h"

static void servo_console_task(void *arg)
{
    (void)arg;
    static servo_console_t con;
    servo_console_start(&con);
    TickType_t last_wake = xTaskGetTickCount();
    const TickType_t period = pdMS_TO_TICKS(1000 / SERVO_CAL_TICK_HZ);
    for (;;) {
        vTaskDelayUntil(&last_wake, period);
        int ch;
        while ((ch = servo_hal_getchar()) >= 0) {
            servo_console_feed(&con, ch);
        }
        servo_console_tick(&con);
    }
}

void app_main(void)
{
    ESP_LOGI(TAG, "サーボの試験・較正モードで起動（方策は動かさない）");
    /* 応答の文字列を作るので、スタックの大きいタスクで動かす */
    xTaskCreate(servo_console_task, "servo_console", 8192, NULL, 5, NULL);
}

#else /* Q8_RUN_POLICY */

#include "driver/ledc.h"
#include "esp_cpu.h"

#include "control.h"
#include "policy.h"

#define CONTROL_HZ 50
#define CONTROL_DT_S (1.0f / (float)CONTROL_HZ)
/* 歩行位相の周波数 [Hz]。export_c.py --task がタスクの位相ゴールの周期を
 * POLICY_CLOCK_PERIOD_S に書く。不明（0）のときは手で決めること。TODO */
#define GAIT_HZ (POLICY_CLOCK_PERIOD_S > 0.0f ? 1.0f / POLICY_CLOCK_PERIOD_S : 2.0f)
/* 1 なら int8 版で推論する（policy_weights.h に int8 が入っている必要がある）。 */
#ifndef USE_INT8_POLICY
#define USE_INT8_POLICY 0
#endif

/* サーボの GPIO（方策の行動の順）。servo_map.h の表。基板の配線に合わせること（今は仮の値）。 */
static const int SERVO_GPIO[SERVO_MAP_N] = SERVO_MAP_GPIO;
_Static_assert(POLICY_N_OUT == SERVO_MAP_N, "方策の出力の数と servo_map.h の関節の数が違う");

static void servo_init(void)
{
    const ledc_timer_config_t timer = {
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .duty_resolution = (ledc_timer_bit_t)SERVO_LEDC_BITS,
        .timer_num = LEDC_TIMER_0,
        .freq_hz = SERVO_PWM_HZ,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    ESP_ERROR_CHECK(ledc_timer_config(&timer));
    for (int j = 0; j < POLICY_N_OUT; ++j) {
        const ledc_channel_config_t ch = {
            .gpio_num = SERVO_GPIO[j],
            .speed_mode = LEDC_LOW_SPEED_MODE,
            .channel = (ledc_channel_t)j,
            .timer_sel = LEDC_TIMER_0,
            .duty = 0, /* 起動直後はパルスを出さない */
            .hpoint = 0,
        };
        ESP_ERROR_CHECK(ledc_channel_config(&ch));
    }
}

static void servo_write(const float *cmd_deg)
{
    for (int j = 0; j < POLICY_N_OUT; ++j) {
        const uint32_t duty = servo_ledc_duty(servo_pulse_us(j, cmd_deg[j]));
        ledc_set_duty(LEDC_LOW_SPEED_MODE, (ledc_channel_t)j, duty);
        ledc_update_duty(LEDC_LOW_SPEED_MODE, (ledc_channel_t)j);
    }
}

static void imu_init(void)
{
    /* TODO: MPU-6050 を I2C で初期化（±500 dps、DLPF 約 42 Hz、サンプル 200 Hz 以上）。 */
}

static bool imu_read(imu_sample_t *out)
{
    /*
     * TODO: ジャイロ [rad/s] を機体座標（シミュレーションの IMU ボディと同じ軸）で読み、
     * 相補フィルタ／Madgwick で姿勢クォータニオン wxyz を推定する。
     * 軸の向き・符号の合わせ方は docs/esp32-deployment.md を参照。
     */
    const imu_sample_t still = {.quat_wxyz = {1.0f, 0.0f, 0.0f, 0.0f},
                                .gyro_rad_s = {0.0f, 0.0f, 0.0f}};
    *out = still;
    return true;
}

static void goal_read(float *goal)
{
    /* TODO: 無線（ESP-NOW / BLE）などから目標速度を受け取る。単位はタスクのゴールと同じ。 */
    for (int k = 0; k < CONTROL_MAX_GOAL; ++k) {
        goal[k] = 0.0f;
    }
}

void app_main(void)
{
    servo_init();
    imu_init();

    float cmd[POLICY_N_OUT];
    float before[POLICY_N_OUT];
    float raw[POLICY_N_OUT];
    float obs[POLICY_N_IN];
    float goal[CONTROL_MAX_GOAL];
    imu_sample_t imu;
    command_filter_t filter;

    /* 学習時のリセット姿勢と同じく、最初の観測では指令＝初期指令、変化量 0。 */
    memcpy(cmd, policy_initial_command, sizeof cmd);
    memcpy(before, cmd, sizeof before);
    command_filter_reset(&filter);

    const bool drive = !POLICY_SYNTHETIC;
    if (!drive) {
        ESP_LOGW(TAG, "合成ポリシーなのでサーボへ出力しない");
    } else {
        servo_write(cmd);
    }

    float phase = 0.0f;
    uint32_t worst_cycles = 0;
    int64_t sum_cycles = 0;
    int steps = 0;
    TickType_t last_wake = xTaskGetTickCount();
    const TickType_t period = pdMS_TO_TICKS(1000 / CONTROL_HZ);

    for (;;) {
        vTaskDelayUntil(&last_wake, period);

        if (!imu_read(&imu)) {
            ESP_LOGE(TAG, "IMU 読み取り失敗: 指令を保持");
            continue;
        }
        goal_read(goal);
        obs_build(&imu, cmd, before, goal, phase, CONTROL_DT_S, obs);

        const uint32_t t0 = esp_cpu_get_cycle_count();
#if USE_INT8_POLICY && POLICY_HAS_INT8
        policy_forward_i8(obs, raw, NULL);
#else
        policy_forward_f32(obs, raw);
#endif
        const uint32_t cycles = esp_cpu_get_cycle_count() - t0;

        memcpy(before, cmd, sizeof before);
        command_filter_step(&filter, raw, cmd);
        if (drive) {
            servo_write(cmd);
        }

        phase += 2.0f * (float)M_PI * GAIT_HZ * CONTROL_DT_S;
        if (phase >= 2.0f * (float)M_PI) {
            phase -= 2.0f * (float)M_PI;
        }

        if (cycles > worst_cycles) {
            worst_cycles = cycles;
        }
        sum_cycles += cycles;
        if (++steps == CONTROL_HZ) {
            ESP_LOGI(TAG, "推論 平均 %lld / 最大 %lu サイクル（%s）",
                     (long long)(sum_cycles / steps), (unsigned long)worst_cycles,
                     USE_INT8_POLICY ? "int8" : "float32");
            steps = 0;
            sum_cycles = 0;
            worst_cycles = 0;
        }
    }
}

#endif /* Q8_RUN_POLICY */
