/*
 * servo_hal_esp.c — servo_hal.h の ESP32-S3 版（ESP-IDF v5 系）。
 *
 *   PWM:   LEDC（50 Hz、14 bit）。パルスを止めるときはデューティ 0（信号が出ない = サーボの力が抜ける）
 *   保存:  NVS（名前空間 "q8"、キー "servo_cal"）
 *   入出力: 標準入出力（ESP-IDF のコンソール。既定は UART0）。読み取りは待たない形にする
 *
 * ESP-IDF ではまだビルドしていない。tests/test_servo_cal_c.py が仮のヘッダで構文と型だけを確かめる。
 */
#include "servo_hal.h"

#include <fcntl.h>
#include <stdio.h>

#include "driver/ledc.h"
#include "esp_err.h"
#include "esp_log.h"
#include "nvs.h"
#include "nvs_flash.h"

#include "servo_cal.h"

#define SERVO_NVS_NAMESPACE "q8"
#define SERVO_NVS_KEY "servo_cal"

static const char *TAG = "servo_hal";
static int nvs_ready = 0;

static int nvs_start(void)
{
    if (nvs_ready) {
        return 0;
    }
    esp_err_t e = nvs_flash_init();
    if (e == ESP_ERR_NVS_NO_FREE_PAGES || e == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        /* NVS の区画が古い形式か満杯: 消して作り直す（保存した較正も消える） */
        ESP_LOGW(TAG, "NVS を初期化し直す（%s）", esp_err_to_name(e));
        if (nvs_flash_erase() != ESP_OK) {
            return -1;
        }
        e = nvs_flash_init();
    }
    if (e != ESP_OK) {
        ESP_LOGE(TAG, "NVS を使えない（%s）", esp_err_to_name(e));
        return -1;
    }
    nvs_ready = 1;
    return 0;
}

int servo_hal_init(const int *gpio, int n)
{
    const ledc_timer_config_t timer = {
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .duty_resolution = LEDC_TIMER_14_BIT,
        .timer_num = LEDC_TIMER_0,
        .freq_hz = 50,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    if (ledc_timer_config(&timer) != ESP_OK) {
        return -1;
    }
    for (int k = 0; k < n; ++k) {
        const ledc_channel_config_t ch = {
            .gpio_num = gpio[k],
            .speed_mode = LEDC_LOW_SPEED_MODE,
            .channel = (ledc_channel_t)k,
            .timer_sel = LEDC_TIMER_0,
            .duty = 0, /* 起動直後はパルスを出さない */
            .hpoint = 0,
        };
        if (ledc_channel_config(&ch) != ESP_OK) {
            return -2;
        }
    }
    /* コンソールの読み取りで止まらないようにする（文字が無ければ EOF が返る） */
    const int flags = fcntl(fileno(stdin), F_GETFL, 0);
    if (flags >= 0) {
        fcntl(fileno(stdin), F_SETFL, flags | O_NONBLOCK);
    }
    nvs_start();
    return 0;
}

void servo_hal_write(int ch, float pulse_us, int on)
{
    const uint32_t duty = on ? servo_cal_duty(pulse_us) : 0u;
    ledc_set_duty(LEDC_LOW_SPEED_MODE, (ledc_channel_t)ch, duty);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, (ledc_channel_t)ch);
}

int servo_hal_store_save(const uint8_t *blob, size_t len)
{
    if (nvs_start() != 0) {
        return -1;
    }
    nvs_handle_t h;
    if (nvs_open(SERVO_NVS_NAMESPACE, NVS_READWRITE, &h) != ESP_OK) {
        return -2;
    }
    esp_err_t e = nvs_set_blob(h, SERVO_NVS_KEY, blob, len);
    if (e == ESP_OK) {
        e = nvs_commit(h);
    }
    nvs_close(h);
    return e == ESP_OK ? 0 : -3;
}

int servo_hal_store_load(uint8_t *blob, size_t cap, size_t *len)
{
    if (nvs_start() != 0) {
        return -1;
    }
    nvs_handle_t h;
    esp_err_t e = nvs_open(SERVO_NVS_NAMESPACE, NVS_READONLY, &h);
    if (e == ESP_ERR_NVS_NOT_FOUND) {
        return 1; /* まだ一度も保存していない */
    }
    if (e != ESP_OK) {
        return -2;
    }
    size_t size = cap;
    e = nvs_get_blob(h, SERVO_NVS_KEY, blob, &size);
    nvs_close(h);
    if (e == ESP_ERR_NVS_NOT_FOUND) {
        return 1;
    }
    if (e != ESP_OK) {
        return -3;
    }
    *len = size;
    return 0;
}

void servo_hal_print(const char *line)
{
    fputs(line, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

int servo_hal_getchar(void)
{
    const int c = fgetc(stdin);
    if (c == EOF) {
        clearerr(stdin);
        return -1;
    }
    return c;
}
