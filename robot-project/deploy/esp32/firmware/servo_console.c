/*
 * servo_console.c — servo_console.h の実装。ハードウェアには servo_hal.h を通してだけ触る。
 */
#include "servo_console.h"

#include <stdio.h>

#include "servo_hal.h"

static void emit(void *ctx, const char *line)
{
    (void)ctx;
    servo_hal_print(line);
}

static int save(void *ctx, const uint8_t *blob, size_t len)
{
    (void)ctx;
    return servo_hal_store_save(blob, len);
}

static int load(void *ctx, uint8_t *blob, size_t cap, size_t *len)
{
    (void)ctx;
    return servo_hal_store_load(blob, cap, len);
}

int servo_console_start(servo_console_t *c)
{
    c->io.emit = emit;
    c->io.save = save;
    c->io.load = load;
    c->io.ctx = c;
    c->len = 0;
    c->overflow = 0;
    const int r = servo_hal_init(servo_cal_gpio, SERVO_CAL_N);
    /* 初期化に失敗しても、パルスを出さないだけで、コンソールは使えるようにする */
    if (r != 0) {
        servo_hal_print("err PWM の初期化に失敗した");
    }
    for (int j = 0; j < SERVO_CAL_N; ++j) {
        servo_hal_write(j, 0.0f, 0);
    }
    servo_cal_init(&c->cal, &c->io);
    return r;
}

void servo_console_line(servo_console_t *c, const char *line)
{
    char echo[112];
    snprintf(echo, sizeof echo, "> %s", line);
    servo_hal_print(echo);
    servo_cal_command(&c->cal, line, &c->io);
}

void servo_console_feed(servo_console_t *c, int ch)
{
    if (ch == '\r' || ch == '\n') {
        if (c->overflow) {
            servo_hal_print("err 行が長すぎる");
        } else if (c->len > 0) {
            c->line[c->len] = '\0';
            servo_console_line(c, c->line);
        }
        c->len = 0;
        c->overflow = 0;
        return;
    }
    if (ch == 0x08 || ch == 0x7F) { /* バックスペース */
        if (c->len > 0) {
            --c->len;
        }
        return;
    }
    if (c->len + 1 >= sizeof c->line) {
        c->overflow = 1;
        return;
    }
    c->line[c->len++] = (char)ch;
}

void servo_console_tick(servo_console_t *c)
{
    float us[SERVO_CAL_N];
    uint8_t on[SERVO_CAL_N];
    servo_cal_tick(&c->cal, us, on);
    for (int j = 0; j < SERVO_CAL_N; ++j) {
        servo_hal_write(j, us[j], on[j]);
    }
}
