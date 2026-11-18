/*
 * servo_console.h — 試験・較正モードのつなぎ（servo_cal と servo_hal を組み合わせる）。
 *
 *   servo_console_start  ハードの初期化、保存した較正の読み出し、案内の表示
 *   servo_console_feed   シリアルの 1 文字を渡す（改行で 1 行のコマンドになる）
 *   servo_console_line   1 行のコマンドを直接渡す
 *   servo_console_tick   20 ms ごとに呼ぶ（速さを制限して目標へ近づけ、パルスを出す）
 */
#ifndef SERVO_CONSOLE_H
#define SERVO_CONSOLE_H

#include <stddef.h>

#include "servo_cal.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    servo_cal_t cal;
    servo_cal_io_t io;
    char line[96];
    size_t len;
    int overflow;
} servo_console_t;

int servo_console_start(servo_console_t *c);
void servo_console_feed(servo_console_t *c, int ch);
void servo_console_line(servo_console_t *c, const char *line);
void servo_console_tick(servo_console_t *c);

#ifdef __cplusplus
}
#endif

#endif /* SERVO_CONSOLE_H */
