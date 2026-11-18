/*
 * servo_hal.h — 試験・較正モードがハードウェアに触る所だけを集めた薄い層。
 *
 *   ESP32-S3:  main/servo_hal_esp.c（LEDC の PWM、NVS、USB シリアルのコンソール）
 *   ホスト:    host/servo_host_main.c（テスト用の偽物。パルスは記録するだけ、保存はファイル）
 *
 * ロジック（servo_cal.c、servo_console.c）はこの関数だけを呼ぶ。
 */
#ifndef SERVO_HAL_H
#define SERVO_HAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* n 個のチャネルを gpio[k] に割り当てる。パルスはまだ出さない。0 = 成功。 */
int servo_hal_init(const int *gpio, int n);
/* チャネル ch に pulse_us のパルスを出す。on が 0 ならパルスを止める（サーボの力が抜ける）。 */
void servo_hal_write(int ch, float pulse_us, int on);
/* 較正のバイト列を保存する。0 = 成功。 */
int servo_hal_store_save(const uint8_t *blob, size_t len);
/* 保存したバイト列を読む。0 = 成功、1 = 保存が無い、負 = 失敗。 */
int servo_hal_store_load(uint8_t *blob, size_t cap, size_t *len);
/* 1 行を出す（改行は付けて出す）。 */
void servo_hal_print(const char *line);
/* シリアルから 1 文字。来ていなければ -1（待たない）。 */
int servo_hal_getchar(void);

#ifdef __cplusplus
}
#endif

#endif /* SERVO_HAL_H */
