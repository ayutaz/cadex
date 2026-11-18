/*
 * servo_host_main.c — 試験・較正モードをホスト（macOS/Linux）で動かすハーネスと、servo_hal.h の偽物。
 * tests/test_servo_cal_c.py が clang でコンパイルして使う。ESP-IDF のビルドには入らない。
 *
 *   servo_host STORE      STORE は NVS の代わりのファイル
 *
 * 標準入力の 1 行ずつ:
 *   #tick N      20 ms の周期を N 回進め、毎回 "T p0 p1 ... p7" を出す（パルスを出していなければ "-"）
 *   #reboot      電源を入れ直したことにする（STORE は残る）
 *   #corrupt K   STORE の K バイト目を壊す
 *   それ以外     シリアルから届いた 1 行として、1 文字ずつ servo_console_feed に渡す
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "servo_console.h"
#include "servo_hal.h"

static const char *store_path;
static float pulse[SERVO_CAL_N];
static int on[SERVO_CAL_N];

int servo_hal_init(const int *gpio, int n)
{
    (void)gpio;
    for (int k = 0; k < n && k < SERVO_CAL_N; ++k) {
        on[k] = 0;
    }
    return 0;
}

void servo_hal_write(int ch, float pulse_us, int enable)
{
    pulse[ch] = pulse_us;
    on[ch] = enable;
}

int servo_hal_store_save(const uint8_t *blob, size_t len)
{
    FILE *f = fopen(store_path, "wb");
    if (!f) {
        return -1;
    }
    const size_t w = fwrite(blob, 1, len, f);
    fclose(f);
    return w == len ? 0 : -2;
}

int servo_hal_store_load(uint8_t *blob, size_t cap, size_t *len)
{
    FILE *f = fopen(store_path, "rb");
    if (!f) {
        return 1;
    }
    *len = fread(blob, 1, cap, f);
    fclose(f);
    return 0;
}

void servo_hal_print(const char *line)
{
    printf("%s\n", line);
}

int servo_hal_getchar(void)
{
    return -1;
}

static void corrupt(long k)
{
    FILE *f = fopen(store_path, "r+b");
    if (!f) {
        return;
    }
    if (fseek(f, k, SEEK_SET) == 0) {
        const int c = fgetc(f);
        if (c != EOF && fseek(f, k, SEEK_SET) == 0) {
            fputc(c ^ 0x5A, f);
        }
    }
    fclose(f);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "使い方: %s STORE\n", argv[0]);
        return 2;
    }
    store_path = argv[1];
    static servo_console_t con;
    servo_console_start(&con);
    char line[256];
    while (fgets(line, sizeof line, stdin)) {
        line[strcspn(line, "\r\n")] = '\0';
        if (strncmp(line, "#tick", 5) == 0) {
            const int n = atoi(line + 5);
            for (int t = 0; t < n; ++t) {
                servo_console_tick(&con);
                printf("T");
                for (int k = 0; k < SERVO_CAL_N; ++k) {
                    if (on[k]) {
                        printf(" %.4f", (double)pulse[k]);
                    } else {
                        printf(" -");
                    }
                }
                printf("\n");
            }
        } else if (strcmp(line, "#reboot") == 0) {
            servo_console_start(&con);
        } else if (strncmp(line, "#corrupt", 8) == 0) {
            corrupt(atol(line + 8));
        } else {
            for (const char *p = line; *p; ++p) {
                servo_console_feed(&con, (unsigned char)*p);
            }
            servo_console_feed(&con, '\n');
        }
    }
    return 0;
}
