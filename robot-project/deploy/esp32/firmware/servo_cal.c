/*
 * servo_cal.c — servo_cal.h の実装。ハードウェアに触らない（PWM・保存・シリアルは呼び出し側）。
 * Python 側の対応物: deploy/esp32/servo_calibration.py（dump の読み取り、servo.json、servo_map.h）。
 */
#include "servo_cal.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *const servo_cal_names[SERVO_CAL_N] = SERVO_MAP_NAMES;
const int servo_cal_gpio[SERVO_CAL_N] = SERVO_MAP_GPIO;

static const float map_sign[SERVO_CAL_N] = SERVO_MAP_SIGN;
static const float map_center[SERVO_CAL_N] = SERVO_MAP_CENTER_US;
static const float map_upd[SERVO_CAL_N] = SERVO_MAP_US_PER_DEG;
static const float map_lo[SERVO_CAL_N] = SERVO_MAP_LO_US;
static const float map_hi[SERVO_CAL_N] = SERVO_MAP_HI_US;
static const float model_low[SERVO_CAL_N] = SERVO_MAP_MODEL_LOW_DEG;
static const float model_high[SERVO_CAL_N] = SERVO_MAP_MODEL_HIGH_DEG;

/* ------------------------------------------------------------------------- */
/* 換算と範囲                                                                 */
/* ------------------------------------------------------------------------- */

static float clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

float servo_cal_angle_to_us(const servo_joint_cal_t *c, float deg)
{
    return c->center_us + c->sign * deg * c->us_per_deg;
}

float servo_cal_us_to_angle(const servo_joint_cal_t *c, float us)
{
    return (us - c->center_us) / (c->sign * c->us_per_deg);
}

void servo_cal_window(const servo_cal_t *s, int j, float *lo_us, float *hi_us)
{
    const servo_joint_cal_t *c = &s->cal[j];
    const float a = c->lo_us > 0.0f
                        ? c->lo_us
                        : servo_cal_angle_to_us(c, model_low[j] - SERVO_MAP_EXPLORE_MARGIN_DEG);
    const float b = c->hi_us > 0.0f
                        ? c->hi_us
                        : servo_cal_angle_to_us(c, model_high[j] + SERVO_MAP_EXPLORE_MARGIN_DEG);
    *lo_us = clampf(a < b ? a : b, SERVO_MAP_HARD_MIN_US, SERVO_MAP_HARD_MAX_US);
    *hi_us = clampf(a < b ? b : a, SERVO_MAP_HARD_MIN_US, SERVO_MAP_HARD_MAX_US);
}

uint32_t servo_cal_duty(float pulse_us)
{
    const float per_us = (float)(1u << 14) / (1.0e6f / 50.0f);
    long d = lrintf(clampf(pulse_us, SERVO_MAP_HARD_MIN_US, SERVO_MAP_HARD_MAX_US) * per_us);
    const long top = (1L << 14) - 1;
    return (uint32_t)(d < 0 ? 0 : (d > top ? top : d));
}

void servo_cal_defaults(servo_joint_cal_t cal[SERVO_CAL_N])
{
    for (int j = 0; j < SERVO_CAL_N; ++j) {
        cal[j].center_us = map_center[j];
        cal[j].sign = map_sign[j];
        cal[j].us_per_deg = map_upd[j];
        cal[j].lo_us = map_lo[j];
        cal[j].hi_us = map_hi[j];
    }
}

/* ------------------------------------------------------------------------- */
/* 保存用のバイト列                                                           */
/* ------------------------------------------------------------------------- */

static uint32_t crc32(const uint8_t *p, size_t n)
{
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) {
        c ^= p[i];
        for (int k = 0; k < 8; ++k) {
            c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
        }
    }
    return ~c;
}

static void put_u32(uint8_t *p, uint32_t v)
{
    for (int k = 0; k < 4; ++k) {
        p[k] = (uint8_t)(v >> (8 * k));
    }
}

static uint32_t get_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void put_f32(uint8_t *p, float v)
{
    uint32_t u;
    memcpy(&u, &v, sizeof u);
    put_u32(p, u);
}

static float get_f32(const uint8_t *p)
{
    const uint32_t u = get_u32(p);
    float v;
    memcpy(&v, &u, sizeof v);
    return v;
}

size_t servo_cal_serialize(const servo_joint_cal_t cal[SERVO_CAL_N], uint8_t *buf, size_t cap)
{
    if (cap < SERVO_CAL_BLOB_SIZE) {
        return 0;
    }
    memcpy(buf, "Q8SC", 4);
    buf[4] = (uint8_t)(SERVO_CAL_BLOB_VERSION & 0xFF);
    buf[5] = (uint8_t)(SERVO_CAL_BLOB_VERSION >> 8);
    buf[6] = (uint8_t)(SERVO_CAL_N & 0xFF);
    buf[7] = (uint8_t)(SERVO_CAL_N >> 8);
    uint8_t *p = buf + 8;
    for (int j = 0; j < SERVO_CAL_N; ++j) {
        const float v[5] = {cal[j].center_us, cal[j].sign, cal[j].us_per_deg, cal[j].lo_us,
                            cal[j].hi_us};
        for (int k = 0; k < 5; ++k, p += 4) {
            put_f32(p, v[k]);
        }
    }
    put_u32(p, crc32(buf, (size_t)(p - buf)));
    return SERVO_CAL_BLOB_SIZE;
}

static int pulse_ok(float us)
{
    return isfinite(us) && us >= SERVO_MAP_HARD_MIN_US && us <= SERVO_MAP_HARD_MAX_US;
}

int servo_cal_deserialize(const uint8_t *buf, size_t len, servo_joint_cal_t cal[SERVO_CAL_N])
{
    if (len != SERVO_CAL_BLOB_SIZE || memcmp(buf, "Q8SC", 4) != 0 ||
        (buf[4] | (buf[5] << 8)) != SERVO_CAL_BLOB_VERSION || (buf[6] | (buf[7] << 8)) != SERVO_CAL_N) {
        return -1;
    }
    if (crc32(buf, len - 4) != get_u32(buf + len - 4)) {
        return -2;
    }
    servo_joint_cal_t tmp[SERVO_CAL_N];
    const uint8_t *p = buf + 8;
    for (int j = 0; j < SERVO_CAL_N; ++j, p += 20) {
        tmp[j].center_us = get_f32(p);
        tmp[j].sign = get_f32(p + 4);
        tmp[j].us_per_deg = get_f32(p + 8);
        tmp[j].lo_us = get_f32(p + 12);
        tmp[j].hi_us = get_f32(p + 16);
        const servo_joint_cal_t *c = &tmp[j];
        if (!pulse_ok(c->center_us) || (c->sign != 1.0f && c->sign != -1.0f) ||
            !isfinite(c->us_per_deg) || c->us_per_deg < SERVO_CAL_UPD_MIN ||
            c->us_per_deg > SERVO_CAL_UPD_MAX || (c->lo_us != 0.0f && !pulse_ok(c->lo_us)) ||
            (c->hi_us != 0.0f && !pulse_ok(c->hi_us))) {
            return -3;
        }
    }
    memcpy(cal, tmp, sizeof tmp);
    return 0;
}

/* ------------------------------------------------------------------------- */
/* 応答                                                                       */
/* ------------------------------------------------------------------------- */

static void say(const servo_cal_io_t *io, const char *fmt, ...)
{
    char line[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    if (io && io->emit) {
        io->emit(io->ctx, line);
    }
}

static int fail(const servo_cal_io_t *io, const char *fmt, ...)
{
    char msg[224];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    say(io, "err %s", msg);
    return -1;
}

static void dump(const servo_cal_t *s, const servo_cal_io_t *io)
{
    say(io, "CAL begin v=1 n=%d saved=%d", SERVO_CAL_N, (s->in_store && !s->dirty) ? 1 : 0);
    for (int j = 0; j < SERVO_CAL_N; ++j) {
        const servo_joint_cal_t *c = &s->cal[j];
        say(io, "CAL %d %s gpio=%d center=%.2f sign=%d upd=%.6f lo=%.2f hi=%.2f", j,
            servo_cal_names[j], servo_cal_gpio[j], (double)c->center_us, c->sign > 0 ? 1 : -1,
            (double)c->us_per_deg, (double)c->lo_us, (double)c->hi_us);
    }
    say(io, "CAL end");
}

static void help(const servo_cal_io_t *io)
{
    static const char *const lines[] = {
        "コマンド（1行ずつ。J は 0〜7 か関節名 hip_lf など）:",
        "  sel J      関節 J を選び、パルスを出す（初めてなら中立へ）",
        "  center     中立（関節角 0）へ動かす",
        "  + / -      少し動かす（+ は関節角を増やす = 足先が後ろへ）。+ 3 のように回数も可",
        "  step U     + / - の1回ぶん [µs]（1〜50、既定 5 ≒ 0.5°）",
        "  deg A      関節角 A[度] へ動かす     us P   パルス幅 P[µs] へ動かす",
        "  speed V    動く速さ [度/s]（5〜90、既定 30）",
        "  test       中立 → +10° → -10° → 中立 とゆっくり動かす（動作確認）",
        "  zero       今の位置を中立（立った姿勢）として記録",
        "  flip       向き（sign）を反転",
        "  lo / hi    今の位置を範囲の端として記録（lo = 足先が前の端、hi = 後ろの端）",
        "  clear      選んだ関節の lo / hi を消す     reset  選んだ関節を既定値に戻す",
        "  mark A     今の位置が関節角 A[度] だと記録し、µs/度 と向きを決める（|A| >= 15）",
        "  status     選んだ関節の状態     dump / list   全関節の較正を出す",
        "  save       保存する（電源を切っても残る）     load   保存した値に戻す",
        "  off        すべてのパルスを止める（サーボの力が抜ける）",
    };
    for (size_t k = 0; k < sizeof lines / sizeof lines[0]; ++k) {
        say(io, "%s", lines[k]);
    }
}

/* ------------------------------------------------------------------------- */
/* 初期化・読み出し                                                           */
/* ------------------------------------------------------------------------- */

static void load_from_store(servo_cal_t *s, const servo_cal_io_t *io)
{
    uint8_t blob[SERVO_CAL_BLOB_SIZE + 8];
    size_t len = 0;
    const int got = (io && io->load) ? io->load(io->ctx, blob, sizeof blob, &len) : 1;
    servo_cal_defaults(s->cal);
    s->in_store = 0;
    s->dirty = 0;
    if (got == 0) {
        const int r = servo_cal_deserialize(blob, len, s->cal);
        if (r == 0) {
            s->in_store = 1;
            say(io, "ok 保存した較正を読み出した");
            return;
        }
        servo_cal_defaults(s->cal);
        say(io, "warn 保存した較正が壊れていた（%d）ので既定値（servo_map.h）で始める", r);
        return;
    }
    if (got == 1) {
        say(io, "ok 保存が無いので既定値（servo_map.h%s）で始める",
            SERVO_MAP_CALIBRATED ? "、較正済み" : "、名目値");
        return;
    }
    say(io, "warn 保存を読めなかった（%d）ので既定値（servo_map.h）で始める", got);
}

void servo_cal_init(servo_cal_t *s, const servo_cal_io_t *io)
{
    memset(s, 0, sizeof *s);
    s->selected = -1;
    s->wiggle_joint = -1;
    s->step_us = SERVO_CAL_STEP_DEFAULT;
    s->speed_deg_s = SERVO_CAL_SPEED_DEFAULT;
    say(io, "サーボの試験・較正モード。方策は動かさない。help でコマンドの一覧");
    say(io, "起動直後はどのサーボにもパルスを出さない。sel で1つずつ選ぶ");
    load_from_store(s, io);
}

/* ------------------------------------------------------------------------- */
/* コマンド                                                                   */
/* ------------------------------------------------------------------------- */

static int find_joint(const char *arg)
{
    if (!arg || !*arg) {
        return -1;
    }
    char *end = NULL;
    const long k = strtol(arg, &end, 10);
    if (end && *end == '\0') {
        return (k >= 0 && k < SERVO_CAL_N) ? (int)k : -1;
    }
    for (int j = 0; j < SERVO_CAL_N; ++j) {
        if (strcmp(arg, servo_cal_names[j]) == 0) {
            return j;
        }
    }
    return -1;
}

static int parse_float(const char *arg, float *out)
{
    if (!arg || !*arg) {
        return 0;
    }
    char *end = NULL;
    const float v = strtof(arg, &end);
    if (!end || *end != '\0' || !isfinite(v)) {
        return 0;
    }
    *out = v;
    return 1;
}

/* 目標を範囲に収めて置く。収めたら知らせる。 */
static void set_target(servo_cal_t *s, int j, float us, const servo_cal_io_t *io)
{
    float lo, hi;
    servo_cal_window(s, j, &lo, &hi);
    const float t = clampf(us, lo, hi);
    s->target_us[j] = t;
    const servo_joint_cal_t *c = &s->cal[j];
    if (t != us) {
        say(io, "ok %s → %.1f µs（%+.1f°）。範囲 %.1f〜%.1f µs の端で止める", servo_cal_names[j],
            (double)t, (double)servo_cal_us_to_angle(c, t), (double)lo, (double)hi);
    } else {
        say(io, "ok %s → %.1f µs（%+.1f°）", servo_cal_names[j], (double)t,
            (double)servo_cal_us_to_angle(c, t));
    }
}

static int need_joint(const servo_cal_t *s, const servo_cal_io_t *io)
{
    if (s->selected < 0) {
        fail(io, "関節を選んでいない（sel 0〜7 か sel hip_lf のように）");
        return 0;
    }
    return 1;
}

static int need_still(const servo_cal_t *s, int j, const servo_cal_io_t *io)
{
    if (fabsf(s->current_us[j] - s->target_us[j]) > 0.01f || s->wiggle_joint >= 0) {
        fail(io, "まだ動いている。止まってからもう一度");
        return 0;
    }
    return 1;
}

static void check_ends(const servo_cal_t *s, int j, const servo_cal_io_t *io)
{
    const servo_joint_cal_t *c = &s->cal[j];
    if (c->lo_us > 0.0f && c->hi_us > 0.0f &&
        !(servo_cal_us_to_angle(c, c->lo_us) < servo_cal_us_to_angle(c, c->hi_us))) {
        say(io, "warn %s: lo（足先が前）と hi（後ろ）が向きと合わない。flip してから測り直す",
            servo_cal_names[j]);
    }
}

static void status(const servo_cal_t *s, int j, const servo_cal_io_t *io)
{
    const servo_joint_cal_t *c = &s->cal[j];
    float lo, hi;
    servo_cal_window(s, j, &lo, &hi);
    say(io, "ok %s（GPIO %d）: 目標 %.1f µs（%+.1f°）、今 %.1f µs、出力 %s", servo_cal_names[j],
        servo_cal_gpio[j], (double)s->target_us[j],
        (double)servo_cal_us_to_angle(c, s->target_us[j]), (double)s->current_us[j],
        s->enabled[j] ? "あり" : "なし");
    say(io, "ok   中立 %.1f µs、向き %+d、%.3f µs/度、lo %.1f、hi %.1f（0 = 未測定）、"
        "許す範囲 %.1f〜%.1f µs、学習の範囲 %+.0f〜%+.0f°",
        (double)c->center_us, c->sign > 0 ? 1 : -1, (double)c->us_per_deg, (double)c->lo_us,
        (double)c->hi_us, (double)lo, (double)hi, (double)model_low[j], (double)model_high[j]);
}

int servo_cal_command(servo_cal_t *s, const char *line, const servo_cal_io_t *io)
{
    char buf[96];
    size_t n = strlen(line);
    if (n >= sizeof buf) {
        return fail(io, "行が長すぎる");
    }
    memcpy(buf, line, n + 1);
    while (n > 0 && (buf[n - 1] == ' ' || buf[n - 1] == '\t' || buf[n - 1] == '\r')) {
        buf[--n] = '\0';
    }
    char *cmd = buf;
    while (*cmd == ' ' || *cmd == '\t') {
        ++cmd;
    }
    if (*cmd == '\0') {
        return 0;
    }
    char *arg = cmd;
    while (*arg && *arg != ' ' && *arg != '\t') {
        ++arg;
    }
    if (*arg) {
        *arg++ = '\0';
        while (*arg == ' ' || *arg == '\t') {
            ++arg;
        }
    }
    const int j = s->selected;
    float v = 0.0f;

    if (strcmp(cmd, "help") == 0 || strcmp(cmd, "?") == 0) {
        help(io);
        return 0;
    }
    if (strcmp(cmd, "dump") == 0 || strcmp(cmd, "list") == 0) {
        dump(s, io);
        return 0;
    }
    if (strcmp(cmd, "save") == 0) {
        uint8_t blob[SERVO_CAL_BLOB_SIZE];
        const size_t len = servo_cal_serialize(s->cal, blob, sizeof blob);
        const int r = (io && io->save) ? io->save(io->ctx, blob, len) : -1;
        if (r != 0) {
            return fail(io, "保存できなかった（%d）", r);
        }
        s->in_store = 1;
        s->dirty = 0;
        say(io, "ok 保存した");
        return 0;
    }
    if (strcmp(cmd, "load") == 0) {
        load_from_store(s, io);
        return 0;
    }
    if (strcmp(cmd, "off") == 0) {
        for (int k = 0; k < SERVO_CAL_N; ++k) {
            s->enabled[k] = 0;
        }
        s->selected = -1;
        s->wiggle_joint = -1;
        say(io, "ok すべてのパルスを止めた");
        return 0;
    }
    if (strcmp(cmd, "speed") == 0) {
        if (!parse_float(arg, &v) || v < SERVO_CAL_SPEED_MIN || v > SERVO_CAL_SPEED_MAX) {
            return fail(io, "speed は %.0f〜%.0f 度/s", (double)SERVO_CAL_SPEED_MIN,
                        (double)SERVO_CAL_SPEED_MAX);
        }
        s->speed_deg_s = v;
        say(io, "ok 速さ %.1f 度/s", (double)v);
        return 0;
    }
    if (strcmp(cmd, "step") == 0) {
        if (!parse_float(arg, &v) || v < 1.0f || v > SERVO_CAL_STEP_MAX) {
            return fail(io, "step は 1〜%.0f µs", (double)SERVO_CAL_STEP_MAX);
        }
        s->step_us = v;
        say(io, "ok 1回 %.1f µs", (double)v);
        return 0;
    }
    if (strcmp(cmd, "sel") == 0) {
        const int k = find_joint(arg);
        if (k < 0) {
            return fail(io, "関節 '%s' が無い（0〜7 か hip_lf などの名前）", arg);
        }
        s->selected = k;
        s->wiggle_joint = -1;
        if (!s->enabled[k]) {
            if (s->current_us[k] <= 0.0f) {
                float lo, hi;
                servo_cal_window(s, k, &lo, &hi);
                s->current_us[k] = clampf(s->cal[k].center_us, lo, hi);
            }
            s->target_us[k] = s->current_us[k];
            s->enabled[k] = 1;
            say(io, "ok %s（GPIO %d）にパルスを出す: %.1f µs。サーボは最初の1回だけ速く動く",
                servo_cal_names[k], servo_cal_gpio[k], (double)s->current_us[k]);
        } else {
            say(io, "ok %s を選んだ", servo_cal_names[k]);
        }
        return 0;
    }

    /* ここから下は関節を選んでいることが前提 */
    if (strcmp(cmd, "+") == 0 || strcmp(cmd, "-") == 0) {
        if (!need_joint(s, io)) {
            return -1;
        }
        int times = 1;
        if (*arg) {
            times = atoi(arg);
            if (times < 1 || times > 20) {
                return fail(io, "回数は 1〜20");
            }
        }
        s->wiggle_joint = -1;
        const float dir = (cmd[0] == '+') ? 1.0f : -1.0f;
        set_target(s, j, s->target_us[j] + dir * s->cal[j].sign * s->step_us * (float)times, io);
        return 0;
    }
    if (strcmp(cmd, "us") == 0 || strcmp(cmd, "deg") == 0 || strcmp(cmd, "center") == 0) {
        if (!need_joint(s, io)) {
            return -1;
        }
        if (cmd[0] != 'c' && !parse_float(arg, &v)) {
            return fail(io, "%s には数値が要る", cmd);
        }
        s->wiggle_joint = -1;
        const float us = cmd[0] == 'u' ? v
                         : cmd[0] == 'd' ? servo_cal_angle_to_us(&s->cal[j], v)
                                         : s->cal[j].center_us;
        set_target(s, j, us, io);
        return 0;
    }
    if (strcmp(cmd, "status") == 0) {
        if (!need_joint(s, io)) {
            return -1;
        }
        status(s, j, io);
        return 0;
    }
    if (strcmp(cmd, "test") == 0) {
        if (!need_joint(s, io)) {
            return -1;
        }
        s->wiggle_joint = j;
        s->wiggle_stage = 0;
        s->wiggle_hold = 0;
        set_target(s, j, servo_cal_angle_to_us(&s->cal[j], SERVO_CAL_WIGGLE_DEG), io);
        return 0;
    }
    if (strcmp(cmd, "zero") == 0) {
        if (!need_joint(s, io) || !need_still(s, j, io)) {
            return -1;
        }
        s->cal[j].center_us = s->target_us[j];
        s->dirty = 1;
        say(io, "ok %s の中立を %.1f µs にした（関節角 0 = 立った姿勢）", servo_cal_names[j],
            (double)s->cal[j].center_us);
        return 0;
    }
    if (strcmp(cmd, "flip") == 0) {
        if (!need_joint(s, io)) {
            return -1;
        }
        s->cal[j].sign = -s->cal[j].sign;
        s->dirty = 1;
        say(io, "ok %s の向きを %+d にした", servo_cal_names[j], s->cal[j].sign > 0 ? 1 : -1);
        check_ends(s, j, io);
        return 0;
    }
    if (strcmp(cmd, "lo") == 0 || strcmp(cmd, "hi") == 0) {
        if (!need_joint(s, io) || !need_still(s, j, io)) {
            return -1;
        }
        const int is_lo = cmd[0] == 'l';
        if (is_lo) {
            s->cal[j].lo_us = s->target_us[j];
        } else {
            s->cal[j].hi_us = s->target_us[j];
        }
        s->dirty = 1;
        say(io, "ok %s の %s を %.1f µs（%+.1f°）にした", servo_cal_names[j], cmd,
            (double)s->target_us[j], (double)servo_cal_us_to_angle(&s->cal[j], s->target_us[j]));
        check_ends(s, j, io);
        return 0;
    }
    if (strcmp(cmd, "clear") == 0) {
        if (!need_joint(s, io)) {
            return -1;
        }
        s->cal[j].lo_us = 0.0f;
        s->cal[j].hi_us = 0.0f;
        s->dirty = 1;
        say(io, "ok %s の lo / hi を消した", servo_cal_names[j]);
        return 0;
    }
    if (strcmp(cmd, "reset") == 0) {
        if (!need_joint(s, io)) {
            return -1;
        }
        servo_joint_cal_t d[SERVO_CAL_N];
        servo_cal_defaults(d);
        s->cal[j] = d[j];
        s->dirty = 1;
        say(io, "ok %s を既定値に戻した", servo_cal_names[j]);
        return 0;
    }
    if (strcmp(cmd, "mark") == 0) {
        if (!need_joint(s, io) || !need_still(s, j, io)) {
            return -1;
        }
        if (!parse_float(arg, &v) || fabsf(v) < SERVO_CAL_MARK_MIN_DEG) {
            return fail(io, "mark の角度は ±%.0f° 以上（小さいと誤差が大きい）",
                        (double)SERVO_CAL_MARK_MIN_DEG);
        }
        const float d = s->target_us[j] - s->cal[j].center_us;
        const float upd = fabsf(d) / fabsf(v);
        if (upd < SERVO_CAL_UPD_MIN || upd > SERVO_CAL_UPD_MAX) {
            return fail(io, "%.2f µs/度 になる（%.0f〜%.0f のはず）。中立（zero）と角度を確かめる",
                        (double)upd, (double)SERVO_CAL_UPD_MIN, (double)SERVO_CAL_UPD_MAX);
        }
        s->cal[j].us_per_deg = upd;
        s->cal[j].sign = (d / v > 0.0f) ? 1.0f : -1.0f;
        s->dirty = 1;
        say(io, "ok %s: %.3f µs/度、向き %+d", servo_cal_names[j], (double)upd,
            s->cal[j].sign > 0 ? 1 : -1);
        check_ends(s, j, io);
        return 0;
    }
    return fail(io, "知らないコマンド '%s'（help で一覧）", cmd);
}

/* ------------------------------------------------------------------------- */
/* 周期の処理                                                                 */
/* ------------------------------------------------------------------------- */

static void wiggle_step(servo_cal_t *s)
{
    const int j = s->wiggle_joint;
    if (j < 0 || fabsf(s->current_us[j] - s->target_us[j]) > 0.01f) {
        return;
    }
    if (++s->wiggle_hold < SERVO_CAL_WIGGLE_HOLD_TICKS) {
        return;
    }
    s->wiggle_hold = 0;
    ++s->wiggle_stage;
    float lo, hi;
    servo_cal_window(s, j, &lo, &hi);
    if (s->wiggle_stage == 1) {
        s->target_us[j] = clampf(servo_cal_angle_to_us(&s->cal[j], -SERVO_CAL_WIGGLE_DEG), lo, hi);
    } else if (s->wiggle_stage == 2) {
        s->target_us[j] = clampf(s->cal[j].center_us, lo, hi);
    } else {
        s->wiggle_joint = -1;
    }
}

void servo_cal_tick(servo_cal_t *s, float out_us[SERVO_CAL_N], uint8_t out_on[SERVO_CAL_N])
{
    wiggle_step(s);
    for (int j = 0; j < SERVO_CAL_N; ++j) {
        if (!s->enabled[j]) {
            out_us[j] = 0.0f;
            out_on[j] = 0;
            continue;
        }
        const float rate = s->speed_deg_s * s->cal[j].us_per_deg / (float)SERVO_CAL_TICK_HZ;
        const float d = s->target_us[j] - s->current_us[j];
        if (fabsf(d) <= rate) {
            s->current_us[j] = s->target_us[j];
        } else {
            s->current_us[j] += d > 0.0f ? rate : -rate;
        }
        s->current_us[j] = clampf(s->current_us[j], SERVO_MAP_HARD_MIN_US, SERVO_MAP_HARD_MAX_US);
        out_us[j] = s->current_us[j];
        out_on[j] = 1;
    }
}
