/*
 * servo_map.h — deploy/esp32/servo_calibration.py が生成。手で編集しない。
 * 元: nominal（名目値。未較正）  タスク: robot/mjcf/mini-r2-p7-eff/cut/walk-task.json
 *
 * 方策の行動の順。角度[度] → パルス幅: center_us + sign * angle * us_per_deg
 * 関節角 0 は立った姿勢、正の角度で足先が後ろへ動く。lo/hi は範囲の端のパルス幅（0 = 未測定）。
 *
 *   #  関節      GPIO  向き  中立[µs]  µs/度    lo[µs]   hi[µs]  学習の範囲[度]
 *   0  hip_lf       4  +1     1450.0  10.556      0.0      0.0  -45〜+20
 *   1  knee_lf      5  +1     1450.0  10.556      0.0      0.0  -60〜+45
 *   2  hip_rf       6  -1     1450.0  10.556      0.0      0.0  -45〜+20
 *   3  knee_rf      7  -1     1450.0  10.556      0.0      0.0  -60〜+45
 *   4  hip_lh      15  +1     1450.0  10.556      0.0      0.0  -35〜+45
 *   5  knee_lh     16  +1     1450.0  10.556      0.0      0.0  -60〜+45
 *   6  hip_rh      17  -1     1450.0  10.556      0.0      0.0  -35〜+45
 *   7  knee_rh     18  -1     1450.0  10.556      0.0      0.0  -60〜+45
 */

#ifndef SERVO_MAP_H
#define SERVO_MAP_H

#define SERVO_MAP_N 8
#define SERVO_MAP_CALIBRATED 0
#define SERVO_MAP_HARD_MIN_US 500.0f
#define SERVO_MAP_HARD_MAX_US 2400.0f
#define SERVO_MAP_EXPLORE_MARGIN_DEG 10.0f

#define SERVO_MAP_NAMES {"hip_lf", "knee_lf", "hip_rf", "knee_rf", "hip_lh", "knee_lh", "hip_rh", "knee_rh"}
#define SERVO_MAP_GPIO {4, 5, 6, 7, 15, 16, 17, 18}
#define SERVO_MAP_SIGN {1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f}
#define SERVO_MAP_CENTER_US {1450.0f, 1450.0f, 1450.0f, 1450.0f, 1450.0f, 1450.0f, 1450.0f, 1450.0f}
#define SERVO_MAP_US_PER_DEG {10.555556f, 10.555556f, 10.555556f, 10.555556f, 10.555556f, 10.555556f, 10.555556f, 10.555556f}
#define SERVO_MAP_LO_US {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
#define SERVO_MAP_HI_US {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
#define SERVO_MAP_MODEL_LOW_DEG {-45.0f, -60.0f, -45.0f, -60.0f, -35.0f, -60.0f, -35.0f, -60.0f}
#define SERVO_MAP_MODEL_HIGH_DEG {20.0f, 45.0f, 20.0f, 45.0f, 45.0f, 45.0f, 45.0f, 45.0f}

#endif /* SERVO_MAP_H */
