/* テスト専用の仮のヘッダ（ESP-IDF v5 の esp_err.h のうち使う形だけ）。 */
#pragma once
#include <stdint.h>
typedef int esp_err_t;
#define ESP_OK 0
#define ESP_FAIL -1
const char *esp_err_to_name(esp_err_t code);
void esp_error_check_failed_stub(esp_err_t rc);
#define ESP_ERROR_CHECK(x) do { esp_err_t rc_ = (x); if (rc_ != ESP_OK) esp_error_check_failed_stub(rc_); } while (0)
