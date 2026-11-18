/* テスト専用の仮のヘッダ（ESP-IDF v5 の FreeRTOS のうち使う形だけ）。 */
#pragma once
#include <stdint.h>
#include "freertos/FreeRTOS.h"
typedef void (*TaskFunction_t)(void *);
typedef void *TaskHandle_t;
TickType_t xTaskGetTickCount(void);
void vTaskDelayUntil(TickType_t *previous_wake_time, TickType_t time_increment);
BaseType_t xTaskCreate(TaskFunction_t task_code, const char *name, uint32_t stack_depth,
                       void *parameters, UBaseType_t priority, TaskHandle_t *created_task);
