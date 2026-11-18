# policy.c / control.c / servo_cal.c / servo_console.c は firmware/ 直下にあり、ホストのテストでもそのまま使う。
# ハードウェアに触るのは main.c と servo_hal_esp.c だけ。
idf_component_register(SRCS "main.c" "servo_hal_esp.c" "../policy.c" "../control.c"
                            "../servo_cal.c" "../servo_console.c"
                       INCLUDE_DIRS "." ".."
                       REQUIRES driver esp_hw_support log freertos nvs_flash)
# 方策の制御ループを動かすときだけ: idf.py -DQ8_RUN_POLICY=1 build（既定は試験・較正モード）
if(Q8_RUN_POLICY)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE Q8_RUN_POLICY=1)
endif()
