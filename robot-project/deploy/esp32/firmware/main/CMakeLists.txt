# policy.c / control.c は firmware/ 直下にあり、ホストのテストでもそのまま使う。
idf_component_register(SRCS "main.c" "../policy.c" "../control.c"
                       INCLUDE_DIRS "." ".."
                       REQUIRES driver esp_hw_support log freertos)
