# ESP-IDF の仮のヘッダ（テスト専用）

`tests/test_servo_cal_c.py` が、`deploy/esp32/firmware/main/main.c` と `servo_hal_esp.c` を
ホストの clang で `-fsyntax-only` にかけるときだけ使う。ESP-IDF v5 の関数のうち、使うものの形だけを書いた。
本物の ESP-IDF でのビルドの代わりにはならない（関数の中身も、構造体の全部のメンバも無い）。
