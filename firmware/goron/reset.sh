#!/usr/bin/env bash
# Core2 をリセットして起動ログを流す。
#
#   ./reset.sh                     # /dev/ttyUSB0 を 20 秒ぶん読む
#   ./reset.sh /dev/ttyUSB1 60     # ポートと秒数を指定
#   ./reset.sh /dev/ttyUSB0 0      # リセットするだけ
#   ./reset.sh /dev/ttyUSB0 20 T   # テレメトリ(T,...)も表示する
#
# テレメトリは 20Hz で流れていて人が読むものではないので、既定では伏せる。
# 中身を見たいときは scripts/mirror.py を使う。
#
# pio device monitor は端末(tty)を要求するので、スクリプトから呼ぶとこける。
# ここでは pyserial で直接開いて RTS を叩く。

set -euo pipefail

PORT="${1:-/dev/ttyUSB0}"
READ_SEC="${2:-20}"
SHOW_TELEM="${3:-}"

PYTHON="$HOME/.platformio/penv/bin/python"  # pyserial が入っている PlatformIO の環境
[ -x "$PYTHON" ] || PYTHON=python3

exec "$PYTHON" - "$PORT" "$READ_SEC" "$SHOW_TELEM" <<'PY'
import sys, time, serial

port, read_sec = sys.argv[1], float(sys.argv[2])
show_telemetry = bool(sys.argv[3]) if len(sys.argv) > 3 else False
s = serial.Serial(port, 115200, timeout=1)

# DTR/RTS は USB-UART 経由で ESP32 のブート回路に繋がっている。
# DTR=False(IO0 を離す) のまま RTS をパルスさせると EN が落ちて通常起動でリセットされる。
s.setDTR(False)
s.setRTS(True)
time.sleep(0.1)
s.setRTS(False)

end = time.time() + read_sec
while time.time() < end:
    line = s.readline()
    if not line:
        continue
    text = line.decode("utf-8", "replace").rstrip()
    if text.startswith("T,") and not show_telemetry:
        continue
    print(text, flush=True)
PY
