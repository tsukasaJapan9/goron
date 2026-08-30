"""Measure the floor's friction coefficient with the robot's own IMU.

    uv run python -m scripts.friction_test --port /dev/ttyUSB0 --surface belly

Rest the robot on a board, tilt the board slowly, and press Enter the moment it
starts to slide. At the point of slipping the friction force is at its limit,
so mu = tan(theta) -- and theta is exactly what the accelerometer reads, since
the robot lies flat on the board.

With `--method slide` the board is not needed at all, which matters when the
floor the robot actually runs on cannot be tilted. Shove the robot along that
floor and let it coast to a stop: while it slides, the only horizontal force is
friction, so the accelerometer's "up" tips away from vertical by exactly
atan(mu). It is the same measurement as the ramp, done on the real surface --
and being a ratio of two accelerometer axes, it does not care about scale
error. The firmware records at ~4 kHz for this, since a slide lasts under a
second and the 20 Hz telemetry would catch only a handful of samples.

Which surface touches the floor is set by the crank angle, so the two contacts
in the model can be measured apart:

    belly  crank 130 deg, legs at their highest -- only the torso touches,
           giving the floor/torso pair (`floor_friction`)
    feet   crank 340 deg, legs at their lowest -- the robot stands on the leg
           tips, giving the leg/floor pair (`foot_friction`)
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import serial

IMU_MAP = Path(__file__).resolve().parent.parent / "hardware" / "imu_map.json"
CRANK = {"belly": 130.0, "feet": 340.0}


def tilt_deg(up_body: np.ndarray) -> float:
    """Angle between the body's up axis and the world's, in degrees."""
    n = np.linalg.norm(up_body)
    return math.degrees(math.acos(np.clip(up_body[2] / (n or 1.0), -1.0, 1.0)))


def capture(port: serial.Serial, ms: int) -> np.ndarray:
    """Ask the firmware for a high-rate accelerometer burst: (t_s, ax, ay, az)."""
    port.reset_input_buffer()
    port.write(f"A {ms}\n".encode()); port.flush()
    rows = []
    deadline = time.time() + ms / 1000 + 25
    while time.time() < deadline:
        line = port.readline().decode("utf-8", "replace").strip()
        if line == "A,end":
            break
        if not line.startswith("A,"):
            continue
        f = line.split(",")
        if len(f) == 5 and f[1].isdigit():
            rows.append([int(f[1]) / 1e6] + [int(v) / 10000.0 for v in f[2:]])
    return np.array(rows)


def slide_estimate(trace: np.ndarray, up_map: np.ndarray,
                   verbose: bool = False) -> tuple[float, float, int]:
    """mu from the slide, measured against how the robot reads while at rest.

    Referencing the resting orientation rather than world vertical removes both
    the floor's own slope and any leftover error in the IMU calibration: what is
    wanted is how far the specific force tips *while sliding*, not where "up" is.

    The slide is the longest run of samples whose deviation stays well above the
    resting noise -- not the longest steady run, which is the robot sitting
    still and is what an earlier version of this locked onto, happily reporting
    the noise floor as a friction coefficient of 0.01.
    """
    up = trace[:, 1:4] @ up_map.T
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    t = trace[:, 0]

    # Rest is the quietest half-second: least movement, so the truest reference.
    win = max(10, min(len(up) // 4, int(0.5 / np.median(np.diff(t)))))
    spread = np.array([up[i:i + win].std(axis=0).sum()
                       for i in range(0, len(up) - win)])
    if len(spread) == 0:
        return 0.0, 0.0, 0
    rest = up[(q := int(spread.argmin())):q + win].mean(axis=0)
    rest /= np.linalg.norm(rest)
    dev = np.degrees(np.arccos(np.clip(up @ rest, -1.0, 1.0)))
    noise = float(np.percentile(dev[q:q + win], 95))

    thresh = max(3.0 * noise, 2.0)
    above = dev > thresh
    best, i = (0.0, 0.0, 0), 0
    while i < len(above):
        if not above[i]:
            i += 1
            continue
        j = i
        while j < len(above) and above[j]:
            j += 1
        if (dur := t[j - 1] - t[i]) > best[1]:
            best = (float(np.median(dev[i:j])), float(dur), j - i)
        i = j

    if verbose:
        print(f"     静止時のノイズ {noise:.2f} 度、しきい値 {thresh:.2f} 度、"
              f"全体の最大 {dev.max():.2f} 度")
        step = max(1, len(dev) // 20)
        prof = "".join(" .:-=+*#%@"[min(9, int(dev[k:k + step].max() / 3))]
                       for k in range(0, len(dev), step))
        print(f"     経過({t[-1]:.1f}s): |{prof}|  (1目盛3度)")
    return best


def run_slide(port: serial.Serial, up_map: np.ndarray, args) -> None:
    results = []
    for trial in range(1, args.trials + 1):
        input(f"\n--- {trial} 回目 ---  Enter を押したら床の上で機体を押してください: ")
        trace = capture(port, args.window_ms)
        if len(trace) < 100:
            print("  サンプルが足りません。テレメトリと接続を確認してください")
            continue
        rate = len(trace) / (trace[-1, 0] - trace[0, 0])
        theta, dur, n = slide_estimate(trace, up_map, verbose=True)
        if dur < 0.05 or theta < 2.0:
            print(f"  滑走が見つかりません（記録 {len(trace)} 点 / {rate:.0f} Hz）。"
                  "もう少し強く、まっすぐ押してください")
            continue
        mu = math.tan(math.radians(theta))
        print(f"  滑走 {dur*1000:.0f} ms ({n} 点)  傾き {theta:5.2f} 度  ->  mu = {mu:.3f}")
        results.append(mu)

    if results:
        r = np.array(results)
        print(f"\n{args.surface}: mu = {r.mean():.3f} ± {r.std():.3f}  "
              f"（{len(r)} 回、{', '.join(f'{v:.3f}' for v in r)}）")
        key = "floor_friction" if args.surface == "belly" else "foot_friction"
        print(f"RobotParams.{key} に入れる値です")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--surface", choices=sorted(CRANK), default="belly")
    ap.add_argument("--method", choices=("slide", "tilt"), default="slide",
                    help="slide: shove it along the real floor (default). "
                         "tilt: put it on a board and tip the board over")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--window-ms", type=int, default=3000,
                    help="slide only: how long to record after the go signal")
    args = ap.parse_args()

    m = np.array(json.loads(IMU_MAP.read_text())["matrix"]) if IMU_MAP.exists() \
        else np.eye(3)
    if not IMU_MAP.exists():
        print("警告: hardware/imu_map.json がありません。先に scripts.imu_calib を実行してください")

    port = serial.Serial(args.port, args.baud, timeout=1)
    crank = CRANK[args.surface]
    print(f"脚を crank {crank:.0f} 度へ動かします（{args.surface} が接地する角度）")
    port.write(f"G {crank}\n".encode()); port.flush()
    time.sleep(4.0)

    if args.method == "slide":
        run_slide(port, m, args)
        return

    results = []
    for trial in range(1, args.trials + 1):
        print(f"\n--- {trial} 回目 ---")
        print("  板の上に置いて、ゆっくり傾けてください。")
        print("  滑り出した瞬間に Enter を押します。")
        trace: list[tuple[float, float, float]] = []
        stop = threading.Event()

        def collect() -> None:
            port.reset_input_buffer()
            while not stop.is_set():
                line = port.readline().decode("utf-8", "replace").strip()
                if not line.startswith("T,"):
                    continue
                f = line.split(",")
                if len(f) != 12:
                    continue
                a = np.array([float(f[6]), float(f[7]), float(f[8])])
                up = m @ a
                trace.append((time.time(), tilt_deg(up), float(np.linalg.norm(a))))

        t = threading.Thread(target=collect, daemon=True)
        t.start()
        input("  （傾けて、滑ったら Enter）")
        stop.set(); t.join(timeout=2)

        if len(trace) < 10:
            print("  データが足りません。テレメトリが流れているか確認してください")
            continue
        arr = np.array(trace)
        now = arr[-1, 0]
        recent = arr[arr[:, 0] > now - 1.0]       # 直前1秒のピークを滑り出しとみなす
        theta = float(recent[:, 1].max())
        mu = math.tan(math.radians(theta))
        # 滑走中は接触力が変わるので |a| が 1g からずれる。自動検出の補助に見る
        wobble = float(np.abs(recent[:, 2] - 1.0).max())
        print(f"  傾き {theta:5.2f} 度  ->  mu = {mu:.3f}   "
              f"(|a| の 1g からのずれ 最大 {wobble:.3f} g)")
        results.append(mu)

    if results:
        r = np.array(results)
        print(f"\n{args.surface}: mu = {r.mean():.3f} ± {r.std():.3f}  "
              f"（{len(r)} 回、{', '.join(f'{v:.3f}' for v in r)}）")
        key = "floor_friction" if args.surface == "belly" else "foot_friction"
        print(f"RobotParams.{key} に入れる値です")


if __name__ == "__main__":
    main()
