"""Work out how the M5Stack's IMU axes sit in the robot's body frame.

    uv run python -m scripts.imu_calib --port /dev/ttyUSB0

At rest the accelerometer reads the world's up direction expressed in the IMU's
own axes. Hold the robot in a pose whose body-frame up direction is known and
the reading *is* that body axis, written in IMU coordinates. Two poses pin down
two axes; the third is their cross product, so the frame is fully determined.

The result is written to hardware/imu_map.json and picked up by scripts/mirror
automatically. It maps IMU axes onto the **simulation** body frame (+X forward,
+Y left, +Z up) -- the frame `GoronEnv` observes in, so the same matrix serves
the policy later. The URDF viewer applies the fixed CAD rotation on top.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import serial

OUT_PATH = Path(__file__).resolve().parent.parent / "hardware" / "imu_map.json"

# Each pose names the body-frame direction that points at the sky while held.
# The body frame's +X points AWAY from the screen: m5stack.stl sits at CAD
# Z -37.5..-21 and CAD Z is the simulation's X, so the display faces -X. Poses
# are therefore described by where the screen ends up, never by "the front" --
# get fore/aft backwards and left/right goes with it, which leaves the third
# pose agreeing with the other two and the error silently 180 degrees out.
POSES = (
    ("腹を下にして普通に置く", "+Z", np.array([0.0, 0.0, 1.0])),
    ("画面が真上を向くように立てる", "-X", np.array([-1.0, 0.0, 0.0])),
    ("画面を自分に向けたとき右に見える面を、上にして寝かせる",
     "+Y", np.array([0.0, 1.0, 0.0])),
)


def average_accel(port: serial.Serial, seconds: float = 1.5) -> np.ndarray:
    """Mean accelerometer vector over a short window, from the telemetry."""
    import time
    samples = []
    port.reset_input_buffer()
    end = time.time() + seconds
    while time.time() < end:
        line = port.readline().decode("utf-8", "replace").strip()
        if not line.startswith("T,"):
            continue
        parts = line.split(",")
        if len(parts) == 12:
            samples.append([float(parts[6]), float(parts[7]), float(parts[8])])
    if len(samples) < 5:
        raise RuntimeError(
            f"サンプルが {len(samples)} 個しか取れませんでした。"
            "実機がテレメトリを流しているか確認してください")
    v = np.mean(samples, axis=0)
    spread = np.std(samples, axis=0)
    if np.linalg.norm(spread) > 0.05:
        print(f"    注意: 読み値がばらついています ({np.round(spread, 3)})。静止させてください")
    return v


def axis_name(v: np.ndarray) -> str:
    """Nearest signed IMU axis, for the --accel-map shorthand."""
    i = int(np.argmax(np.abs(v)))
    return f"{'-' if v[i] < 0 else ''}{'xyz'[i]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    port = serial.Serial(args.port, args.baud, timeout=1)
    readings = {}
    for label, up_name, _ in POSES:
        input(f"\n{label}\n  （body {up_name} が上を向く姿勢）Enter: ")
        a = average_accel(port)
        n = np.linalg.norm(a)
        print(f"    加速度 = {np.round(a, 3)}   大きさ = {n:.3f} g")
        if not 0.85 < n < 1.15:
            print("    注意: 1g から外れています。動いていませんでしたか")
        readings[up_name] = a / (n or 1.0)

    # Rows of M are the body axes written in IMU coordinates.
    z = readings["+Z"]
    x = -readings["-X"]
    x = x - np.dot(x, z) * z          # make it perpendicular to z
    skew = math.degrees(math.asin(min(1.0, np.linalg.norm(x))))
    x /= np.linalg.norm(x) or 1.0
    y = np.cross(z, x)
    m = np.array([x, y, z])

    measured_y = readings["+Y"]
    err = math.degrees(math.acos(np.clip(np.dot(y, measured_y), -1, 1)))

    print("\nIMU軸 → ボディ座標系の変換行列（各行が body の +X/+Y/+Z を IMU 軸で表したもの）:")
    for row, name in zip(m, ("+X 前", "+Y 左", "+Z 上")):
        print(f"    {name:6s} {np.round(row, 4)}")
    print(f"\n  1つ目と2つ目の姿勢は {skew:.1f} 度離れています（90度が理想）")
    print(f"  3つ目の姿勢は外積から求めた軸と {err:.1f} 度ずれています")

    perm = ",".join(axis_name(r) for r in m)
    off = max(math.degrees(math.acos(np.clip(abs(r[np.argmax(np.abs(r))]), -1, 1)))
              for r in m)
    print(f"\n  もっとも近い軸並び: {perm}   （最大 {off:.1f} 度の誤差）")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"comment": "IMU axes -> simulation body frame (+X fore, +Y left, +Z up)",
         "matrix": m.round(6).tolist(),
         "nearest_axis_map": perm,
         "third_pose_error_deg": round(err, 2)}, indent=2) + "\n")
    print(f"\n{args.out} に保存しました")
    print("scripts/mirror が自動で読み込みます")


if __name__ == "__main__":
    main()
