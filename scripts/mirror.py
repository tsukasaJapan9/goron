"""Mirror the real robot into the MuJoCo model, live.

    uv run python -m scripts.mirror --port /dev/ttyUSB0

The firmware streams one CSV line per sample (20 Hz):

    T,millis,raw_l,raw_r,crank_l,crank_r,ax,ay,az,gx,gy,gz

Crank angles are already measured from the calibrated zero, so they drop
straight into `qpos[7:9]`. Body attitude comes from the accelerometer: at rest
it reads world up expressed in the body frame, which is exactly the model's
`gravity_body` observation, so the pose is recovered by rotating that vector
onto world +Z. Yaw is unobservable and is left at zero -- the robot does not
observe it either.

This exists to settle the conventions that cannot be read off the code: which
way the printed leg points at crank zero, whether the servo turns the same way
as the model's hinge, and how the IMU axes sit in the body. Move the real robot
by hand and check that the drawing follows; where it does not, the mapping is
wrong.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import serial

from goron.model import RobotParams, build_mjcf

TELEM_FIELDS = 12  # tag + 11 values
HARDWARE = Path(__file__).resolve().parent.parent / "hardware"
URDF_PATH = HARDWARE / "goron.urdf"
IMU_MAP_PATH = HARDWARE / "imu_map.json"

# Simulation frame -> CAD assembly frame: CAD_x = sim_y, CAD_y = sim_z,
# CAD_z = sim_x. The URDF is drawn in CAD axes, so a vector calibrated into the
# simulation frame needs this on top before it is used to pose that model.
SIM_TO_CAD = np.array([[0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [1.0, 0.0, 0.0]])


def load_imu_map() -> np.ndarray | None:
    """The matrix scripts/imu_calib wrote, if it has been run."""
    if not IMU_MAP_PATH.exists():
        return None
    return np.array(json.loads(IMU_MAP_PATH.read_text())["matrix"], dtype=float)


def load_urdf() -> mujoco.MjModel:
    """The printed design, with the real STLs as visual geometry.

    Two edits are needed to mirror into it. The mesh directory is relative to
    the URDF, but `from_xml_string` resolves against the working directory, so
    it is made absolute. And a URDF root is welded to the world, which leaves
    nowhere to put the measured attitude -- a floating joint is inserted so the
    body can be posed.
    """
    xml = URDF_PATH.read_text()
    xml = xml.replace('meshdir="stl/"', f'meshdir="{URDF_PATH.parent / "stl"}/"')
    xml = xml.replace(
        '<link name="base_link">',
        '<link name="world_link"/>\n'
        '  <joint name="floating" type="floating">\n'
        '    <parent link="world_link"/><child link="base_link"/>\n'
        '  </joint>\n'
        '  <link name="base_link">',
    )
    return mujoco.MjModel.from_xml_string(xml)


def free_joint_qposadr(model: mujoco.MjModel) -> int:
    for j in range(model.njnt):
        if model.joint(j).type[0] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.joint(j).qposadr[0])
    raise RuntimeError("model has no free joint; the body cannot be posed")


def parse_axis_map(spec: str) -> np.ndarray:
    """"-y,x,z" -> a 3x3 matrix taking IMU axes to body axes."""
    axes = {"x": 0, "y": 1, "z": 2}
    m = np.zeros((3, 3))
    parts = [s.strip().lower() for s in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"axis map needs 3 entries, got {spec!r}")
    for row, part in enumerate(parts):
        sign = -1.0 if part.startswith("-") else 1.0
        name = part.lstrip("+-")
        if name not in axes:
            raise ValueError(f"unknown axis {part!r} in {spec!r}")
        m[row, axes[name]] = sign
    return m


def quat_from_up(up_body: np.ndarray) -> np.ndarray:
    """Orientation (w,x,y,z) whose world-up, seen from the body, is `up_body`.

    The minimal rotation carrying `up_body` onto world +Z. It leaves yaw free,
    which is what we want: the accelerometer cannot see yaw.
    """
    a = up_body / (np.linalg.norm(up_body) or 1.0)
    b = np.array([0.0, 0.0, 1.0])
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    c = float(np.dot(a, b))
    if s < 1e-8:  # already aligned, or exactly upside down
        return np.array([1.0, 0.0, 0.0, 0.0]) if c > 0 else np.array([0.0, 1.0, 0.0, 0.0])
    axis /= s
    angle = math.atan2(s, c)
    return np.concatenate([[math.cos(angle / 2)], axis * math.sin(angle / 2)])


def lowest_point(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """World z of the robot's lowest point, in the pose already in `data`.

    Bounding spheres (what `GoronEnv._lowest_point` uses) are far too loose
    here: the 80 mm torso's sphere reaches 29 mm below its own underside, which
    is exactly how much the robot would hover. Boxes and meshes are therefore
    measured from their real corners and vertices.
    """
    lows = []
    for g in range(model.ngeom):
        gtype = model.geom_type[g]
        if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        pos, mat = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3)
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            sx, sy, sz = model.geom_size[g]
            corners = np.array([[i * sx, j * sy, k * sz]
                                for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
            lows.append((corners @ mat.T + pos)[:, 2].min())
        elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[g]
            adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            verts = model.mesh_vert[adr:adr + num]
            lows.append((verts @ mat.T + pos)[:, 2].min())
        else:  # sphere, capsule, cylinder: the bounding sphere is tight enough
            lows.append(pos[2] - model.geom_rbound[g])
    return float(min(lows)) if lows else 0.0


def read_latest(port: serial.Serial, previous: list[float] | None) -> list[float] | None:
    """Drain the port and return the newest complete sample, or `previous`.

    Draining matters: the robot streams whether or not we keep up, and a
    backlog would show as the drawing lagging behind the hand.
    """
    sample = previous
    while port.in_waiting:
        line = port.readline().decode("utf-8", "replace").strip()
        if not line.startswith("T,"):
            if line:
                print(line)  # boot log, calibration messages
            continue
        parts = line.split(",")
        if len(parts) != TELEM_FIELDS:
            continue
        try:
            sample = [float(v) for v in parts[1:]]
        except ValueError:
            continue
    return sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--model", choices=("urdf", "mjcf"), default="urdf",
                    help="urdf = the printed design with the real STLs; "
                         "mjcf = the parametric model the policies were trained on")
    ap.add_argument("--leg-shape", default="mesh",
                    help="mjcf only; must match the policy being calibrated for")
    ap.add_argument("--no-visual-stl", action="store_true",
                    help="mjcf only; draw the collision primitives bare")
    ap.add_argument("--accel-map", default=None,
                    help="IMU axes -> simulation body axes, e.g. '-y,x,z'. "
                         "Default: hardware/imu_map.json from scripts.imu_calib, "
                         "or identity if that has never been run")
    ap.add_argument("--crank-sign", type=float, default=1.0,
                    help="-1 if the servo turns opposite to the model hinge")
    ap.add_argument("--crank-offset", type=float, default=0.0,
                    help="degrees added to both crank angles, for trying out a zero")
    ap.add_argument("--bake-offset", type=float, default=None,
                    help="write this offset into the robot's stored zero and exit "
                         "the trial: pass the --crank-offset value that lined up")
    ap.add_argument("--height", type=float, default=None,
                    help="fixed torso height; default is to rest on the floor. "
                         "The real height is not observable -- the robot carries "
                         "no position sensor -- so it is inferred from the pose")
    args = ap.parse_args()

    if args.accel_map:
        axis_map, source = parse_axis_map(args.accel_map), args.accel_map
    elif (calibrated := load_imu_map()) is not None:
        axis_map, source = calibrated, IMU_MAP_PATH.name
    else:
        axis_map, source = np.eye(3), "identity (run scripts.imu_calib)"
    if args.model == "urdf":
        model = load_urdf()
    else:
        # As-built, or the drawing is a 54 mm robot next to an 80 mm one.
        model = mujoco.MjModel.from_xml_string(build_mjcf(RobotParams.asbuilt(
            leg_shape=args.leg_shape, visual_stl=not args.no_visual_stl)))
    data = mujoco.MjData(model)

    # Resolve by name: the two models lay their qpos out differently.
    base = free_joint_qposadr(model)
    hip_l = int(model.joint("hip_left").qposadr[0])
    hip_r = int(model.joint("hip_right").qposadr[0])
    print(f"model {args.model}: base qpos[{base}:{base+7}], "
          f"hips qpos[{hip_l}], qpos[{hip_r}]")

    port = serial.Serial(args.port, args.baud, timeout=0.1)
    print(f"listening on {args.port} at {args.baud}")
    print(f"accel map: {source}; crank sign {args.crank_sign:+.0f}, "
          f"offset {args.crank_offset:+.1f} deg")

    if args.bake_offset is not None:
        # The robot applies it to its own zero, so the correction survives a
        # reboot and every consumer sees it -- not just this viewer.
        time.sleep(2.0)  # let the board finish booting before it can listen
        port.write(f"Z {args.bake_offset:.2f} {args.bake_offset:.2f}\n".encode())
        port.flush()
        print(f"sent zero correction {args.bake_offset:+.2f} deg; "
              f"mirroring with no offset from here")
        args.crank_offset = 0.0

    sample: list[float] | None = None
    last_report = time.time()
    count = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            new = read_latest(port, sample)
            if new is not sample and new is not None:
                count += 1
            sample = new
            if sample is None:
                time.sleep(0.01)
                continue

            _, _, _, crank_l, crank_r, ax, ay, az, *_ = sample
            up = axis_map @ np.array([ax, ay, az])   # simulation body frame
            if args.model == "urdf":
                up = SIM_TO_CAD @ up

            data.qpos[base:base + 3] = [0.0, 0.0, args.height or 0.0]
            data.qpos[base + 3:base + 7] = quat_from_up(up)
            data.qpos[hip_l] = args.crank_sign * math.radians(crank_l + args.crank_offset)
            data.qpos[hip_r] = args.crank_sign * math.radians(crank_r + args.crank_offset)
            mujoco.mj_forward(model, data)  # kinematics only; no stepping
            if args.height is None:
                # Settle it onto the floor, so tilting the real robot shows as
                # the drawing rocking on its edge rather than hovering.
                data.qpos[base + 2] -= lowest_point(model, data)
                mujoco.mj_forward(model, data)
            viewer.sync()

            now = time.time()
            if now - last_report >= 2.0:
                print(f"{count / (now - last_report):5.1f} Hz   "
                      f"crank L {crank_l:6.1f}  R {crank_r:6.1f}   "
                      f"up ({up[0]:+.2f}, {up[1]:+.2f}, {up[2]:+.2f})")
                last_report, count = now, 0
            time.sleep(0.005)


if __name__ == "__main__":
    main()
