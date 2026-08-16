"""What loads does tumbling put on the hardware?

The forward gait somersaults the robot, so before printing anything it is worth
knowing what the impacts do -- especially to the XL330's **engineering-plastic
gearbox**, which is the part most likely to fail first. The leg bolts straight
to the output horn, so every landing on a leg drives an impact torque back
through the gear train.

Sampling matters: peaks must be read at the physics rate (500 Hz), not the
control rate (50 Hz). Reading once per control step misses the spike entirely.

    uv run python -m scripts.loads

CAVEAT. MuJoCo's contacts are soft -- `solref="0.005 1"` spreads an impact over
roughly 5 ms, while real PLA on a hard floor is closer to a fraction of a
millisecond. Peak *force* here is therefore an under-estimate; the *impulse*
(force integrated over the impact) is the trustworthy number, and the peak on
real hardware will be higher for the same impulse.
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

from goron.env import GoronEnv
from goron.model import RobotParams

HIP_DOF = [6, 7]


def measure(params: RobotParams, action: np.ndarray, control_steps: int = 600,
            seed: int = 0) -> dict:
    env = GoronEnv(params, task="forward")
    env.reset(seed=seed)
    m, d = env.model, env.data

    contact_force, hip_torque, foot_impulse = [], [], []
    peak_height, landing_speeds, prev_vz = 0.0, [], 0.0
    f6 = np.zeros(6)

    for _ in range(control_steps):
        # Replicate one env.step, but sample every physics step.
        angle = d.qpos[7:9].copy()
        env.targets = np.clip(env.targets + action * env.max_delta,
                              angle - env.max_lead, angle + env.max_lead)
        d.ctrl[:] = env.targets
        for _ in range(env.frame_skip):
            mujoco.mj_step(m, d)
            total, impulse = 0.0, 0.0
            for c in range(d.ncon):
                mujoco.mj_contactForce(m, d, c, f6)
                total += abs(f6[0])
                impulse += abs(f6[0]) * m.opt.timestep
            contact_force.append(total)
            foot_impulse.append(impulse)
            # Torque seen by the gear train: what the motor applies plus what
            # the ground pushes back through the leg.
            hip_torque.append(
                float(np.abs(d.actuator_force + d.qfrc_constraint[HIP_DOF]).max())
            )
            z = float(d.xpos[env.torso_bid][2])
            peak_height = max(peak_height, z)
            vz = float(d.qvel[2])
            if prev_vz < -0.05 and vz >= 0.0:   # descent arrested = an impact
                landing_speeds.append(abs(prev_vz))
            prev_vz = vz

    return {
        "contact_force": np.array(contact_force),
        "hip_torque": np.array(hip_torque),
        "impulse": np.array(foot_impulse),
        "peak_height": peak_height,
        "landing": np.array(landing_speeds) if landing_speeds else np.zeros(1),
    }


def report(name: str, r: dict, params: RobotParams) -> None:
    weight = params.total_mass * 9.81
    stall = params.servo_stall_torque
    f, t = r["contact_force"], r["hip_torque"]
    print(f"\n{name}  (weight {weight:.2f} N, stall torque {stall:.2f} N.m)")
    print(f"  contact force   p50 {np.percentile(f, 50):6.1f}  "
          f"p99 {np.percentile(f, 99):6.1f}  max {f.max():7.1f} N "
          f"= {f.max() / weight:.0f} x body weight")
    print(f"  hip torque      p50 {np.percentile(t, 50):6.3f}  "
          f"p99 {np.percentile(t, 99):6.3f}  max {t.max():7.3f} N.m "
          f"= {t.max() / stall:.1f} x stall")
    print(f"  peak impulse    {r['impulse'].max() * 1000:.2f} N.ms per timestep")
    print(f"  torso rises to  {r['peak_height'] * 1000:.0f} mm, "
          f"lands at {np.percentile(r['landing'], 99):.2f} m/s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()
    print("Worst case: both cranks driven at full speed, continuous somersaults.")
    for shape in ("bar", "c_leg"):
        p = RobotParams(leg_shape=shape)
        report(shape, measure(p, np.array([1.0, 1.0]), args.steps), p)


if __name__ == "__main__":
    main()
