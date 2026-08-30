"""Gymnasium environment for the two-crank robot.

The environment owns the robot: the servo model, the observation vector and
the contact queries. *What* the robot is being asked to do lives in
`goron.tasks`, so self-righting, locomotion, rolling and jumping all share one
observation and action contract and one set of physics.

Observation (14 values, all obtainable on the real robot from the M5Stack's
6-axis IMU plus the XL330's feedback over TTL):

    [0:3]   gravity direction in the body frame   (IMU attitude / accelerometer)
    [3:6]   body angular velocity, rad/s          (gyro)
    [6:10]  sin/cos of each crank angle           (XL330 absolute encoder)
    [10:12] crank angular velocity, normalised    (XL330 Present Velocity)
    [12:14] servo torque, normalised              (XL330 Present Current)

Crank angle enters as sin/cos rather than the raw angle because the joints turn
continuously -- a raw angle would jump at every wrap and grow without bound.

Deliberately absent: torso position and yaw. Neither is measurable on the real
robot, and no task here needs them.

Action (2 values in [-1, 1]): per-step increments to the Extended Position
Control Mode targets, scaled so that a saturated action commands exactly the
servo's no-load speed. The target is also clamped to stay within `max_lead` of
the measured angle -- the standard anti-windup guard, without which a blocked
crank lets the target run arbitrarily far ahead and the robot then spends
seconds unwinding it.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from goron.model import RobotParams, build_mjcf
from goron.tasks import TASKS, Task

CORE_OBS = 14


class GoronEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        params: RobotParams | None = None,
        task: Task | str = "selfright",
        *,
        frame_skip: int = 10,        # 500 Hz physics -> 50 Hz control
        max_lead: float = 0.5,       # rad the target may run ahead of the crank
        randomize: bool = False,
        obs_noise: float = 0.0,
        render_mode: str | None = None,
    ):
        self.params = params or RobotParams()
        self.task = TASKS[task]() if isinstance(task, str) else task
        self.frame_skip = frame_skip
        self.control_dt = self.params.timestep * frame_skip
        self.max_lead = max_lead
        self.randomize = randomize
        self.obs_noise = obs_noise
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_string(build_mjcf(self.params))
        self.data = mujoco.MjData(self.model)

        self.torso_bid = self.model.body("torso").id
        self.torso_gid = self.model.geom("torso").id
        self.floor_gid = self.model.geom("floor").id
        # Every geom carried by a leg body, rather than the ones named "foot_*":
        # the mesh leg is a set of convex wedges with no single foot geom.
        leg_bids = {self.model.body(f"leg_{s}").id for s in ("left", "right")}
        self.foot_gids = {
            g for g in range(self.model.ngeom)
            if self.model.geom_bodyid[g] in leg_bids
        }
        # A saturated action commands exactly the servo's no-load speed.
        self.max_delta = self.params.servo_no_load_speed * self.control_dt

        self._nominal = {
            name: getattr(self.model, name).copy()
            for name in ("body_mass", "body_inertia", "geom_friction",
                         "dof_damping", "actuator_gainprm", "actuator_biasprm",
                         "actuator_forcerange")
        }

        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (CORE_OBS + self.task.obs_size,), np.float32
        )

        self._renderer: mujoco.Renderer | None = None
        self._camera: mujoco.MjvCamera | None = None

    # -------------------------------------------------------------- state
    @property
    def rot(self) -> np.ndarray:
        return self.data.xmat[self.torso_bid].reshape(3, 3)

    @property
    def up_z(self) -> float:
        """Torso up axis projected on world up. +1 prone, -1 belly-up."""
        return float(self.rot[2, 2])

    @property
    def gravity_body(self) -> np.ndarray:
        """World up in the body frame == accelerometer reading at rest."""
        return self.rot[2, :].copy()

    @property
    def gyro(self) -> np.ndarray:
        return self.data.qvel[3:6].copy()

    @property
    def torso_pos(self) -> np.ndarray:
        return self.data.xpos[self.torso_bid].copy()

    @property
    def torso_vel(self) -> np.ndarray:
        return self.data.qvel[0:3].copy()

    @property
    def crank_angle(self) -> np.ndarray:
        return self.data.qpos[7:9].copy()

    @property
    def crank_vel(self) -> np.ndarray:
        return self.data.qvel[6:8].copy()

    def belly_contact(self) -> bool:
        return self._touching({self.torso_gid})

    def foot_contact(self) -> bool:
        return self._touching(self.foot_gids)

    def _touching(self, gids: set[int]) -> bool:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if self.floor_gid in pair and pair & gids:
                return True
        return False

    # --------------------------------------------------------- reset tools
    @property
    def spawn_height(self) -> float:
        """Lowest height at which nothing can be intersecting the floor yet.

        Spawning below this makes the reset start from an interpenetration that
        the solver has to push apart, which shows up as a random kick.
        """
        p = self.params
        return p.torso_height / 2 + p.leg_length + p.foot_radius + 0.005

    def _lowest_point(self) -> float:
        """World z of the lowest point of the robot, using bounding spheres."""
        mujoco.mj_kinematics(self.model, self.data)
        z = self.data.geom_xpos[:, 2] - self.model.geom_rbound
        return float(np.min(np.delete(z, self.floor_gid)))

    def place(self, quat: np.ndarray, *, yaw: float = 0.0,
              tilt_noise: float = 0.0, height: float | None = None,
              clearance: float = 0.002) -> None:
        """Put the torso down in orientation `quat`, yawed and jittered.

        With `height=None` the robot is lowered until its lowest point sits
        `clearance` above the floor, so the episode starts from rest instead of
        from a drop. Dropping the robot in adds impact energy that has nothing
        to do with the task and makes the starting pose a lottery.
        """
        q_yaw = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, q_yaw, quat)
        if tilt_noise > 0:
            axis = self.np_random.normal(0, 1, 3)
            axis /= np.linalg.norm(axis) + 1e-12
            q_tilt = np.zeros(4)
            mujoco.mju_axisAngle2Quat(
                q_tilt, axis, self.np_random.normal(0, tilt_noise)
            )
            mujoco.mju_mulQuat(q, q, q_tilt)
        self.data.qpos[0:3] = [0.0, 0.0, self.spawn_height if height is None else height]
        self.data.qpos[3:7] = q
        if height is None:
            self.data.qpos[2] -= self._lowest_point() - clearance

    def settle(self, steps: int) -> None:
        """Let the robot come to rest before the episode proper begins."""
        self.targets = self.data.qpos[7:9].copy()
        self.data.ctrl[:] = self.targets
        mujoco.mj_forward(self.model, self.data)
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_kinematics(self.model, self.data)

    # ------------------------------------------------------------ observe
    def _obs(self) -> np.ndarray:
        torque = self.data.actuator_force / self.params.servo_stall_torque
        core = np.concatenate([
            self.gravity_body,
            self.gyro,
            np.sin(self.crank_angle), np.cos(self.crank_angle),
            self.crank_vel / self.params.servo_no_load_speed,
            np.clip(torque, -1.0, 1.0),
        ])
        obs = np.concatenate([core, self.task.obs(self)]).astype(np.float32)
        if self.obs_noise:
            obs = obs + self.np_random.normal(0, self.obs_noise, obs.shape)
        return obs.astype(np.float32)

    # ---------------------------------------------------------- randomise
    def _apply_randomisation(self) -> None:
        for name, value in self._nominal.items():
            getattr(self.model, name)[:] = value
        if not self.randomize:
            return
        u = self.np_random.uniform
        self.model.body_mass[self.torso_bid] *= u(0.85, 1.15)
        self.model.geom_friction[:, 0] *= u(0.7, 1.4)
        kp_scale = u(0.7, 1.3)
        # position actuator: gain = [kp], bias = [0, -kp, -kv]
        self.model.actuator_gainprm[:, 0] *= kp_scale
        self.model.actuator_biasprm[:, 1] *= kp_scale
        self.model.actuator_forcerange *= u(0.8, 1.2)
        self.model.dof_damping[6:8] *= u(0.7, 1.4)   # gearbox friction spread

    # ------------------------------------------------------------ gym api
    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._apply_randomisation()
        mujoco.mj_resetData(self.model, self.data)
        self.prev_action = np.zeros(2)
        self.step_count = 0
        self.task.reset(self)
        self.targets = self.crank_angle.copy()
        self.data.ctrl[:] = self.targets
        # Recompute after setting ctrl, so the torque channel of the first
        # observation reflects the new command instead of the last settle step.
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
        angle = self.crank_angle
        # Advance the extended-position target, then hold it within max_lead of
        # the crank so a blocked leg cannot wind the target up indefinitely.
        self.targets = np.clip(
            self.targets + action * self.max_delta,
            angle - self.max_lead, angle + self.max_lead,
        )
        self.data.ctrl[:] = self.targets
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        # mj_step integrates *after* its forward pass, so xpos/xmat still
        # describe the previous state. Without this the whole observation --
        # attitude included -- is one substep stale.
        mujoco.mj_kinematics(self.model, self.data)

        reward = self.task.reward(self, action)
        terminated = self.task.terminated(self)
        self.prev_action = action
        self.step_count += 1
        truncated = self.step_count >= self.task.max_steps

        info = {
            "up_z": self.up_z,
            "tilt_deg": math.degrees(math.acos(float(np.clip(self.up_z, -1, 1)))),
            **self.task.info(self),
        }
        info.setdefault("is_success", False)
        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------- render
    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
            self._camera = mujoco.MjvCamera()
            self._camera.distance = 0.45
            self._camera.elevation = -15
            self._camera.azimuth = 110
        self._camera.lookat[:] = self.torso_pos
        self._renderer.update_scene(self.data, self._camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# Backwards-compatible alias: the self-righting task was the original env.
def SelfRightEnv(params: RobotParams | None = None, **kwargs) -> GoronEnv:
    return GoronEnv(params, task="selfright", **kwargs)
