"""Tasks the same robot can be trained for.

The robot is fixed hardware; only the objective changes. A `Task` owns three
things and nothing else:

* `reset()`   -- how the episode starts (pose, velocities)
* `reward()`  -- what is being optimised
* `terminated()` -- when the objective is met (or hopelessly lost)

Everything else -- observations, the servo model, contacts -- lives in
`GoronEnv`, so tasks stay short and comparable. Add a task by subclassing
`Task` and registering it in `TASKS`.

`SelfRight` is the tuned one. `Forward`, `Roll` and `Jump` are working first
cuts for the later goals: their reward terms are stated plainly rather than
tuned, and they are the natural things to iterate on.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from goron.env import GoronEnv

# Belly-up = 180 deg about the axis the robot does *not* flip around, so the
# somersault happens in the plane the legs actually work in.
BELLY_UP = {
    "sagittal": np.array([0.0, 0.0, 1.0, 0.0]),  # 180 deg about +Y
    "lateral": np.array([0.0, 1.0, 0.0, 0.0]),   # 180 deg about +X
}


class Task:
    name = "base"
    max_steps = 400          # control steps; 50 Hz -> 8 s
    obs_size = 0             # extra observations beyond the core 14
    has_success = False      # does this task define a pass/fail goal at all?

    def reset(self, env: "GoronEnv") -> None:
        raise NotImplementedError

    def obs(self, env: "GoronEnv") -> np.ndarray:
        return np.zeros(0, np.float32)

    def reward(self, env: "GoronEnv", action: np.ndarray) -> float:
        raise NotImplementedError

    def terminated(self, env: "GoronEnv") -> bool:
        return False

    def info(self, env: "GoronEnv") -> dict:
        return {}

    # -- shared helpers ----------------------------------------------------
    @staticmethod
    def effort_cost(env: "GoronEnv", action: np.ndarray) -> float:
        """Discourage thrashing the servos. Small, so it never dominates."""
        return (0.02 * float(np.sum(action ** 2))
                + 0.05 * float(np.sum((action - env.prev_action) ** 2)))


class SelfRight(Task):
    """Belly-up -> belly-down ("伏せ", pattern A).

    Success needs all three of:
      1. the torso's up axis within `prone_tilt` of vertical,
      2. the torso box actually touching the floor, and
      3. the body nearly at rest,
    held for `hold_steps` consecutive control steps.

    Condition 2 is the important one: without it, "propped up at an angle on
    the legs" counts as success, which is the attractor the passive drop test
    falls into. Condition 1 rules out lying on an edge.
    """

    name = "selfright"
    max_steps = 400
    has_success = True

    # Per-step reward while the goal conditions hold. Success pays out this
    # rate for every step the episode would still have had -- see `reward()`.
    goal_rate = 2.0
    posture_rate = 1.0

    def __init__(
        self,
        prone_tilt: float = math.radians(25.0),
        settle_omega: float = 3.0,
        hold_steps: int = 25,
    ):
        self.prone_cos = math.cos(prone_tilt)
        self.settle_omega = settle_omega
        self.hold_steps = hold_steps

    def reset(self, env: "GoronEnv") -> None:
        env.data.qpos[7:9] = env.np_random.uniform(-math.pi, math.pi, 2)
        env.place(
            quat=BELLY_UP[env.params.swing],
            yaw=env.np_random.uniform(-math.pi, math.pi),
            tilt_noise=0.05,
        )
        env.settle(150)
        self.hold_count = 0
        self.prev_up_z = env.up_z

    def at_goal(self, env: "GoronEnv") -> bool:
        return (
            env.up_z > self.prone_cos
            and env.belly_contact()
            and float(np.linalg.norm(env.gyro)) < self.settle_omega
        )

    def reward(self, env: "GoronEnv", action: np.ndarray) -> float:
        up_z, at_goal = env.up_z, self.at_goal(env)
        self.hold_count = self.hold_count + 1 if at_goal else 0
        # Potential-based shaping telescopes to 10*(up_z_end - up_z_start), so
        # it guides the climb out of belly-up without biasing the optimum.
        r = (10.0 * (up_z - self.prev_up_z)
             + self.posture_rate * up_z
             + self.goal_rate * float(at_goal)
             - self.effort_cost(env, action)
             - 0.02)
        if self.hold_count >= self.hold_steps:
            # Succeeding ends the episode, which forfeits every remaining step
            # of `goal_rate` + `posture_rate`. A flat bonus is therefore worth
            # far less than loitering at the goal, and the policy learns to
            # break the hold every few steps to farm the per-step bonus
            # forever: measured 0% success at reward 1000, versus 100% success
            # at reward 200 early in the same run.
            #
            # Paying out the rest of the episode makes finishing weakly better
            # than stalling at every point in time, so the exploit disappears.
            remaining = max(0, self.max_steps - env.step_count - 1)
            r += (self.goal_rate + self.posture_rate) * remaining
        self.prev_up_z = up_z
        return r

    def terminated(self, env: "GoronEnv") -> bool:
        return self.hold_count >= self.hold_steps

    def info(self, env: "GoronEnv") -> dict:
        return {"at_goal": self.at_goal(env),
                "is_success": self.hold_count >= self.hold_steps}


class Forward(Task):
    """Travel as far from the start as possible, by any means.

    The obvious formulation -- forward speed along the heading, while staying
    upright -- turned out to be very nearly infeasible on this body. Two legs
    cranking in the sagittal plane cannot hold the torso up, so any vigorous
    cranking somersaults the robot: with random open-loop commands, 118 of 120
    episodes ended inverted, and cranking both legs at full speed flipped it in
    0.66 s. Requiring "upright" leaves almost nothing the robot is allowed to do.

    Tumbling end-over-end is a perfectly good way for this machine to travel,
    and it is the mode that actually uses the 360 degree cranks -- self-righting
    only ever needed 124 degrees. So the upright constraint is dropped and the
    objective is simply displacement.

    The reward is potential-based on distance from the start, so the return over
    an episode telescopes to exactly `rate * final_distance`. There is nothing
    to farm: distance from the start cannot be accumulated by loitering, and
    driving in a circle stops paying as soon as the robot curves back.
    """

    name = "forward"
    max_steps = 600
    rate = 20.0

    def reset(self, env: "GoronEnv") -> None:
        env.data.qpos[7:9] = env.np_random.uniform(-math.pi, math.pi, 2)
        env.place(quat=np.array([1.0, 0.0, 0.0, 0.0]),
                  yaw=env.np_random.uniform(-math.pi, math.pi),
                  tilt_noise=0.03)
        env.settle(300)
        self.start = env.torso_pos[:2].copy()
        self.prev_dist = 0.0
        self.flips = 0
        self.was_inverted = env.up_z < 0.0

    def _distance(self, env: "GoronEnv") -> float:
        return float(np.linalg.norm(env.torso_pos[:2] - self.start))

    def reward(self, env: "GoronEnv", action: np.ndarray) -> float:
        dist = self._distance(env)
        r = self.rate * (dist - self.prev_dist) - self.effort_cost(env, action)
        self.prev_dist = dist
        inverted = env.up_z < 0.0
        self.flips += inverted is not self.was_inverted
        self.was_inverted = inverted
        return r

    def info(self, env: "GoronEnv") -> dict:
        return {
            "distance": self.prev_dist,
            "speed": self.prev_dist / max(1e-9, env.step_count * env.control_dt),
            "half_turns": self.flips,
        }


class Crawl(Task):
    """Travel *without* letting the body tumble.

    `Forward` discovers somersaulting, which is fast (0.44 m/s with C-legs) but
    means the body is rotating end over end the whole time. This task asks for
    the opposite: get somewhere while staying belly-down.

    The cost of that constraint is steep, and worth knowing before choosing it.
    Measured with open-loop sinusoidal paddling:

        stays fully prone (tilt < 45 deg)   0.007 m/s
        rocks but never inverts (< 90 deg)  0.04 m/s
        unconstrained somersaulting         0.44 m/s

    The constraint is imposed as a **termination**, not a reward penalty.
    Balancing "distance earned" against "uprightness lost" is a knife-edge: a
    little too much penalty and the best policy is to sit perfectly still, a
    little too little and it just tumbles anyway, because tumbling is ~60x
    faster and can outrun any penalty small enough to leave crawling positive.
    Terminating removes the trade entirely -- tumbling forfeits the rest of the
    episode, so the only way to accumulate reward is to move while upright.

    `rate` is 10x the `Forward` value because the attainable speeds are ~10-60x
    lower; it keeps the returns in a comparable range for the same PPO settings.
    """

    name = "crawl"
    max_steps = 600
    rate = 200.0
    max_tilt = math.radians(60.0)

    def reset(self, env: "GoronEnv") -> None:
        env.data.qpos[7:9] = env.np_random.uniform(-math.pi, math.pi, 2)
        env.place(quat=np.array([1.0, 0.0, 0.0, 0.0]),
                  yaw=env.np_random.uniform(-math.pi, math.pi),
                  tilt_noise=0.03)
        env.settle(300)
        self.start = env.torso_pos[:2].copy()
        self.prev_dist = 0.0
        self.worst_tilt = 0.0

    def reward(self, env: "GoronEnv", action: np.ndarray) -> float:
        dist = float(np.linalg.norm(env.torso_pos[:2] - self.start))
        r = self.rate * (dist - self.prev_dist) - self.effort_cost(env, action)
        self.prev_dist = dist
        self.worst_tilt = max(self.worst_tilt,
                              math.degrees(math.acos(np.clip(env.up_z, -1, 1))))
        return r

    def terminated(self, env: "GoronEnv") -> bool:
        return env.up_z < math.cos(self.max_tilt)

    def info(self, env: "GoronEnv") -> dict:
        return {
            "distance": self.prev_dist,
            "speed": self.prev_dist / max(1e-9, env.step_count * env.control_dt),
            "worst_tilt": self.worst_tilt,
        }


class Roll(Task):
    """Keep rolling sideways -- rotate continuously about the body's X axis.

    Deliberately hard: sagittal cranks produce pitch directly but no roll, so
    the policy has to find an indirect route (asymmetric leg contact, momentum
    from the 288:1 reflected inertia). Reward is the accumulated roll angle,
    measured by integrating the body-X angular rate.
    """

    name = "roll"
    max_steps = 600

    def reset(self, env: "GoronEnv") -> None:
        env.place(quat=np.array([1.0, 0.0, 0.0, 0.0]),
                  yaw=env.np_random.uniform(-math.pi, math.pi),
                  tilt_noise=0.03)
        env.settle(300)
        self.rolled = 0.0
        self.direction = 1.0 if env.np_random.random() < 0.5 else -1.0

    def reward(self, env: "GoronEnv", action: np.ndarray) -> float:
        rate = float(env.gyro[0]) * self.direction
        self.rolled += rate * env.control_dt
        return 1.0 * rate - 0.2 * abs(float(env.gyro[1])) \
            - self.effort_cost(env, action)

    def info(self, env: "GoronEnv") -> dict:
        return {"revolutions": self.rolled / (2 * math.pi)}


class Jump(Task):
    """Get the whole robot off the ground.

    Reward is peak torso height gained plus airborne time. The 288:1 reflected
    inertia is the thing to exploit here: spinning both cranks up and then
    stopping them against the ground dumps that stored momentum into the body.
    """

    name = "jump"
    max_steps = 300

    def reset(self, env: "GoronEnv") -> None:
        env.data.qpos[7:9] = env.np_random.uniform(-math.pi, math.pi, 2)
        env.place(quat=np.array([1.0, 0.0, 0.0, 0.0]),
                  yaw=env.np_random.uniform(-math.pi, math.pi),
                  tilt_noise=0.02)
        env.settle(300)
        self.rest_height = float(env.torso_pos[2])
        self.peak = 0.0

    def reward(self, env: "GoronEnv", action: np.ndarray) -> float:
        height = float(env.torso_pos[2]) - self.rest_height
        airborne = not (env.belly_contact() or env.foot_contact())
        gain, self.peak = max(0.0, height - self.peak), max(self.peak, height)
        return (50.0 * gain                 # only new records pay
                + 1.0 * float(airborne)
                + 0.5 * env.up_z            # land the right way up
                - self.effort_cost(env, action))

    def info(self, env: "GoronEnv") -> dict:
        return {"peak_height_mm": 1000 * self.peak}


TASKS: dict[str, type[Task]] = {
    t.name: t for t in (SelfRight, Forward, Crawl, Roll, Jump)
}
