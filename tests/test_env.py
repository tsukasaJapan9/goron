"""Sanity checks for the mistakes that silently ruin an RL run.

Several of these encode bugs that actually happened during development:
the servo cases protruding below the belly (making the goal unreachable), and
observations leaking un-measurable state.
"""

from __future__ import annotations

import argparse
import json
import math

import mujoco
import numpy as np
import pytest

from goron.env import CORE_OBS, GoronEnv
from goron.model import RobotParams, build_mjcf
from goron.tasks import TASKS
from goron.train import add_robot_args, build_params


@pytest.mark.parametrize("swing", ["sagittal", "lateral"])
@pytest.mark.parametrize("shape", ["bar", "c_leg"])
def test_model_loads(swing, shape):
    m = mujoco.MjModel.from_xml_string(
        build_mjcf(RobotParams(swing=swing, leg_shape=shape))
    )
    assert m.nu == 2
    assert m.njnt == 3  # free root + two hips


def test_cranks_are_unlimited():
    """360 degree rotation means the hip joints must have no range at all."""
    p = RobotParams()
    m = mujoco.MjModel.from_xml_string(build_mjcf(p))
    assert not m.jnt_limited[1:].any()

    # And the leg must physically clear the torso all the way round.
    assert p.leg_clears_torso
    d = mujoco.MjData(m)
    d.qpos[2] = 0.5
    for q in np.linspace(0, 2 * math.pi, 24, endpoint=False):
        d.qpos[7:9] = q
        mujoco.mj_forward(m, d)
        assert d.ncon == 0, f"self-collision at crank angle {math.degrees(q):.0f} deg"


def test_servo_case_does_not_protrude_below_the_belly():
    """The XL330 case (26 mm) is taller than the M5Stack (16 mm). Mounted at
    mid-height it sticks out below the belly, the robot rests on its servo
    cases, and no belly-contact goal is ever reachable."""
    assert RobotParams().servo_clears_belly
    assert not RobotParams(hip_z_frac=0.0).servo_clears_belly


def test_prone_pose_puts_the_belly_on_the_floor():
    env = GoronEnv(task="selfright")
    env.reset(seed=0)
    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[0:3] = [0, 0, env.spawn_height]
    env.data.qpos[3:7] = [1, 0, 0, 0]
    env.data.qpos[7:9] = math.pi          # cranks parked pointing up
    env.settle(800)
    assert env.belly_contact()
    assert env.torso_pos[2] == pytest.approx(env.params.torso_height / 2, abs=1e-3)


@pytest.mark.parametrize("swing", ["sagittal", "lateral"])
def test_leg_pairing_matches_the_swing_plane(swing):
    p = RobotParams(swing=swing)
    m = mujoco.MjModel.from_xml_string(build_mjcf(p))
    d = mujoco.MjData(m)
    # The "both legs do the same thing" command, in each convention.
    d.qpos[7:9] = [0.4, 0.4] if swing == "sagittal" else [0.4, -0.4]
    mujoco.mj_forward(m, d)
    left = d.site("foot_site_left").xpos - d.body("leg_left").xpos
    right = d.site("foot_site_right").xpos - d.body("leg_right").xpos
    if swing == "sagittal":
        # Parallel oars: the same command moves both legs identically.
        assert np.allclose(left, right, atol=1e-9)
        assert abs(left[1]) < 1e-9
    else:
        # Mirrored: the same command moves them to opposite sides.
        assert np.allclose(left * [1, -1, 1], right, atol=1e-9)
        assert abs(left[0]) < 1e-9


def test_episode_starts_belly_up():
    """The robot is placed upside down and released. Legs sticking out at random
    crank angles sometimes let it topple onto an edge while it settles, which is
    a real starting condition -- but it must never start already prone, and the
    inverted case must dominate."""
    env = GoronEnv(task="selfright")
    ups = []
    for seed in range(30):
        env.reset(seed=seed)
        assert not env.task.at_goal(env), f"seed {seed}: starts at the goal"
        assert env.up_z < 0.5, f"seed {seed}: starts prone (up_z={env.up_z})"
        ups.append(env.up_z)
    assert np.mean(np.array(ups) < -0.8) > 0.6


def test_observation_is_hardware_realisable():
    """Observations must not leak what the real robot cannot measure: no world
    position, no yaw. Crank angle enters as sin/cos so it survives wrapping."""
    env = GoronEnv(task="selfright")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (CORE_OBS,)
    env.settle(400)
    assert np.isclose(np.linalg.norm(env._obs()[0:3]), 1.0, atol=0.05), (
        "at rest the accelerometer should read one g"
    )

    before = env._obs().copy()
    yaw = 1.1
    q = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    rot = np.zeros(4)
    mujoco.mju_mulQuat(rot, q, env.data.qpos[3:7].copy())
    env.data.qpos[3:7] = rot
    env.data.qpos[0:2] += [0.5, -0.3]
    mujoco.mj_forward(env.model, env.data)
    assert np.allclose(before, env._obs(), atol=1e-5)

    # A full extra turn of a crank -- and of the target that tracks it -- must
    # be invisible to the policy. This is why the angle enters as sin/cos.
    env.data.qpos[7] += 2 * math.pi
    env.targets[0] += 2 * math.pi
    env.data.ctrl[:] = env.targets
    mujoco.mj_forward(env.model, env.data)
    assert np.allclose(before, env._obs(), atol=1e-5)


def test_saturated_action_commands_the_no_load_speed():
    env = GoronEnv(task="selfright")
    p = env.params
    assert env.max_delta == pytest.approx(p.servo_no_load_speed * env.control_dt)


def test_target_cannot_wind_up_against_a_blocked_crank():
    """Anti-windup: with the leg held, the target must stay within max_lead."""
    env = GoronEnv(task="selfright", max_lead=0.5)
    env.reset(seed=0)
    for _ in range(200):
        env.data.qvel[6:8] = 0.0          # pin the cranks as if jammed
        env.data.qpos[7:9] = 0.0
        env.step(np.array([1.0, 1.0]))
    assert np.all(np.abs(env.targets - env.crank_angle) <= 0.5 + 1e-9)


@pytest.mark.parametrize("name", sorted(TASKS))
def test_every_task_runs(name):
    env = GoronEnv(task=name)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (CORE_OBS + env.task.obs_size,)
    for _ in range(20):
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs)), f"{name}: non-finite observation"
        assert math.isfinite(reward), f"{name}: non-finite reward"
        if term or trunc:
            break
    assert "is_success" in info


def test_success_is_never_worth_less_than_loitering_at_the_goal():
    """Regression for a reward-hacking bug.

    Succeeding terminates the episode, so a *flat* success bonus was worth far
    less than staying near the goal and collecting the per-step bonus for the
    remaining ~380 steps. PPO duly learned to reach the prone pose and then
    break the hold every 1-4 steps, farming the bonus forever: success went
    100% -> 0% while mean reward went 193 -> 1006 in the same run.

    The invariant that kills the exploit: finishing at step t must pay at least
    as much as continuing to sit at the goal until the episode ends.
    """
    env = GoronEnv(task="selfright")
    env.reset(seed=0)
    task = env.task
    env.step_count = 100
    zero = np.zeros(2)

    # Pin the goal test to True so we compare the two branches at one state.
    task.at_goal = lambda _env: True

    task.hold_count, task.prev_up_z = 0, env.up_z
    loiter = task.reward(env, zero)                 # at goal, hold incomplete

    task.hold_count, task.prev_up_z = task.hold_steps - 1, env.up_z
    finish = task.reward(env, zero)                 # this step completes it

    remaining = task.max_steps - env.step_count - 1
    assert finish - loiter >= loiter * remaining, (
        f"stalling at the goal for {remaining} more steps pays "
        f"{loiter * remaining:.0f}, finishing pays only {finish - loiter:.0f}"
    )


def test_randomisation_does_not_compound():
    env = GoronEnv(task="selfright", randomize=True)
    nominal = env._nominal["body_mass"][env.torso_bid]
    for seed in range(5):
        env.reset(seed=seed)
        assert 0.8 * nominal < env.model.body_mass[env.torso_bid] < 1.2 * nominal


def test_run_parameters_survive_a_round_trip():
    """A run's robot must be recoverable from what training wrote down.

    Without this, `eval` and `export_policy` silently fall back to the default
    robot -- which is how a policy ends up judged against, or scaled for, a
    machine it was never trained on.
    """
    p = RobotParams.asbuilt(leg_shape="mesh")
    assert RobotParams.from_dict(json.loads(json.dumps(p.to_dict()))) == p


def test_asbuilt_flag_is_the_measured_robot_and_flags_still_override():
    ap = argparse.ArgumentParser()
    add_robot_args(ap)
    assert build_params(ap.parse_args([])) == RobotParams()
    assert build_params(ap.parse_args(["--asbuilt"])) == RobotParams.asbuilt()
    p = build_params(ap.parse_args(["--asbuilt", "--leg-shape", "c_leg"]))
    assert p.leg_shape == "c_leg"
    assert p.servo_kp == RobotParams.asbuilt().servo_kp  # the rest is untouched


def test_the_leg_stays_put_when_the_servo_is_off():
    """Dry friction has to exceed the gravity torque on a leg.

    Measured on the robot: with torque off the leg does not move at all, at any
    crank angle. A joint carrying only viscous damping cannot reproduce that,
    and a leg that flops under gravity in simulation is a different machine.
    """
    p = RobotParams.asbuilt()
    m = p.leg_mass + p.foot_mass
    gravity_torque = m * 9.81 * 0.0167  # centre of mass 16.7 mm from the hinge
    assert p.joint_frictionloss > gravity_torque


def test_the_accelerometer_carries_motion_not_just_attitude():
    """obs[0:3] must be specific force, the way the real IMU delivers it.

    This encodes a sim2real break that actually shipped. The observation used
    to be the torso's up axis, which is a unit vector by construction; the real
    accelerometer measures gravity *plus* acceleration and ranged over 0.40 to
    3.66 g while the robot tried to stand. The policy met inputs it had never
    seen exactly when it was working hardest, and only flapped its legs.

    The old assertion -- that the vector always has unit length -- is what kept
    this invisible, so the property is inverted here on purpose.
    """
    env = GoronEnv(RobotParams.asbuilt(), task="stand")
    env.reset(seed=0)
    mags = []
    for _ in range(200):
        env.step(np.array([1.0, -1.0]))          # thrash the cranks
        mags.append(float(np.linalg.norm(env.gravity_body)))
    assert max(mags) > 1.5, "no dynamic term: this is attitude, not an accelerometer"
