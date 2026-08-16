"""Where does the trained policy actually break?

Both the nominal and the domain-randomised evaluation sit at 100%, which tells
you the policy is fine inside the training distribution but nothing about the
margin beyond it. This script pushes one physical parameter at a time until
success collapses, so the sim2real headroom is a number rather than a hope.

    uv run python -m scripts.stress --run runs/selfright

Each axis is applied by overwriting the environment's *nominal* parameters,
which is what `GoronEnv.reset` restores from -- so the change survives resets
and is in force during the settle phase too.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from goron.env import GoronEnv
from goron.model import RobotParams
from goron.train import add_robot_args, build_params


def make(run: Path, obs_noise: float = 0.0, params: RobotParams | None = None):
    params = params or RobotParams()

    def _init():
        return GoronEnv(params, task="selfright", obs_noise=obs_noise)

    venv = DummyVecEnv([_init])
    venv = VecNormalize.load(str(run / "vecnormalize.pkl"), venv)
    venv.training = False
    venv.norm_reward = False
    inner = venv.venv.envs[0].unwrapped
    return venv, inner, PPO.load(run / "best_model", device="cpu")


def success_rate(venv, model, episodes: int) -> float:
    hits = 0
    for _ in range(episodes):
        obs = venv.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = venv.step(action)
            done = bool(dones[0])
        hits += bool(infos[0].get("is_success"))
    return hits / episodes


def scale_friction(env: GoronEnv, s: float) -> None:
    env._nominal["geom_friction"][:, 0] *= s


def scale_torque(env: GoronEnv, s: float) -> None:
    env._nominal["actuator_forcerange"] *= s


def scale_mass(env: GoronEnv, s: float) -> None:
    """Scale mass *and* rotational inertia together.

    Scaling body_mass alone leaves body_inertia untouched, which models a robot
    that is heavy to lift but just as easy to spin -- not a heavier robot, and
    it makes the policy look far more mass-tolerant than it is.
    """
    env._nominal["body_mass"] *= s
    env._nominal["body_inertia"] *= s


def scale_damping(env: GoronEnv, s: float) -> None:
    """Gearbox friction -- also caps the free speed at torque/damping."""
    env._nominal["dof_damping"][6:8] *= s


# Ranges deliberately run past the cliff: the interesting number is where the
# policy dies, not that it survives its own training distribution.
AXES = {
    "floor friction": ((0.0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
                       scale_friction),
    "servo torque": ((0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0),
                     scale_torque),
    "total mass": ((1.0, 2.0, 4.0, 8.0, 12.0, 16.0), scale_mass),
    "gearbox damping": ((1.0, 2.0, 4.0, 8.0, 16.0, 32.0), scale_damping),
}


def bar(rate: float) -> str:
    return "#" * int(round(20 * rate))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("runs/selfright"))
    ap.add_argument("--episodes", type=int, default=30)
    add_robot_args(ap)
    args = ap.parse_args()
    params = build_params(args)

    for axis, (scales, apply) in AXES.items():
        print(f"\n{axis}")
        for s in scales:
            # fresh env each time, so it starts from nominal values
            venv, inner, model = make(args.run, params=params)
            apply(inner, s)
            rate = success_rate(venv, model, args.episodes)
            print(f"  x{s:<5.2f} {100 * rate:5.0f}%  {bar(rate)}")
            venv.close()

    print("\nobservation noise (sigma, on the raw observation)")
    for sigma in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8):
        venv, _, model = make(args.run, obs_noise=sigma, params=params)
        rate = success_rate(venv, model, args.episodes)
        print(f"  {sigma:<5.2f} {100 * rate:5.0f}%  {bar(rate)}")
        venv.close()


if __name__ == "__main__":
    main()
