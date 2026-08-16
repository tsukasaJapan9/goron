"""PPO training.

    uv run python -m goron.train --task selfright --timesteps 2000000
    uv run python -m goron.train --task forward --leg-shape c_leg

Checkpoints, the VecNormalize statistics and tensorboard logs land in
`runs/<name>/`. The statistics file matters: the policy is trained on
normalised observations, so evaluation and any hardware export must reuse it.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from goron.env import GoronEnv
from goron.model import RobotParams
from goron.tasks import TASKS

RUNS = Path(__file__).resolve().parent.parent / "runs"


class SaveVecNormalize(BaseCallback):
    """Snapshot the observation statistics whenever a new best model is saved.

    `EvalCallback` writes `best_model.zip` the moment evaluation improves, but
    the VecNormalize statistics are only written when training ends. Pairing an
    early `best_model` with end-of-run statistics feeds the policy observations
    normalised by a distribution it never saw: on the crawl task that turned
    1.7 m per episode into 0.4 m. Saving both at the same instant keeps them
    consistent.
    """

    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def _on_step(self) -> bool:
        vec = self.model.get_vec_normalize_env()
        if vec is not None:
            vec.save(str(self.path))
        return True


def add_robot_args(ap: argparse.ArgumentParser) -> None:
    """Geometry flags shared by train and eval, so a run can be reproduced."""
    ap.add_argument("--swing", default="sagittal", choices=("sagittal", "lateral"))
    ap.add_argument("--leg-shape", default="bar",
                    choices=("bar", "c_leg", "spiral", "mesh"))
    ap.add_argument("--leg", type=float, default=50.0, help="leg length, mm")
    ap.add_argument("--hip-x", type=float, default=0.0)
    ap.add_argument("--hip-z", type=float, default=1.0)
    ap.add_argument("--hip-gap", type=float, default=12.0, help="hip outboard gap, mm")
    ap.add_argument("--leg-phase", type=float, default=0.0,
                    help="degrees to rotate the mesh leg about its hinge")
    # Enclosure, so a measured CAD design can be simulated as-built.
    ap.add_argument("--torso-len", type=float, default=54.0, help="fore/aft, mm")
    ap.add_argument("--torso-width", type=float, default=54.0, help="lateral, mm")
    ap.add_argument("--torso-height", type=float, default=16.0, help="vertical, mm")
    ap.add_argument("--body-mass", type=float, default=None,
                    help="body mass in grams: shell + board + battery + servos, "
                         "i.e. everything that is not a leg")
    ap.add_argument("--leg-mass", type=float, default=None,
                    help="mass of one leg in grams, including its servo horn")


def build_params(args: argparse.Namespace) -> RobotParams:
    p = RobotParams(
        swing=args.swing,
        leg_shape=args.leg_shape,
        leg_length=args.leg / 1000,
        hip_x_frac=args.hip_x,
        hip_z_frac=args.hip_z,
        hip_y_gap=args.hip_gap / 1000,
        leg_phase=math.radians(args.leg_phase),
        torso_len=args.torso_len / 1000,
        torso_width=args.torso_width / 1000,
        torso_height=args.torso_height / 1000,
    )
    if args.leg_mass is not None:
        # Two thirds in the limb, one third at the tip -- a C-leg's horn end is
        # heavier than its toe, and the split barely matters next to the 288:1
        # reflected inertia, which is ~6x the whole leg.
        p = p.with_(leg_mass=args.leg_mass / 1000 * 2 / 3,
                    foot_mass=args.leg_mass / 1000 / 3)
    if args.body_mass is not None:
        # The servos sit inside the body, so their mass is already in the
        # measured figure and must not be counted twice.
        p = p.with_(board_mass=args.body_mass / 1000 - 2 * p.servo_mass,
                    extra_mass=0.0)
    if not p.servo_clears_belly:
        raise SystemExit(
            f"--hip-z {args.hip_z} puts the servo case {-p.servo_bottom * 1000:.1f} mm "
            f"below the torso centre, past the belly at "
            f"{p.torso_height / 2 * 1000:.1f} mm. The robot would rest on its servo "
            f"cases and never reach a belly-down pose. Use --hip-z >= "
            f"{(p.servo_size[2] - p.torso_height) / p.torso_height:.2f}."
        )
    if not p.leg_clears_torso:
        raise SystemExit("--hip-gap too small: the leg cannot rotate past the torso.")
    return p


def make_env_fn(args: argparse.Namespace, *, randomize: bool):
    params = build_params(args)

    def _init():
        env = GoronEnv(
            params,
            task=args.task,
            randomize=randomize,
            obs_noise=args.obs_noise if randomize else 0.0,
        )
        return Monitor(env, info_keywords=("is_success",))

    return _init


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="selfright", choices=sorted(TASKS))
    ap.add_argument("--name", default=None, help="run directory (default: task name)")
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomize", action="store_true",
                    help="domain randomisation (mass, friction, servo gain)")
    ap.add_argument("--obs-noise", type=float, default=0.01)
    ap.add_argument("--init-from", type=Path, default=None,
                    help="warm-start from another run's policy and observation "
                         "statistics, instead of training from scratch")
    add_robot_args(ap)
    args = ap.parse_args()

    out = RUNS / (args.name or args.task)
    out.mkdir(parents=True, exist_ok=True)

    venv = make_vec_env(make_env_fn(args, randomize=args.randomize),
                        n_envs=args.n_envs, seed=args.seed,
                        vec_env_cls=SubprocVecEnv)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Evaluation always runs on the nominal robot, so the reported number stays
    # comparable between randomised and non-randomised runs.
    eval_venv = make_vec_env(make_env_fn(args, randomize=False),
                             n_envs=4, seed=args.seed + 1000,
                             vec_env_cls=SubprocVecEnv)
    eval_venv = VecNormalize(eval_venv, norm_obs=True, norm_reward=False,
                             clip_obs=10.0, training=False)

    if args.init_from:
        # A gait found on simpler geometry can beat anything PPO discovers from
        # scratch on the harder one: the mesh leg plateaued at 0.25 m/s, while
        # the policy trained on the primitive C-leg reached 0.31 m/s on that
        # same mesh. Warm-starting keeps the better gait and refines it.
        stats = args.init_from / "vecnormalize.pkl"
        if stats.exists():
            venv = VecNormalize.load(str(stats), venv.venv)
            venv.training, venv.norm_reward = True, True
        model = PPO.load(args.init_from / "best_model", env=venv, device="auto",
                         tensorboard_log=str(out / "tb"))
        print(f"warm-started from {args.init_from}")
    else:
        model = PPO(
            "MlpPolicy", venv,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.002,
            policy_kwargs=dict(net_arch=[128, 128]),
            tensorboard_log=str(out / "tb"),
            seed=args.seed,
            verbose=1,
        )

    callbacks = [
        EvalCallback(eval_venv, best_model_save_path=str(out),
                     log_path=str(out), eval_freq=max(1, 20_000 // args.n_envs),
                     n_eval_episodes=20, deterministic=True,
                     callback_on_new_best=SaveVecNormalize(
                         out / "vecnormalize_best.pkl")),
        # save_vecnormalize matters: the policy is trained on normalised
        # observations, so a checkpoint without its statistics is unusable.
        # Without this, killing a run before it ends loses every checkpoint.
        CheckpointCallback(save_freq=max(1, 200_000 // args.n_envs),
                           save_path=str(out), name_prefix="ckpt",
                           save_vecnormalize=True),
    ]
    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                progress_bar=True)

    model.save(out / "final")
    venv.save(str(out / "vecnormalize.pkl"))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
