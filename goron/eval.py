"""Evaluate a trained policy and optionally record a video.

    uv run python -m goron.eval --run runs/selfright --episodes 50 --video sr.mp4

Report the nominal number and the randomised one; the gap between them is the
honest sim2real indicator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from goron.env import GoronEnv
from goron.tasks import TASKS
from goron.train import add_robot_args, build_params


def make_eval_venv(args, *, render: bool) -> VecNormalize:
    params = build_params(args)

    def _init():
        return GoronEnv(
            params,
            task=args.task,
            randomize=args.randomize,
            obs_noise=args.obs_noise if args.randomize else 0.0,
            render_mode="rgb_array" if render else None,
        )

    venv = DummyVecEnv([_init])
    # Use the statistics saved at the same moment as the model being loaded --
    # `best_model` is written mid-run, `vecnormalize.pkl` only at the end, and
    # mixing the two feeds the policy a distribution it never trained on.
    run = Path(args.run)
    best_stats = run / "vecnormalize_best.pkl"
    stats = (best_stats if args.model == "best_model" and best_stats.exists()
             else run / "vecnormalize.pkl")
    if stats.exists():
        venv = VecNormalize.load(str(stats), venv)
        venv.training = False
        venv.norm_reward = False
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, training=False)
    return venv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/selfright")
    ap.add_argument("--task", default="selfright", choices=sorted(TASKS))
    ap.add_argument("--model", default="best_model")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--randomize", action="store_true")
    ap.add_argument("--obs-noise", type=float, default=0.01)
    add_robot_args(ap)
    args = ap.parse_args()

    venv = make_eval_venv(args, render=args.video is not None)
    model = PPO.load(Path(args.run) / args.model, device="cpu")
    inner = venv.venv.envs[0].unwrapped

    successes, steps_to_goal, frames = 0, [], []
    metrics: dict[str, list[float]] = {}
    for ep in range(args.episodes):
        obs = venv.reset()
        done, n = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = venv.step(action)
            done, info, n = bool(dones[0]), infos[0], n + 1
            if args.video is not None and ep == 0:
                frames.append(inner.render())
        # DummyVecEnv auto-resets, but the info it returns is the one from the
        # terminating step, so these values are the real terminal ones.
        if info.get("is_success"):
            successes += 1
            steps_to_goal.append(n)
        for k, v in info.items():
            if isinstance(v, (int, float)) and k != "is_success":
                metrics.setdefault(k, []).append(float(v))

    print(f"task         : {args.task}")
    if inner.task.has_success:
        print(f"success rate : {successes}/{args.episodes} "
              f"({100 * successes / args.episodes:.0f}%)")
    if steps_to_goal:
        print(f"steps to goal: mean {np.mean(steps_to_goal):.0f} "
              f"({np.mean(steps_to_goal) / 50:.1f} s), "
              f"median {np.median(steps_to_goal):.0f}")
    for k, v in metrics.items():
        print(f"{k:13s}: mean {np.mean(v):8.2f}  median {np.median(v):8.2f}")

    if args.video is not None and frames:
        import imageio.v2 as imageio

        imageio.mimsave(args.video, frames, fps=50)
        print(f"wrote {args.video} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
