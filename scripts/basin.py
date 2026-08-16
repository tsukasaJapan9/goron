"""Design sweep: which crank geometry gives the widest basin of success?

The metric is the fraction of random piecewise-constant action sequences that
satisfy the task's own goal test -- see `feasibility.py` for why the rate
matters and the single best hit does not.

The torso is fixed hardware (the M5Stack, 54 x 54 x 16 mm) and so is the servo
(XL330-M288-T), so the free variables are the leg shape and where the hips sit.
"""

from __future__ import annotations

import itertools
from multiprocessing import Pool

from goron.model import RobotParams
from scripts.feasibility import open_loop_search

TRIALS = 400

GRID: dict[str, tuple] = {
    "shape": ("bar", "c_leg"),
    "leg": (40, 50, 60, 70),      # mm, hip to tip
    "hipx": (-1.0, 0.0, 1.0),     # tail / mid / nose
    "hipz": (-1.0, 0.0, 1.0),     # belly plate / mid / top plate
}


def to_params(cfg: dict) -> RobotParams:
    return RobotParams(
        leg_shape=cfg["shape"],
        leg_length=cfg["leg"] / 1000,
        hip_x_frac=cfg["hipx"],
        hip_z_frac=cfg["hipz"],
    )


def evaluate(cfg: dict) -> dict:
    best, _, rate = open_loop_search(to_params(cfg), trials=TRIALS, seed=0)
    return {**cfg, "rate": rate, "best_tilt": best.best_tilt}


def main() -> None:
    keys = list(GRID)
    grid = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    with Pool(12) as pool:
        rows = pool.map(evaluate, grid)

    rows.sort(key=lambda r: (-r["rate"], r["best_tilt"]))
    print(f"{'shape':>6} {'leg':>4} {'hipx':>5} {'hipz':>5} "
          f"{'success%':>9} {'best tilt':>10}")
    print("-" * 44)
    for r in rows:
        print(f"{r['shape']:>6} {r['leg']:>4} {r['hipx']:>5.0f} {r['hipz']:>5.0f} "
              f"{100 * r['rate']:>9.2f} {r['best_tilt']:>10.1f}")


if __name__ == "__main__":
    main()
