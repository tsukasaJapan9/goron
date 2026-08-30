"""Parallel design sweep: for each geometry, how far can *any* open-loop action
sequence roll the robot out of belly-up?

This is the go/no-go gate before RL. The metric is the best peak roll angle
reached over `TRIALS` random piecewise-constant control sequences, plus whether
the prone goal (belly contact + up_z > cos 25 deg) was ever satisfied.
"""

from __future__ import annotations

import itertools
import math
from multiprocessing import Pool

import numpy as np

from goron.model import RobotParams
from scripts.feasibility import Sim, roll_of

TRIALS = 1500
HOLD = 40        # sim steps per control segment (0.08 s)
STEPS = 1500     # 3.0 s episode


def evaluate(cfg: dict) -> dict:
    p = RobotParams(
        hip_q_max=math.radians(cfg["qmax"]),
        splay=math.radians(cfg["splay"]),
        leg_length=cfg["leg"] / 1000,
        hip_z_frac=cfg["hipz"],
        torso_width=cfg["width"] / 1000,
    )
    sim = Sim(p)
    rng = np.random.default_rng(0)
    best_peak, best_final, ever_prone = -1.0, -1.0, False
    for _ in range(TRIALS):
        n = int(rng.integers(2, 8))
        seq = rng.uniform(sim.lo, sim.hi, size=(n, 2))
        sim.reset_belly_up()
        r = sim.run(seq, HOLD, STEPS)
        ever_prone |= r.prone
        if r.peak_up_z > best_peak:
            best_peak, best_final = r.peak_up_z, r.final_up_z
    return {
        **cfg,
        "peak_roll": roll_of(best_peak),
        "final_roll": roll_of(best_final),
        "prone": ever_prone,
    }


def main() -> None:
    grid = [
        dict(zip(("qmax", "splay", "leg", "hipz", "width"), v))
        for v in itertools.product(
            (97, 130, 160),      # hip travel, degrees
            (10, 34, 60),        # rest splay, degrees
            (45, 60, 75),        # leg length, mm
            (-1.0, 0.0, 1.0),    # hip anchor: belly / mid / back
            (30,),               # torso width, mm
        )
    ]
    with Pool(12) as pool:
        rows = pool.map(evaluate, grid)

    rows.sort(key=lambda r: (not r["prone"], r["peak_roll"]))
    print(f"{'qmax':>5} {'splay':>6} {'leg':>4} {'hipz':>5} {'width':>6} "
          f"{'peak_roll':>10} {'final':>7} {'prone':>6}")
    print("-" * 60)
    for r in rows[:25]:
        print(f"{r['qmax']:>5} {r['splay']:>6} {r['leg']:>4} {r['hipz']:>5.0f} "
              f"{r['width']:>6} {r['peak_roll']:>10.1f} {r['final_roll']:>7.1f} "
              f"{str(r['prone']):>6}")
    print(f"\n(lower peak_roll is better; prone goal needs < 25 deg. "
          f"belly-up = 180, side = 90)")


if __name__ == "__main__":
    main()
