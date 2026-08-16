"""Design analysis, to be run before spending time on RL.

Two questions:

1. `flip_ladder()` -- what tilt angles must the torso climb? Treating the torso
   cross-section as a rectangle rolling on its corners gives the unstable
   equilibria between belly-up (180 deg), on-edge (90 deg) and prone (0 deg).
   Sagittal cranks somersault about Y and pivot on the nose/tail edges.

2. `open_loop_search()` -- can *any* action sequence get there, and how *often*?
   The search runs in the same action space the policy uses (piecewise-constant
   crank commands), so a hit here is something PPO can plausibly find.

The success **rate** is the number that matters, not the single best hit. In a
contact-rich simulation a lone success is usually a knife-edge trajectory: on
the earlier limited-joint prototype, the one sequence that reached prone
stopped working under 1e-4 rad/s of initial velocity noise. A configuration
where a few percent of random sequences succeed has a real basin; one with a
single hit in thousands does not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from goron.env import GoronEnv
from goron.model import RobotParams

PRONE_TILT_DEG = 25.0


def flip_ladder(p: RobotParams) -> dict[str, float]:
    """Unstable equilibria, in degrees of tilt away from prone."""
    a, h = p.flip_half_extent, p.torso_height / 2
    corner = math.degrees(math.atan2(h, a))
    return {
        "prone": 0.0,
        "prone -> edge tip": 90.0 - corner,     # exceed this to leave prone
        "on edge": 90.0,
        "edge -> belly_up tip": 90.0 + corner,  # drop below this to escape
        "belly_up": 180.0,
    }


def tilt_of(up_z: float) -> float:
    """Degrees away from prone. 0 = belly down, 180 = belly up."""
    return math.degrees(math.acos(float(np.clip(up_z, -1.0, 1.0))))


@dataclass
class Rollout:
    best_tilt: float      # closest approach to prone, degrees
    final_tilt: float
    success: bool         # the task's own goal test


def run_sequence(env: GoronEnv, seq: np.ndarray, hold: int, seed: int) -> Rollout:
    """Play a piecewise-constant action sequence, each row held `hold` steps."""
    env.reset(seed=seed)
    best, success, info = 180.0, False, {}
    for i in range(env.task.max_steps):
        _, _, term, trunc, info = env.step(seq[min(i // hold, len(seq) - 1)])
        best = min(best, info["tilt_deg"])
        success |= bool(info.get("is_success"))
        if term or trunc:
            break
    return Rollout(best, info["tilt_deg"], success)


def open_loop_search(
    p: RobotParams,
    task: str = "selfright",
    trials: int = 500,
    hold: int = 20,       # control steps per segment (0.4 s)
    seed: int = 0,
) -> tuple[Rollout, np.ndarray | None, float]:
    """Returns the best rollout, its sequence, and the success *rate*."""
    env = GoronEnv(p, task=task)
    rng = np.random.default_rng(seed)
    best, best_seq, hits = Rollout(180.0, 180.0, False), None, 0
    for t in range(trials):
        seq = rng.uniform(-1.0, 1.0, size=(int(rng.integers(2, 8)), 2))
        r = run_sequence(env, seq, hold, seed=t)
        hits += r.success
        if (r.success, -r.best_tilt) > (best.success, -best.best_tilt):
            best, best_seq = r, seq
    return best, best_seq, hits / trials


if __name__ == "__main__":
    p = RobotParams()
    print(f"swing={p.swing} leg={p.leg_shape} "
          f"tips over +/-{p.flip_half_extent * 1000:.0f} mm, "
          f"mass {p.total_mass * 1000:.0f} g")
    for k, v in flip_ladder(p).items():
        print(f"  {k:22s} {v:6.1f} deg")
    best, _, rate = open_loop_search(p, trials=300)
    print(f"\nopen-loop (300 random sequences): closest approach "
          f"{best.best_tilt:.1f} deg, success rate {100 * rate:.1f}%")
