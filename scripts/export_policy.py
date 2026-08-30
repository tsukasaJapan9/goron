"""Export a trained policy as a C header for the M5Stack.

    uv run python -m scripts.export_policy --run runs/mesh_selfright \\
        --out firmware/policy_selfright.h

The policy is a 14 -> 128 -> 128 -> 2 MLP with tanh activations: 18,692
parameters, 73 KB as float32, and about 19k multiply-accumulates per inference.
An ESP32-S3 at 240 MHz runs that in well under a millisecond, against a 20 ms
budget at 50 Hz, so no quantisation or inference framework is needed -- a plain
loop over the weights is enough.

The header carries three things the robot cannot work without:

1. the network weights;
2. the observation normalisation (mean and variance) that the policy was
   trained with -- feeding it raw observations produces confident nonsense;
3. the action semantics (increment scaling and the anti-windup clamp).

Export correctness is verified here against Stable-Baselines3 itself: the same
observations go through both this file's NumPy reimplementation and
`model.predict`, and the maximum difference is printed. If that number is not
tiny, the header is wrong and the robot will not behave like the simulation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from goron.env import GoronEnv
from goron.model import RobotParams
from goron.train import load_run_params


def load(run: Path, model_name: str) -> tuple[PPO, VecNormalize]:
    model = PPO.load(run / model_name, device="cpu")
    stats = run / ("vecnormalize_best.pkl" if model_name == "best_model"
                   else "vecnormalize.pkl")
    if not stats.exists():
        stats = run / "vecnormalize.pkl"
    venv = VecNormalize.load(
        str(stats), DummyVecEnv([lambda: GoronEnv(RobotParams(), task="selfright")])
    )
    return model, venv


def weights(model: PPO) -> dict[str, np.ndarray]:
    p = model.policy
    net = p.mlp_extractor.policy_net
    return {
        "w0": net[0].weight.detach().numpy(), "b0": net[0].bias.detach().numpy(),
        "w1": net[2].weight.detach().numpy(), "b1": net[2].bias.detach().numpy(),
        "w2": p.action_net.weight.detach().numpy(),
        "b2": p.action_net.bias.detach().numpy(),
    }


def forward(w: dict[str, np.ndarray], obs_n: np.ndarray) -> np.ndarray:
    """The exact computation the C code performs."""
    h = np.tanh(w["w0"] @ obs_n + w["b0"])
    h = np.tanh(w["w1"] @ h + w["b1"])
    return np.clip(w["w2"] @ h + w["b2"], -1.0, 1.0)


def normalise(venv: VecNormalize, obs: np.ndarray) -> np.ndarray:
    return np.clip((obs - venv.obs_rms.mean) / np.sqrt(venv.obs_rms.var + venv.epsilon),
                   -venv.clip_obs, venv.clip_obs)


def verify(model: PPO, venv: VecNormalize, w: dict, n: int = 2000) -> float:
    """Compare this file's arithmetic against SB3's own inference."""
    rng = np.random.default_rng(0)
    lo = venv.obs_rms.mean - 3 * np.sqrt(venv.obs_rms.var)
    hi = venv.obs_rms.mean + 3 * np.sqrt(venv.obs_rms.var)
    raw = rng.uniform(lo, hi, size=(n, len(venv.obs_rms.mean))).astype(np.float32)
    theirs, _ = model.predict(normalise(venv, raw), deterministic=True)
    mine = np.stack([forward(w, o) for o in normalise(venv, raw)])
    return float(np.abs(np.clip(theirs, -1, 1) - mine).max())


def c_array(name: str, a: np.ndarray, width: int = 84) -> str:
    """Emit a C initialiser.

    Two things this has to get right: wrap *between* values, never inside one
    (slicing the joined text on a character count cuts literals in half and the
    header will not compile), and brace each row of a 2-D array so the compiler
    does not warn about a flat initialiser.
    """
    def row(values: np.ndarray, indent: str) -> list[str]:
        lines, line = [], indent
        for v in values:
            token = f" {v:.8g}f,"
            if len(line) + len(token) > width:
                lines.append(line)
                line = indent
            line += token
        lines.append(line)
        return lines

    dims = "".join(f"[{d}]" for d in a.shape)
    if a.ndim == 1:
        body = row(a, "   ")
        body[-1] = body[-1].rstrip(",")
    else:
        body = []
        for i, r in enumerate(a):
            chunk = row(r, "     ")
            chunk[0] = "    {" + chunk[0][5:]
            chunk[-1] = chunk[-1].rstrip(",") + ("}," if i < len(a) - 1 else "}")
            body += chunk
    return f"static const float {name}{dims} = {{\n" + "\n".join(body) + "\n};\n"


def _touch_firmware(out: Path) -> None:
    """Force the firmware to rebuild after the policy header changes.

    main.cpp pulls the policy in as `#include GORON_POLICY`, a macro the
    PlatformIO dependency scanner cannot expand, so it never learns that the
    header is a dependency: exporting a new policy and reflashing silently
    leaves the *old* weights on the robot -- which is indistinguishable from a
    policy that simply behaves badly.

    Touching main.cpp does not help either, because SCons decides what to
    rebuild from content hashes rather than timestamps. Deleting the object
    file is what actually works.
    """
    build = out.resolve().parent / "goron" / ".pio" / "build"
    stale = list(build.glob("*/src/main.cpp.o")) if build.exists() else []
    for obj in stale:
        obj.unlink()
    if stale:
        print(f"removed {len(stale)} stale object file(s) so the firmware rebuilds")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("runs/mesh_selfright"))
    ap.add_argument("--model", default="best_model")
    ap.add_argument("--out", type=Path, default=Path("firmware/policy.h"))
    ap.add_argument("--max-delta", type=float, default=None,
                    help="rad per control step; default = servo no-load speed / 50 Hz")
    ap.add_argument("--max-lead", type=float, default=0.5)
    ap.add_argument("--crank-zero", type=float, default=0.0,
                    help="degrees the servo zero must be offset from the CAD "
                         "assembly pose for this policy")
    args = ap.parse_args()

    model, venv = load(args.run, args.model)
    w = weights(model)
    err = verify(model, venv, w)
    print(f"export check vs stable-baselines3: max |difference| = {err:.3e} "
          f"{'OK' if err < 1e-5 else 'FAILED -- do not flash this'}")

    # The action scale must come from the robot this policy was trained on:
    # a saturated action is meant to command exactly that servo's no-load speed.
    p = load_run_params(args.run) or RobotParams()
    if load_run_params(args.run) is None:
        print(f"warning: no params.json in {args.run} -- action scale falls back "
              f"to the default robot, which may not be what it was trained on")
    max_delta = args.max_delta or p.servo_no_load_speed * 0.02
    mean, std = venv.obs_rms.mean, np.sqrt(venv.obs_rms.var + venv.epsilon)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _touch_firmware(args.out)
    with args.out.open("w") as f:
        f.write(f"""// Generated by scripts/export_policy.py from {args.run}/{args.model}
// Verified against stable-baselines3: max difference {err:.2e}
//
// Observation layout (all available on the robot):
//   [0:3]   gravity direction in the body frame (IMU attitude)
//   [3:6]   body angular velocity, rad/s        (gyro)
//   [6:10]  sin(qL), sin(qR), cos(qL), cos(qR)  (XL330 present position)
//   [10:12] crank velocity / {p.servo_no_load_speed:.2f}          (XL330 present velocity)
//   [12:14] servo torque / {p.servo_stall_torque:.2f}             (XL330 present current)
//
// CRANK ZERO: this policy expects servo angle 0 to be the CAD assembly pose
// rotated by GORON_CRANK_ZERO_DEG. Calibrate the servo origin there. A
// mismatched zero is not a small loss of quality -- measured on this robot, a
// 240 degree offset took self-righting from 100%% to 27%% and travel from
// 0.31 m/s to 0.04 m/s.
#pragma once
#include <math.h>

#define GORON_OBS 14
#define GORON_ACT 2
#define GORON_HID 128
#define GORON_CONTROL_HZ 50
// Target increment for a saturated action, and how far the target may run
// ahead of the measured angle (anti-windup; required, not optional).
#define GORON_MAX_DELTA {max_delta:.6f}f
#define GORON_MAX_LEAD  {args.max_lead:.6f}f
#define GORON_OBS_CLIP  {venv.clip_obs:.1f}f
#define GORON_CRANK_ZERO_DEG {args.crank_zero:.1f}f

""")
        f.write(c_array("goron_obs_mean", mean.astype(np.float32)))
        f.write(c_array("goron_obs_std", std.astype(np.float32)))
        for k in ("w0", "b0", "w1", "b1", "w2", "b2"):
            f.write(c_array(f"goron_{k}", w[k].astype(np.float32)))
        f.write("""
// raw_obs -> action, both caller-allocated. No dynamic memory, no framework.
static inline void goron_policy(const float raw[GORON_OBS],
                                float action[GORON_ACT]) {
  float x[GORON_OBS], h0[GORON_HID], h1[GORON_HID];
  for (int i = 0; i < GORON_OBS; ++i) {
    float v = (raw[i] - goron_obs_mean[i]) / goron_obs_std[i];
    x[i] = v > GORON_OBS_CLIP ? GORON_OBS_CLIP
         : (v < -GORON_OBS_CLIP ? -GORON_OBS_CLIP : v);
  }
  for (int j = 0; j < GORON_HID; ++j) {
    float s = goron_b0[j];
    for (int i = 0; i < GORON_OBS; ++i) s += goron_w0[j][i] * x[i];
    h0[j] = tanhf(s);
  }
  for (int j = 0; j < GORON_HID; ++j) {
    float s = goron_b1[j];
    for (int i = 0; i < GORON_HID; ++i) s += goron_w1[j][i] * h0[i];
    h1[j] = tanhf(s);
  }
  for (int j = 0; j < GORON_ACT; ++j) {
    float s = goron_b2[j];
    for (int i = 0; i < GORON_HID; ++i) s += goron_w2[j][i] * h1[i];
    action[j] = s > 1.0f ? 1.0f : (s < -1.0f ? -1.0f : s);
  }
}
""")
    size = args.out.stat().st_size
    print(f"wrote {args.out} ({size / 1024:.0f} KB source, "
          f"{sum(a.size for a in w.values()) * 4 / 1024:.0f} KB of weights)")


if __name__ == "__main__":
    main()
