"""Convex-decompose the printed leg so it can be used as collision geometry.

    uv run python -m scripts.split_leg

MuJoCo collides a mesh as its **convex hull**. The hull of the spiral leg is a
filled disc, which throws away the one property the shape exists for: that the
ground contact point migrates along an arc as the crank turns. Handing the raw
mesh to the collider would look more faithful and silently simulate a wheel.

The fix is to cut the leg into angular wedges about the hub axis. Each wedge is
thin enough that its convex hull is close to the wedge itself, and the union of
the hulls follows the spiral. The wedges are written as separate STLs for
`leg_shape="mesh"` in `goron.model` to reference.

Frames. The STLs are in the CAD assembly frame (mm), with the hinge axis along
CAD +X through (Y, Z) = (35, 0). The simulation puts the hinge along its own +Y
with the leg swinging in XZ, so each vertex is mapped

    sim_x = CAD_z              (fore/aft, the swing plane)
    sim_y = CAD_x + 52         (along the hinge; 52 mm centres the 8 mm plate)
    sim_z = CAD_y - 35         (up, measured from the hinge)

and converted to metres. Both legs share one set of meshes: the printed left
and right legs have the same profile in the swing plane and differ only in
which side of the body they bolt to.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np

from scripts.stl_report import load_stl

HIP_Y_CAD, PLATE_X_CAD = 35.0, 52.0


def to_sim_frame(tris: np.ndarray) -> np.ndarray:
    """CAD assembly millimetres -> simulation leg-body metres."""
    out = np.empty_like(tris)
    out[..., 0] = tris[..., 2]
    out[..., 1] = tris[..., 0] + PLATE_X_CAD
    out[..., 2] = tris[..., 1] - HIP_Y_CAD
    return out / 1000.0


def write_binary_stl(path: Path, tris: np.ndarray) -> None:
    n = len(tris)
    with path.open("wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", n))
        for t in tris:
            normal = np.cross(t[1] - t[0], t[2] - t[0])
            norm = np.linalg.norm(normal)
            normal = normal / norm if norm else np.zeros(3)
            f.write(struct.pack("<3f", *normal))
            for v in t:
                f.write(struct.pack("<3f", *v))
            f.write(b"\0\0")


def split(tris: np.ndarray, wedges: int,
          hub_radius: float) -> tuple[np.ndarray, list[np.ndarray]]:
    """Separate the hub, then cut the arm into angular wedges.

    Splitting by angle alone is not enough: near the arm's root a single wedge
    spans radius 8..74 mm, and its convex hull is a filled pie slice reaching
    from the hub to the tip. Pulling the hub out first leaves each arm wedge as
    a thin curved bar, whose hull tracks the real shape closely.
    """
    radius = np.linalg.norm(tris[:, :, [0, 2]], axis=2)      # per vertex
    is_hub = radius.max(axis=1) <= hub_radius
    hub, arm = tris[is_hub], tris[~is_hub]

    centre = arm.mean(axis=1)
    theta = np.arctan2(centre[:, 2], centre[:, 0]) % (2 * math.pi)
    idx = np.minimum((theta / (2 * math.pi) * wedges).astype(int), wedges - 1)
    return hub, [arm[idx == k] for k in range(wedges)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", type=Path, default=Path("hardware/stl/leg_left.stl"))
    ap.add_argument("--out", type=Path, default=Path("hardware/stl/wedges"))
    ap.add_argument("--wedges", type=int, default=24)
    ap.add_argument("--hub-radius", type=float, default=0.020, help="metres")
    args = ap.parse_args()

    tris = to_sim_frame(load_stl(args.stl))
    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("*.stl"):
        old.unlink()

    hub, wedges = split(tris, args.wedges, args.hub_radius)
    write_binary_stl(args.out / "hub.stl", hub)
    print(f"  hub        {len(hub):5d} tris")

    kept = ["hub"]
    for k, group in enumerate(wedges):
        if len(group) < 4:          # too few facets to form a solid hull
            continue
        name = f"wedge_{k:02d}"
        write_binary_stl(args.out / f"{name}.stl", group)
        kept.append(name)
        r = np.linalg.norm(group.reshape(-1, 3)[:, [0, 2]], axis=1) * 1000
        # A wedge whose hull spans a wide radial band is a filled pie slice --
        # the exact failure this decomposition exists to avoid.
        flag = "  <- wide" if (r.max() - r.min()) > 25 else ""
        print(f"  {name}  {len(group):5d} tris  "
              f"radius {r.min():5.1f}..{r.max():5.1f} mm{flag}")
    print(f"\nwrote {len(kept)} pieces to {args.out}")


if __name__ == "__main__":
    main()
