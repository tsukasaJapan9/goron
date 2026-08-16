"""Build a URDF from the printed design's STL files.

    uv run python -m scripts.make_urdf --body-mass 270 --leg-mass 20

What this buys over the parametric MJCF: real mass properties. Inertia is
integrated over the actual meshes instead of approximated by a box, and the
centre of mass lands where the CAD puts it, not where a uniform-density
primitive would.

**Collision geometry is deliberately NOT the mesh.** MuJoCo (and most engines)
collide a mesh as its *convex hull*, and the convex hull of a C-leg is a filled
half-disc -- the robot would roll on a solid wheel instead of an arc, which is
precisely the property that makes the C-leg interesting. The URDF therefore
carries the meshes as `<visual>` only; `<collision>` keeps the primitive
capsule chain that the MJCF builder generates. Feeding the raw mesh to a
collider would silently change the physics while looking more accurate.

Geometry read out of the CAD (all parts share one assembly frame, in mm):

    enclosure   80 (X, hinge axis) x 72 (Y, vertical) x 80 (Z, fore/aft)
    hinge axis  along +X, passing through (Y, Z) = (35, 0)
    leg plates  at |X| = 48..56, tip 73.7 mm from the axis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scripts.stl_report import load_stl

# Hinge axis, in the CAD assembly frame (mm). Taken from servo_joint_*.stl,
# whose 30 x 30 mm cross-section is centred on (Y, Z) = (35, 0) -- and confirmed
# by the leg's 7 mm shaft bore sitting on the same centre.
HIP_Y, HIP_Z = 35.0, 0.0

# Everything rigidly attached to the body. Masses in grams: known figures for
# the bought parts, and the printed shell takes whatever is left of the
# measured body mass. `all.stl` is a preview of the whole assembly and is
# deliberately not used -- it would double every part.
BODY_PARTS = ("body_bottom", "body_upper", "m5stack", "bat",
              "servo_left", "servo_right")
KNOWN_MASS = {"m5stack": 73.3, "servo_left": 18.0, "servo_right": 18.0}
SHELL_PARTS = ("body_bottom", "body_upper")

# The rotating side: the leg and the horn adapter that clamps it to the output.
LEG_PARTS = {"left": ("leg_left", "servo_joint_left"),
             "right": ("leg_right", "servo_joint_right")}


def mesh_properties(tris: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Volume, centroid and inertia tensor about the centroid, at density 1.

    Integrates over the tetrahedra spanned by the origin and each facet, which
    is exact for a closed, outward-oriented mesh.
    """
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    det = np.einsum("ij,ij->i", a, np.cross(b, c))      # 6 * signed volume
    volume = det.sum() / 6.0
    # Each tetrahedron's centroid is (0+a+b+c)/4, weighted by its signed volume
    # det/6 -- so the normaliser is sum(det)/6, i.e. the volume itself.
    centroid = ((a + b + c) / 4.0 * det[:, None]).sum(axis=0) / det.sum()

    # Second-moment covariance of a tetrahedron (origin, a, b, c).
    cov = np.zeros((3, 3))
    for p in (a, b, c):
        for q in (a, b, c):
            cov += np.einsum("i,ij,ik->jk", det, p, q)
        cov += np.einsum("i,ij,ik->jk", det, p, p)
    cov /= 120.0

    # Shift to the centroid, then convert covariance to an inertia tensor.
    cov -= volume * np.outer(centroid, centroid)
    inertia = np.trace(cov) * np.eye(3) - cov
    return volume, centroid, inertia


def combined(parts: list[str], stl_dir: Path) -> tuple[float, np.ndarray, np.ndarray]:
    """Merge several parts into one rigid body, at uniform density."""
    return combined_with_masses({p: None for p in parts}, stl_dir)[1:]


def combined_with_masses(
    masses_g: dict[str, float | None], stl_dir: Path
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Mass properties of a rigid group whose parts have *different* densities.

    Treating the body as uniform would put its centre of mass in the middle of
    the shell. In reality the M5Stack is bolted to one wall, the battery to the
    other and the servos sit between them, so the real centre of mass and
    inertia only come out right if each part carries its own mass.

    Returns (total mass in kg, total volume in mm^3, centre of mass in mm,
    inertia about that centre in kg m^2). A `None` mass means "unknown" and is
    only valid when every part is unknown, in which case unit density is used.
    """
    pieces = []
    for name, mass_g in masses_g.items():
        v, c, i = mesh_properties(load_stl(stl_dir / f"{name}.stl"))
        pieces.append((mass_g, v, c, i))

    if all(m is None for m, *_ in pieces):          # uniform-density fallback
        total_v = sum(v for _, v, _, _ in pieces)
        pieces = [(v / total_v, v, c, i) for _, v, c, i in pieces]

    total_m = sum(m for m, *_ in pieces)
    total_v = sum(v for _, v, _, _ in pieces)
    com = sum(m * c for m, _, c, _ in pieces) / total_m

    inertia = np.zeros((3, 3))
    for m, v, c, i in pieces:
        # i is the unit-density integral in mm^5; scale to this part's density,
        # convert mm^5 -> m^5, then shift to the group's centre of mass.
        rho = (m / 1000.0) / (v * 1e-9)             # kg/m^3
        d = (c - com) / 1000.0                      # mm -> m
        inertia += (i * 1e-15 * rho
                    + (m / 1000.0) * (np.dot(d, d) * np.eye(3) - np.outer(d, d)))
    return total_m / 1000.0, total_v, com, inertia


def _self_check(stl_dir: Path) -> None:
    """m5stack.stl is a plain box, so its inertia has a closed form to check."""
    v, _, i = mesh_properties(load_stl(stl_dir / "m5stack.stl"))
    tris = load_stl(stl_dir / "m5stack.stl").reshape(-1, 3)
    w, h, d = tris.max(axis=0) - tris.min(axis=0)
    want = np.diag([(h * h + d * d), (w * w + d * d), (w * w + h * h)]) * v / 12.0
    err = np.abs(np.diag(i) - np.diag(want)).max() / np.diag(want).max()
    print(f"self-check (box inertia): relative error {err:.2e}"
          f" {'OK' if err < 1e-6 else 'FAILED'}")


def body_masses(body_mass_g: float, battery_g: float,
                stl_dir: Path) -> dict[str, float]:
    """Split the measured body mass across the parts that make it up.

    The bought parts have known masses; the printed shell takes the remainder,
    divided between its two halves by volume.
    """
    remainder = body_mass_g - sum(KNOWN_MASS.values()) - battery_g
    if remainder <= 0:
        raise SystemExit(
            f"body mass {body_mass_g:.0f} g is already used up by the M5Stack, "
            f"servos and a {battery_g:.0f} g battery -- nothing left for the "
            f"printed shell. Raise --body-mass or lower --battery-mass.")
    vols = {p: mesh_properties(load_stl(stl_dir / f"{p}.stl"))[0]
            for p in SHELL_PARTS}
    total = sum(vols.values())
    masses = dict(KNOWN_MASS)
    masses["bat"] = battery_g
    for p, v in vols.items():
        masses[p] = remainder * v / total
    return masses


def build(stl_dir: Path, body_mass_g: float, battery_g: float,
          leg_mass_g: float) -> str:
    out = ['<?xml version="1.0"?>', '<robot name="goron">',
           # MuJoCo reads this block when loading a URDF: it locates the meshes
           # and keeps visual-only geoms, which it otherwise discards.
           '  <mujoco>',
           '    <compiler meshdir="stl/" discardvisual="false" balanceinertia="true"/>',
           '  </mujoco>']

    def inertial(mass_kg: float, com_mm: np.ndarray, inertia: np.ndarray,
                 origin_mm: np.ndarray) -> str:
        c = (com_mm - origin_mm) / 1000.0
        return (f'    <inertial>\n'
                f'      <origin xyz="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}"/>\n'
                f'      <mass value="{mass_kg:.5f}"/>\n'
                f'      <inertia ixx="{inertia[0,0]:.4e}" ixy="{inertia[0,1]:.4e}" '
                f'ixz="{inertia[0,2]:.4e}" iyy="{inertia[1,1]:.4e}" '
                f'iyz="{inertia[1,2]:.4e}" izz="{inertia[2,2]:.4e}"/>\n'
                f'    </inertial>')

    masses = body_masses(body_mass_g, battery_g, stl_dir)
    m, _, com, inertia = combined_with_masses(
        {p: masses[p] for p in BODY_PARTS}, stl_dir)

    out.append('  <link name="base_link">')
    out.append(inertial(m, com, inertia, np.zeros(3)))
    for part in BODY_PARTS:
        out.append(f'    <visual><origin xyz="0 0 0"/><geometry>'
                   f'<mesh filename="{part}.stl" scale="0.001 0.001 0.001"/>'
                   f'</geometry></visual>')
    out.append('  </link>')

    for side, parts in LEG_PARTS.items():
        # The horn adapter is small next to the leg, so splitting the measured
        # leg mass between them by volume is accurate enough.
        vols = {p: mesh_properties(load_stl(stl_dir / f"{p}.stl"))[0]
                for p in parts}
        tv = sum(vols.values())
        m, _, c, i = combined_with_masses(
            {p: leg_mass_g * v / tv for p, v in vols.items()}, stl_dir)
        hip = np.array([0.0, HIP_Y, HIP_Z])
        out.append(f'  <joint name="hip_{side}" type="continuous">')
        out.append(f'    <parent link="base_link"/>')
        out.append(f'    <child link="leg_{side}"/>')
        out.append(f'    <origin xyz="0 {HIP_Y / 1000:.6f} {HIP_Z / 1000:.6f}"/>')
        out.append(f'    <axis xyz="1 0 0"/>')
        out.append(f'    <limit effort="0.52" velocity="10.79"/>')
        out.append(f'    <dynamics damping="0.0482"/>')
        out.append(f'  </joint>')
        out.append(f'  <link name="leg_{side}">')
        out.append(inertial(m, c, i, hip))
        for part in parts:
            out.append(f'    <visual>'
                       f'<origin xyz="0 {-HIP_Y / 1000:.6f} {-HIP_Z / 1000:.6f}"/>'
                       f'<geometry><mesh filename="{part}.stl" '
                       f'scale="0.001 0.001 0.001"/></geometry></visual>')
        out.append('  </link>')

    out.append('</robot>')
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl-dir", type=Path, default=Path("hardware/stl"))
    ap.add_argument("--out", type=Path, default=Path("hardware/goron.urdf"))
    ap.add_argument("--body-mass", type=float, default=270.0,
                    help="grams, everything that is not a leg (measured)")
    ap.add_argument("--battery-mass", type=float, default=60.0, help="grams")
    ap.add_argument("--leg-mass", type=float, default=20.0, help="grams per leg")
    args = ap.parse_args()

    _self_check(args.stl_dir)
    urdf = build(args.stl_dir, args.body_mass, args.battery_mass, args.leg_mass)
    args.out.write_text(urdf)
    print(f"wrote {args.out}\n")

    masses = body_masses(args.body_mass, args.battery_mass, args.stl_dir)
    print("body mass breakdown:")
    for part, g in sorted(masses.items(), key=lambda kv: -kv[1]):
        v = mesh_properties(load_stl(args.stl_dir / f"{part}.stl"))[0] / 1000
        print(f"  {part:<14} {g:6.1f} g   {v:6.2f} cm3   "
              f"{g / v:5.2f} g/cm3{'  (fitted)' if part in SHELL_PARTS else ''}")

    m, v, c, i = combined_with_masses(
        {p: masses[p] for p in BODY_PARTS}, args.stl_dir)
    print(f"\nbase_link  {m * 1000:.1f} g  com ({c[0]:6.1f},{c[1]:6.1f},{c[2]:6.1f}) mm"
          f"  I_diag {np.diag(i)[0]:.3e} {np.diag(i)[1]:.3e} {np.diag(i)[2]:.3e}")
    vols = {p: mesh_properties(load_stl(args.stl_dir / f"{p}.stl"))[0]
            for p in LEG_PARTS["left"]}
    tv = sum(vols.values())
    m, v, c, i = combined_with_masses(
        {p: args.leg_mass * vv / tv for p, vv in vols.items()}, args.stl_dir)
    print(f"leg_left   {m * 1000:.1f} g  com ({c[0]:6.1f},{c[1]:6.1f},{c[2]:6.1f}) mm"
          f"  I_diag {np.diag(i)[0]:.3e} {np.diag(i)[1]:.3e} {np.diag(i)[2]:.3e}")


if __name__ == "__main__":
    main()
