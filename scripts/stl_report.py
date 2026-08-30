"""Measure the printed design and compare it with the simulated one.

    uv run python -m scripts.stl_report hardware/stl

Reports, per part: bounding box, mesh volume, centroid and triangle count.
Volume times material density gives the printed mass, which is the number the
simulation actually needs -- everything in `RobotParams` downstream of mass
(tipping torque, impact loads, servo margin) was estimated, not measured.

Volume assumes a closed, correctly-oriented mesh; the sign of the result is a
cheap check on that, so a negative volume is reported rather than hidden.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

# g/cm^3 -- for turning mesh volume into printed mass.
DENSITY = {"PLA": 1.24, "PETG": 1.27, "ABS": 1.04, "TPU": 1.21}


def load_stl(path: Path) -> np.ndarray:
    """Return an (n, 3, 3) array of triangle vertices, in millimetres."""
    raw = path.read_bytes()
    # An ASCII STL starts with "solid", but so do some binary ones, so trust the
    # binary triangle count against the file length instead of the header.
    if len(raw) >= 84:
        n = struct.unpack_from("<I", raw, 80)[0]
        if len(raw) == 84 + n * 50:
            data = np.frombuffer(raw, dtype=np.uint8, count=n * 50, offset=84)
            data = data.reshape(n, 50)[:, 12:48].copy()
            return data.view("<f4").reshape(n, 3, 3).astype(np.float64)

    tris, cur = [], []
    for line in raw.decode("utf-8", "replace").splitlines():
        parts = line.split()
        if parts and parts[0] == "vertex":
            cur.append([float(x) for x in parts[1:4]])
            if len(cur) == 3:
                tris.append(cur)
                cur = []
    return np.asarray(tris, dtype=np.float64)


def measure(tris: np.ndarray) -> dict:
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    # Signed volume of the tetrahedron from the origin to each facet.
    sv = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    volume = float(sv.sum())
    centroid = ((a + b + c) / 4.0 * sv[:, None]).sum(axis=0) / volume if volume else None
    lo, hi = tris.reshape(-1, 3).min(axis=0), tris.reshape(-1, 3).max(axis=0)
    return {
        "triangles": len(tris),
        "bbox_min": lo,
        "bbox_max": hi,
        "size": hi - lo,
        "volume_mm3": volume,
        "centroid": centroid,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path, nargs="?",
                    default=Path("hardware/stl"))
    ap.add_argument("--material", default="PETG", choices=sorted(DENSITY))
    args = ap.parse_args()

    rho = DENSITY[args.material]
    files = sorted(args.directory.glob("*.stl"))
    print(f"{'part':<22} {'X':>7} {'Y':>7} {'Z':>7} {'volume':>10} "
          f"{'solid g':>8} {'tris':>7}")
    print("-" * 72)
    for f in files:
        tris = load_stl(f)
        if not len(tris):
            print(f"{f.stem:<22} (読み取れず)")
            continue
        m = measure(tris)
        sx, sy, sz = m["size"]
        v_cm3 = m["volume_mm3"] / 1000.0
        print(f"{f.stem:<22} {sx:7.1f} {sy:7.1f} {sz:7.1f} "
              f"{v_cm3:9.2f}c {v_cm3 * rho:8.1f} {m['triangles']:7d}")
    print(f"\n寸法は mm。'solid g' は中実として {args.material} 密度 "
          f"{rho} g/cm3 を掛けた値で、インフィル率を掛ける前の上限。")


if __name__ == "__main__":
    main()
