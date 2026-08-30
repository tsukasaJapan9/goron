"""MJCF builder for the two-leg robot.

Topology (NOT the serial chain of `poco`):

        torso (M5Stack + battery, free joint)
       /                              \\
   hip_left (hinge, axis = Y)    hip_right (hinge, axis = Y)
   continuous rotation           continuous rotation

The legs hang off the left and right sides of the torso (at +/-Y) but *swing in
the sagittal plane*: the hinge axes are parallel to the torso's lateral (Y)
axis, so each leg sweeps fore and aft. Both hips rotate **continuously through
360 degrees** -- they are cranks, not limited joints. This is the RHex layout,
reduced to two legs.

Consequences that drive the whole design:

* Both legs cranking together produces a **pitch** torque, so self-righting is
  an end-over-end somersault about Y, not a roll. The torso dimension that sets
  the difficulty is therefore `torso_len` (X), not `torso_width`.
* No left/right mirroring: both hips turn the same way about +Y, so
  `ctrl = [+a, +a]` cranks the legs in phase (pitch / hopping) and
  `ctrl = [+a, -a]` cranks them in anti-phase (yaw / scissoring).
* The hips must sit **outboard of the torso** (`hip_y_gap`), otherwise a leg
  cannot pass the torso and 360 degree rotation is impossible.

Hardware modelled
-----------------
* Servo: **Dynamixel XL330-M288-T**. 288.4:1, 12-bit absolute encoder,
  0.52 N.m stall / 103 rpm no-load at 5.0 V (0.42 N.m / 76 rpm at 3.7 V), 18 g.
  Driven in **Extended Position Control Mode** (+/-256 rev), which is what makes
  continuous rotation *and* absolute angle feedback available at the same time.
* Controller: **M5Stack Core-series**, 54 x 54 x 16 mm, 6-axis IMU on board.
  `board_mass` defaults to the CoreS3's 73.3 g (battery included); a Core2 is
  about 52 g.

The 288:1 gearbox matters more than it looks: reflected rotor inertia is
`ratio^2 * I_rotor`, which lands around 8e-4 kg.m^2 -- roughly fifty times the
inertia of the leg itself. It dominates the swing dynamics, so it is modelled
explicitly as joint `armature` rather than ignored.
"""

from __future__ import annotations

import math
import dataclasses
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@dataclass(frozen=True)
class RobotParams:
    """Every geometric / actuator quantity worth sweeping.

    Lengths are metres, angles radians, masses kilograms, torques N.m.
    """

    # --- which plane the legs swing in -------------------------------------
    swing: Literal["sagittal", "lateral"] = "sagittal"

    # --- torso = the M5Stack controller ------------------------------------
    torso_len: float = 0.054      # X, fore/aft -- the tipping dimension
    torso_width: float = 0.054    # Y, lateral
    torso_height: float = 0.016   # Z
    board_mass: float = 0.0733    # M5Stack CoreS3 incl. its 500 mAh battery
    extra_mass: float = 0.020     # frame, wiring, fasteners

    # --- legs --------------------------------------------------------------
    # "spiral" reproduces the printed leg measured off the STL: a hub disc with
    # an Archimedean arm winding out from it. `bar` and `c_leg` are the earlier
    # study shapes, kept so the comparison stays reproducible.
    # "mesh" uses the convex-decomposed printed leg from hardware/stl/wedges,
    # produced by scripts/split_leg.py. It is the highest-fidelity option and
    # the slowest; "spiral" is its primitive-geometry stand-in.
    leg_shape: Literal["bar", "c_leg", "spiral", "mesh"] = "bar"
    wedge_dir: Path = MODELS_DIR.parent / "hardware" / "stl" / "wedges"
    # Rotation applied to the mesh leg about its hinge, to line the printed
    # part's zero angle up with the primitive shapes' convention. Without it a
    # policy trained on a primitive reads its own crank-angle observation
    # against a leg pointing somewhere else entirely.
    leg_phase: float = 0.0
    # Draw the printed parts on top of the collision primitives. Visual only:
    # the geoms carry no mass and no contact, so the physics is byte-identical
    # with and without it. For comparing the model against the real robot.
    visual_stl: bool = False
    # Spiral parameters, all measured from hardware/stl/leg_left.stl.
    hub_radius: float = 0.018
    spiral_r0: float = 0.019      # arm radius where it leaves the hub
    spiral_sweep: float = math.radians(120.0)
    spiral_thickness: float = 0.005   # half the arm's radial width
    leg_length: float = 0.050     # hip to foot (bar) / hip to tip (c_leg)
    leg_radius: float = 0.004
    leg_mass: float = 0.008
    foot_radius: float = 0.006
    foot_mass: float = 0.003
    c_leg_segments: int = 6
    c_leg_sweep: float = math.pi  #半円

    # --- hip placement -----------------------------------------------------
    hip_x_frac: float = 0.0       # 0 = mid-length, +1 = nose
    # +1 = top plate. The XL330 case (26 mm) is *taller* than the M5Stack
    # (16 mm), so mounting the servos at mid-height leaves them protruding
    # 5 mm below the belly -- the robot then rests on its servo cases and the
    # belly never reaches the floor. See `servo_clears_belly`.
    hip_z_frac: float = 1.0
    # Outboard gap between the torso side wall and the hip axis. Must be big
    # enough that the rotating leg clears the torso.
    hip_y_gap: float = 0.012

    # --- servo: Dynamixel XL330-M288-T -------------------------------------
    servo_mass: float = 0.018
    servo_size: tuple[float, float, float] = (0.020, 0.034, 0.026)
    servo_stall_torque: float = 0.52    # N.m at 5.0 V
    servo_no_load_speed: float = 10.79  # rad/s (103 rpm) at 5.0 V
    servo_gear_ratio: float = 288.4
    servo_rotor_inertia: float = 1.0e-8  # kg.m^2, estimated -- identify on HW
    servo_kp: float = 5.0               # N.m/rad, internal position loop
    # Viscous damping at the joint. This used to be derived as stall torque over
    # no-load speed, but that quantity describes the motor's *electrical*
    # behaviour, not mechanical drag, and measurement on the robot showed the
    # real viscous term is far smaller. The old value is kept as the default so
    # existing runs reproduce; identification overrides it.
    joint_damping: float = 0.0482       # = 0.52 / 10.79, the old derivation
    # Coulomb friction in the gearbox, N.m. Not a refinement: with torque off
    # the leg does not fall at all, so dry friction exceeds the 1.5e-3 N.m that
    # gravity applies to it. A viscous-only joint cannot reproduce that.
    joint_frictionloss: float = 0.0

    # --- contact -----------------------------------------------------------
    floor_friction: float = 1.0
    foot_friction: float = 1.2

    timestep: float = 0.002

    def with_(self, **kwargs) -> "RobotParams":
        return replace(self, **kwargs)

    def to_dict(self) -> dict:
        """Serialisable form, for recording alongside a training run.

        `wedge_dir` is left out on purpose: it points at repository data, not at
        a property of the robot, and writing an absolute path into a run would
        break the moment the checkout moved.
        """
        d = asdict(self)
        d.pop("wedge_dir", None)
        d["servo_size"] = list(self.servo_size)
        return d

    @staticmethod
    def from_dict(d: dict) -> "RobotParams":
        """Rebuild from `to_dict`, ignoring keys this version no longer has.

        Tolerating unknown keys matters: a run recorded by a later version must
        still load, with a warning, rather than crash a comparison.
        """
        fields = {f.name for f in dataclasses.fields(RobotParams)}
        unknown = set(d) - fields
        if unknown:
            print(f"warning: ignoring unknown robot parameters {sorted(unknown)}")
        kw = {k: v for k, v in d.items() if k in fields}
        if "servo_size" in kw:
            kw["servo_size"] = tuple(kw["servo_size"])
        return RobotParams(**kw)

    # --- derived -----------------------------------------------------------
    @property
    def torso_half(self) -> tuple[float, float, float]:
        return (self.torso_len / 2, self.torso_width / 2, self.torso_height / 2)

    @staticmethod
    def asbuilt(**kwargs) -> "RobotParams":
        """The robot that was actually printed, measured off hardware/stl.

        Dimensions are read from the CAD assembly: the enclosure is 80 (fore /
        aft) x 80 (lateral) x 72 (vertical) mm, and the hinge sits 1.5 mm above
        its centre -- `hip_z_frac=0.04`, nowhere near the top plate the default
        assumes.

        Masses are weighed: 303.9 g assembled, 9 g per leg. The legs came in at
        less than half the 20 g that hardware/goron.urdf was first built with --
        the printed parts are sparse-infill, so mesh volume times material
        density overestimates them badly. That matters more than the 2 percent
        error in the total, because leg inertia dominates the swing.

        Only the total and the legs were weighed. The split of the remaining
        249.9 g between board, battery and printed shell is not measured, so it
        is carried whole in `board_mass`; hardware/goron.urdf distributes it
        over the parts by volume to get the centre of mass.
        """
        defaults = dict(
            torso_len=0.080, torso_width=0.080, torso_height=0.072,
            hip_z_frac=0.04, leg_shape="mesh",
            board_mass=0.2499, extra_mass=0.0,    # + 2 servos = 285.9 g
            leg_mass=0.006, foot_mass=0.003,      # 9 g per leg
            # Identified by matching simulated step responses against ten
            # measured ones (5 to 120 degrees, both directions) on the real
            # servo. Trajectory error fell from 17.0% to 2.5%; on the other
            # leg's step, which was held out of the fit, from 3.74 to 0.32 deg
            # RMS. The joint is much softer and heavier than the spec-derived
            # guesses, and it has dry friction the model did not have at all.
            servo_kp=0.673,                # was 5.0 -- 7.4x too stiff
            servo_rotor_inertia=2.256e-8,  # was 1.0e-8
            joint_damping=0.0554,          # was 0.0482, barely moved
            joint_frictionloss=0.00877,    # was absent
            # Measured by shoving the robot along the floor it runs on and
            # reading the tilt of the specific force while it slid: 0.302 over
            # three trials that agreed to 0.01. The old 1.0 was a placeholder.
            floor_friction=0.302,
            # Not measured -- the robot cannot stand on two legs to be shoved.
            # Set equal to the belly because both surfaces are the same printed
            # PLA on the same floor, and dry friction depends on the material
            # pair rather than on contact area. It is worth distrusting: with
            # the floor at 0.302, moving this from 0.30 to 0.50 cost the crawl
            # policy 4x its reward, so it is the first thing to re-check if the
            # real robot crawls worse than the simulation says. To measure it,
            # tilt any board and take the ratio of leg-on-board to belly-on-
            # board, then scale the belly's measured 0.302 by it.
            foot_friction=0.302,
        )
        return RobotParams(**{**defaults, **kwargs})  # caller wins

    @property
    def stl_dir(self) -> Path:
        """The printed parts, in the CAD assembly frame (mm)."""
        return self.wedge_dir.parent

    @property
    def hip_y(self) -> float:
        """Lateral distance from the centre plane to a hip axis."""
        return self.torso_width / 2 + self.hip_y_gap

    @property
    def total_mass(self) -> float:
        return (self.board_mass + self.extra_mass
                + 2 * (self.servo_mass + self.leg_mass + self.foot_mass))

    @property
    def joint_armature(self) -> float:
        """Rotor inertia reflected through the gearbox: ratio^2 * I_rotor."""
        return self.servo_gear_ratio ** 2 * self.servo_rotor_inertia

    @property
    def flip_half_extent(self) -> float:
        """Half-extent of the torso along the axis it must tip over.

        Sagittal legs somersault about Y and pivot on the nose/tail edges;
        lateral legs roll about X and pivot on the side edges.
        """
        return (self.torso_len if self.swing == "sagittal" else self.torso_width) / 2

    @property
    def leg_clears_torso(self) -> bool:
        """A leg can only rotate 360 degrees if it passes outboard of the torso."""
        return self.hip_y > self.torso_width / 2 + self.leg_radius

    @property
    def servo_bottom(self) -> float:
        """Height of the lowest point of a servo case, relative to torso centre."""
        return self.hip_z_frac * self.torso_height / 2 - self.servo_size[2] / 2

    @property
    def servo_clears_belly(self) -> bool:
        """The belly must be the lowest surface, or the robot rests on its servos
        and can never satisfy a belly-contact goal."""
        return self.servo_bottom >= -self.torso_height / 2


def _leg_geoms(p: RobotParams, side_sign: int) -> str:
    """Leg geometry at joint angle 0: pointing straight down (-Z).

    Sagittal legs sweep in the XZ plane; lateral legs in the YZ plane. `u` is
    the in-plane unit direction that the leg curls towards as it rotates.
    """
    sagittal = p.swing == "sagittal"

    def vec(along: float, down: float) -> tuple[float, float, float]:
        """Build a point `along` the swing direction and `down` from the hip."""
        if sagittal:
            return (along, 0.0, -down)
        return (0.0, side_sign * along, -down)

    if p.leg_shape == "bar":
        tip = vec(0.0, p.leg_length)
        return f"""        <geom name="thigh_{'left' if side_sign > 0 else 'right'}"
              type="capsule" mass="{p.leg_mass}"
              fromto="0 0 0 {tip[0]:.5f} {tip[1]:.5f} {tip[2]:.5f}"
              size="{p.leg_radius}" rgba="0.25 0.55 0.85 1"/>
        <geom name="foot_{'left' if side_sign > 0 else 'right'}"
              type="sphere" mass="{p.foot_mass}"
              pos="{tip[0]:.5f} {tip[1]:.5f} {tip[2]:.5f}" size="{p.foot_radius}"
              friction="{p.foot_friction} 0.02 0.002" rgba="0.95 0.45 0.15 1"/>
        <site name="foot_site_{'left' if side_sign > 0 else 'right'}"
              pos="{tip[0]:.5f} {tip[1]:.5f} {tip[2]:.5f}" size="0.002"/>"""

    side = "left" if side_sign > 0 else "right"

    if p.leg_shape == "mesh":
        # The printed leg, cut into convex pieces. The meshes are already in the
        # leg-body frame (see scripts/split_leg.py); the right leg reuses them
        # because the printed legs share one profile in the swing plane and
        # differ only in which side they bolt to.
        pieces = sorted(p.wedge_dir.glob("*.stl"))
        if not pieces:
            raise FileNotFoundError(
                f"no wedge meshes in {p.wedge_dir}. Run: "
                f"uv run python -m scripts.split_leg")
        each = (p.leg_mass + p.foot_mass) / len(pieces)
        parts = []
        spin = (f'euler="0 {p.leg_phase:.5f} 0"' if sagittal
                else f'euler="{p.leg_phase:.5f} 0 0"')
        for f in pieces:
            parts.append(
                f"""        <geom name="{f.stem}_{side}" type="mesh" mesh="{f.stem}"
              mass="{each}" {spin} friction="{p.foot_friction} 0.02 0.002"
              rgba="0.25 0.55 0.85 1"/>"""
            )
        parts.append(
            f"""        <site name="foot_site_{side}" pos="0 0 {-p.leg_length:.5f}"
              size="0.002"/>"""
        )
        return "\n".join(parts)

    if p.leg_shape == "spiral":
        # The printed leg: a hub disc plus an arm whose radius grows linearly
        # with angle. Built as a capsule chain rather than as the mesh, because
        # a mesh collides as its convex hull -- which for this shape is a solid
        # disc, exactly the property the spiral is meant not to have.
        n = p.c_leg_segments
        parts = [
            f"""        <geom name="hub_{side}" type="cylinder" mass="{p.leg_mass / 3}"
              fromto="{-p.leg_radius if sagittal else 0} 0 0
                      {p.leg_radius if sagittal else 0} 0 0"
              size="{p.hub_radius}" rgba="0.20 0.22 0.26 1"/>"""
        ]

        def spiral_point(t: float) -> tuple[float, float, float]:
            """t in [0, 1] along the arm; angle 0 points straight down."""
            phi = t * p.spiral_sweep
            r = p.spiral_r0 + t * (p.leg_length - p.spiral_r0)
            return vec(r * math.sin(phi), r * math.cos(phi))

        seg_mass = p.leg_mass * 2 / 3 / n
        for i in range(n):
            a, b = spiral_point(i / n), spiral_point((i + 1) / n)
            parts.append(
                f"""        <geom name="arm{i}_{side}" type="capsule" mass="{seg_mass}"
              fromto="{a[0]:.5f} {a[1]:.5f} {a[2]:.5f} {b[0]:.5f} {b[1]:.5f} {b[2]:.5f}"
              size="{p.spiral_thickness}" friction="{p.foot_friction} 0.02 0.002"
              rgba="0.25 0.55 0.85 1"/>"""
            )
        tip = spiral_point(1.0)
        parts.append(
            f"""        <site name="foot_site_{side}" pos="{tip[0]:.5f} {tip[1]:.5f} {tip[2]:.5f}"
              size="0.002"/>
        <geom name="foot_{side}" type="sphere" mass="{p.foot_mass}"
              pos="{tip[0]:.5f} {tip[1]:.5f} {tip[2]:.5f}"
              size="{p.spiral_thickness}"
              friction="{p.foot_friction} 0.02 0.002" rgba="0.95 0.45 0.15 1"/>"""
        )
        return "\n".join(parts)

    # C-leg: an arc of radius r that starts at the hip and curls round, so that
    # the ground contact point moves smoothly as the crank turns.
    r = p.leg_length / (2 * math.sin(p.c_leg_sweep / 2)) if p.c_leg_sweep else 0.0
    n = p.c_leg_segments
    seg_mass = p.leg_mass / n

    def arc_point(t: float) -> tuple[float, float, float]:
        phi = t * p.c_leg_sweep
        return vec(r * (1 - math.cos(phi)), r * math.sin(phi))

    parts = []
    for i in range(n):
        a, b = arc_point(i / n), arc_point((i + 1) / n)
        parts.append(
            f"""        <geom name="cleg{i}_{side}" type="capsule" mass="{seg_mass}"
              fromto="{a[0]:.5f} {a[1]:.5f} {a[2]:.5f} {b[0]:.5f} {b[1]:.5f} {b[2]:.5f}"
              size="{p.leg_radius}" friction="{p.foot_friction} 0.02 0.002"
              rgba="0.25 0.55 0.85 1"/>"""
        )
    tip = arc_point(1.0)
    parts.append(
        f"""        <geom name="foot_{side}" type="sphere" mass="{p.foot_mass}"
              pos="{tip[0]:.5f} {tip[1]:.5f} {tip[2]:.5f}" size="{p.foot_radius}"
              friction="{p.foot_friction} 0.02 0.002" rgba="0.95 0.45 0.15 1"/>
        <site name="foot_site_{side}" pos="{tip[0]:.5f} {tip[1]:.5f} {tip[2]:.5f}"
              size="0.002"/>"""
    )
    return "\n".join(parts)


# Printed parts drawn as visual overlay. Torso parts ride the body; leg parts
# turn with the crank (matching how hardware/goron.urdf splits them).
VISUAL_TORSO = ("body_bottom", "body_upper", "m5stack", "bat",
                "servo_left", "servo_right")
VISUAL_LEG = ("leg_left", "servo_joint_left", "leg_right", "servo_joint_right")

# CAD assembly frame -> simulation frame. The same mapping scripts/split_leg.py
# applies to the wedges:  sim_x = CAD_z,  sim_y = CAD_x,  sim_z = CAD_y.
# That is the cyclic permutation x->y->z->x, i.e. a 120 degree turn about
# (1,1,1) -- a rotation, not a mirror, so rotation senses are preserved.
CAD_TO_SIM_QUAT = "0.5 0.5 0.5 0.5"
HINGE_Y_CAD = 0.035   # the hinge sits 35 mm up the CAD frame
PLATE_X_CAD = 0.052   # and 52 mm out along it


def _visual_geoms(names: tuple[str, ...], pos: str) -> str:
    """Visual-only geoms: no contact, no mass, no effect on the physics."""
    return "\n".join(
        f"""        <geom name="{n}_vis" type="mesh" mesh="{n}" mass="0"
              contype="0" conaffinity="0" group="2"
              pos="{pos}" quat="{CAD_TO_SIM_QUAT}" rgba="0.75 0.78 0.82 1"/>"""
        for n in names
    )


def _mesh_assets(p: RobotParams) -> str:
    """Declare the wedge meshes, with absolute paths so that a model built as a
    string (rather than loaded from a file) still resolves them."""
    out = []
    # The wedges were written in metres by scripts/split_leg.py; the printed
    # parts are raw CAD exports and are still in millimetres.
    if p.leg_shape == "mesh":
        out += [f'    <mesh name="{f.stem}" file="{f.resolve()}"/>'
                for f in sorted(p.wedge_dir.glob("*.stl"))]
    if p.visual_stl:
        out += [f'    <mesh name="{n}" file="{(p.stl_dir / f"{n}.stl").resolve()}"'
                f' scale="0.001 0.001 0.001"/>'
                for n in VISUAL_TORSO + VISUAL_LEG]
    return "\n".join(out)


def build_mjcf(p: RobotParams = RobotParams()) -> str:
    hx, hy, hz = p.torso_half
    sx, sy, sz = p.servo_size
    axis = "0 1 0" if p.swing == "sagittal" else "1 0 0"
    spawn_z = max(hx, hy, hz) + p.leg_length + p.foot_radius + 0.02
    # Anchor the printed body on the hinge, which is the one point the model and
    # the CAD must agree on. Any disagreement then shows as a visible offset.
    torso_visual = _visual_geoms(
        VISUAL_TORSO, f"0 0 {p.hip_z_frac * hz - HINGE_Y_CAD:.5f}"
    ) if p.visual_stl else ""

    def leg(side: str, s: int) -> str:
        return f"""
      <!-- servo case, carried by the torso -->
      <geom name="servo_{side}" type="box" mass="{p.servo_mass}"
            pos="{p.hip_x_frac * hx:.5f} {s * p.hip_y:.5f} {p.hip_z_frac * hz:.5f}"
            size="{sx / 2:.5f} {sy / 2:.5f} {sz / 2:.5f}" rgba="0.2 0.2 0.22 1"/>
      <body name="leg_{side}"
            pos="{p.hip_x_frac * hx:.5f} {s * p.hip_y:.5f} {p.hip_z_frac * hz:.5f}">
        <!-- no range attribute: the crank turns through a full 360 degrees -->
        <joint name="hip_{side}" type="hinge" axis="{axis}" limited="false"
               damping="{p.joint_damping:.6f}" armature="{p.joint_armature:.6e}"
               frictionloss="{p.joint_frictionloss:.6f}"/>
{_leg_geoms(p, s)}
{_visual_geoms(
    (f"leg_{side}", f"servo_joint_{side}"),
    f"0 {s * PLATE_X_CAD:.5f} {-HINGE_Y_CAD:.5f}") if p.visual_stl else ""}
      </body>"""

    # Extended Position Control Mode is +/-256 revolutions.
    ctrl_limit = 256 * 2 * math.pi

    return f"""<mujoco model="goron_{p.swing}_{p.leg_shape}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{p.timestep}" integrator="implicitfast" cone="elliptic"/>

  <default>
    <geom condim="4" friction="{p.floor_friction} 0.01 0.001" solref="0.005 1"
          solimp="0.9 0.95 0.001"/>
    <position kp="{p.servo_kp}" kv="0"
              forcerange="{-p.servo_stall_torque} {p.servo_stall_torque}"
              ctrlrange="{-ctrl_limit:.3f} {ctrl_limit:.3f}"/>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.22 0.24 0.27"
             rgb2="0.30 0.32 0.36" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.05"/>
{_mesh_assets(p)}
  </asset>

  <worldbody>
    <light pos="0 0 0.6" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="5 5 0.05" material="grid"
          friction="{p.floor_friction} 0.01 0.001"/>

    <body name="torso" pos="0 0 {spawn_z:.5f}">
      <freejoint name="root"/>
      <geom name="torso" type="box" size="{hx:.5f} {hy:.5f} {hz:.5f}"
            mass="{p.board_mass + p.extra_mass}" rgba="0.85 0.85 0.88 1"/>
      <site name="imu" pos="0 0 0" size="0.002" rgba="1 0 0 1"/>
      <site name="nose" pos="{hx:.5f} 0 0" size="0.002" rgba="0 1 1 1"/>
{torso_visual}
{leg("left", +1)}
{leg("right", -1)}
    </body>
  </worldbody>

  <actuator>
    <position name="hip_left"  joint="hip_left"/>
    <position name="hip_right" joint="hip_right"/>
  </actuator>

  <sensor>
    <!-- Everything the real robot delivers: M5Stack 6-axis IMU, plus the
         XL330's absolute encoder and velocity feedback over TTL. -->
    <framezaxis name="torso_up" objtype="site" objname="imu"/>
    <gyro name="gyro" site="imu"/>
    <accelerometer name="accel" site="imu"/>
    <jointpos name="hip_left_pos" joint="hip_left"/>
    <jointpos name="hip_right_pos" joint="hip_right"/>
    <jointvel name="hip_left_vel" joint="hip_left"/>
    <jointvel name="hip_right_vel" joint="hip_right"/>
    <framepos name="torso_pos" objtype="site" objname="imu"/>
  </sensor>
</mujoco>
"""


def save_mjcf(p: RobotParams = RobotParams(), path: Path | None = None) -> Path:
    """Write the MJCF out for viewing (`python -m mujoco.viewer --mjcf=...`).

    Nothing reads these files back: `GoronEnv` always builds its model in memory
    from `RobotParams`, so that geometry can be swept. `models/` is gitignored
    for that reason -- editing a written-out file has no effect on training.
    """
    path = path or (MODELS_DIR / f"goron_{p.swing}_{p.leg_shape}.xml")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_mjcf(p))
    return path


if __name__ == "__main__":
    for shape in ("bar", "c_leg"):
        p = RobotParams(leg_shape=shape)
        out = save_mjcf(p)
        print(f"{out.name}")
        print(f"  total mass      {p.total_mass * 1000:6.1f} g")
        print(f"  stall torque    {p.servo_stall_torque:6.2f} N.m x2")
        print(f"  no-load speed   {p.servo_no_load_speed:6.2f} rad/s "
              f"({p.servo_no_load_speed * 60 / (2 * math.pi):.0f} rpm)")
        print(f"  joint damping   {p.joint_damping:8.4f} N.m.s/rad")
        print(f"  armature        {p.joint_armature:8.2e} kg.m^2 "
              f"(leg alone ~ {p.leg_mass * p.leg_length ** 2:.1e})")
        print(f"  leg clears torso: {p.leg_clears_torso}")
