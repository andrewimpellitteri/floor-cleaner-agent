"""Physical configuration for the wash-bay floor simulation.

Everything in here is in SI units and is meant to correspond to a real
measurable quantity. If a number is a guess, it says so and names the knob you
would turn to calibrate it against a stopwatch on the real floor.

Array/index convention used everywhere in this package:

    field[i, j]  <->  x = i * dx,  y = j * dx

`x` runs ALONG the trough (the long axis of the bay), `y` runs ACROSS it. The
trough sits at y = Ly/2 and the floor falls toward it from both sides, which is
the "sloped in two directions" geometry. The trough itself falls along +x
toward the drain end.

Renderers transpose for display so that x is horizontal; no other module is
allowed to be clever about axes.
"""

from __future__ import annotations

import dataclasses
import math

# --- Unit helpers -----------------------------------------------------------

PSI = 6894.757  # Pa per psi
GPM = 3.785411784e-3 / 60.0  # m^3/s per US gallon/minute
FT = 0.3048  # m per foot
INCH = 0.0254  # m per inch


@dataclasses.dataclass(frozen=True)
class FloorConfig:
    """Geometry of the wash bay section being cleaned.

    The real bay is symmetric about its central trough: the slab falls to the
    trough from both sides, roughly 20-25 ft of run on each.

    The FULL width is modelled, not half of it, even though that doubles the
    grid. Cutting the domain at the trough would put a wall there, making it
    impossible to push slurry past the trough and up the far slope -- which is
    a failure mode that actually happens, and one of the things the policy has
    to learn to avoid. Overshoot needs no penalty term in the reward: grit that
    lands on the far side simply has 23 ft to travel back, and the time cost
    takes care of itself.
    """

    length_x: float = 4.2  # m, along the trough (a worked section)
    length_y: float = 14.0  # m, wall to wall (~23 ft of run on each side)
    dx: float = 0.07  # m per cell -> 60 x 200 grid

    # y position of the trough: the middle of the bay.
    trough_y: float = 7.0

    # Cross slope: fall from the wall down to the trough. 1.5% over 7 m is
    # about 4 inches of fall, which is a normal wash-bay pitch.
    cross_slope: float = 0.015  # m fall per m run

    # The trough itself falls along +x toward the drain.
    trough_slope: float = 0.005
    trough_width: float = 0.20  # m
    trough_depth: float = 0.04  # m below the surrounding slab

    # Manning's n for shallow flow. A grit-broadcast epoxy hangar floor sits
    # between smooth epoxy (~0.011) and broom-finished concrete (~0.030): the
    # anti-slip aggregate is the roughness that matters at film depths.
    manning_n: float = 0.018

    # Depth below which the film is treated as immobile (surface tension and
    # roughness hold it in the texture). Prevents divide-by-zero and gives the
    # physically right behaviour: a thin film does not transport grit.
    h_min: float = 2.0e-4  # m (0.2 mm)

    # The slab is not a clean plane. Two years past its recoat it has settled
    # and worn into random flat spots that hold standing water -- which matter
    # twice over: water is what carries slurry, so a puddle is a help, but a
    # flat spot has no slope to drain it, so it is also where dirt sits down
    # and stays. Amplitude is comparable to the cross-slope fall over a metre,
    # which is what it takes to actually defeat the drainage.
    flat_spot_amplitude: float = 0.012  # m
    flat_spot_length: float = 1.2  # m, correlation length of the undulation

    # The floor is worked wet: hoses are running for rinse and the bay rarely
    # dries out mid-job. Starting dry would starve transport and badly
    # misrepresent the task -- how far a push carries is set by u*h, so the
    # water already on the floor is a first-order term, not a detail.
    initial_film_mean: float = 2.5e-3  # m
    initial_film_std: float = 1.2e-3
    max_ponding: float = 0.020  # m, depth a flat spot can hold

    @property
    def nx(self) -> int:
        return int(round(self.length_x / self.dx))

    @property
    def ny(self) -> int:
        return int(round(self.length_y / self.dx))

    @property
    def cell_area(self) -> float:
        return self.dx * self.dx


@dataclasses.dataclass(frozen=True)
class WasherConfig:
    """1200 psi soft-wash unit with a wide fan tip."""

    pressure: float = 1200.0 * PSI  # Pa
    flow: float = 4.0 * GPM  # m^3/s
    discharge_coeff: float = 0.95

    # 25 deg green tip: the usual choice for floor work. Wider tips rinse but
    # spread the momentum too thin to lift anything.
    fan_angle: float = math.radians(25.0)

    # The fan is a thin sheet leaving the orifice, and it thickens with
    # distance through turbulent entrainment of air. These two set how fast the
    # jet loses punch as you lift the wand -- the single most important pair of
    # numbers in the model, and the first thing to calibrate.
    fan_thickness_0: float = 3.0e-3  # m at the tip
    fan_spread_angle: float = math.radians(2.0)  # full angle of thickness growth

    # Air drag bleeds jet velocity over distance. e-folding length.
    velocity_decay_length: float = 1.6  # m

    # Operator limits.
    standoff_min: float = 0.05  # m -- tip almost on the slab
    standoff_max: float = 1.10  # m -- arm fully extended
    tilt_max: float = math.radians(80.0)  # from vertical; 80 deg is nearly flat
    walk_speed: float = 1.0  # m/s, operator translation
    azimuth_rate: float = 3.0  # rad/s, how fast the wand can be swung
    standoff_rate: float = 1.2  # m/s, how fast the tip can be raised/lowered
    tilt_rate: float = 3.0  # rad/s

    @property
    def jet_velocity(self) -> float:
        """Discharge velocity at the orifice, from Bernoulli."""
        return self.discharge_coeff * math.sqrt(2.0 * self.pressure / 1000.0)

    @property
    def momentum_flux(self) -> float:
        """Thrust of the jet, rho * Q * V. Also the kickback the operator feels."""
        return 1000.0 * self.flow * self.jet_velocity


@dataclasses.dataclass(frozen=True)
class DirtConfig:
    """Three-phase sediment, following Hairsine-Rose.

        bound      grit stuck to the epoxy. Held by ADHESION, so it takes normal
                   stress to break loose: only jet impingement will do it, at
                   kilopascals. Flowing water never shifts it.
        deposited  grit that has been knocked loose and has settled again. It is
                   lying ON the floor, not stuck to it, so it moves under
                   ordinary tangential stress -- pascals, not kilopascals.
        suspended  grit currently carried by the film.

    Keeping `bound` and `deposited` separate is essential, not a refinement.
    With a single threshold, loosened slurry would be as hard to move as virgin
    adhered grit, and the jet would have to physically strike every square inch
    of the bay -- which is neither how the model should behave nor how the job
    is actually done. Splitting them reproduces the real two-part technique:
    stand the wand up close to break grit loose, then lay it over to sweep the
    loosened slurry to the trough.
    """

    # Areal loading of dirt at the start, kg/m^2.
    #
    # This is a thin film of dirt and mould, not bulk grit. Published dust
    # loadings for paved surfaces run 1-10 g/m^2 in decent condition and up to
    # ~50 g/m^2 badly soiled; a ~20 um film at soil density is ~30 g/m^2. So
    # 20 g/m^2 is a dirty-but-routine floor: about 1.2 kg (2.6 lb) over the
    # whole 59 m^2 section, which is a few pounds of dirt off 630 sq ft.
    #
    # Earlier values of 0.30 and then 0.10 kg/m^2 were 40 lb and 13 lb on that
    # same floor -- both far outside anything a working bay carries, and the
    # main reason the simulated job ran several times too long.
    #
    # Note what this does to the character of the problem. At a realistic
    # loading a single pass strips whatever it touches, so job time stops being
    # set by how fast the jet cuts and becomes set by how fast you can physically
    # COVER the floor. That is the right regime: it is why swath width (and so
    # standoff) dominates, and it lands the job near the observed 15 minutes
    # without tuning a rate constant to get there.
    load_mean: float = 0.020
    load_std: float = 0.009

    # Yield stress holding grit to the floor, Pa. The bay is a grit-broadcast
    # epoxy (hangar-type) floor, so the anti-slip aggregate gives it real
    # texture to hold dirt in -- less than a broom finish, far more than smooth
    # epoxy.
    yield_mean: float = 2.5e3
    yield_std: float = 1.0e3
    yield_min: float = 0.8e3

    # The coating is two years overdue for a recoat, so it is worn through in
    # the traffic lanes where pieces get dragged. Grit ground into worn epoxy
    # takes substantially more to shift, and the lanes run along the bay. This
    # patchiness is what makes "thorough" harder than "fast".
    worn_lane_boost: float = 3.0e3  # Pa added at the centre of a worn lane
    worn_lane_length_x: float = 6.0  # m, lanes are long in x ...
    worn_lane_length_y: float = 0.8  # m, ... and narrow across the bay

    # Rate at which jet impingement tears ADHERED grit loose: kg per m^2 per
    # second per Pa of excess normal stress.
    #
    # Calibrated against the thing you can actually check by eye: ONE steady
    # pass at a normal working height and a normal walking pace should take
    # most of the grit off, and the stubborn worn lanes should need a second
    # pass or a slower one. Note how short the dwell really is -- the impact
    # patch is ~1.5 cm across the direction of travel, so at 0.3 m/s any given
    # spot is under the jet for only about 50 ms, and the rate has to be high
    # enough to do the work in that time. `scripts/calibrate.py` prints the
    # single-pass removal fraction so this stays checkable rather than assumed.
    #
    # Set so that the whole 59 m^2 section comes clean in roughly the 15 minutes
    # Andrew reports for one person on a floor that is not unusually dirty. That
    # is the one end-to-end number available to calibrate against, so it anchors
    # this constant rather than the other way round.
    entrainment_rate: float = 2.5e-3

    # The loose DEPOSITED layer instead responds to tangential stress -- the
    # film's own bed shear plus whatever the jet is pushing with. Threshold is a
    # Shields-type criterion for grit already lying free on a smooth floor, so
    # it is three orders of magnitude below the adhesion yield stress.
    deposit_threshold: float = 0.5  # Pa
    deposit_entrainment_rate: float = 1.0e-2  # kg/m^2/s per Pa of excess

    # Settling velocity of the suspended grit, m/s.
    #
    # This single number sets the TRANSPORT LENGTH -- how far a slurry travels
    # before it drops out -- via  L ~ u * h / v_settle. Wash-bay dirt is mostly
    # silt and fine sand, a few mm/s, giving roughly a metre of travel in a
    # 2 mm film at 1 m/s. Coarse-sand values (10+ mm/s) strand the slurry within
    # a foot of the jet, which is neither realistic nor navigable.
    #
    # Note what the formula says: transport length is proportional to film
    # depth. More water on the floor carries dirt further per push. Andrew
    # confirmed exactly this -- how far a push carries "depends on standing
    # water" -- so this relationship must survive any recalibration.
    settling_velocity: float = 0.003

    # Cleanliness threshold: areal loading below this counts as clean, kg/m^2.
    # Must scale WITH `load_mean` -- it is "looks clean", roughly a tenth of the
    # starting film. At the old 0.02 value against a 0.020 kg/m^2 load the floor
    # would begin the episode already clean by definition.
    clean_threshold: float = 0.002


@dataclasses.dataclass(frozen=True)
class SimConfig:
    """Time-stepping and episode structure."""

    control_dt: float = 0.20  # s between agent decisions (5 Hz, human-ish)
    # 16 substeps puts the advective CFL limit at ~1.3 m/s per face-pair
    # (cfl * dx / dt / 2 for the two dimensions -- see physics.py), which is
    # fast enough for gravity-driven sheet flow and most of a bow wave; a
    # faster jet-driven surge is clipped rather than allowed to go unstable,
    # and the clip is positivity-preserving so it cannot manufacture water.
    # Retaining inertia in the flow solver is what makes the advective limit
    # the binding one; gravity waves in a millimetre film are far slower.
    physics_substeps: int = 16

    # An episode is a five-minute WINDOW of work, not a whole floor. Cleaning
    # 34 m^2 to completion takes 20-40 minutes, which is far too long a horizon
    # to assign credit over. Technique is local and repeatable, so the policy is
    # trained on five-minute windows started from every phase of the job (see
    # `env.reset`, which randomises how far along the floor already is) and is
    # then evaluated to completion in `scripts/benchmark.py`.
    max_steps: int = 1500  # 1500 * 0.20 s = 5 minutes

    # Explicit advection is CFL-limited. Velocities are clipped to keep the
    # substep stable regardless of what the jet does.
    cfl: float = 0.45

    @property
    def physics_dt(self) -> float:
        return self.control_dt / self.physics_substeps

    @property
    def episode_seconds(self) -> float:
        return self.max_steps * self.control_dt


@dataclasses.dataclass(frozen=True)
class Config:
    floor: FloorConfig = dataclasses.field(default_factory=FloorConfig)
    washer: WasherConfig = dataclasses.field(default_factory=WasherConfig)
    dirt: DirtConfig = dataclasses.field(default_factory=DirtConfig)
    sim: SimConfig = dataclasses.field(default_factory=SimConfig)

    gravity: float = 9.81
    water_density: float = 1000.0


DEFAULT = Config()
