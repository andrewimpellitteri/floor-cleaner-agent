"""Scripted cleaning strategies -- the methods people actually argue about.

The learned policy is only interesting next to these. Each one is a real
technique somebody defends in a wash bay, written so it can be run in exactly
the same simulator, under exactly the same clock, as the trained agent.

GEOMETRY. A pass runs ACROSS the bay, from the outside in toward the trough,
because that is the direction the slurry has to go. The fan is held broadside to
that direction, so one pass clears a strip about as wide as the fan, and lanes
step along the bay. Walking ALONG the fan's long axis instead would clear a
strip only as wide as the fan is thick -- a few centimetres -- which is the kind
of mistake that is invisible in a reward curve and obvious in a rendered video.

THE ARGUMENT WORTH SETTLING:

  FarToNear   Start at the wall and drive the whole load the full 23 ft to the
              trough in one march. Every gram is handled once. But the slurry
              wave grows as it goes, and whatever outruns its water strands and
              has to be picked up again.

  NearToFar   Clean the strip nearest the trough first, then work outward, each
              pass carrying only its own strip's load across ground that is
              already clean. Strictly MORE total push distance, but a lighter
              wave each time and a clean runway in front of it.

  BlastThenSweep  Treat it as two jobs. First pass wand-up and close to break
              the grit loose without trying to move it; second pass laid over
              to sweep the loosened slurry in. Costs a second pass over every
              square foot, and buys the right tool for each half of the work.

Which of these wins is a race between settling velocity and transport speed, and
it depends on how far the trough is. That is exactly the kind of question a
calibrated simulation can settle and a break room cannot.

`standoff_variants` holds everything else fixed and changes only how high the
wand is held, to isolate the single variable the operator most directly
controls.

Every baseline is a pure function pair (`init`, `act`) over a small array carry,
so they `jit` and `vmap` exactly like the neural policy and cost the same to
evaluate.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .env import CleaningEnv, EnvState


class SweepCarry(NamedTuple):
    lane_x: jnp.ndarray  # x of the strip currently being worked
    segment: jnp.ndarray  # which band out from the trough (NearToFar only)
    phase: jnp.ndarray  # 0 = repositioning to the start of a pass, 1 = pushing
    side: jnp.ndarray  # +1 working the far side of the trough, -1 the near side


def _wrap(a):
    return jnp.arctan2(jnp.sin(a), jnp.cos(a))


def _walk(env: CleaningEnv, state: EnvState, target_x, target_y, speed=1.0):
    """Proportional controller on operator position, saturating at walking pace."""
    step = env.cfg.washer.walk_speed * env.cfg.sim.control_dt
    return (
        jnp.clip((target_x - state.tip_x) / step, -1.0, 1.0) * speed,
        jnp.clip((target_y - state.tip_y) / step, -1.0, 1.0) * speed,
    )


def _aim(env: CleaningEnv, state: EnvState, target_az):
    rate = env.cfg.washer.azimuth_rate * env.cfg.sim.control_dt
    return jnp.clip(_wrap(target_az - state.azimuth) / rate, -1.0, 1.0)


def _standoff(env: CleaningEnv, target):
    wc = env.cfg.washer
    frac = (target - wc.standoff_min) / (wc.standoff_max - wc.standoff_min)
    return jnp.clip(frac * 2.0 - 1.0, -1.0, 1.0)


def _tilt(env: CleaningEnv, target):
    return jnp.clip(target / env.cfg.washer.tilt_max * 2.0 - 1.0, -1.0, 1.0)


def _start_side(env: CleaningEnv, state: EnvState):
    return jnp.where(state.tip_y >= env.cfg.floor.trough_y, 1.0, -1.0)


def _side_is_done(env: CleaningEnv, state: EnvState, side, factor: float):
    """Whether the half of the bay currently being worked is clean enough to leave.

    Judged on the FRACTION OF CELLS still dirty, not the mean loading. The mean
    drops below the cleanliness threshold long before the floor is actually
    clean -- the stubborn worn-lane patches are a small share of the area and
    barely move the average. Using the mean made every baseline declare both
    sides finished at around 2.3 kg remaining and then ping-pong across the
    trough doing nothing, which looked exactly like a physics plateau and was
    not one.
    """
    fc = env.cfg.floor
    residual = state.fields.bound + state.fields.deposited + state.fields.suspended
    on_side = jnp.where(side > 0, env.floor.y >= fc.trough_y, env.floor.y <= fc.trough_y)
    dirty = (residual > env.cfg.dirt.clean_threshold) & on_side
    return jnp.sum(dirty) < factor * 0.01 * jnp.sum(on_side)


@dataclasses.dataclass(frozen=True)
class PushSweep:
    """Lane-by-lane passes toward the trough.

    `near_to_far=False` starts every pass at the wall (the long march).
    `near_to_far=True` works outward in bands, nearest the trough first.
    """

    name: str = "far_to_near"
    near_to_far: bool = False
    # Lane spacing must match the swath the jet ACTUALLY cuts at this standoff
    # and speed, not the fan's nominal width. `scripts/calibrate.py` prints the
    # effective swath; stepping further than that leaves uncut stripes between
    # passes, which is invisible in aggregate mass and obvious in a render.
    lane_width: float = 0.18  # m stepped along the bay between passes
    segment: float = 1.75  # m of reach per band, when working outward
    standoff: float = 0.40
    # The compromise angle. It must be upright enough to actually CUT: at a
    # 40 cm standoff, 0.55 rad (32 deg) delivers ~4x adhesion while still
    # putting real tangential momentum into the slurry. An earlier default of
    # 0.95 rad (54 deg) pushed well but delivered only 1.1x adhesion, so it
    # swept the loose layer to the trough and left the adhered grit untouched
    # -- the floor stopped getting cleaner after about two minutes.
    tilt: float = 0.55
    push_speed: float = 0.45  # you push slower than you walk

    # A side counts as finished when its mean residual falls to this multiple
    # of the cleanliness threshold; the operator then crosses to the other side.
    # Without this the run can only ever clean half the bay.
    side_done_factor: float = 2.0

    def init(self, env: CleaningEnv, state: EnvState) -> SweepCarry:
        return SweepCarry(
            lane_x=jnp.array(0.0),
            segment=jnp.array(0.0),
            phase=jnp.array(0.0),
            side=_start_side(env, state),
        )

    def _pass_start_y(self, env: CleaningEnv, carry: SweepCarry):
        """Where a pass begins: the wall, or the outer edge of the current band."""
        fc = env.cfg.floor
        wall = jnp.where(carry.side > 0, fc.length_y, 0.0)
        band = fc.trough_y + carry.side * self.segment * (carry.segment + 1.0)
        start = jnp.where(self.near_to_far, band, wall)
        return jnp.clip(start, jnp.minimum(wall, fc.trough_y), jnp.maximum(wall, fc.trough_y))

    def act(self, env: CleaningEnv, state: EnvState, carry: SweepCarry):
        fc = env.cfg.floor
        start_y = self._pass_start_y(env, carry)
        # Push all the way in: the slurry has to actually reach the trough.
        end_y = fc.trough_y

        at_start = jnp.abs(state.tip_y - start_y) < 0.12
        reached_trough = jnp.abs(state.tip_y - end_y) < 0.12

        pushing = carry.phase > 0.5
        # Repositioning finishes when we arrive; a push finishes at the trough.
        phase = jnp.where(
            pushing, jnp.where(reached_trough, 0.0, 1.0), jnp.where(at_start, 1.0, 0.0)
        )

        # Finishing a push advances the lane; running off the end of the bay
        # starts the next band (or repeats, for the long march).
        finished = pushing & reached_trough
        lane_x = jnp.where(finished, carry.lane_x + self.lane_width, carry.lane_x)
        wrapped = lane_x > fc.length_x
        lane_x = jnp.where(wrapped, 0.0, lane_x)
        segment = jnp.where(wrapped & finished, carry.segment + 1.0, carry.segment)
        # Cap the band index so it cannot walk off past the wall.
        segment = jnp.minimum(segment, jnp.floor(fc.length_y * 0.5 / self.segment))

        # Cross to the other half of the bay once this one is covered, otherwise
        # the run can only ever finish half the floor. A side never reaches
        # "clean" on one coverage -- the worn lanes need several -- so a
        # cleanliness test means the operator stays on one half of the bay
        # forever. Alternating after each full sweep is also what a person
        # actually does. Working outward (near_to_far), a "full sweep" means
        # all bands, not just the first: crossing earlier resets the band
        # index and the outer floor never gets worked.
        if self.near_to_far:
            n_bands = jnp.maximum(
                jnp.ceil(fc.length_y * 0.5 / self.segment), 1.0)
            side_covered = carry.segment >= n_bands - 1.0
        else:
            side_covered = True
        cross = wrapped & finished & side_covered
        side = jnp.where(cross, -carry.side, carry.side)
        lane_x = jnp.where(cross, 0.0, lane_x)
        segment = jnp.where(cross, 0.0, segment)
        carry = carry._replace(side=side)
        start_y = self._pass_start_y(env, carry)

        target_y = jnp.where(phase > 0.5, end_y, start_y)
        speed = jnp.where(phase > 0.5, self.push_speed, 1.0)
        ax, ay = _walk(env, state, lane_x, target_y, speed)

        # Always push down the slope, toward the trough.
        target_az = jnp.where(carry.side > 0, -jnp.pi / 2, jnp.pi / 2)

        # Lift the wand while repositioning so the return trip does not undo
        # the pass just made.
        standoff = jnp.where(phase > 0.5, self.standoff, env.cfg.washer.standoff_max)

        action = jnp.stack([
            ax, ay,
            _aim(env, state, target_az),
            _standoff(env, standoff),
            _tilt(env, self.tilt),
        ])
        return SweepCarry(lane_x=lane_x, segment=segment, phase=phase, side=carry.side), action


@dataclasses.dataclass(frozen=True)
class BlastThenSweep(PushSweep):
    """Two passes per lane: one to break grit loose, one to sweep it in.

    The first pass is wand-up and close, which maximises normal stress and
    barely pushes at all. The second is laid over, which pushes hard and cuts
    nothing. The bet is that doing each job with the right tool beats doing both
    at once with a compromise angle.
    """

    name: str = "blast_then_sweep"
    # The blast standoff is chosen so the FAN WIDTH matches the lane spacing:
    # at 40 cm and near-upright, the 25 deg fan is ~18 cm across. Holding the
    # wand closer would hit harder but cut a strip narrower than the lane step,
    # leaving uncut stripes between passes -- which barely shows up in total
    # mass and is glaring in a render. There is pressure to spare here anyway
    # (~7x adhesion, ~3x even in the worn lanes), so width is the thing worth
    # buying with the standoff.
    blast_standoff: float = 0.40
    blast_tilt: float = 0.22  # nearly upright: all of the momentum into cutting
    blast_speed: float = 0.30  # slower: the cutting pass needs the dwell
    sweep_standoff: float = 0.35
    sweep_tilt: float = 1.25  # laid well over: all of it into pushing

    def act(self, env: CleaningEnv, state: EnvState, carry: SweepCarry):
        fc = env.cfg.floor
        start_y = self._pass_start_y(env, carry)
        end_y = fc.trough_y

        at_start = jnp.abs(state.tip_y - start_y) < 0.12
        reached_trough = jnp.abs(state.tip_y - end_y) < 0.12

        # Four phases, because BOTH working passes run inward and so the wand
        # has to be walked back out between them:
        #   0  reposition to the start of the lane   (wand lifted)
        #   1  BLAST inward: upright, cutting        -> ends at the trough
        #   2  return to the start                   (wand lifted)
        #   3  SWEEP inward: laid over, pushing      -> ends at the trough
        # An earlier three-phase version had the sweep target the trough while
        # waiting on `at_start` to leave that phase, so on arrival it swept at
        # the trough forever and never advanced a lane.
        phase = carry.phase
        repositioning = jnp.abs(phase - 0.0) < 0.5
        blasting = jnp.abs(phase - 1.0) < 0.5
        returning = jnp.abs(phase - 2.0) < 0.5
        sweeping = jnp.abs(phase - 3.0) < 0.5

        next_phase = jnp.where(
            repositioning, jnp.where(at_start, 1.0, 0.0),
            jnp.where(
                blasting, jnp.where(reached_trough, 2.0, 1.0),
                jnp.where(
                    returning, jnp.where(at_start, 3.0, 2.0),
                    jnp.where(reached_trough, 0.0, 3.0),
                ),
            ),
        )

        # A lane is done once its sweep has driven the slurry to the trough.
        finished = sweeping & reached_trough
        lane_x = jnp.where(finished, carry.lane_x + self.lane_width, carry.lane_x)
        wrapped = lane_x > fc.length_x
        lane_x = jnp.where(wrapped, 0.0, lane_x)
        segment = jnp.where(wrapped & finished, carry.segment + 1.0, carry.segment)
        segment = jnp.minimum(segment, jnp.floor(fc.length_y * 0.5 / self.segment))

        # Cross over once a full set of lanes has been covered, rather than
        # waiting for this side to come clean. A side never reaches "clean" on
        # one coverage -- the worn lanes need several -- so a cleanliness test
        # means the operator stays on one half of the bay forever and can only
        # ever finish 50% of the floor. Alternating after each full sweep is
        # also what a person actually does.
        cross = wrapped & finished
        side = jnp.where(cross, -carry.side, carry.side)
        lane_x = jnp.where(cross, 0.0, lane_x)
        segment = jnp.where(cross, 0.0, segment)
        next_phase = jnp.where(cross, 0.0, next_phase)
        carry = carry._replace(side=side)
        start_y = self._pass_start_y(env, carry)

        # Working passes head for the trough; the two travel phases head out.
        working = blasting | sweeping
        target_y = jnp.where(working, end_y, start_y)
        speed = jnp.where(blasting, self.blast_speed,
                          jnp.where(sweeping, self.push_speed, 1.0))
        ax, ay = _walk(env, state, lane_x, target_y, speed)

        target_az = jnp.where(side > 0, -jnp.pi / 2, jnp.pi / 2)
        standoff = jnp.where(blasting, self.blast_standoff,
                             jnp.where(sweeping, self.sweep_standoff,
                                       env.cfg.washer.standoff_max))
        tilt = jnp.where(blasting, self.blast_tilt, self.sweep_tilt)

        action = jnp.stack([
            ax, ay,
            _aim(env, state, target_az),
            _standoff(env, standoff),
            _tilt(env, tilt),
        ])
        return SweepCarry(lane_x=lane_x, segment=segment,
                          phase=next_phase, side=side), action


@dataclasses.dataclass(frozen=True)
class ChaseWorst:
    """Always attack the heaviest remaining patch on your side of the trough.

    The intuitive greedy strategy, and the one most people describe doing.
    It ignores transport entirely, which is exactly what makes it worth
    measuring rather than assuming.
    """

    name: str = "chase_worst"
    standoff: float = 0.25
    tilt: float = 0.9

    def init(self, env: CleaningEnv, state: EnvState) -> SweepCarry:
        return SweepCarry(lane_x=jnp.array(0.0), segment=jnp.array(0.0),
                          phase=jnp.array(0.0), side=_start_side(env, state))

    def act(self, env: CleaningEnv, state: EnvState, carry: SweepCarry):
        fc = env.cfg.floor
        residual = (state.fields.bound + state.fields.deposited
                    + state.fields.suspended)

        own_side = jnp.where(
            carry.side > 0, env.floor.y >= fc.trough_y, env.floor.y <= fc.trough_y
        )
        score = jnp.where(own_side, residual, -1.0)
        idx = jnp.argmax(score)
        tx = env.floor.x.ravel()[idx]
        ty = env.floor.y.ravel()[idx]

        ax, ay = _walk(env, state, tx, ty)
        target_az = jnp.where(carry.side > 0, -jnp.pi / 2, jnp.pi / 2)

        action = jnp.stack([
            ax, ay,
            _aim(env, state, target_az),
            _standoff(env, self.standoff),
            _tilt(env, self.tilt),
        ])
        return carry, action


@dataclasses.dataclass(frozen=True)
class RandomPolicy:
    """Uncorrelated random actions. The floor that any real method must beat."""

    name: str = "random"

    def init(self, env: CleaningEnv, state: EnvState) -> SweepCarry:
        return SweepCarry(lane_x=jnp.array(0.0), segment=jnp.array(0.0),
                          phase=jnp.array(0.0), side=jnp.array(1.0))

    def act(self, env: CleaningEnv, state: EnvState, carry: SweepCarry):
        key = jax.random.fold_in(state.key, state.step)
        return carry, jax.random.uniform(key, (env.action_dim,), minval=-1.0, maxval=1.0)


def standoff_variants(heights=(0.10, 0.20, 0.30, 0.50, 0.80)) -> list[PushSweep]:
    """The same far-to-near sweep at a range of wand heights.

    Everything else is held fixed, so the spread across these is a clean read on
    how much standoff alone is worth -- the question the operator has the most
    direct control over and the one the whole exercise started from.
    """
    return [PushSweep(name=f"standoff_{h:.2f}m", standoff=h) for h in heights]


def default_suite() -> list:
    """The strategies compared in the benchmark."""
    return [
        RandomPolicy(),
        ChaseWorst(),
        PushSweep(name="far_to_near", near_to_far=False),
        PushSweep(name="near_to_far", near_to_far=True),
        BlastThenSweep(),
        *standoff_variants(),
    ]
