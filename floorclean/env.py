"""The cleaning environment: a batched, fully-jittable JAX RL environment.

Design notes that matter:

REWARD. The only thing that truly counts is grit reaching the trough, but that
signal arrives twenty feet and many seconds after the action that caused it, so
it is hopeless on its own. The shaping term is a strict potential-based one,

    F = gamma * Phi(s') - Phi(s),      Phi = -(transport work still outstanding)

which Ng et al. (1999) prove leaves the optimal policy unchanged. So the agent
can be given dense feedback for loosening grit and for moving it closer to the
trough without any risk of it learning to farm the shaping term instead of
cleaning the floor. Every other reward component is a real cost: elapsed time,
and a bonus for finishing.

That guarantee only holds because the physics conserves mass exactly. If grit
could quietly evaporate through numerical diffusion -- as it did in the previous
version of this project -- the potential would fall for free and the agent would
be paid for nothing. `tests/test_physics.py` is what keeps that honest.

AN EPISODE IS A WHOLE JOB, NOT A WINDOW. It was a five-minute window while the
modelled floor carried 40 lb of grit and took 20-40 minutes to clean, where that
really was too long a horizon for credit assignment. Once the dirt loading was
corrected to a realistic 20 g/m^2 the job became ~15 minutes, and the window
turned into an active mistake: the objective is fast AND THOROUGH, thoroughness
means getting every cell under the threshold, and that requires covering the
whole floor -- which takes longer than the window did. So no policy could ever
finish, the finish bonus was unreachable, and the thoroughness term barely
moved. Measured over a five-minute window, a random policy and the best scripted
sweep were indistinguishable in return, despite the sweep removing 23% more
grit. `reset` still randomises how far along the job starts, so the policy sees
fresh floors, half-done floors, and floors down to the last stubborn worn-lane
patches -- but now it can actually finish them.

THE POTENTIAL HAS TWO TERMS, and the second is not optional. Bulk mass is
dominated by the easy 90%, which random flailing collects nearly as well as a
systematic sweep. What separates techniques is the fraction of the floor still
above the cleanliness threshold -- a saturating term that pays only for getting
cells CLEAN, never for skimming the heaviest patches. See `_potential`.

BEWARE OF RANKING POLICIES BY TOTAL RETURN. Shaping contributes
REWARD_SCALE*(gamma-1)*Phi every step (+0.54 on a fresh floor at Phi~-2.72,
measured 2026-09-19), which for a dirty floor (Phi < 0) is a positive drift far
larger than genuine progress (~0.12/step averaged over a 15-min job). The
critic learns it as a predictable offset (-REWARD_SCALE*Phi, Wiewiora 2003),
so it does not bias gradients once learned -- Ng's optimum guarantee holds --
but it swamps undiscounted return, and while the critic is learning it the
signal-to-noise collapses (issue #4: EV->1 with adv_std->0 means the advantage
is drizzle residue, not progress). Compare policies on the physical metrics --
fraction of cells clean, time to finish, grit delivered -- not on summed reward.

TIME LIMITS ARE TRUNCATION, NOT TERMINATION. They are reported separately so
the value function bootstraps through the cut-off. Conflating the two (which the
previous version did) teaches the agent the world ends at 1500 steps and makes
it hoard reward early.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import Config
from .geometry import (
    ambient_source,
    build_floor,
    episode_elevation,
    initial_dirt,
    initial_water,
)
from .jet import jet_impact, jet_peak_pressure
from .physics import FieldState, initial_fields, physics_substep, residual_map


@dataclasses.dataclass(frozen=True)
class ObsConfig:
    """Shape of what the policy sees."""

    pool: int = 4  # global map downsample factor
    crop: int = 24  # egocentric crop, in cells (24 * 0.07 m = 1.7 m)

    # Channels in both maps: adhered grit, loose grit, film depth, yield
    # stress. Adhered and loose are shown separately because they call for
    # completely different actions -- get close and stand the wand up, versus
    # lay it over and sweep -- so collapsing them would hide the distinction
    # the policy most needs to make.
    n_channels: int = 4


class EnvState(NamedTuple):
    fields: FieldState
    yield_stress: jnp.ndarray  # (nx, ny)
    z: jnp.ndarray  # (nx, ny) this episode's floor elevation, m
    tip_x: jnp.ndarray
    tip_y: jnp.ndarray
    standoff: jnp.ndarray
    tilt: jnp.ndarray
    azimuth: jnp.ndarray
    step: jnp.ndarray
    potential: jnp.ndarray  # cached Phi(s) so it is computed once per step
    initial_mass: jnp.ndarray  # kg of grit the episode started with
    water_used: jnp.ndarray  # m^3, for the water-cost diagnostic
    key: jax.Array


class Obs(NamedTuple):
    global_map: jnp.ndarray  # (nx//pool, ny//pool, C)
    local_map: jnp.ndarray  # (crop, crop, C)
    vector: jnp.ndarray  # (N,)


# --- Reward weights ---------------------------------------------------------
# Relative cost of grit in each state, in "units of work outstanding". Adhered
# grit has to be blasted loose AND then carried; loose deposit only has to be
# swept; suspended grit is already moving. The gaps between them are what pay
# the agent for making progress through the three phases. ALPHA converts
# distance-to-trough into the same units, so moving a kilogram one metre closer
# is worth ALPHA.
COST_BOUND = 1.0
COST_DEPOSITED = 0.70
COST_SUSPENDED = 0.55
ALPHA_DISTANCE = 0.22  # per metre

# The two halves of "fastest cleaning while still being thorough". Transport is
# the bulk mass still to move; thoroughness is the fraction of the floor still
# above the cleanliness threshold. See `CleaningEnv._potential`.
WEIGHT_TRANSPORT = 1.0
WEIGHT_THOROUGHNESS = 1.0

# Both potential terms are normalised, so a full clean is worth about
# (1 + ALPHA_DISTANCE * mean distance) + 1 ~ 2.6 units of potential regardless
# of the dirt loading. At this scale that is ~520 of reward against ~300 for the
# five-minute window's elapsed-time cost, so finishing the floor is clearly
# worth more than the time it takes, without the sparse finish bonus having to
# carry the whole signal.
REWARD_SCALE = 200.0
TIME_COST = 1.0  # per second of simulated work -- the objective being minimised

# Cost of grit still on the floor, per kg per second. THIS IS WHAT MAKES THE
# OBJECTIVE NON-EMPTY, and it was added on direct evidence rather than taste.
#
# Before it, the genuine (unshaped) reward was `-TIME_COST*dt + FINISH_BONUS*
# done_clean`. Measured over 3000 steps, that came to a single unique value:
# mean -0.2000, std 0.000000. The finish bonus fired exactly zero times in
# 750M steps across five training configurations, and a 90-minute run of the
# best scripted strategy showed why -- fraction_clean flatlines at 0.923 with
# a hard 7.7% residual, so "every cell under threshold" is not merely unreached
# but UNREACHABLE. A constant reward has identically zero advantages, which is
# precisely what the ablation measured, and no value of gamma, ent_coef or the
# Wiewiora offset can recover a gradient that is not there.
#
# Note this is a genuine cost, NOT more shaping: potential-based shaping is
# policy-invariant by construction (Ng et al. 1999) and therefore cannot supply
# the missing objective. Paying per second for dirt still on the floor makes
# the agent minimise the time-integral of remaining grit, which is the actual
# job. It is not farmable -- the only way to reduce it is to remove grit.
#
# Scale: a fresh floor carries ~1.18 kg, so at 1.0 the dirt and time terms
# start out comparable and the dirt term decays as the floor cleans.
DIRT_COST = 1.0

# Kept, but note it currently never fires: no policy tested, scripted or
# learned, reaches "every cell under clean_threshold". See issue #2 -- the
# criterion is stricter than the operator's real "the floor looks good"
# standard, which the 90-minute curve puts at roughly 70% of cells at 30 min.
FINISH_BONUS = 400.0


class CleaningEnv:
    """Single-environment pure functions. Use `jax.vmap` for a batch."""

    def __init__(self, cfg: Config | None = None, obs_cfg: ObsConfig | None = None,
                 discount: float = 0.9998):
        self.cfg = cfg or Config()
        self.obs_cfg = obs_cfg or ObsConfig()
        # Discount for the potential-based shaping term F = gamma*Phi(s') - Phi(s).
        # MUST equal the trainer's gamma (PPOConfig.gamma): with gamma=1 here the
        # per-step error (1-gamma)*Phi dwarfs TIME_COST (B2). Kept as a parameter
        # (rather than imported) because ppo imports this module, not vice versa.
        self.discount = discount
        self.floor = build_floor(self.cfg)

        # Where the ambient rinse lands (issue #1). Built once: it depends only
        # on config and the base floor, never on episode state.
        self.ambient = ambient_source(self.cfg, self.floor)

        # Distance from each cell to the trough: the transport cost map.
        self.dist_to_trough = jnp.abs(self.floor.y - self.cfg.floor.trough_y)

        # THE FLOOR, as distinct from the drain channel running through it.
        # Cleanliness is a property of the floor: grit that has reached the
        # trough has left the floor, which is the whole job. This never mattered
        # while the trough was a perfect sink and always empty, but it now
        # retains a puddle (issue #3), so grit can sit in it -- and counting
        # those cells would make `done_clean` unreachable and would have the
        # shaping term penalise the agent for delivering.
        self.floor_mask = (self.floor.trough < 0.5).astype(jnp.float32)
        self.n_floor = jnp.maximum(jnp.sum(self.floor_mask), 1.0)

        fc = self.cfg.floor
        assert fc.nx % self.obs_cfg.pool == 0 and fc.ny % self.obs_cfg.pool == 0, (
            f"grid {fc.nx}x{fc.ny} must divide by pool={self.obs_cfg.pool}"
        )

    # -- spaces ------------------------------------------------------------
    @property
    def action_dim(self) -> int:
        return 5

    @property
    def obs_shapes(self):
        fc, oc = self.cfg.floor, self.obs_cfg
        return {
            "global_map": (fc.nx // oc.pool, fc.ny // oc.pool, oc.n_channels),
            "local_map": (oc.crop, oc.crop, oc.n_channels),
            "vector": (11,),
        }

    # -- potential ---------------------------------------------------------
    def _potential(self, fields, initial_mass) -> jnp.ndarray:
        """Negative outstanding work, as a fraction of the job.

        TWO terms, because bulk mass alone does not describe this job.

        `transport` is the mass still on the floor, weighted by how hard each
        phase is to shift and by how far it has to travel. It is the obvious
        term and it is not sufficient: most of the mass is the easy 90%, and a
        policy flailing at random collects nearly as much of it as a systematic
        sweep does. Measured over a five-minute window, random removed 8.1% and
        the best scripted sweep 10.0% -- a real difference, but swamped.

        `thoroughness` is the fraction of the floor still ABOVE the cleanliness
        threshold, and it is what actually separates techniques. It saturates:
        a cell four times over the threshold counts the same as one just over
        it, so the term only pays for getting cells CLEAN, not for skimming the
        heaviest patches. Random coverage cannot finish cells; systematic
        coverage can. This is the "still being thorough" half of Andrew's
        objective, and without it the reward cannot express it at all.

        Both are normalised -- by the episode's starting mass, and by area -- so
        the scale is invariant to the dirt loading. That matters: the loading
        constant has already been corrected twice, and each time it silently
        rescaled every reward in the run.

        Still a strict potential: a function of the state alone (`initial_mass`
        is carried in `EnvState`), so Ng et al.'s guarantee holds and the
        optimal policy is unchanged.
        """
        weighted = (
            COST_BOUND * fields.bound
            + COST_DEPOSITED * fields.deposited
            + COST_SUSPENDED * fields.suspended
        )
        # Masked to the floor: grit in the trough is delivered, not outstanding.
        work = weighted * (1.0 + ALPHA_DISTANCE * self.dist_to_trough) * self.floor_mask
        transport = jnp.sum(work) * self.cfg.floor.cell_area / jnp.maximum(
            initial_mass, 1e-6
        )

        residual = fields.bound + fields.deposited + fields.suspended
        thoroughness = jnp.sum(
            jnp.clip(residual / self.cfg.dirt.clean_threshold, 0.0, 1.0)
            * self.floor_mask
        ) / self.n_floor

        return -(WEIGHT_TRANSPORT * transport + WEIGHT_THOROUGHNESS * thoroughness)

    # -- reset -------------------------------------------------------------
    def reset(self, key: jax.Array) -> tuple[EnvState, Obs]:
        cfg = self.cfg
        k_dirt, k_prog, k_pos, k_z, k_water, k_next = jax.random.split(key, 6)

        z = episode_elevation(k_z, cfg, self.floor)
        bound0, yield_stress = initial_dirt(k_dirt, cfg, self.floor)

        # Start the floor at a uniformly random point through the job, so the
        # policy is trained on the whole distribution of situations rather than
        # only on a fresh floor.
        bound, deposited = self._apply_progress(k_prog, bound0, yield_stress)

        fields = initial_fields(cfg.floor.nx, cfg.floor.ny)._replace(
            bound=bound, deposited=deposited, h=initial_water(k_water, cfg, z)
        )

        # The operator starts somewhere random along the bay, on one side.
        kx, ky = jax.random.split(k_pos)
        tip_x = jax.random.uniform(kx, (), minval=0.0, maxval=cfg.floor.length_x)
        tip_y = jax.random.uniform(ky, (), minval=0.0, maxval=cfg.floor.length_y)

        # Includes the loose layer: a part-done floor's outstanding work is
        # everything still on it, not just what is still stuck down.
        initial_mass = jnp.sum(bound + deposited) * cfg.floor.cell_area

        state = EnvState(
            fields=fields,
            yield_stress=yield_stress,
            z=z,
            tip_x=tip_x,
            tip_y=tip_y,
            standoff=jnp.array(0.4),
            tilt=jnp.array(0.6),
            azimuth=jnp.array(0.0),
            step=jnp.array(0, dtype=jnp.int32),
            potential=self._potential(fields, initial_mass),
            initial_mass=initial_mass,
            water_used=jnp.array(0.0),
            key=k_next,
        )
        return state, self._observe(state)

    def fresh_state(self, key: jax.Array) -> EnvState:
        """A fresh, uniformly dirty floor, operator at the wall.

        Bypasses `reset`'s random job-progress: completion runs (benchmark,
        T1 calibration) must start the whole job, not a random slice of one.
        Canonical copy -- tests and scripts must use this, not their own.
        """
        cfg = self.cfg
        k_dirt, k_z, k_water, k_next = jax.random.split(key, 4)
        z = episode_elevation(k_z, cfg, self.floor)
        bound, yield_stress = initial_dirt(k_dirt, cfg, self.floor)
        fields = initial_fields(cfg.floor.nx, cfg.floor.ny)._replace(
            bound=bound, h=initial_water(k_water, cfg, z)
        )
        initial_mass = jnp.sum(bound) * cfg.floor.cell_area
        return EnvState(
            fields=fields,
            yield_stress=yield_stress,
            z=z,
            tip_x=jnp.array(0.15),
            tip_y=jnp.array(cfg.floor.length_y - 0.15),
            standoff=jnp.array(0.4),
            tilt=jnp.array(0.6),
            azimuth=jnp.array(0.0),
            step=jnp.array(0, dtype=jnp.int32),
            potential=self._potential(fields, initial_mass),
            initial_mass=initial_mass,
            water_used=jnp.array(0.0),
            key=k_next,
        )

    def _apply_progress(self, key, bound, yield_stress):
        """Wind the floor forward to a random point in the job.

        Cells are cleaned in the order a competent operator would get to them:
        easy grit near the trough first, stubborn worn-lane grit far from it
        last. A soft mask rather than a hard one, so the boundary is a realistic
        gradient rather than a cliff.

        Grit that has been blasted off does not vanish -- a fraction of it is
        still lying loose on the floor waiting to be swept to the trough, piled
        up nearer the trough because that is the way it has been pushed. So a
        partly-done floor presents BOTH outstanding jobs at once, which is what
        the operator actually walks into.
        """
        k_p, k_jitter, k_ret = jax.random.split(key, 3)
        progress = jax.random.uniform(k_p, (), minval=0.0, maxval=0.95)

        ease = -(yield_stress / self.cfg.dirt.yield_mean) - 0.25 * self.dist_to_trough
        # Per-episode jitter so the cleaning order is not a fixed function of
        # the terrain.
        ease = ease + 0.3 * jax.random.normal(k_jitter, ease.shape)

        cutoff = jnp.quantile(ease, 1.0 - progress)
        remaining = jax.nn.sigmoid((cutoff - ease) / 0.15)

        loosened = bound * (1.0 - remaining)
        # How much of the loosened grit is still sitting on the floor rather
        # than already down the trough.
        retention = jax.random.uniform(k_ret, (), minval=0.05, maxval=0.5)
        # Piled toward the trough: the further out, the more of it has already
        # been driven inward.
        pile = jnp.exp(-self.dist_to_trough / 2.5)
        deposited = loosened * retention * pile

        return bound * remaining, deposited

    # -- observation -------------------------------------------------------
    def _channels(self, state: EnvState):
        """The physical fields the policy sees, each scaled to ~O(1)."""
        f = state.fields
        load = self.cfg.dirt.load_mean
        adhered = f.bound / load
        loose = (f.deposited + f.suspended) / load
        film = f.h / 3.0e-3
        grip = state.yield_stress / self.cfg.dirt.yield_mean
        return jnp.stack([adhered, loose, film, grip], axis=-1)  # (nx, ny, 4)

    def _observe(self, state: EnvState) -> Obs:
        cfg, oc = self.cfg, self.obs_cfg
        ch = self._channels(state)
        nx, ny, nc = ch.shape

        # Global view: mean-pool so nothing is lost to aliasing.
        p = oc.pool
        global_map = ch.reshape(nx // p, p, ny // p, p, nc).mean(axis=(1, 3))

        # Egocentric crop at full resolution. Pad first so a crop near the wall
        # is well defined: no grit and no water outside the bay, and a grip
        # value high enough to read as "not cleanable".
        half = oc.crop // 2
        pad_vals = jnp.array([0.0, 0.0, 0.0, 4.0])
        padded = jnp.stack(
            [
                jnp.pad(ch[..., c], half, mode="constant", constant_values=pad_vals[c])
                for c in range(nc)
            ],
            axis=-1,
        )
        i = jnp.clip((state.tip_x / cfg.floor.dx).astype(jnp.int32), 0, nx - 1)
        j = jnp.clip((state.tip_y / cfg.floor.dx).astype(jnp.int32), 0, ny - 1)
        local_map = jax.lax.dynamic_slice(padded, (i, j, 0), (oc.crop, oc.crop, nc))

        wc = cfg.washer
        remaining = jnp.sum(residual_map(state.fields)) * cfg.floor.cell_area
        signed_to_trough = (state.tip_y - cfg.floor.trough_y) / cfg.floor.length_y

        vector = jnp.stack(
            [
                state.tip_x / cfg.floor.length_x * 2.0 - 1.0,
                state.tip_y / cfg.floor.length_y * 2.0 - 1.0,
                signed_to_trough,  # sign tells the agent which side it is on
                (state.standoff - wc.standoff_min) / (wc.standoff_max - wc.standoff_min),
                state.tilt / wc.tilt_max,
                jnp.sin(state.azimuth),
                jnp.cos(state.azimuth),
                state.step / cfg.sim.max_steps,
                remaining / jnp.maximum(state.initial_mass, 1e-6),
                jnp.log10(jnp.maximum(self._current_pressure(state), 1.0)) / 5.0,
                jnp.sum(state.fields.h) * cfg.floor.cell_area * 100.0,  # water on floor
            ]
        )
        return Obs(global_map=global_map, local_map=local_map, vector=vector)

    def _current_pressure(self, state: EnvState) -> jnp.ndarray:
        return jet_peak_pressure(self.cfg, state.standoff, state.tilt)

    # -- step --------------------------------------------------------------
    def step(self, state: EnvState, action: jnp.ndarray):
        """Advance one control step. `action` is 5 values in [-1, 1].

        0,1: operator translation, as a fraction of walking speed
        2:   wand azimuth rate
        3:   target standoff  (absolute, mapped onto the reachable range)
        4:   target tilt from vertical (absolute)

        Standoff and tilt are commanded as targets rather than rates because
        that is how an operator thinks about them -- "hold it about a foot off,
        laid over" -- and it makes the trained policy directly readable as
        technique. Both then slew toward the target at a finite rate, so the
        wand still cannot teleport.
        """
        cfg, wc = self.cfg, self.cfg.washer
        dt = cfg.sim.control_dt
        action = jnp.clip(action, -1.0, 1.0)

        # -- operator and wand kinematics ---------------------------------
        tip_x = jnp.clip(state.tip_x + action[0] * wc.walk_speed * dt, 0.0, cfg.floor.length_x)
        tip_y = jnp.clip(state.tip_y + action[1] * wc.walk_speed * dt, 0.0, cfg.floor.length_y)
        azimuth = state.azimuth + action[2] * wc.azimuth_rate * dt
        azimuth = jnp.arctan2(jnp.sin(azimuth), jnp.cos(azimuth))  # wrap to [-pi, pi]

        standoff_target = wc.standoff_min + (action[3] * 0.5 + 0.5) * (
            wc.standoff_max - wc.standoff_min
        )
        standoff = _slew(state.standoff, standoff_target, wc.standoff_rate * dt)

        tilt_target = (action[4] * 0.5 + 0.5) * wc.tilt_max
        tilt = _slew(state.tilt, tilt_target, wc.tilt_rate * dt)

        # -- physics -------------------------------------------------------
        impact = jet_impact(cfg, self.floor, tip_x, tip_y, standoff, tilt, azimuth)

        floor = self.floor._replace(z=state.z)

        # Ambient rinse water reaching the floor while it is worked. It is what
        # keeps a film on the slab between passes, and transport length scales
        # with film depth -- see FloorConfig.ambient_inflow_gpm, still the
        # model's dominant uncertainty in MAGNITUDE.
        #
        # Its SHAPE is no longer a guess: this used to divide the total by the
        # section area, i.e. a uniform rain, which is the one thing the real bay
        # never produces (issue #1). `self.ambient` is now built from
        # `FloorConfig.ambient_layout` and integrates to the same total for
        # every layout, so geometry and amount are independent knobs.
        water_source = impact.water_source + self.ambient

        def substep(fields, _):
            return physics_substep(
                cfg, floor, fields, state.yield_stress,
                water_source, impact.coverage, impact.intensity,
                impact.p_normal, impact.tau_x, impact.tau_y,
                cfg.sim.physics_dt,
            ), None

        fields, _ = jax.lax.scan(substep, state.fields, None, length=cfg.sim.physics_substeps)

        new_state = EnvState(
            fields=fields,
            yield_stress=state.yield_stress,
            z=state.z,
            tip_x=tip_x,
            tip_y=tip_y,
            standoff=standoff,
            tilt=tilt,
            azimuth=azimuth,
            step=state.step + 1,
            potential=self._potential(fields, state.initial_mass),
            initial_mass=state.initial_mass,
            water_used=state.water_used + wc.flow * dt,
            key=state.key,
        )

        # -- reward --------------------------------------------------------
        residual = residual_map(fields)
        # Judged on the floor, not on the drain channel -- see `floor_mask`.
        worst = jnp.max(residual * self.floor_mask)
        done_clean = worst < cfg.dirt.clean_threshold

        # Strict potential-based shaping F = gamma*Phi(s') - Phi(s) (Ng et al.
        # 1999): dense feedback that leaves the optimal policy unchanged. The
        # gamma here must be the trainer's discount -- the difference telescopes
        # to the true objective only then, and only because mass is conserved.
        shaping = REWARD_SCALE * (self.discount * new_state.potential - state.potential)

        remaining = jnp.sum(residual * self.floor_mask) * cfg.floor.cell_area

        # The genuine (unshaped) reward. DIRT_COST is the term that makes this
        # objective well-posed at all; see its definition for the measurement
        # that forced it. Without it the unshaped reward is -TIME_COST*dt plus
        # an event that never occurs, i.e. a constant, and PPO's advantages are
        # then identically zero no matter how the critic is parameterised.
        reward = (shaping
                  - TIME_COST * dt
                  - DIRT_COST * remaining * dt
                  + FINISH_BONUS * done_clean)

        truncated = new_state.step >= cfg.sim.max_steps
        terminated = done_clean
        # In the trough but not yet down the drain: the pond's sediment load.
        # `drained_kg` is what LEFT; `delivered_kg` is what reached the trough.
        # They were the same number while the trough was a perfect sink.
        in_trough = jnp.sum(residual * (1.0 - self.floor_mask)) * cfg.floor.cell_area
        info = {
            "remaining_kg": remaining,
            "drained_kg": fields.drained,
            "fraction_removed": 1.0 - remaining / jnp.maximum(state.initial_mass, 1e-6),
            "worst_residual": worst,
            "fraction_clean": jnp.sum(
                (residual < cfg.dirt.clean_threshold) * self.floor_mask
            ) / self.n_floor,
            "delivered_kg": fields.drained + in_trough,
            "trough_grit_kg": in_trough,
            "pond_m3": jnp.sum(fields.h * (1.0 - self.floor_mask)) * cfg.floor.cell_area,
            "standoff": standoff,
            "tilt": tilt,
            "pressure": impact.p_normal,
            "water_m3": new_state.water_used,
            "suspended_kg": jnp.sum(fields.suspended) * cfg.floor.cell_area,
            "deposited_kg": jnp.sum(fields.deposited) * cfg.floor.cell_area,
            "adhered_kg": jnp.sum(fields.bound) * cfg.floor.cell_area,
            "is_clean": done_clean,
        }
        return new_state, self._observe(new_state), reward, terminated, truncated, info


def _slew(current, target, max_delta):
    """Move `current` toward `target` by at most `max_delta`."""
    return current + jnp.clip(target - current, -max_delta, max_delta)
