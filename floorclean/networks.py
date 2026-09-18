"""Actor-critic network for the cleaning policy.

Three input streams, because the task has structure at two scales:

  global_map  a coarse view of the whole bay -- where the grit still is and how
              far it has to travel. Drives the strategic choice of which band to
              work and which way to push.
  local_map   a full-resolution crop around the wand tip. Drives aim, standoff
              and the decision to dwell on a stubborn patch.
  vector      proprioception: where the operator is, how the wand is held, how
              much of the job is left, how hard the jet is currently hitting.

Deliberately NO BatchNorm. The previous feature extractor used it throughout,
which quietly breaks PPO: the rollout is collected in eval mode on single
states while the update runs in train mode on shuffled minibatches, so the
running statistics are estimated from one distribution and applied to another,
and the deterministic-evaluation path sees different activations again from the
ones that were trained. LayerNorm has none of those problems and is used here
instead.

Initialisation follows the usual on-policy recipe: orthogonal with gain sqrt(2)
in the trunk, 0.01 on the policy mean so the initial policy is near-uniform and
does not commit before it has seen anything, and 1.0 on the value head.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
import numpy as np


def _ortho(scale=np.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class MapEncoder(nn.Module):
    """Small conv stack. Strided rather than pooled to keep it cheap."""

    features: tuple[int, ...] = (32, 64, 64)

    @nn.compact
    def __call__(self, x):
        for i, f in enumerate(self.features):
            x = nn.Conv(
                f,
                kernel_size=(3, 3),
                strides=(2, 2) if i > 0 else (1, 1),
                kernel_init=_ortho(),
                bias_init=nn.initializers.zeros,
            )(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
        return x.reshape((x.shape[0], -1))


class ActorCritic(nn.Module):
    action_dim: int
    hidden: int = 256

    @nn.compact
    def __call__(self, obs):
        g = MapEncoder(name="global_encoder")(obs.global_map)
        l = MapEncoder(name="local_encoder")(obs.local_map)
        v = nn.relu(nn.Dense(64, kernel_init=_ortho())(obs.vector))

        h = jnp.concatenate([g, l, v], axis=-1)
        h = nn.relu(nn.Dense(self.hidden, kernel_init=_ortho())(h))
        h = nn.relu(nn.Dense(self.hidden, kernel_init=_ortho())(h))

        # Separate heads on a shared trunk. The value head gets its own layer so
        # its gradients do not dominate the shared representation.
        actor = nn.relu(nn.Dense(self.hidden, kernel_init=_ortho())(h))
        mean = nn.Dense(self.action_dim, kernel_init=_ortho(0.01))(actor)

        # State-independent log-std, the standard on-policy choice: it keeps
        # exploration from collapsing early on states the critic is still
        # wrong about.
        log_std = self.param("log_std", nn.initializers.constant(-0.5), (self.action_dim,))

        critic = nn.relu(nn.Dense(self.hidden, kernel_init=_ortho())(h))
        value = nn.Dense(1, kernel_init=_ortho(1.0))(critic)

        return mean, jnp.broadcast_to(log_std, mean.shape), jnp.squeeze(value, -1)


def log_prob(mean, log_std, action):
    """Diagonal Gaussian log-density, summed over the action dimensions."""
    var = jnp.exp(2.0 * log_std)
    return jnp.sum(
        -0.5 * ((action - mean) ** 2 / var + 2.0 * log_std + jnp.log(2.0 * jnp.pi)),
        axis=-1,
    )


def entropy(log_std):
    return jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1)
