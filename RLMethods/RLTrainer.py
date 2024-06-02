from functools import partial
from typing import Dict, Tuple
import chex
import flax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax import struct

from utils.config import TrainConfig




class RLTrainer(object):
    def get_models(
            self,
            obs_dim: int,
            act_dim: int,
            max_action: float,
            config: TrainConfig,
            rng: chex.PRNGKey,
    ):
        raise NotImplementedError()

    def sample_buff_and_update_n_times(
        self,
        train_state,
        buffer,
        num_existing_samples: int,
        max_action: float,
        action_dim: int,
        rng: chex.PRNGKey,
        config: TrainConfig,
    ) -> Tuple[TrainState, Dict[str, float]]:
        raise NotImplementedError()

    def get_action(
        self,
        train_state: TrainState,
        obs: chex.Array,
        rng: chex.PRNGKey,
        config: TrainConfig,
        exploration_noise: bool = True,
    ) -> jnp.ndarray:
        raise NotImplementedError()
