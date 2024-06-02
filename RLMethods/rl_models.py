from functools import partial
import math

from typing import Any, Callable, Optional, Sequence, Tuple
import chex
import flax
import jax

import jax.numpy as jnp
import flax.linen as nn
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


activation_aliases = {
    'relu' : nn.relu,
    'tanh' : nn.tanh,
    'sigmoid' : nn.sigmoid,
    'swish': nn.swish,
}

def orthogonal_init(scale: Optional[float] = math.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class DoubleCritic(nn.Module):
    num_hidden_units: int
    num_hidden_layers: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    prefix: str = "critic"
    model_name: str = "double_critic"
    use_layernorm: bool = False
    use_dropout: bool = False
    dropout_rate: float = 0.01
    orthogonal_init: bool = True

    @nn.compact
    def __call__(self, state, action, rng):
        sa = jnp.concatenate([state, action], axis=-1)
        # critic 1
        if self.orthogonal_init:
            dense_with_init = partial(nn.Dense, kernel_init=orthogonal_init())
        else:
            dense_with_init = nn.Dense
        x_q = dense_with_init(
                self.num_hidden_units,
                name=self.prefix + "1_fc_1",
            )(sa)
        if self.use_dropout:
            x_q = nn.Dropout(self.dropout_rate)(x_q, deterministic=False)
        if self.use_layernorm:
            x_q = nn.LayerNorm()(x_q)
        x_q = self.activation(x_q)
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x_q = dense_with_init(
                    self.num_hidden_units,
                    name=self.prefix + f"1_fc_{i+1}",
                )(x_q)
            if self.use_dropout:
                x_q = nn.Dropout(self.dropout_rate)(x_q, deterministic=False)
            if self.use_layernorm:
                x_q = nn.LayerNorm()(x_q)
            x_q = self.activation(x_q)
        q1 = dense_with_init(  
            1,
            name=self.prefix + "1_fc_v",
        )(x_q)

        x_q = dense_with_init(
                self.num_hidden_units,
                name=self.prefix + "2_fc_1",
            )(sa)
        if self.use_dropout:
            x_q = nn.Dropout(self.dropout_rate)(x_q, deterministic=False)
        if self.use_layernorm:
            x_q = nn.LayerNorm()(x_q)
        x_q = self.activation(x_q)
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x_q = dense_with_init(
                    self.num_hidden_units,
                    name=self.prefix + f"2_fc_{i+1}",
                )(x_q)
            if self.use_dropout:
                x_q = nn.Dropout(self.dropout_rate)(x_q, deterministic=False)
            if self.use_layernorm:
                x_q = nn.LayerNorm()(x_q)
            x_q = self.activation(x_q)
            # print('USING LAYERNORM')
        q2 = nn.Dense(  
            1,
            name=self.prefix + "2_fc_v",
            #bias_init=default_mlp_init(),
        )(x_q)

        return q1, q2

class TD3Actor(nn.Module):
    action_dim: int
    num_hidden_units: int
    num_hidden_layers: int
    max_action: float
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    prefix: str = "actor"
    model_name: str = "actor"
    orthogonal_init: bool = True

    @nn.compact
    def __call__(self, state, rng):
        if self.orthogonal_init:
            dense_with_init = partial(nn.Dense, kernel_init=orthogonal_init())
        else:
            dense_with_init = nn.Dense
        x_a = self.activation(
            dense_with_init(
                self.num_hidden_units,
                name=self.prefix + "_fc_1",
            )(state)
        )
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x_a = self.activation(
                dense_with_init(
                    self.num_hidden_units,
                    name=self.prefix + f"_fc_{i+1}",
                )(x_a)
            )
        action = dense_with_init(
            self.action_dim,
            name=self.prefix + "_fc_mu",
        )(x_a)
        action = self.max_action * jnp.tanh(action)

        return action

