from collections import defaultdict
from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Tuple
import warnings

import chex
import flax
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax import struct

from utils.config import TrainConfig
from .RLTrainer import RLTrainer
from .rl_models import TD3Actor, DoubleCritic
from .TD3Trainer import get_critic_loss

@struct.dataclass
class TD3BCTrainState:
    critic: TrainState
    actor: TrainState
    critic_params_target: flax.core.FrozenDict
    actor_params_target: flax.core.FrozenDict


class TD3BCTrainer(RLTrainer):

    def __init__(self, config: TrainConfig, action_space) -> None:
        super().__init__()
        self.config = config
        self.action_space = action_space

    def get_models(
            self,
            obs_dim: int,
            act_dim: int,
            max_action: float,
            config: TrainConfig,
            rng: chex.PRNGKey,
        ) -> TD3BCTrainState:
        critic_model = DoubleCritic(
            num_hidden_layers=config.offline.network_config.num_hidden_layers, 
            num_hidden_units=config.offline.network_config.num_hidden_units, 
            use_layernorm=config.offline.network_config.use_layer_norm
        )
        actor_model = TD3Actor(
            act_dim, 
            num_hidden_layers=config.offline.network_config.num_hidden_layers, 
            num_hidden_units=config.offline.network_config.num_hidden_units, 
            max_action=max_action
        )
        
        # Initialize the network based on the observation shape
        rng, rng1, rng2 = jax.random.split(rng, 3)
        critic_params = critic_model.init(rng1, state=jnp.zeros(obs_dim), action=jnp.zeros(act_dim), rng=rng1)
        critic_params_target = critic_model.init(rng1, jnp.zeros(obs_dim), jnp.zeros(act_dim), rng=rng1)
        actor_params = actor_model.init(rng2, jnp.zeros(obs_dim), rng=rng2)
        actor_params_target = actor_model.init(rng2, jnp.zeros(obs_dim), rng=rng2)

        critic_train_state = TrainState.create(
            apply_fn=critic_model.apply,
            params=critic_params,
            tx=optax.adam(config.offline.critic_lr),
        )
        actor_train_state = TrainState.create(
            apply_fn=actor_model.apply,
            params=actor_params,
            tx=optax.adam(config.offline.actor_lr),
        )

        return TD3BCTrainState(
            critic=critic_train_state,
            actor=actor_train_state,
            critic_params_target=critic_params_target,
            actor_params_target=actor_params_target,
        )


    @partial(jax.jit, static_argnames=('self', 'config', 'num_existing_samples'))
    def sample_buff_and_update_n_times(
        self,
        offline_train_state: TD3BCTrainState,
        buffer,
        num_existing_samples: int,
        max_action: float,
        action_dim: int,
        rng: chex.PRNGKey,
        config: TrainConfig,
    ) -> Tuple[dict, TD3BCTrainState]:
        avg_metrics_dict = defaultdict(int)
        train_state_critic = offline_train_state.critic
        train_state_actor = offline_train_state.actor
        critic_params_target = offline_train_state.critic_params_target
        actor_params_target = offline_train_state.actor_params_target

        for update_idx in range(config.offline.n_updates_jit):
            rng, subkey = jax.random.split(rng)
            obs, action, reward, next_obs, done = sample_batch(buffer, num_existing_samples, config, subkey)

            rng, subkey = jax.random.split(rng)

            critic_grad_fn = jax.value_and_grad(get_critic_loss, has_aux=True)
            critic_loss, critic_grads = critic_grad_fn(
                train_state_critic.params,
                critic_params_target,
                actor_params_target,
                train_state_critic.apply_fn,
                train_state_actor.apply_fn,
                obs,
                action,
                reward,
                done,
                next_obs,
                config.offline.gamma,
                config.offline.td3_policy_noise_std,
                config.offline.td3_policy_noise_clip,
                max_action,
                subkey,
            )
            train_state_critic = train_state_critic.apply_gradients(grads=critic_grads)
            avg_metrics_dict["offline/critic_grad_norm"] += jnp.mean(jnp.array(jax.tree_util.tree_flatten(jax.tree_map(jnp.linalg.norm, critic_grads))[0]))
            avg_metrics_dict["offline/value_loss_1"] += (critic_loss[1][0])
            avg_metrics_dict["offline/value_loss_2"] += (critic_loss[1][1])
            avg_metrics_dict["offline/target"] += (critic_loss[1][2])

            if update_idx % config.offline.policy_freq == 0:
                actor_grad_fn = jax.value_and_grad(get_actor_loss, has_aux=True)
                actor_loss, actor_grads = actor_grad_fn(
                    train_state_actor.params,
                    train_state_critic.params,
                    train_state_actor.apply_fn,
                    train_state_critic.apply_fn,
                    obs,
                    action,
                    config.offline.td3_alpha
                )
                train_state_actor = train_state_actor.apply_gradients(grads=actor_grads)
                avg_metrics_dict["offline/actor_loss"] += (actor_loss[0])
                avg_metrics_dict["offline/actor_loss_bc"] += (actor_loss[1][0])
                avg_metrics_dict["offline/actor_loss_td3_xlambda"] += (actor_loss[1][1])
                avg_metrics_dict["offline/actor_grad_norm"] += jnp.mean(jnp.array(jax.tree_util.tree_flatten(jax.tree_map(jnp.linalg.norm, actor_grads))[0]))

            # update target network
            critic_params_target = jax.tree_map(
                lambda target, live: config.offline.polyak * target + (1.0 - config.offline.polyak) * live,
                critic_params_target,
                train_state_critic.params
            )
            actor_params_target = jax.tree_map(
                lambda target, live: config.offline.polyak * target + (1.0 - config.offline.polyak) * live,
                actor_params_target,
                train_state_actor.params
            )

        for k, v in avg_metrics_dict.items():
            if 'offline/actor' in k:
                avg_metrics_dict[k] = v / (config.offline.n_updates_jit / config.offline.policy_freq)
            else:
                avg_metrics_dict[k] = v / config.offline.n_updates_jit

        offline_train_state = TD3BCTrainState(
            critic=train_state_critic,
            actor=train_state_actor,
            critic_params_target=critic_params_target,
            actor_params_target=actor_params_target,
        )

        return avg_metrics_dict, offline_train_state
    
    @partial(jax.jit, static_argnames=('self', 'config', 'exploration_noise'))
    def get_action(
        self,
        train_state: TD3BCTrainState,
        obs: chex.Array,
        rng: chex.PRNGKey,
        config: TrainConfig,
        exploration_noise: bool = True,
    ) -> jnp.ndarray:
        action = train_state.actor.apply_fn(train_state.actor.params, obs, rng=None)
        
        if exploration_noise:
            warnings.warn('TD3BC with exploration noise is probably not useful.')
            noise = config.online.exploration_std * self.action_space.high * jax.random.normal(rng, action.shape)
            action = action + noise.clip(-self.config.online.exploration_clip, self.config.online.exploration_clip)
        
        action = action.clip(self.action_space.low, self.action_space.high)
        return action
            

    
@partial(jax.jit, static_argnames=('config'))
def sample_batch(buffer, num_existing_samples, config, rng):
    idxes = jax.random.randint(rng, (config.offline.batch_size,), 0, num_existing_samples)
    obs = buffer["states"][idxes]
    action  = buffer["actions"][idxes]
    reward =  buffer["rewards"][idxes]
    next_obs = buffer["next_states"][idxes]
    done = buffer["dones"][idxes]
    return obs, action, reward, next_obs, done

def get_actor_loss(
    actor_params: flax.core.frozen_dict.FrozenDict,
    critic_params: flax.core.frozen_dict.FrozenDict,
    actor_apply_fn: Callable[..., Any],
    critic_apply_fn: Callable[..., Any],
    obs: chex.Array,
    action: chex.Array,
    td3_alpha: Optional[float] = None,
) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:

    predicted_action = actor_apply_fn(actor_params, obs, rng=None)
    critic_params = jax.lax.stop_gradient(critic_params)
    q_value, _ = critic_apply_fn(critic_params, obs, predicted_action, rng=None) 


    if td3_alpha is None:
        loss_actor = -1.0 * q_value.mean()
        bc_loss = 0.0
        loss_lambda = 1.0
    else:
        mean_abs_q = jax.lax.stop_gradient(jnp.abs(q_value).mean())
        loss_lambda = td3_alpha / mean_abs_q

        bc_loss = jnp.square(predicted_action - action).mean()
        loss_actor = -1.0 * q_value.mean() * loss_lambda  + bc_loss
    return loss_actor, (
        bc_loss, -1.0 * q_value.mean() * loss_lambda,
    )
