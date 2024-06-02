from collections import defaultdict
from functools import partial
from typing import Any, Callable, Tuple

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState
import optax

from utils.config import NetworkConfig, TrainConfig
from .RLTrainer import RLTrainer
from .rl_models import DoubleCritic, TD3Actor, activation_aliases


@struct.dataclass
class TD3TrainState:
    critic: TrainState
    actor: TrainState
    critic_params_target: flax.core.FrozenDict
    actor_params_target: flax.core.FrozenDict


class TD3Trainer(RLTrainer):

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
    ) -> TD3TrainState:
        critic_model = DoubleCritic(
            num_hidden_layers=config.online.critic_network_config.num_hidden_layers, 
            num_hidden_units=config.online.critic_network_config.num_hidden_units, 
            use_layernorm=False,
            activation=activation_aliases[config.online.critic_network_config.activation],
        )
        actor_model = TD3Actor(
            act_dim, 
            num_hidden_layers=config.online.actor_network_config.num_hidden_layers, 
            num_hidden_units=config.online.actor_network_config.num_hidden_units, 
            max_action=max_action,
            activation=activation_aliases[config.online.actor_network_config.activation],
        )

        # Initialize the network based on the observation shape
        rng, subkey1, subkey2 = jax.random.split(rng, 3)

        critic_params = critic_model.init(subkey1, state=jnp.zeros(obs_dim), action=jnp.zeros(act_dim), rng=subkey1)
        critic_params_target = critic_model.init(subkey1, jnp.zeros(obs_dim), jnp.zeros(act_dim), rng=subkey1)
        actor_params = actor_model.init(subkey2, jnp.zeros(obs_dim), rng=subkey2)
        actor_params_target = actor_model.init(subkey2, jnp.zeros(obs_dim), rng=subkey2)

        tx = optax.chain(
            # optax.clip_by_global_norm(config.online.max_grad_norm),
            optax.adam(config.online.lr),
            # optax.scale_by_schedule(lr_sched),
        )

        critic_train_state = TrainState.create(
            apply_fn=critic_model.apply,
            params=critic_params,
            # tx=optax.adam(config.lr),
            tx=tx,
        )
        actor_train_state = TrainState.create(
            apply_fn=actor_model.apply,
            params=actor_params,
            # tx=optax.adam(config.online.lr),
            tx=tx,
        )


        return TD3TrainState(
            critic=critic_train_state,
            actor=actor_train_state,
            critic_params_target=critic_params_target,
            actor_params_target=actor_params_target,
        )

    @partial(jax.jit, static_argnames=('self', 'config'))
    def sample_buff_and_update_n_times(
        self,
        train_state,
        buffer,
        num_existing_samples: int,
        max_action: float,
        action_dim: int,
        rng: chex.PRNGKey,
        config: TrainConfig,
    ):
        train_state_critic = train_state.critic
        train_state_actor = train_state.actor
        critic_params_target = train_state.critic_params_target
        actor_params_target = train_state.actor_params_target

        avg_metrics_dict = defaultdict(float)
        for update_idx in range(config.online.n_updates_jit):
            rng, subkey = jax.random.split(rng)
            obs, action, reward, next_obs, done = sample_batch(
                buffer, num_existing_samples, config.online.batch_size, config.online.num_train_envs, subkey
            )

            rng, subkey = jax.random.split(rng)
            train_state_critic, critic_loss, critic_grad_norm = critic_update(
                train_state_critic, critic_params_target, train_state_actor, actor_params_target,
                max_action, subkey, obs, action, reward, next_obs, done, config
            )

            if update_idx % config.online.policy_freq == 0:
                train_state_actor, actor_loss, actor_grad_norm = update_actor(train_state_critic, train_state_actor, obs, action)
                # update target network

                critic_params_target, actor_params_target = target_updates(
                    train_state_critic.params, train_state_actor.params,
                    critic_params_target, actor_params_target,
                    config.online.polyak
                )
                avg_metrics_dict["online/actor_loss"] += actor_loss[0]
                avg_metrics_dict["online/actor_grad_norm"] += actor_grad_norm


            avg_metrics_dict["online/value_loss_1"] += critic_loss[1][0]
            avg_metrics_dict["online/value_loss_2"] += critic_loss[1][1]
            avg_metrics_dict["online/target"] += critic_loss[1][2]
            avg_metrics_dict["online/critic_grad_norm"] += critic_grad_norm

        for k, v in avg_metrics_dict.items():
            if 'online/actor' in k:
                avg_metrics_dict[k] = v / (config.online.n_updates_jit / config.online.policy_freq)
            else:
                avg_metrics_dict[k] = v / config.online.n_updates_jit

        new_trainstate = TD3TrainState(
            critic=train_state_critic,
            actor=train_state_actor,
            critic_params_target=critic_params_target,
            actor_params_target=actor_params_target,
        )
        return new_trainstate, avg_metrics_dict
    
    @partial(jax.jit, static_argnames=('self','config', 'exploration_noise'))
    def get_action(
        self,
        train_state: TD3TrainState,
        obs: chex.Array,
        rng: chex.PRNGKey,
        config: TrainConfig,
        exploration_noise: bool = True,
    ) -> jnp.ndarray:
        action = train_state.actor.apply_fn(train_state.actor.params, obs, rng=None)
        
        if exploration_noise:
            noise = config.online.exploration_std * self.action_space.high * jax.random.normal(rng, action.shape)
            action = action + noise.clip(-self.config.online.exploration_clip, self.config.online.exploration_clip)
        
        action = action.clip(self.action_space.low, self.action_space.high)
        return action



def get_critic_loss(
    critic_params: flax.core.frozen_dict.FrozenDict,
    critic_target_params: flax.core.frozen_dict.FrozenDict,
    actor_target_params: flax.core.frozen_dict.FrozenDict,
    critic_apply_fn: Callable[..., Any],
    actor_apply_fn: Callable[..., Any],
    obs: chex.ArrayDevice,
    action: chex.ArrayDevice,
    reward: chex.ArrayDevice,
    done: chex.ArrayDevice,
    next_obs: chex.ArrayDevice,
    decay: float,
    policy_noise_std: float,
    policy_noise_clip: float,
    max_action: float,
    rng_key: chex.PRNGKey,
) -> Tuple[chex.Array, chex.ArrayTree]:

    q_pred_1, q_pred_2 = critic_apply_fn(critic_params, obs, action, rng=None)

    target_next_action = actor_apply_fn(actor_target_params, next_obs, rng=None)
    policy_noise = policy_noise_std * max_action * jax.random.normal(rng_key, action.shape)
    target_next_action = target_next_action + policy_noise.clip(-policy_noise_clip, policy_noise_clip)
    target_next_action = target_next_action.clip(-max_action, max_action)

    q_next_1, q_next_2 = critic_apply_fn(critic_target_params, next_obs, target_next_action, rng=None)

    target = reward[...,None] + decay * jnp.minimum(q_next_1, q_next_2) * (1 - done[...,None])
    target = jax.lax.stop_gradient(target)

    value_loss_1 = jnp.square(q_pred_1 - target)
    value_loss_2 = jnp.square(q_pred_2 - target)
    value_loss = (value_loss_1 + value_loss_2).mean()


    return value_loss, (
        value_loss_1.mean(),
        value_loss_2.mean(),
        target.mean(),
    )


def get_actor_loss(
    actor_params: flax.core.frozen_dict.FrozenDict,
    critic_params: flax.core.frozen_dict.FrozenDict,
    actor_apply_fn: Callable[..., Any],
    critic_apply_fn: Callable[..., Any],
    obs: chex.Array,
    action: chex.Array,
) -> Tuple[chex.Array, Tuple[None]]:

    action = actor_apply_fn(actor_params, obs, rng=None)
    critic_params = jax.lax.stop_gradient(critic_params)
    q_value, _ = critic_apply_fn(critic_params, obs, action, rng=None)

    loss_actor = -1.0 * q_value.mean()

    return loss_actor, (
        None,
    )


@partial(jax.jit, static_argnames=('batch_size', 'num_train_envs'))
def sample_batch(buffer, num_existing_samples, batch_size, num_train_envs, rng):
    subkey, subkey2 = jax.random.split(rng)
    idxes = jax.random.randint(subkey, (batch_size,), 0, num_existing_samples)
    subidxes = jax.random.randint(subkey2, (batch_size,), 0, num_train_envs)
    obs = buffer["states"][idxes,subidxes]
    action = buffer["actions"][idxes,subidxes]
    reward = buffer["rewards"][idxes,subidxes]
    next_obs = buffer["next_states"][idxes,subidxes]
    done = buffer["dones"][idxes,subidxes]

    return obs, action, reward, next_obs, done


@jax.jit
def target_updates(critic_params, actor_params, critic_params_target, actor_params_target, polyak):
    critic_params_target = jax.tree_map(
                lambda target, live: polyak * target + (1.0 - polyak) * live,
                critic_params_target,
                critic_params
            )
    actor_params_target = jax.tree_map(
                lambda target, live: polyak * target + (1.0 - polyak) * live,
                actor_params_target,
                actor_params
            )

    return critic_params_target, actor_params_target


@jax.jit
def update_actor(train_state_critic, train_state_actor, obs, action):
    actor_grad_fn = jax.value_and_grad(get_actor_loss, has_aux=True)
    actor_loss, actor_grads = actor_grad_fn(
                train_state_actor.params,
                train_state_critic.params,
                train_state_actor.apply_fn,
                train_state_critic.apply_fn,
                obs,
                action
            )
    train_state_actor = train_state_actor.apply_gradients(grads=actor_grads)
    actor_grad_norm = jnp.mean(jnp.array(jax.tree_util.tree_flatten(jax.tree_map(jnp.linalg.norm, actor_grads))[0]))
    return train_state_actor, actor_loss, actor_grad_norm


@partial(jax.jit, static_argnames=('config'))
def critic_update(
    train_state_critic, critic_params_target, train_state_actor, actor_params_target,
    max_action, rng, obs, action, reward, next_obs, done, config: TrainConfig
):
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
            config.online.gamma,
            config.online.policy_noise_std,
            config.online.policy_noise_clip,
            max_action,
            rng,
        )
    train_state_critic = train_state_critic.apply_gradients(grads=critic_grads)
    critic_grad_norm = jnp.mean(jnp.array(jax.tree_util.tree_flatten(jax.tree_map(jnp.linalg.norm, critic_grads))[0]))
    return train_state_critic, critic_loss, critic_grad_norm




def selective_stop_grad(variables, prefix):
    # https://github.com/google/flax/discussions/1931#discussioncomment-2227393
    flat_vars = flax.traverse_util.flatten_dict(variables)
    new_vars = {k: (jax.lax.stop_gradient(v)) if k[1].startswith(prefix) else v for k, v in flat_vars.items()}
    return flax.traverse_util.unflatten_dict(new_vars)


