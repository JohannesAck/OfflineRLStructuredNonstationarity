"""
Buffer mangament, based on the original gymnax-baselines code.
Huge mess now.
"""

from functools import partial
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from utils.DeploymentDataset import Dataset

class BufferManager:
    def __init__(
        self,
        size: int,
        num_envs: int,
        action_size,
        state_space,
    ):
        self.num_envs = num_envs
        self.action_size = action_size
        self.buffer_size = num_envs * size
        self.size = size

        try:
            temp = state_space.shape[0]
            self.state_shape = state_space.shape
        except Exception:
            self.state_shape = [state_space]

    @partial(jax.jit, static_argnums=0)
    def reset(self):
        return {
            "states": jnp.empty(
                (self.size, self.num_envs, *self.state_shape),
                dtype=jnp.float32,
            ),
            "hips": jnp.empty((self.size, self.num_envs), dtype=jnp.float32),
            "actions": jnp.empty(
                (self.size, self.num_envs, *self.action_size),
            ),
            "rewards": jnp.empty(
                (self.size, self.num_envs), dtype=jnp.float32
            ),
            "next_states": jnp.empty(
                (self.size, self.num_envs, *self.state_shape),
                dtype=jnp.float32,
            ),
            "dones": jnp.empty((self.size, self.num_envs), dtype=jnp.uint8),
            "_p": 0,
            "_valid": 0,
        }

    @partial(jax.jit, static_argnums=0)
    def append(self, buffer, state, hip, action, reward, next_state, done):
        return {
                "states":  buffer["states"].at[buffer["_p"]].set(state),
                "hips":  buffer["hips"].at[buffer["_p"]].set(hip),
                "actions": buffer["actions"].at[buffer["_p"]].set(action),
                "rewards": buffer["rewards"].at[buffer["_p"]].set(reward.squeeze()),
                "next_states":  buffer["next_states"].at[buffer["_p"]].set(next_state),
                "dones": buffer["dones"].at[buffer["_p"]].set(done.squeeze()),
                "_p": (buffer["_p"] + 1) % self.size,
                "_valid": jnp.minimum(buffer["_valid"] + 1, self.size),
            }


    def convert_dataset(self, dataset: Dataset) -> Dict[str, jnp.ndarray]:
        trans = dataset.transition
        if trans.done[:,:,:-1].sum() > 0:
            # if dones occur earlier on in the state, the rest of the "episode" has to be discarded
            # basically that's just padding
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            hips = []
            deployment_steps = []
            deployment_ids = []
            valid_ends = trans.done.argmax(axis=2) + 1
            for idx_deployment in tqdm.trange(trans.obs.shape[0], desc='removing padding'):
                for idx_episode in range(trans.obs.shape[1]):
                    end = valid_ends[idx_deployment, idx_episode]
                    states.append(trans.obs[idx_deployment, idx_episode, :end])
                    actions.append(trans.act[idx_deployment, idx_episode, :end])
                    rewards.append(trans.rew[idx_deployment, idx_episode, :end])
                    next_states.append(trans.next_obs[idx_deployment, idx_episode, :end])
                    dones.append(trans.done[idx_deployment, idx_episode, :end])
                    hips.append(trans.hip[idx_deployment, idx_episode, :end])
                    deployment_steps.append(trans.deployment_step[idx_deployment, idx_episode, :end])
                    deployment_ids.append(trans.deployment_id[idx_deployment, idx_episode, :end])
            buffer = {
                "states": np.concatenate(states, axis=0),
                "actions": np.concatenate(actions, axis=0),
                "rewards": np.concatenate(rewards, axis=0),
                "next_states": np.concatenate(next_states, axis=0),
                "dones": np.concatenate(dones, axis=0),
                "hips": np.concatenate(hips, axis=0),
                "deployment_steps": np.concatenate(deployment_steps, axis=0),
                "deployment_ids": np.concatenate(deployment_ids, axis=0),
            }
        else:
            buffer = {
                "states": trans.obs.reshape([-1, trans.obs.shape[-1]]),
                "actions": trans.act.reshape([-1, trans.act.shape[-1]]),
                "rewards": trans.rew.reshape([-1,]),
                "next_states": trans.next_obs.reshape([-1, trans.next_obs.shape[-1]]),
                "dones": trans.done.reshape([-1]),
                "hips": trans.hip.reshape([-1]),
                "deployment_steps": trans.deployment_step.reshape([-1]),
                "deployment_ids": trans.deployment_id.reshape([-1]),
            }
        buffer["_p"] = buffer["states"].shape[0]
        buffer["_valid"] = buffer["states"].shape[0]
        return buffer
