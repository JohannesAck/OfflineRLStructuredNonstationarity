
import functools
from typing import Optional, Tuple, Union
from gymnax.environments.environment import EnvState, EnvParams
from gymnax.environments import spaces

from jax import numpy as jnp
import brax
import brax.envs, brax.envs.base
from flax import struct
import brax.io.image
import chex
import jax
from custom_env.leglen_ant import BraxAntLeglen
from brax.envs.wrappers.training import EpisodeWrapper

from custom_env.hip_environment import HiPEnvironment
from custom_env.environment_scheduler import EnvScheduler
from custom_env.utils import ParameterizedEnvParams


class EpisodeWrapperLegLen(EpisodeWrapper):
    """Maintains episode step count and sets done at episode end."""
    def reset(self, rng: jnp.ndarray, config_id: int) -> brax.envs.base.State:
        state = self.env.reset(rng, config_id=config_id)
        state.info['steps'] = jnp.zeros(rng.shape[:-1])
        state.info['truncation'] = jnp.zeros(rng.shape[:-1])
        return state

    def step(self, state: brax.envs.base.State, action: jnp.ndarray, config_id: int) -> brax.envs.base.State:
        def f(state, _):
            nstate = self.env.step(state, action, config_id=config_id)
            return nstate, nstate.reward

        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jnp.sum(rewards, axis=0))
        steps = state.info['steps'] + self.action_repeat
        one = jnp.ones_like(state.done)
        zero = jnp.zeros_like(state.done)
        episode_length = jnp.array(self.episode_length, dtype=jnp.int32)
        done = jnp.where(steps >= episode_length, one, state.done)
        state.info['truncation'] = jnp.where(
            steps >= episode_length, 1 - state.done, zero
        )
        state.info['steps'] = steps
        return state.replace(done=done)
    
    @property
    def observation_size(self) -> int:
        return self.env.get_observation_size(config_id=0)

    @property
    def action_size(self) -> int:
        return self.env.get_action_size(config_id=0)

@struct.dataclass
class AntParams(ParameterizedEnvParams):
    config_id: int = 1


@struct.dataclass
class AntState(EnvState):
    time: int
    hip: float
    brax_state: brax.envs.base.State

class AntLegLen(HiPEnvironment):
    video_subsample: int = 10
    fps_after_subsample: int = 3

    def __init__(self, scheduler: EnvScheduler, remove_z_obs: bool = True, possible_leg_lengths=(0.15, 0.2, 0.25)) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.remove_z_obs = remove_z_obs
        self.brax_env: BraxAntLeglen = EpisodeWrapperLegLen(BraxAntLeglen(possible_leg_lengths=possible_leg_lengths), episode_length=1000, action_repeat=1)
    
    @property
    def default_params(self) -> EnvParams:
        return AntParams(max_steps_in_episode=1000, config_id=1)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: AntState,
        action: chex.Array,
        params: AntParams,
    ) -> Tuple[chex.Array, AntState, float, bool, dict]:
        """Environment-specific step transition."""
        brax_state = self.brax_env.step(state.brax_state, action, config_id=params.config_id)
        state = AntState(brax_state=brax_state, time=state.time + 1, hip=params.config_id)
        reward = self.get_reward(state, params)
        done = self.is_terminal(state, params)
        return self.get_obs(state), state, reward, done, {**brax_state.info, **brax_state.metrics}

    def reset_env(
        self, key: chex.PRNGKey, params: AntParams
    ) -> Tuple[chex.Array, AntState]:
        """Environment-specific reset."""
        brax_state = self.brax_env.reset(key, config_id=params.config_id)
        state = AntState(brax_state=brax_state, time=0, hip=params.config_id)
        return self.get_obs(state), state

    def get_obs(self, state: AntState) -> chex.Array:
        """Applies observation function to state."""
        if self.remove_z_obs:
            return state.brax_state.obs[..., 1:]
        return state.brax_state.obs

    def get_reward(self, state: AntState, params: AntParams) -> chex.Array:
        return state.brax_state.reward

    def is_terminal(self, state: AntState, params: AntParams) -> bool:
        """Check whether state transition is terminal."""
        return state.brax_state.done.astype(bool)
    
    def _render(self, state: AntState) -> chex.Array:
        sys = jax.tree_map(lambda x: x[state.hip], self.brax_env.configs_stacked)
        return brax.io.image.render_array(
            sys=sys,
            state=state.brax_state.pipeline_state,
            width=128,
            height=128,
        )

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.brax_env.action_size

    def action_space(self, params: Optional[AntParams] = None):
        """Action space of the environment."""
        return spaces.Box(-1, 1, (self.brax_env.get_action_size(),))

    def observation_space(self, params: Optional[AntParams] = None):
        """Observation space of the environment."""
        obs_size = self.brax_env.get_observation_size()
        if self.remove_z_obs:
            obs_size -= 1
        return spaces.Box(-jnp.inf, jnp.inf, (obs_size,))

    def state_space(self, params: Optional[AntParams] = None):
        """State space of the environment."""
        raise NotImplementedError


