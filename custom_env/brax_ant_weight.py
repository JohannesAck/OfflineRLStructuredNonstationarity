
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
from custom_env.weight_ant import BraxAntWeight
from custom_env.brax_ant_leglen import EpisodeWrapperLegLen

from custom_env.hip_environment import HiPEnvironment
from custom_env.environment_scheduler import EnvScheduler
from custom_env.utils import ParameterizedEnvParams


@struct.dataclass
class AntParams(ParameterizedEnvParams):
    config_id: int = 1


@struct.dataclass
class AntState(EnvState):
    time: int
    hip: float
    brax_state: brax.envs.base.State

class AntWeight(HiPEnvironment):
    video_subsample: int = 10
    fps_after_subsample: int = 3

    def __init__(self, scheduler: EnvScheduler, remove_z_obs: bool = True, possible_mass_mults=(0.9,1.0,1.1)) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.remove_z_obs = remove_z_obs
        self.brax_env: BraxAntWeight = EpisodeWrapperLegLen(BraxAntWeight(possible_mass_mults=possible_mass_mults), episode_length=1000, action_repeat=1)
    
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


