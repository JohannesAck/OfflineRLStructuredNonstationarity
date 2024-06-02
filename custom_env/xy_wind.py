from typing import List, Optional, Tuple
import chex
from gym.utils import seeding
from gymnax.environments.environment import EnvState
import jax
import numpy as np
import jax.numpy as jnp
from custom_env.environment_scheduler import EnvScheduler
from custom_env.hip_environment import HiPEnvironment
from custom_env.utils import ParameterizedEnvParams
from flax import struct

from custom_env.xy_env import XYEnv


@struct.dataclass
class EnvState:
    pos: float
    last_pos: float
    time: int
    hip: float = 0.0
    wind_strength: float = 0.09


@struct.dataclass
class EnvParams(ParameterizedEnvParams):
    max_steps_in_episode: int = 50
    phi: float = 0.0
    episode_idx: int = 0


class XYWindEnv(XYEnv):
    parameter_names = ['phi']
    jittable_render = True

    def __init__(self, scheduler: EnvScheduler):
        """
        Environment where the agent starts randomly, the goal location is on a circle around 
        it and the reward is the negative euclidean distance to the goal.

        Here phi represents the direction of the wind. This is inspired by Borel but unlike borel 
        we use a fixed strength and only change the direction.

        Actions are to move in increments in the x or y direction.
        """
        super().__init__()
        self.scheduler = scheduler
        self.max_action = 0.1
        self.action_scaling = 1.0  # internally reduces action to [-0.1,0.1], externally looks like [-1.0,1.0]
        self.space_size = 2.0
        self.sparse_reward = True

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Pendulum-v0."""
        return EnvParams()

    def get_goal(self, state, params):
        return jnp.array([1, 0])

    def step_env(
            self,
            key: chex.PRNGKey,
            state: EnvState,
            action: float,
            params: EnvParams,
        ) -> Tuple[chex.Array, EnvState, float, bool, dict]:

        action = jnp.clip(action, -self.max_action, self.max_action)
        action = self.action_scaling * action
        reward = self.get_reward(state, params)

        wind_dir = jnp.array([jnp.sin(params.phi), jnp.cos(params.phi)])
        new_pos = jnp.clip(state.pos + action + state.wind_strength * wind_dir , -self.space_size, self.space_size)
        state = EnvState(
            new_pos, state.pos, state.time + 1, hip=params.phi
        )
        done = self.is_terminal(state, params)

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            jax.lax.stop_gradient(reward),
            jax.lax.stop_gradient(done),
            {}
        )

    def get_reward(self, state, params):
        if self.sparse_reward:
            return jax.lax.cond(
                jnp.linalg.norm(state.pos - self.get_goal(state, params)) < 0.2,
                lambda _: 1.0,
                lambda _: 0.0,
                operand=None,
            )
            # return 1.0 if jnp.linalg.norm(state.pos - self.get_goal(state, params)) < 0.2 else 0.0
        else:
            return -jnp.linalg.norm(state.pos - self.get_goal(state, params))
    
    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        # pos = jax.random.uniform(key,  shape=(2,), minval=-self.space_size, maxval=self.space_size,)
        pos = jnp.array([0.0, 0.0])
        state = EnvState(
            pos=pos,
            last_pos=pos,
            time=0,
            hip=params.phi
        )

        return jax.lax.stop_gradient(self.get_obs(state)), jax.lax.stop_gradient(state)
    

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = state.time >= params.max_steps_in_episode
        return done


    def _render(self, state: EnvState) -> jnp.ndarray:
        size = 256
        radius = 5
        rgb_array = jnp.full([size,size,3], 64, dtype=jnp.uint8)
    
        def draw_circle(edit_array, pos, radius, color):
            color = jnp.array(color, dtype=jnp.uint8)
            color = jnp.tile(color,[256,256, 1])
            pos = (pos + self.space_size) / (2 * self.space_size) * size
            y, x = jnp.mgrid[:size, :size]
            y_diff = y - pos[0]
            x_diff = x - pos[1]
            pixel_dists = jnp.sqrt(x_diff**2 + y_diff**2)
            pixel_dists = jnp.repeat(pixel_dists[:,:,None],3,2)
            return jnp.where(pixel_dists < radius, color, edit_array)

        rgb_array = draw_circle(rgb_array, state.last_pos, radius, [0, 125, 0])
        rgb_array = draw_circle(rgb_array, state.pos, radius, [0, 255, 0])
        goal = jnp.array([1, 0])
        rgb_array = draw_circle(rgb_array, goal, radius, [0, 0, 255])
        wind_dir = jnp.array([jnp.sin(state.hip), jnp.cos(state.hip)])
        rgb_array = draw_circle(rgb_array, jnp.array([0.0, 0.0]), radius, [200, 0, 0])
        rgb_array = draw_circle(rgb_array, 0.2 * wind_dir, radius, [125, 0, 0])

        return rgb_array

    def close(self):
        pass

