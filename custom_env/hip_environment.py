import dataclasses
import time
from collections import namedtuple
from functools import partial
from typing import List, Optional, Tuple, Union

import chex
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments.environment import EnvState, EnvParams
from gymnax.environments.environment import Environment

from custom_env.environment_scheduler import DummyScheduler, EnvScheduler
from custom_env.utils import ParameterizedEnvParams


class HiPEnvironment(Environment):
    """
    Same as the gymnax environment, but returns EnvParams as well as EnvState, to allow
    transitions in Environment Params dynamically.
    """
    scheduler: EnvScheduler
    jittable_render: bool = False
    video_subsample: int = 1
    fps_after_subsample: int = 10

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        env_params: ParameterizedEnvParams,
    ) -> Tuple[jax.Array, EnvState, jax.Array, jax.Array, dict, ParameterizedEnvParams]:
        """
        Performs step transitions in the environment. This involves resets and environment parameter changes.
        """
        # Use default env parameters if no others specified
        if env_params is None:
            env_params = self.default_params
        key_step, key_reset, key_sched = jax.random.split(key, 3)
        obs_st, state_st, reward, done, info = self.step_env(
            key_step, state, action, env_params
        )
        obs_re, state_re = self.reset_env(key_reset, env_params)
        # Auto-reset environment based on termination
        # jax.debug.breakpoint()
        state = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.lax.select(done, obs_re, obs_st)
        
        env_params_next = self.scheduler.step(env_params, key_sched)
        env_params = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), env_params_next, env_params
        )

        return obs, state, reward, done, info, env_params

    @property
    def default_params(self) -> ParameterizedEnvParams:
        raise NotImplementedError()

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[ParameterizedEnvParams] = None
    ) -> Tuple[jax.Array, EnvState, ParameterizedEnvParams]:
        """
        Resets environment but does not change the environment parameters.
        """
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        obs, state = self.reset_env(key, params)
        return obs, state, params
    
    def _render(self, state: EnvState) -> jax.Array:
        raise NotImplementedError("This method must be implemented by the subclass.")
    
    def render_statelist(self, envstate_list: List[EnvState], video_fp_list: List[str]):
        """
        Renders a list of trajectories, each of which is one EnvState object.
        We assume that the elements of the envstate have the timestep as first index-dimension.
        """
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        for envstate, video_fp in zip(envstate_list, video_fp_list):
            #subsample envstate
            envstate = jax.tree_map(lambda x: x[::self.video_subsample], envstate)
            
            
            # rgb_start_time = time.time()
            if self.jittable_render:
                rgb_frames = jax.jit(jax.vmap(self._render), backend='cpu')(envstate)
            else:
                rgb_frames = [self._render(jax.tree_map(lambda x: x[i], envstate)) for i in range(envstate.time.shape[0])]

            rgb_frames = list(np.array(rgb_frames))
            # print(f"RGB frames took {time.time() - rgb_start_time} seconds")
            # mp4_start_time = time.time()
            clip = ImageSequenceClip(rgb_frames, fps=self.fps_after_subsample)
            clip.write_videofile(video_fp, logger=None, preset='ultrafast')

@struct.dataclass
class WrappedEnvState:
    state: EnvState
    hip: float
    time: int = 0

@struct.dataclass
class WrappedEnvParams:
    params: EnvParams
    episode_idx: int
    max_steps_in_episode: int

class HipEnvironmentWrapper(HiPEnvironment):
    def __init__(self, env: gymnax.environments.environment.Environment) -> None:
        self.env = env
        self.scheduler = DummyScheduler()

    @property
    def default_params(self) -> ParameterizedEnvParams:
        env_params = self.env.default_params
        env_params_wrapped = WrappedEnvParams(
            params=env_params, 
            episode_idx=0,
            max_steps_in_episode=env_params.max_steps_in_episode
        )
        return env_params_wrapped

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: WrappedEnvState,
        action: Union[int, float],
        params: WrappedEnvParams,
    ) -> Tuple[jax.Array, WrappedEnvState, float, bool, dict]:
        """Environment-specific step transition."""
        obs_st, internal_state, reward, done, info = self.env.step_env(key, state.state, action, params.params)
        state = state.replace(state=internal_state, time = state.time + 1)
        return obs_st, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(
        self, key: chex.PRNGKey, params: ParameterizedEnvParams
    ) -> Tuple[jax.Array, EnvState]:
        """Environment-specific reset."""
        obs, state = self.env.reset_env(key, params.params)
        wrapped_state = WrappedEnvState(state=state, hip=0.0, time=0)
        return obs, wrapped_state

    def get_obs(self, state: WrappedEnvState) -> jax.Array:
        """Applies observation function to state."""
        return self.env.get_obs(state.state)

    def is_terminal(self, state: WrappedEnvState, params: WrappedEnvParams) -> bool:
        """Check whether state transition is terminal."""
        return self.env.is_terminal(state.state, params.params)

    def discount(self, state: WrappedEnvState, params: WrappedEnvParams) -> float:
        """Return a discount of zero if the episode has terminated."""
        return self.env.discount(state.state, params.params)

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.env.num_actions

    def action_space(self, params: Optional[EnvParams] = None):
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return self.env.action_space(params.params)

    def observation_space(self, params: WrappedEnvParams):
        """Observation space of the environment."""
        return self.env.observation_space(params.params)

    def state_space(self, params: WrappedEnvParams):
        """State space of the environment."""
        return self.env.state_space(params.params)

    @partial(jax.jit, static_argnums=(0,))
    def _render(self, state: EnvState) -> jax.Array:
        # FIXME this is only for the gymnax-reacher env.
        with jax.ensure_compile_time_eval():
            size = 256
            radius = 5.0
            space_size = 2.0
        rgb_array = jnp.full([size,size,3], 64, dtype=jnp.uint8)
        state = state.state


        joint_xy = [jnp.array([0.0, 0.0])]
        for idx_joint in range(self.env.num_joints):
            x = jnp.sum(jnp.cos(state.angles[:idx_joint + 1]), axis=-1)
            y = jnp.sum(jnp.sin(state.angles[:idx_joint + 1]), axis=-1)
            xy = jnp.concatenate(
                [jnp.expand_dims(x, axis=0), jnp.expand_dims(y, axis=0)], axis=0
            )
            joint_xy.append(xy)
        
        # draw links:
        n_intermediates = 10
        for idx_joint in range(self.env.num_joints):
            xy1 = joint_xy[idx_joint]
            xy2 = joint_xy[idx_joint + 1]
            for i in range(n_intermediates):
                xy = (xy1 * (n_intermediates - i) + xy2 * i) / n_intermediates
                rgb_array = draw_circle(rgb_array, xy, radius / 2.0, [125, 125, 125], space_size, size)

        # draw joints
        for idx_joint, xy in enumerate(joint_xy):
            if idx_joint == self.env.num_joints:  # end effector
                rgb_array = draw_circle(rgb_array, xy, radius, [0, 0, 255], space_size, size)
            else:
                rgb_array = draw_circle(rgb_array, xy, radius, [255, 0, 0], space_size, size)

        rgb_array = draw_circle(rgb_array, state.goal_xy, radius, [0, 255, 0], space_size, size)

        return rgb_array

@partial(jax.jit, static_argnames=('space_size', 'size'))
def draw_circle(edit_array, pos, radius, color, space_size, size):
    color = jnp.array(color, dtype=jnp.uint8)
    color = jnp.tile(color,[256,256, 1])
    pos = (pos + space_size) / (2 * space_size) * size
    y, x = jnp.mgrid[:size, :size]
    y_diff = y - pos[0]
    x_diff = x - pos[1]
    pixel_dists = jnp.sqrt(x_diff**2 + y_diff**2)
    pixel_dists = jnp.repeat(pixel_dists[:,:,None],3,2)
    return jnp.where(pixel_dists < radius, color, edit_array)
