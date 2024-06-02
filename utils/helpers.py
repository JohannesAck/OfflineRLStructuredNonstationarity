from typing import Tuple, Union

import numpy as np
import jax
from gymnax.environments.environment import EnvParams

from custom_env import XYEnv
from custom_env.brax_ant_leglen import AntLegLen
from custom_env.brax_ant_weight import AntWeight
from custom_env.environment_scheduler import NoisyFlipFlopScheduler, SawToothIntScheduler, SawToothScheduler
from custom_env.environment_scheduler import TriangleScheduler
from custom_env.hip_environment import HiPEnvironment
from custom_env.x_leftrightenv import XLeftRightEnv
from custom_env.xy_wind import XYWindEnv


np_or_jnp_array = Union[np.ndarray, jax.Array]

def get_env(env_name: str, env_kwargs) -> Tuple[HiPEnvironment, EnvParams]:
    if env_name == 'Xleftright-evolvediscretelong-v1':
        env = XLeftRightEnv(
            scheduler=SawToothScheduler({'phi': 0.0}, {'phi': 2.0}, deployment_period=10, saw_tooth_period=2),
            **env_kwargs)
        return env, env.default_params
    if env_name.startswith('Xleftright-evolvenoisy') and env_name.endswith('-v1'):  # looks like Xleftright-evolvenoisy0.1-v1
        noise_rate = float(env_name.removeprefix('Xleftright-evolvenoisy').removesuffix('-v1'))
        env = XLeftRightEnv(
            scheduler=NoisyFlipFlopScheduler({'phi': 0.0}, {'phi': 1.0}, deployment_period=10, noise_rate=noise_rate),
            **env_kwargs)
        return env, env.default_params
    elif env_name == 'XY-evolvediscretelong-v3':
        env = XYEnv(
            scheduler=TriangleScheduler({'phi': 0.0}, {'phi': 0.75 * 6.28}, deployment_period=24, triangle_period=8),
            **env_kwargs)
        return env, env.default_params
    elif env_name == 'XY-wind-v1':
        env = XYWindEnv(
            scheduler=SawToothScheduler({'phi': 0.0}, {'phi': 6.28}, deployment_period=20, saw_tooth_period=5),
            **env_kwargs)
        return env, env.default_params
    elif env_name == 'Ant-leglen-v2':
        possible_leg_lengths = (0.15, 0.175, 0.2, 0.225, 0.25)
        print('possible leg lengths', possible_leg_lengths)
        env = AntLegLen(
            scheduler=SawToothIntScheduler({'config_id': 0}, {'config_id': 5}, deployment_period=20, saw_tooth_period=5),
            possible_leg_lengths=possible_leg_lengths,
            **env_kwargs
        )
        return env, env.default_params
    elif env_name == 'Ant-weight-v0':
        possible_mass_mults = (0.5, 1.0, 1.5, 2.0, 2.5)
        print('possible mass mults', possible_mass_mults)
        env = AntWeight(
            scheduler=SawToothIntScheduler({'config_id': 0}, {'config_id': 5}, deployment_period=20, saw_tooth_period=5),
            possible_mass_mults=possible_mass_mults,
            **env_kwargs
        )
        return env, env.default_params
