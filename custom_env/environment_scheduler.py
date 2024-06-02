from functools import partial
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import chex
import jax
import jax.numpy as jnp
from custom_env.utils import ParameterizedEnvParams


class EnvScheduler(object):
    def __init__(self, parameter_names: List[str], period = None) -> None:
        self.parameter_names = parameter_names
        self.period: Optional[int] = period
        self.possible_hips: Optional[Dict[str,List[float]]] = None
    
    @partial(jax.jit, static_argnames='self')
    def step(self, old_env_params: ParameterizedEnvParams, rng: chex.PRNGKey) -> ParameterizedEnvParams:
        prev_episode_idx = old_env_params.episode_idx
        new_params: Dict[str, float] = {
            'episode_idx' : prev_episode_idx + 1
        }
        for pname in self.parameter_names:
            new_params[pname] = self._get_val_for_step(pname, prev_episode_idx + 1, rng)

        new_params = old_env_params.replace(**new_params)
        return new_params

    @partial(jax.jit, static_argnames='self')
    def reset(self, rng: chex.PRNGKey, env_params: ParameterizedEnvParams) -> ParameterizedEnvParams:
        """
        Resets the parameters that are controlled by a scheduler, others are copied from the input env_params.
        """
        new_params = {
            'episode_idx' : 0
        }
        for pname in self.parameter_names:
            rng, subkey = jax.random.split(rng)
            new_params[pname] = self._get_val_for_step(pname, 0, subkey)

        new_params = env_params.replace(**new_params)
        return new_params

    # float hints includes bools and ints, apparently https://peps.python.org/pep-0484/#the-numeric-tower
    @partial(jax.jit, static_argnums=(0, 1, 2))
    def _get_val_for_step(self, pname: str, episode_step: int, rng: chex.PRNGKey) -> float:  
        raise NotImplementedError


class DummyScheduler(EnvScheduler):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__([''])

    def step(self, old_env_params: ParameterizedEnvParams, rng: chex.PRNGKey) -> ParameterizedEnvParams:
        return old_env_params.replace(episode_idx=old_env_params.episode_idx + 1)

    def reset(self, rng: chex.PRNGKey, env_params: ParameterizedEnvParams) -> ParameterizedEnvParams:
        return env_params.replace(episode_idx=0)


class SinusoidalScheduler(EnvScheduler):
    def __init__(
        self, 
        min_val: Dict[str, float], 
        max_val: Dict[str, float], 
        deployment_period: int,
        sine_period: int,
    ) -> None:
        super().__init__(list(min_val.keys()), deployment_period)
        assert sine_period % 4 == 0, 'sine_period must be divisible by 4'
        self.mean = {}
        self.amplitude = {}
        for param_name in self.parameter_names:
            self.mean[param_name] = 0.5 * (max_val[param_name] + min_val[param_name])
            self.amplitude[param_name] = 0.5 * (max_val[param_name] - min_val[param_name])
        self.sine_period = sine_period
        
        # to avoid many different numerical sin values, just take the first quarter period and mirror and negate it
        sine_values_first_quarter = jnp.array([jnp.sin(jnp.pi * 2 * i / self.sine_period) for i in  range(self.sine_period // 4 + 1)])
        sine_values_half = jnp.concatenate(
            [sine_values_first_quarter, sine_values_first_quarter[-2::-1]]
        )
        sine_values_full = jnp.concatenate(
            [sine_values_half, -sine_values_half[1:-1]]
        )

        self.possible_hips = {}
        for pname in self.parameter_names:
            self.possible_hips[pname] = jnp.unique(self.mean[pname] + self.amplitude[pname] * sine_values_full)

    @partial(jax.jit, static_argnums=(0, 1, 2))    
    def _get_val_for_step(self, pname: str, episode_step: int, rng: chex.PRNGKey) -> float:  
        sine_values_first_quarter = jnp.array([jnp.sin(jnp.pi * 2 * i / self.sine_period) for i in  range(self.sine_period // 4 + 1)])
        sine_values_half = jnp.concatenate(
            [sine_values_first_quarter, sine_values_first_quarter[-2::-1]]
        )
        sine_values_full = jnp.concatenate(
            [sine_values_half, -sine_values_half[1:-1]]
        )

        return self.mean[pname] + self.amplitude[pname] * sine_values_full[episode_step % self.sine_period]

    def __repr__(self) -> str:
        return f'SinSched{self.mean}:{self.amplitude}:{self.sine_period}'


class SawToothScheduler(EnvScheduler):
    def __init__(
        self, 
        min_val: Dict[str, float], 
        max_val: Dict[str, float], 
        deployment_period: int,
        saw_tooth_period: int
    ) -> None:
        super().__init__(list(min_val.keys()), period=deployment_period)
        self.min_val = min_val
        self.max_val = max_val
        self.saw_tooth_period = saw_tooth_period
        self.possible_hips = {  # lots of jnp.array here to prevent numerical differences between jax and python
            pname: [
                (jnp.array(self.min_val[pname]) + (jnp.array(self.max_val[pname]) - jnp.array(self.min_val[pname])) * jnp.array(i) / jnp.array(self.saw_tooth_period)).item()
                for i in range(self.saw_tooth_period)
                ] 
            for pname in self.parameter_names}

    @partial(jax.jit, static_argnums=(0, 1, 2))
    def _get_val_for_step(self, pname: str, episode_step: int, rng: chex.PRNGKey) -> float:  
        return self.min_val[pname] + (self.max_val[pname] - self.min_val[pname])  * (episode_step % self.saw_tooth_period) / self.saw_tooth_period

    def __repr__(self) -> str:
        return f'SawSched{self.min_val}:{self.max_val}:{self.saw_tooth_period}'


class NoisyFlipFlopScheduler(EnvScheduler):
    def __init__(
        self, 
        neg_val: Dict[str, float],
        pos_val: Dict[str, float], 
        noise_rate: float,
        deployment_period: int,
    ) -> None:
        super().__init__(list(neg_val.keys()), period=deployment_period)
        self.neg_val = neg_val
        self.pos_val = pos_val
        assert len(pos_val) == len(neg_val) == 1, 'pos_val and neg_val must have length 1'
        self.noise_rate = noise_rate
        self.possible_hips = {pname: [self.neg_val[pname], self.pos_val[pname]] for pname in self.parameter_names}

    @partial(jax.jit, static_argnums=(0, 1, 2))
    def _get_val_for_step(self, pname: str, episode_step: int, rng: chex.PRNGKey) -> float:  
        return_random = jax.random.uniform(rng, shape=(), minval=0, maxval=1) < self.noise_rate
        random_hip = jax.random.choice(rng, jnp.array([self.neg_val[pname], self.pos_val[pname]]))
        expected_hip = jnp.where(episode_step % 2 == 0, self.neg_val[pname], self.pos_val[pname])
        new_hip = jnp.where(return_random, random_hip, expected_hip)
        return new_hip


    def __repr__(self) -> str:
        return f'SawSched{self.min_val}:{self.max_val}:{self.saw_tooth_period}'


class SawToothIntScheduler(EnvScheduler):
    def __init__(
        self, 
        min_val: Dict[str, int], 
        max_val: Dict[str, int], 
        deployment_period: int,
        saw_tooth_period: int
    ) -> None:
        super().__init__(list(min_val.keys()), period=deployment_period)
        self.min_val = min_val
        self.max_val = max_val
        self.saw_tooth_period = saw_tooth_period
        self.possible_hips = {pname: [self.min_val[pname] + (self.max_val[pname] - self.min_val[pname]) * i // self.saw_tooth_period for i in range(self.saw_tooth_period)] for pname in self.parameter_names}

    @partial(jax.jit, static_argnums=(0, 1, 2))
    def _get_val_for_step(self, pname: str, episode_step: int, rng: chex.PRNGKey) -> int:  
        return self.min_val[pname] + (self.max_val[pname] - self.min_val[pname])  * (episode_step % self.saw_tooth_period) // self.saw_tooth_period

    def __repr__(self) -> str:
        return f'SawSched{self.min_val}:{self.max_val}:{self.saw_tooth_period}'


class TriangleScheduler(EnvScheduler):
    def __init__(
        self, 
        min_val: Dict[str, float], 
        max_val: Dict[str, float], 
        deployment_period: int,
        triangle_period: int
    ) -> None:
        super().__init__(list(min_val.keys()), period=deployment_period)
        assert triangle_period % 4 == 0, 'triangle_period must be a multiple of 4'
        self.mean = {}
        self.amplitude = {}
        for param_name in self.parameter_names:
            self.mean[param_name] = 0.5 * (max_val[param_name] + min_val[param_name])
            self.amplitude[param_name] = 0.5 * (max_val[param_name] - min_val[param_name])
        self.triangle_period = triangle_period

        # to avoid many different numerical sin values, just take the first quarter period and mirror and negate it
        triangle_values_first_quarter = jnp.linspace(0, 1, self.triangle_period // 4 + 1)
        triangle_half = jnp.concatenate(
            [triangle_values_first_quarter, triangle_values_first_quarter[-2::-1]]
        )
        triangle_values_full = jnp.concatenate(
            [triangle_half, -triangle_half[1:-1]]
        )

        self.possible_hips = {}
        for pname in self.parameter_names:
            self.possible_hips[pname] = jnp.unique(self.mean[pname] + self.amplitude[pname] * triangle_values_full)

    @partial(jax.jit, static_argnums=(0, 1, 2))
    def _get_val_for_step(self, pname: str, episode_step: int, rng: chex.PRNGKey) -> float:  
        triangle_values_first_quarter = jnp.linspace(0, 1, self.triangle_period // 4 + 1)
        triangle_half = jnp.concatenate(
            [triangle_values_first_quarter, triangle_values_first_quarter[-2::-1]]
        )
        triangle_values_full = jnp.concatenate(
            [triangle_half, -triangle_half[1:-1]]
        )

        return self.mean[pname] + self.amplitude[pname] * triangle_values_full[episode_step % self.triangle_period]

    def __repr__(self) -> str:
        return f'TriangleSched{self.mean}:{self.amplitude}:{self.triangle_period}'


class UniformSamplingScheduler(EnvScheduler):
    def __init__(
        self, 
        min_val: Dict[str, float], 
        max_val: Dict[str, float], 
    ) -> None:
        super().__init__(list(max_val.keys()), None)
        self.min_val = min_val
        self.max_val = max_val

    @partial(jax.jit, static_argnums=(0, 1, 2))
    def _get_val_for_step(self, pname: str, episode_step: int, rng: chex.PRNGKey) -> float:  
        return jax.random.uniform(rng, minval=self.min_val[pname], maxval=self.max_val[pname])

    def __repr__(self) -> str:
        return 'UniSched{}:{}'.format(
            ';'.join([f'{v:.2f}' for v in self.min_val]),
            ';'.join([f'{v:.2f}' for v in self.max_val])
        )

    def get_hip_mean(self, pname: str) -> float:
        return 0.5 * (self.max_val[pname] + self.min_val[pname])

    def get_hip_std(self, pname: str) -> float:
        return np.sqrt(1/12) * (self.max_val[pname] + self.min_val[pname])


class ListScheduler(EnvScheduler):
    def __init__(
        self, 
        value_tuples: Dict[str, Iterable[chex.Array]],
    ) -> None:
        super().__init__(list(value_tuples.keys()), period=len(next(iter((value_tuples.values())))))
        self.value_tuples = value_tuples
        self.possible_hips = {pname: list(value_tuples[pname]) for pname in self.parameter_names}
    
    @partial(jax.jit, static_argnums=(0, 1, 2))
    def _get_val_for_step(self, pname: str, episode_step: int, rng: chex.PRNGKey) -> float:  
        return self.value_tuples[pname][episode_step % self.period]

    def __repr__(self) -> str:
        return f'ListSched{self.value_tuples[0]}:{self.value_tuples[-1]}'

