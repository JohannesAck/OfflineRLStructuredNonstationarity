"""
Here to prevent circular imports
"""

from gymnax.environments.environment import EnvParams
from flax import struct


@struct.dataclass
class ParameterizedEnvParams(EnvParams):
    episode_idx: int = 0
