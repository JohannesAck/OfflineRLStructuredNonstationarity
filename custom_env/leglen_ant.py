"""
Copied from brax.ent.ant and pipelinenv

Modified to include the leg length as parameter for constructor.
"""

from functools import partial
from typing import Any, Tuple
from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax

from brax.generalized import pipeline as g_pipeline
from brax.positional import pipeline as p_pipeline
from brax.spring import pipeline as s_pipeline

from jax import numpy as jnp



def get_ant_xml(leglen: float = 0.2) -> str:
    extf_ant_xml = f"""
    <mujoco model="ant">
    <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
    <option timestep="0.01" iterations="4" />
    <custom>
        <!-- brax custom params -->
        <numeric data="0.0 0.0 0.55 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0" name="init_qpos"/>
        <numeric data="1000" name="constraint_limit_stiffness"/>
        <numeric data="4000" name="constraint_stiffness"/>
        <numeric data="10" name="constraint_ang_damping"/>
        <numeric data="20" name="constraint_vel_damping"/>
        <numeric data="0.5" name="joint_scale_pos"/>
        <numeric data="0.2" name="joint_scale_ang"/>
        <numeric data="0.0" name="ang_damping"/>
        <numeric data="1" name="spring_mass_scale"/>
        <numeric data="1" name="spring_inertia_scale"/>
        <numeric data="15" name="solver_maxls"/>
    </custom>
    <default>
        <joint armature="1" damping="1" limited="true"/>
        <geom contype="0" conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5"/>
    </default>
    <asset>
        <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
        <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
    </asset>
    <worldbody>
        <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
        <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" size="40 40 40" type="plane"/>
        <body name="torso" pos="0 0 0.75">
        <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
        <geom name="torso_geom" pos="0 0 0" size="0.25" type="sphere"/>
        <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" type="free"/>
        <body name="front_left_leg" pos="0 0 0">
            <geom fromto="0.0 0.0 0.0 {leglen:.1f} {leglen:.1f} 0.0" name="aux_1_geom" size="0.08" type="capsule"/>
            <body name="aux_1" pos="{leglen:.1f} {leglen:.1f} 0">
            <joint axis="0 0 1" name="hip_1" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 {leglen:.1f} {leglen:.1f} 0.0" name="left_leg_geom" size="0.08" type="capsule"/>
            <body pos="{leglen:.1f} {leglen:.1f} 0">
                <joint axis="-1 1 0" name="ankle_1" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
                <geom fromto="0.0 0.0 0.0 {2*leglen:.1f} {2*leglen:.1f} 0.0" name="left_ankle_geom" size="0.08" type="capsule"/>
                <geom name="left_foot_geom" contype="1" pos="{2*leglen:.1f} {2*leglen:.1f} 0" size="0.08" type="sphere" mass="0"/>
            </body>
            </body>
        </body>
        <body name="front_right_leg" pos="0 0 0">
            <geom fromto="0.0 0.0 0.0 -{leglen:.1f} {leglen:.1f} 0.0" name="aux_2_geom" size="0.08" type="capsule"/>
            <body name="aux_2" pos="-{leglen:.1f} {leglen:.1f} 0">
            <joint axis="0 0 1" name="hip_2" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -{leglen:.1f} {leglen:.1f} 0.0" name="right_leg_geom" size="0.08" type="capsule"/>
            <body pos="-{leglen:.1f} {leglen:.1f} 0">
                <joint axis="1 1 0" name="ankle_2" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
                <geom fromto="0.0 0.0 0.0 -{2*leglen:.1f} {2*leglen:.1f} 0.0" name="right_ankle_geom" size="0.08" type="capsule"/>
                <geom name="right_foot_geom" contype="1" pos="-{2*leglen:.1f} {2*leglen:.1f} 0" size="0.08" type="sphere" mass="0"/>
            </body>
            </body>
        </body>
        <body name="back_leg" pos="0 0 0">
            <geom fromto="0.0 0.0 0.0 -{leglen:.1f} -{leglen:.1f} 0.0" name="aux_3_geom" size="0.08" type="capsule"/>
            <body name="aux_3" pos="-{leglen:.1f} -{leglen:.1f} 0">
            <joint axis="0 0 1" name="hip_3" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -{leglen:.1f} -{leglen:.1f} 0.0" name="back_leg_geom" size="0.08" type="capsule"/>
            <body pos="-{leglen:.1f} -{leglen:.1f} 0">
                <joint axis="-1 1 0" name="ankle_3" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
                <geom fromto="0.0 0.0 0.0 -{2*leglen:.1f} -{2*leglen:.1f} 0.0" name="third_ankle_geom" size="0.08" type="capsule"/>
                <geom name="third_foot_geom" contype="1" pos="-{2*leglen:.1f} -{2*leglen:.1f} 0" size="0.08" type="sphere" mass="0"/>
            </body>
            </body>
        </body>
        <body name="right_back_leg" pos="0 0 0">
            <geom fromto="0.0 0.0 0.0 {leglen:.1f} -{leglen:.1f} 0.0" name="aux_4_geom" size="0.08" type="capsule"/>
            <body name="aux_4" pos="{leglen:.1f} -{leglen:.1f} 0">
            <joint axis="0 0 1" name="hip_4" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 {leglen:.1f} -{leglen:.1f} 0.0" name="rightback_leg_geom" size="0.08" type="capsule"/>
            <body pos="{leglen:.1f} -{leglen:.1f} 0">
                <joint axis="1 1 0" name="ankle_4" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
                <geom fromto="0.0 0.0 0.0 {2*leglen:.1f} -{2*leglen:.1f} 0.0" name="fourth_ankle_geom" size="0.08" type="capsule"/>
                <geom name="fourth_foot_geom" contype="1" pos="{2*leglen:.1f} -{2*leglen:.1f} 0" size="0.08" type="sphere" mass="0"/>
            </body>
            </body>
        </body>
        </body>
    </worldbody>
    <actuator>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_4" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_4" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_1" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_1" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_2" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_2" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_3" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_3" gear="150"/>
    </actuator>
    </mujoco>
    """
    return extf_ant_xml

class FrozenDict(dict):
    def __init__(self, *args, **kwargs):
        self._hash = None
        super(FrozenDict, self).__init__(*args, **kwargs)

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(tuple(sorted(self.items())))  # iteritems() on py2
        return self._hash

    def _immutable(self, *args, **kws):
        raise TypeError('cannot change object - object is immutable')

    # makes (deep)copy alot more efficient
    def __copy__(self):
        return self

    def __deepcopy__(self, memo=None):
        if memo is not None:
            memo[id(self)] = self
        return self

    __setitem__ = _immutable
    __delitem__ = _immutable
    pop = _immutable
    popitem = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable

class BraxAntLeglen(PipelineEnv):
    """
    Bastardized combination of brax.envs.ant and brax.envs.base.pipelinenv.
    """

    def __init__(
            self,
            possible_leg_lengths: Tuple[float,...] = (0.15, 0.2, 0.25),
            ctrl_cost_weight=0.5,
            use_contact_forces=False,
            contact_cost_weight=5e-4,
            healthy_reward=1.0,
            terminate_when_unhealthy=True,
            healthy_z_range=(0.2, 1.0),
            contact_force_range=(-1.0, 1.0),
            reset_noise_scale=0.1,
            exclude_current_positions_from_observation=True,
            backend='generalized',
            **kwargs,
    ):

        self.possible_leg_lengths = possible_leg_lengths
        self._pipeline = {
            'generalized': g_pipeline,
            'spring': s_pipeline,
            'positional': p_pipeline,
        }[backend]

        config_list = []
        for idx, leg_length in enumerate(self.possible_leg_lengths):
            sys = mjcf.loads(get_ant_xml(leglen=leg_length))

            n_frames = 5

            if backend in ['spring', 'positional']:
                sys = sys.replace(dt=0.005)
                n_frames = 10
            if backend == 'positional':
                # TODO: does the same actuator strength work as in spring
                sys = sys.replace(
                        actuator=sys.actuator.replace(
                                gear=200 * jnp.ones_like(sys.actuator.gear)
                        )
                )
            kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

            self._n_frames = n_frames
            config_list.append(sys)
            # super().__init__(sys=sys, backend=backend, **kwargs)
        self.configs_stacked = jax.tree_map(lambda x, *y: jnp.stack([*y]), config_list[0], *config_list)


        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
                exclude_current_positions_from_observation
        )

        if self._use_contact_forces:
            raise NotImplementedError('use_contact_forces not implemented.')

    @partial(jax.jit, static_argnames=('self', 'config_id'))
    def pipeline_init(self, q: jnp.ndarray, qd: jnp.ndarray, config_id: float = 0.2) -> base.State:
        """Initializes the pipeline state."""
        sys = jax.tree_map(lambda x: x[config_id], self.configs_stacked)
        return self._pipeline.init(sys, q, qd, debug=False)

    def reset(self, rng: jnp.ndarray, config_id: float) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        sys = jax.tree_map(lambda x: x[config_id], self.configs_stacked)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = sys.init_q + jax.random.uniform(
                rng1, (sys.q_size(),), minval=low, maxval=hi
        )
        qd = hi * jax.random.normal(rng2, (sys.qd_size(),))

        pipeline_state = self.pipeline_init(q, qd, config_id)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jnp.zeros(3)
        metrics = {
                'reward_forward': zero,
                'reward_survive': zero,
                'reward_ctrl': zero,
                'reward_contact': zero,
                'x_position': zero,
                'y_position': zero,
                'distance_from_origin': zero,
                'x_velocity': zero,
                'y_velocity': zero,
                'forward_reward': zero,
        }
        return State(pipeline_state, obs, reward, done, metrics)

    @partial(jax.jit, static_argnames=('self', 'config_id'))
    def pipeline_step(
        self, pipeline_state: Any, action: jnp.ndarray, config_id: float
    ) -> base.State:
        """Takes a physics step using the physics pipeline."""
        
        sys = jax.tree_map(lambda x: x[config_id], self.configs_stacked)
        def f(state, _):
            return (
                self._pipeline.step(sys, state, action, False),
                None,
            )

        return jax.lax.scan(f, pipeline_state, (), self._n_frames)[0]

    def step(self, state: State, action: jnp.ndarray, config_id: float) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action, config_id)

        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt   
        forward_reward = velocity[0]

        min_z, max_z = self._healthy_z_range
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] < min_z, x=0.0, y=1.0)
        is_healthy = jnp.where(
                pipeline_state.x.pos[0, 2] > max_z, x=0.0, y=is_healthy
        )
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))
        contact_cost = 0.0

        obs = self._get_obs(pipeline_state)
        reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        state.metrics.update(
                reward_forward=forward_reward,
                reward_survive=healthy_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                x_position=pipeline_state.x.pos[0, 0],
                y_position=pipeline_state.x.pos[0, 1],
                distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
                x_velocity=velocity[0],
                y_velocity=velocity[1],
                forward_reward=forward_reward,
        )
        return state.replace(
                pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State) -> jnp.ndarray:
        """Observe ant body position and velocities."""
        qpos = pipeline_state.q
        qvel = pipeline_state.qd

        if self._exclude_current_positions_from_observation:
            qpos = pipeline_state.q[2:]

        return jnp.concatenate([qpos] + [qvel])
    
    def get_observation_size(self) -> int:
        rng = jax.random.PRNGKey(0)
        reset_state = self.unwrapped.reset(rng, config_id=0)
        return reset_state.obs.shape[-1]
    
    def get_action_size(self) -> int:
        return jax.tree_map(lambda x: x[0], self.configs_stacked).act_size()

    @property
    def dt(self) -> jnp.ndarray:
        """The timestep used for each env step."""
        return jax.tree_map(lambda x: x[0], self.configs_stacked).dt * self._n_frames
