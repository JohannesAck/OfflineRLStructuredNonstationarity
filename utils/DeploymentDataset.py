"""
Implementation of a dataset that records multiple deployments of an agent.

Each deployment consists of multiple episodes with evolving hidden-parameter following 
some dynamics depending on the environment.

Notes:

Implement as single dataclass of shape (obs,act,rew,obs+1,hip,envstate...) and have those shaped as 
[deployment][episode][transition]?

Alternatively: Implement as deployment
"""
from functools import partial
import sys

from typing import Iterable, List, Tuple, Union
import einops
import jax
import jax.numpy as jnp
import numpy as np
import chex
from flax import struct
from gymnax.environments.environment import EnvState
from custom_env.utils import ParameterizedEnvParams
from utils.helpers import np_or_jnp_array

@struct.dataclass
class Transition:
    obs: np_or_jnp_array
    act: np_or_jnp_array
    rew: np_or_jnp_array
    next_obs: np_or_jnp_array
    done: np_or_jnp_array
    state_before: EnvState
    hip: np_or_jnp_array
    episode_step: np_or_jnp_array
    deployment_step: np_or_jnp_array
    deployment_id: np_or_jnp_array
    env_params: ParameterizedEnvParams

@struct.dataclass
class DoubleTrans:
    transition_1: Transition
    transition_2: Transition

@struct.dataclass
class Dataset:
    transition: Transition
    n_deployments: int
    n_episodes_deployment: int
    n_timesteps_episode: int
    obs_shape: Tuple[int, ...]
    act_shape: Tuple[int, ...]

@struct.dataclass
class MinimalTransition:
    obs: np_or_jnp_array
    act: np_or_jnp_array
    rew: np_or_jnp_array
    next_obs: np_or_jnp_array
    done: np_or_jnp_array
    hip: np_or_jnp_array


class DeploymentDataset(object):

    def __init__(
            self,
            n_deployments: int,
            n_episodes_deployment: int,
            n_timesteps_episode: int, 
            obs_shape: Tuple[int,...],
            act_shape: Tuple[int,...],
            env_state: EnvState,
            env_params: ParameterizedEnvParams,
            record_hip_as_state: bool = False,
            on_cpu: bool = False,
        ) -> None:
        """
        Initialize a dataset that records multiple deployments of an agent.
        Only use the self.ds element, which is a Dataset object. All methods are static for jax compatibility,
        so this class really just collects them together.

        env_prams and env_state have to be passed in either in the correct shape for n_deployments, n_episode_deployment, n_timesteps_episode or
        in the shape for a single deployment, episode, timestep. In the latter case, they are automatically tiled to the correct shape.
        """
        self.on_cpu = on_cpu
        def tile_maintain_last_shape(x: chex.Array, tiling: Tuple[int,...]) -> chex.Array:
            # tiling adds an additional dimension to the end of x if x is a scalar, so we test for and remove this
            tiled = np.tile(np.array(x)[None, None, None], tiling)
            if len(x.shape) == 0:
                tiled = np.squeeze(tiled, axis=-1)
            return tiled

        if record_hip_as_state:
            env_state = np.zeros((n_deployments, n_episodes_deployment, n_timesteps_episode))
        else:
            if isinstance(env_state.time, int) or len(env_state.time.shape) == 0:
                env_state = jax.tree_map(lambda x: tile_maintain_last_shape(np.array(x), (n_deployments, n_episodes_deployment, n_timesteps_episode, 1)), env_state)
        
        if isinstance(env_params.episode_idx, int) or len(env_params.episode_idx.shape) == 0:
            env_params = jax.tree_map(lambda x: tile_maintain_last_shape(np.array(x), (n_deployments, n_episodes_deployment, n_timesteps_episode, 1)), env_params)

        transition = Transition(
            obs=np.zeros((n_deployments, n_episodes_deployment, n_timesteps_episode, *obs_shape), dtype=np.float32),
            act=np.zeros((n_deployments, n_episodes_deployment, n_timesteps_episode, *act_shape), dtype=np.float32),
            rew=np.zeros((n_deployments, n_episodes_deployment, n_timesteps_episode), dtype=np.float32),
            next_obs=np.zeros((n_deployments, n_episodes_deployment, n_timesteps_episode, *obs_shape), dtype=np.float32),
            done=np.zeros((n_deployments, n_episodes_deployment, n_timesteps_episode), dtype=bool),
            state_before=env_state,
            hip=np.zeros((n_deployments, n_episodes_deployment, n_timesteps_episode), dtype=np.float32),
            episode_step=np.zeros((n_deployments, n_episodes_deployment, n_timesteps_episode), dtype=np.int32),
            deployment_step=np.zeros((n_deployments, n_episodes_deployment, n_timesteps_episode), dtype=np.int32),
            deployment_id=np.zeros((n_deployments, n_episodes_deployment, n_timesteps_episode), dtype=np.int32),
            env_params=env_params,
        )
        self.ds = Dataset(
            transition=transition,
            n_deployments=n_deployments,
            n_episodes_deployment=n_episodes_deployment,
            n_timesteps_episode=n_timesteps_episode,
            obs_shape=obs_shape,
            act_shape=act_shape,
        )
        if not on_cpu:
            self.ds = jax.tree_map(lambda x: jnp.asarray(x), self.ds)

    @staticmethod
    def add_deployment(dataset: Dataset, transition: Transition, deployment_id: int) -> Dataset:
        new_transitions = jax.tree_map(lambda x, y: x.at[deployment_id].set(y), dataset.transition, transition)
        return dataset.replace(transition=new_transitions)

    def add_deployments(self, dataset: Dataset, new_deployments: Transition, deployment_ids: List[int]) -> Dataset:
        """
        Set a list of deployments, assuming transition is formatted such that the zeroth axis corresponds to deployment ids
        """
        assert new_deployments.obs.shape[0] == len(deployment_ids)
        ds_trans = dataset.transition
        if self.on_cpu:
            def assign(x, y):
                x[deployment_ids] = y
                return x
            ds_trans = jax.tree_map(assign, ds_trans, new_deployments)
        else:
            for idx, deployment_id in enumerate(deployment_ids):
                ds_trans = jax.tree_map(lambda x, y: x.at[deployment_id].set(y[idx]), ds_trans, new_deployments)
        return dataset.replace(transition=ds_trans)
    
    @staticmethod
    def sample_deployment_batch(
            transition: Transition,
            n_deploy_sampled: int,
            n_deployments: int,
            episodes_per_deployment: int,
            steps_per_episode: int,
            rng: chex.PRNGKey,
            replace: bool = False,
            double_transition: bool = False
        ) -> Union[Transition, DoubleTrans]:
        """
        Samples a batch of deployments, then an episode from each deployment and a timestep from this episode.
        If the double_transition flag is set, two transitions are sampled for each deployment and a DoubleTrans object is returned.
        """
        rng1, rng2, rng3 = jax.random.split(rng, 3)
        sampled_deployments = jax.random.choice(rng1, n_deployments, shape=(n_deploy_sampled,), replace=replace)
        sampled_episodes = jax.random.choice(rng2, episodes_per_deployment, shape=(n_deploy_sampled,), replace=True)
    
        if double_transition:
            sampled_timesteps = jax.random.choice(rng3, steps_per_episode, shape=(2,), replace=True)
            sampled_timesteps = jnp.tile(sampled_timesteps, (n_deploy_sampled, 1))
            transition_1 = jax.tree_map(lambda x: x[sampled_deployments, sampled_episodes, sampled_timesteps[:,0]], transition)
            transition_2 = jax.tree_map(lambda x: x[sampled_deployments, sampled_episodes, sampled_timesteps[:,1]], transition)
            return DoubleTrans(transition_1=transition_1, transition_2=transition_2) 
        else:
            sampled_timesteps = jax.random.choice(rng3, steps_per_episode, shape=(n_deploy_sampled,), replace=True)
            transition = jax.tree_map(lambda x: x[sampled_deployments, sampled_episodes, sampled_timesteps], transition)
            return transition


    @staticmethod
    def sample_full_episodes(
        transition: Transition,
        n_episodes_sampled: int,
        n_deployments: int,
        episodes_per_deployment: int,
        rng: chex.PRNGKey,
        replace: bool = False,
    ) -> Transition:
        """
        Returns episodes, such that each for each sampled episode all episode steps are included.
        Episodes are sampled uniformly from deployments and episodes per deployment.
        The indexing of the returned array is [episode, step in episode, obs_shape(e.g.)]
        """
        rng1, rng2, rng3 = jax.random.split(rng, 3)
        sampled_deployments = jax.random.choice(rng1, n_deployments, shape=(n_episodes_sampled,), replace=replace)
        sampled_episodes = jax.random.choice(rng2, episodes_per_deployment, shape=(n_episodes_sampled,), replace=True)
    
        transition = jax.tree_map(lambda x: x[sampled_deployments, sampled_episodes], transition)
        return transition


    @staticmethod
    # @jax.disable_jit()
    @partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 6, 8, 9))
    def sample_contrastive_batch_cpc(
        dataset: Dataset, batch_size: int, context_len: int, pred_offset: int, 
        n_deployments: int, episodes_per_deployment: int, steps_per_episode: int, rng: chex.PRNGKey, random_timesteps: bool = False, replace: bool = False
    ) -> Tuple[Transition, Transition]:
        """
        Return a batch for training of contrastive predictive coding.

        Returns 
        1) context-batch transitions, consiting of multiple partial trajectories of transitions, i.e.[batchsize, context_len]
        2) sample batch transitions, consisting of 4 x (s,a,r,s') [batchsize]
        
        context_len here refers to number of context episodes, and we only return one randomly sampled transition from each episode
        offset is the distance between the context given and the sample to classify as positive or negative

        """
        assert pred_offset > 0, "pred_offset has to be > 0, as pred_offset = 1 means the next step is predicted"
        rng1, rng2, rng3 = jax.random.split(rng, 3)
        sampled_deployments = jax.random.choice(rng1, n_deployments, shape=(batch_size,), replace=replace)
        sampled_deploy_steps = jax.random.randint(
            rng2, (batch_size,), 
            minval=0, maxval=episodes_per_deployment - context_len - pred_offset + 1
        )
        # refers to the start of the context
        
        #sampled time-step/mdp-step:
        sampled_timestep = jax.random.randint(rng3, (1,), minval=0, maxval=steps_per_episode)[0]

        trans_timestep_filtered = jax.tree_map(lambda x: x[:,:,sampled_timestep], dataset.transition)

        def sample_transitions(x, sampled_deployment_id, sampled_deploy_step, context_len):
            x = x[sampled_deployment_id]
            x = jax.lax.dynamic_slice_in_dim(x, sampled_deploy_step, context_len, axis=0)
            return x

        vmapped_sample = jax.vmap(sample_transitions, in_axes=(None, 0, 0, None))

        context_batch = jax.tree_map(
            lambda x: vmapped_sample(x, sampled_deployments, sampled_deploy_steps, context_len),
            trans_timestep_filtered
        )

        positive_sample_batch = jax.tree_map(
            lambda x: x[sampled_deployments, sampled_deploy_steps + context_len + pred_offset - 1],
            trans_timestep_filtered
        )

        return context_batch, positive_sample_batch

    @staticmethod
    @partial(jax.jit, static_argnames = ('batch_size', 'context_len', 'pred_offset', 'n_deployments', 'episodes_per_deployment', 'steps_per_episode', 'random_timesteps', 'replace'))
    def sample_contrastive_batch_faster(
        transition: MinimalTransition, batch_size: int, context_len: int, pred_offset: int, 
        n_deployments: int, episodes_per_deployment: int, steps_per_episode: int, rng: chex.PRNGKey, random_timesteps: bool = False, replace: bool = False
    ) -> Tuple[Transition, Transition]:
        """
        same as the other one, but without using the whole Transition dataclass, so it's faster hopefully
        """
        assert pred_offset > 0, "pred_offset has to be > 0, as pred_offset = 1 means the next step is predicted"
        rng1, rng2, rng3 = jax.random.split(rng, 3)
        sampled_deployments = jax.random.choice(rng1, n_deployments, shape=(batch_size,), replace=replace)
        sampled_deploy_steps = jax.random.randint(
            rng2, (batch_size,), 
            minval=0, maxval=episodes_per_deployment - context_len - pred_offset + 1
        )
        # refers to the start of the context
        
        #sampled time-step/mdp-step:
        sampled_timestep = jax.random.randint(rng3, (1,), minval=0, maxval=steps_per_episode)[0]

        # trans_timestep_filtered = jax.tree_map(lambda x: x[:,:,sampled_timestep], dataset.transition)


        def sample_transitions(x, sampled_deployment_id, sampled_deploy_step, context_len):
            x = x[sampled_deployment_id]
            x = jax.lax.dynamic_slice_in_dim(x, sampled_deploy_step, context_len, axis=0)
            return x

        vmapped_sample = jax.vmap(sample_transitions, in_axes=(None, 0, 0, None))

        context_batch = jax.tree_map(
            lambda x: vmapped_sample(x[:,:,sampled_timestep], sampled_deployments, sampled_deploy_steps, context_len),
            transition
        )

        positive_sample_batch = jax.tree_map(
            lambda x: x[:,:,sampled_timestep][sampled_deployments, sampled_deploy_steps + context_len + pred_offset - 1],
            transition
        )
        return context_batch, positive_sample_batch

    @staticmethod
    # @partial(jax.jit, static_argnames = ('batch_size', 'context_len', 'pred_offset', 'n_deployments', 'episodes_per_deployment', 'steps_per_episode', 'random_timesteps', 'replace'))
    def sample_contrastive_batch_negatives(
        transition: MinimalTransition, batch_size: int, context_len: int, pred_offset: int, 
        n_deployments: int, episodes_per_deployment: int, steps_per_episode: int, rng: chex.PRNGKey, random_timesteps: bool = False, replace: bool = False
    ) -> Tuple[Transition, Transition, Transition]:
        """
        Actually return negative samples, the other method should be used when just shuffling postive samples and using them  as negatives
        """
        assert pred_offset > 0, "pred_offset has to be > 0, as pred_offset = 1 means the next step is predicted"
        rng1, rng2, rng3 = jax.random.split(rng, 3)
        sampled_deployments = jax.random.choice(rng1, n_deployments, shape=(batch_size,), replace=replace)
        sampled_deploy_steps = jax.random.randint(
            rng2, (batch_size,), 
            minval=0, maxval=episodes_per_deployment - context_len - pred_offset + 1
        )
        # refers to the start of the context
        
        #sampled time-step/mdp-step:
        sampled_timestep = jax.random.randint(rng3, (1,), minval=0, maxval=steps_per_episode)[0]

        # trans_timestep_filtered = jax.tree_map(lambda x: x[:, :, sampled_timestep], dataset.transition)


        def sample_transitions(x, sampled_deployment_id, sampled_deploy_step, context_len):
            x = x[sampled_deployment_id]
            x = jax.lax.dynamic_slice_in_dim(x, sampled_deploy_step, context_len, axis=0)
            return x

        vmapped_sample = jax.vmap(sample_transitions, in_axes=(None, 0, 0, None))

        context_batch = jax.tree_map(
            lambda x: vmapped_sample(x[:,:,sampled_timestep], sampled_deployments, sampled_deploy_steps, context_len),
            transition
        )
        positive_sample_batch = jax.tree_map(
            lambda x: x[:,:,sampled_timestep][sampled_deployments, sampled_deploy_steps + context_len + pred_offset - 1],
            transition
        )
        raise NotImplementedError()

        # negative_sample_batch = jax.tree_map(
        # )
        return context_batch, positive_sample_batch, negative_sample_batch


    @staticmethod
    def sample_contrastive_evaluation_batch_cpc(
        dataset: Dataset, batch_size: int, context_len: int, pred_offset: int, 
        n_deployments: int, episodes_per_deployment: int, steps_per_episode: int, n_timestep_samples: int, n_deploystep_samples: int,
        rng: chex.PRNGKey
    ) -> Tuple[Transition, Transition]:
        """
        Similar to sample_contrastive_batch_cpc, but samples multiple deploy-steps and time-steps for each deployment and episode.
        """
        assert pred_offset > 0, "pred_offset has to be > 0, as pred_offset = 1 means the next step is predicted"
        rng, subkey = jax.random.split(rng)
        sampled_deployments = jax.random.choice(subkey, n_deployments, shape=(batch_size,), replace=False)
        
        context_batch, positive_sample_batch = DeploymentDataset.sample_given_deployments_for_cpc(
            dataset,
            sampled_deployments,
            context_len,
            pred_offset, 
            episodes_per_deployment,
            steps_per_episode,
            n_timestep_samples,
            n_deploystep_samples,
            rng
        )
        context_batch = jax.tree_map(jnp.concatenate, context_batch)
        positive_sample_batch = jax.tree_map(jnp.concatenate, positive_sample_batch)


        return context_batch, positive_sample_batch

    @staticmethod
    def sample_given_deployments_for_cpc(
        dataset: Dataset, deployment_ids: Iterable[int], context_len: int, pred_offset: int, 
        episodes_per_deployment: int, steps_per_episode: int, n_timestep_samples: int, n_deploystep_samples: int,
        rng: chex.PRNGKey
    ) -> Tuple[Transition, Transition]:
        """
        Similar to sample_contrastive_batch_cpc, but samples multiple deploy-steps and time-steps for each deployment and episode.
        """
        assert pred_offset > 0, "pred_offset has to be > 0, as pred_offset = 1 means the next step is predicted"

        rng1, rng2 = jax.random.split(rng, 2)
        if n_deploystep_samples == episodes_per_deployment:
            sampled_deploy_steps = jnp.arange(episodes_per_deployment - context_len - pred_offset + 1)
        else:
            sampled_deploy_steps = jax.random.choice(
                rng1, episodes_per_deployment - context_len - pred_offset + 1, shape=(n_deploystep_samples,), replace=False
            )
        # refers to the start of the context
        
        #sampled time-step/mdp-step:
        if n_timestep_samples == steps_per_episode:
            sampled_timesteps = jnp.arange(steps_per_episode)
        else:
            sampled_timesteps = jax.random.randint(rng2, (n_timestep_samples,), minval=0, maxval=steps_per_episode)

        def sample_transitions(x, deployment_ids, sampled_deploy_step, context_len, sampled_timesteps):
            x = x[deployment_ids][:,:,sampled_timesteps]  # this line could be done outside vmap in advance
            x = jax.lax.dynamic_slice_in_dim(x, sampled_deploy_step, context_len, axis=1)
            return x

        vmapped_sample = jax.vmap(sample_transitions, in_axes=(None, None, 0, None, None))

        context_batch = jax.tree_map(
            lambda x: vmapped_sample(x, deployment_ids, sampled_deploy_steps, context_len, sampled_timesteps),
            dataset.transition
        )

        positive_sample_batch = jax.tree_map(
            lambda x: vmapped_sample(x, deployment_ids, sampled_deploy_steps + context_len, 1, sampled_timesteps),
            dataset.transition
        )

        return context_batch, positive_sample_batch
