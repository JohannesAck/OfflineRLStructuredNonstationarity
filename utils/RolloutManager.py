import time
from functools import partial
from typing import Dict, Optional, Tuple, Union
import warnings

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.training.train_state import TrainState

from custom_env.environment_scheduler import DummyScheduler, EnvScheduler
from custom_env.hip_environment import HiPEnvironment, ParameterizedEnvParams
from representation_models import RepresentationTrainer
from representation_models.latent_predictor_model import PredictorTrainer
from utils.BufferManager import BufferManager
from RLMethods import RLTrainer
from utils.config import EvalMode, Methods, TrainConfig
from utils.DeploymentDataset import Transition
from utils.helpers import get_env


@struct.dataclass
class EvalParams:
    do_exploration: bool
    episodes_per_deployment: int
    method: Methods
    binarize_latent: bool
    repr_trainer: RepresentationTrainer
    config: TrainConfig
    pred_trainer: PredictorTrainer


@struct.dataclass
class EvalNormalizers:
    obs_mean: chex.Array
    obs_std: chex.Array
    hip_mean: float
    hip_std: float
    latent_mean: chex.Array
    latent_std: chex.Array
    rew_mean: float
    rew_std: float


class RolloutManager(object):
    def __init__(
        self,
        env_name: str,
        env_kwargs: dict,
        env_params: dict,
        exploration_noise_std: float,
        exploration_noise_clip: float,
        config: TrainConfig,
        rl_trainer: RLTrainer,
    ):
        # Setup functionalities for vectorized batch rollout
        self.env_name = env_name
        self.env, self.env_params = get_env(env_name, env_kwargs)
        self.env_params = self.env_params.replace(**env_params)
        self.observation_space = self.env.observation_space(self.env_params)
        self.action_size = self.env.action_space(self.env_params).shape
        # self.select_action = self.select_action_td3
        self.exploration_noise_std = exploration_noise_std
        self.exploration_noise_clip = exploration_noise_clip
        self.config = config
        self.rl_trainer = rl_trainer

    @partial(jax.jit, static_argnames=('self', 'exploration_noise'))
    def select_action(
        self,
        train_state: TrainState,
        obs: jnp.ndarray,
        rng: chex.PRNGKey,
        exploration_noise: bool = False,
    ) -> jnp.ndarray:
        rng_net, rng_noise = jax.random.split(rng)
        action = self.rl_trainer.get_action(train_state, obs, rng_net, self.config, exploration_noise=exploration_noise)

        if exploration_noise:
            noise = self.exploration_noise_std * self.env.action_space().high * jax.random.normal(rng_noise, action.shape)
            action = action + noise.clip(-self.exploration_noise_clip, self.exploration_noise_clip)

        action = action.clip(self.env.action_space().low, self.env.action_space().high)
        return action

    @partial(jax.jit, static_argnums=0)
    def batch_reset(self, keys):
        return jax.vmap(self.env.reset, in_axes=(0, None))(
            keys, self.env_params
        )

    @partial(jax.jit, static_argnums=0)
    def batch_step(self, keys, state, action):
        return jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            keys, state, action, self.env_params
        )


class HiPRolloutManager(RolloutManager):
    """
    Variant of the RolloutManager that also supports HiPEnvironments, i.e. environments with a hidden-parameter 
    that changes over time according to its scheduler.
    """
    def __init__(
            self,
            env_name: str,
            env_kwargs: dict,
            env_params: Union[dict,ParameterizedEnvParams],
            exploration_noise_std: float,
            exploration_noise_clip: float,
            config: TrainConfig,
            rl_trainer: RLTrainer,
            input_gt_hip: bool = False
        ):
        # Setup functionalities for vectorized batch rollout
        super().__init__(env_name, env_kwargs, env_params, exploration_noise_std, exploration_noise_clip, config, rl_trainer)
        self.env: HiPEnvironment
        self.env_params: ParameterizedEnvParams
        if not isinstance(self.env, HiPEnvironment):
            self.env.scheduler = DummyScheduler(self.env_params)
        self.init_env_params: ParameterizedEnvParams = self.env_params
        self.env_scheduler: EnvScheduler = self.env.scheduler
        if input_gt_hip:
            orig_observation_space = self.observation_space
            if np.isinf(orig_observation_space.low):
                extended_obs_space = type(self.observation_space)(
                    low=orig_observation_space.low,
                    high=orig_observation_space.high,
                    shape=(orig_observation_space.shape[0] + 1,)
                )
            else:
                if orig_observation_space.shape == orig_observation_space.low.shape:
                    extended_obs_space = type(self.observation_space)(
                        low=jnp.concatenate([orig_observation_space.low, orig_observation_space.low[-1:-0]]),
                        high=jnp.concatenate([orig_observation_space.high, orig_observation_space.high[-1][-1:-0]]),
                        shape=(orig_observation_space.shape[0] + 1,)
                    )
                else:  # means gymnax space
                    extended_obs_space = type(self.observation_space)(
                        low=orig_observation_space.low,
                        high=orig_observation_space.high,
                        shape=(orig_observation_space.shape[0] + 1,) 
                    )
            self.observation_space = extended_obs_space

    @partial(jax.jit, static_argnums=0)
    def batch_reset(self, keys, env_params: Optional[ParameterizedEnvParams] = None):
        reset = jax.vmap(self.env.reset, in_axes=(0, 0,))(
            keys, env_params
        )
        return jax.tree_map(jax.lax.stop_gradient, reset)


    @partial(jax.jit, static_argnums=0)
    def batch_step(self, keys, state, action, env_params):
        step = jax.vmap(self.env.step, in_axes=(0, 0, 0, 0))(
            keys, state, action, env_params
        )
        return jax.tree_map(jax.lax.stop_gradient, step)

    # @jax.disable_jit()
    # @partial(jax.jit, static_argnames=('self', 'num_envs', 'method', 'episodes_per_deployment', 'config', 'do_exploration', 'record_hip_only', 'repr_trainer', 'pred_trainer'))
    def batch_record_transitions(
        self,
        rng,
        train_state,
        starting_env_params: ParameterizedEnvParams,
        do_exploration: bool,
        num_envs: int,
        episodes_per_deployment: int,
        config: TrainConfig,
        method: Methods,
        repr_trainstate: Optional[TrainState] = None,
        record_hip_only: bool = False,
        obs_mean: Optional[chex.Array] = None,
        obs_std: Optional[chex.Array] = None,
        transitions: Transition = None,
        hip_mean: Optional[float] = None,
        hip_std: Optional[float] = None,
        rew_mean: Optional[float] = None,
        rew_std: Optional[float] = None,
        binarize_latent: bool = False,
        latent_mean: Optional[chex.Array] = None,
        latent_std: Optional[chex.Array] = None,
        repr_trainer: Optional[RepresentationTrainer] = None,
        pred_trainstate: Optional[TrainState] = None,
        pred_trainer: Optional[PredictorTrainer] = None,
    ) -> Transition:
        """Rollout an episode with lax.scan."""
        # Reset the environment
        if obs_mean is None:
            obs_mean = jnp.zeros(self.observation_space.shape)
        if obs_std is None:
            obs_std = jnp.ones(self.observation_space.shape)
        if hip_mean is None:
            hip_mean = 0.0
        if hip_std is None:
            hip_std = 1.0
        if latent_mean is None:
            latent_mean = 0.0
        if latent_std is None:
            latent_std = 1.0
        if rew_mean is None:
            rew_mean = 0.0
        if rew_std is None:
            rew_std = 1.0

        eval_params = EvalParams(do_exploration=do_exploration,
                                 method=method, binarize_latent=binarize_latent, episodes_per_deployment=episodes_per_deployment,
                                 repr_trainer=repr_trainer, config=config, pred_trainer=pred_trainer)
        eval_normalizers = EvalNormalizers(obs_mean=obs_mean, obs_std=obs_std,
                                           hip_mean=hip_mean, hip_std=hip_std,
                                           latent_mean=latent_mean, latent_std=latent_std,
                                            rew_mean=rew_mean, rew_std=rew_std
                                           )
        rng, rng_sched_reset = jax.random.split(rng)
        rng_sched_reset = jax.random.split(rng_sched_reset, num_envs)
        env_params = jax.vmap(self.env.scheduler.reset, in_axes=(0, None))(rng_sched_reset, starting_env_params)


        # first determine hip for each episode, then rollout deployments in parallel with vmap
        subkeys = jax.random.split(rng, num_envs)

        single_deploy_partial = partial(
            self.single_deployment,
            train_state=train_state, transitions=transitions, eval_params=eval_params,
            eval_normalizers=eval_normalizers, repr_trainstate=repr_trainstate, 
            pred_trainstate=pred_trainstate, record_transitions=True,
            record_hip_only=record_hip_only,
        )
        episode_returns, info, deployments = jax.vmap(single_deploy_partial)(
            initial_env_param=env_params, rng=subkeys
        )

        mean_return = np.mean(episode_returns)
        episode_len = deployments.obs.shape[2]
        deployments = deployments.replace(
            episode_step=jnp.tile(jnp.arange(episode_len)[None, None, :], [num_envs, episodes_per_deployment, 1]),
            deployment_step=jnp.tile(jnp.arange(episodes_per_deployment)[None, :,  None], [num_envs, 1, episode_len]),
        )
        return deployments

    def sample_closest_latent_by_hip(
        self,
        transitions: Transition,
        hip,
        obs_dim,
        rng:chex.PRNGKey,
    ) -> jnp.ndarray:
        """
        sample closest latent from buffer to hip out of a subsample of 1000 random samples.
        uses same sample for all envs in this batch
        """
        randomized_order = jax.random.permutation(rng, transitions.hip.shape[0])
        dists = jnp.abs(transitions.hip[randomized_order,0] - hip)
        closest_index = randomized_order[jnp.argmin(dists, axis=0)]
        return transitions.obs[closest_index, 0, obs_dim:]

    @partial(jax.jit, static_argnames=('self'))
    def sample_matching_episode_by_hip(
        self,
        transitions: Transition,
        hip,
        rng:chex.PRNGKey,
    ) -> Transition:
        """
        sample closest latent from buffer to hip out of a subsample of 1000 random samples.
        uses same sample for all envs in this batch
        """
        randomized_deploy_order = jax.random.permutation(rng, transitions.hip.shape[0])
        episode_hips = transitions.hip[randomized_deploy_order, :, 0]
        dists = jnp.abs(episode_hips - hip)

        closest_index = jnp.argmin(dists, keepdims=True)
        closest_deploy_index = randomized_deploy_order[closest_index[0]][0]
        closest_episode_idx = closest_index[1][0]

        matching_episode = jax.tree_map(lambda x: x[closest_deploy_index, closest_episode_idx], transitions)
        return matching_episode

    @partial(jax.jit, static_argnames=('self', 'context_length'))
    def sample_matching_context_episodes_by_next_hip(
            self,
            transitions: Transition,
            hip: float,
            context_length: int,
            rng: chex.PRNGKey
        ) -> Tuple[Transition, Transition]:
        """
        Sample multiple episodes to provide context for the next sample.
        They're sampled from the same deployment to avoid creating some incorrect hip-dynamics.

        Returned is a Transition object including `context_length` episodes prior to the episode with the given hip.
        Also the target episode is returned, for debugging.
        """
        randomized_deploy_order = jax.random.permutation(rng, transitions.hip.shape[0])
        episode_hips = transitions.hip[randomized_deploy_order, :, 0]
        dists = jnp.abs(episode_hips - hip)
        # set dists of episodes that don't have enough context to inf
        dists = dists.at[:, :context_length].set(jnp.inf)

        closest_index = jnp.argmin(dists, keepdims=True)
        closest_deploy_index = randomized_deploy_order[closest_index[0]][0]
        closest_episode_idx = closest_index[1][0]

        context_transitions = jax.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x[closest_deploy_index], closest_episode_idx - context_length, context_length, axis=0), transitions)

        target_transistion = jax.tree_map(lambda x: x[closest_deploy_index, closest_episode_idx], transitions)

        return context_transitions, target_transistion # for debugging


    # @jax.disable_jit()
    # @partial(jax.jit, static_argnames=('self', 'num_envs', 'episodes_per_deployment', 'method', 'do_exploration', 'binarize_latent', 'config', 'repr_trainer', 'pred_trainer'))
    def batch_evaluate(
        self,
        rng,
        train_state: TrainState,
        starting_env_params: ParameterizedEnvParams,
        do_exploration: bool,
        num_envs: int,
        episodes_per_deployment: int,
        method: Methods,
        config: TrainConfig,
        repr_trainstate: Optional[TrainState] = None,
        obs_mean: Optional[chex.Array] = None,
        obs_std: Optional[chex.Array] = None,
        transitions: Transition = None,
        hip_mean: Optional[float] = None,
        hip_std: Optional[float] = None,
        rew_mean: Optional[float] = None,
        rew_std: Optional[float] = None,
        binarize_latent: bool = False,
        latent_mean: Optional[chex.Array] = None,
        latent_std: Optional[chex.Array] = None,
        repr_trainer: Optional[RepresentationTrainer] = None,
        pred_trainstate: Optional[TrainState] = None,
        pred_trainer: Optional[PredictorTrainer] = None,
        dspol_trainer = None,  # debugging
        ds_pol_trainstate = None  # debugging
    ) -> Tuple[jax.Array, Dict[str, jax.Array], jax.Array, jax.Array]:
        """
        Evaluate in parallel, returns mean reward and info dict.
        """
        # Reset the environment
        if obs_mean is None:
            obs_mean = jnp.zeros(self.observation_space.shape)
        if obs_std is None:
            obs_std = jnp.ones(self.observation_space.shape)
        if hip_mean is None:
            hip_mean = 0.0
        if hip_std is None:
            hip_std = 1.0
        if latent_mean is None:
            latent_mean = 0.0
        if latent_std is None:
            latent_std = 1.0
        if rew_mean is None:
            rew_mean = 0.0
        if rew_std is None:
            rew_std = 1.0

        eval_params = EvalParams(do_exploration=do_exploration,
                                 method=method, binarize_latent=binarize_latent, episodes_per_deployment=episodes_per_deployment,
                                 repr_trainer=repr_trainer, config=config, pred_trainer=pred_trainer)
        eval_normalizers = EvalNormalizers(obs_mean=obs_mean, obs_std=obs_std,
                                           hip_mean=hip_mean, hip_std=hip_std,
                                           latent_mean=latent_mean, latent_std=latent_std,
                                             rew_mean=rew_mean, rew_std=rew_std
                                           )
        rng, rng_sched_reset = jax.random.split(rng)
        rng_sched_reset = jax.random.split(rng_sched_reset, num_envs)
        env_params = jax.vmap(self.env.scheduler.reset, in_axes=(0, None))(rng_sched_reset, starting_env_params)


        # first determine hip for each episode, then rollout deployments in parallel with vmap
        subkeys = jax.random.split(rng, num_envs)

        single_deploy_partial = partial(
            self.single_deployment,
            train_state=train_state, transitions=transitions, eval_params=eval_params,
            eval_normalizers=eval_normalizers, repr_trainstate=repr_trainstate, 
            pred_trainstate=pred_trainstate,
            record_transitions=True, record_hip_only=True
        )
        episode_returns, info, deployments = jax.vmap(single_deploy_partial)(
            initial_env_param=env_params, rng=subkeys
        )
        inferred_latents = None
        hips = deployments.hip[:,:,0]
        episode_returns = episode_returns[:,:,0]

        # mean by hip
        unique_hips = jnp.unique(hips)
        if len(unique_hips) < 10:
            hip_mean_returns = {f'hip={hip:.2f}': episode_returns[hips == hip].mean() for hip in unique_hips}
            info = {**info, **hip_mean_returns}
        else:
            warnings.warn('too many unique hips (>10), not averaging by hip')

        mean_return = jnp.mean(episode_returns)
        return mean_return, info, inferred_latents, hips

    @partial(jax.jit, static_argnames=('self', 'eval_params'))
    def determine_obs_augmentation(
        self,
        rng: chex.PRNGKey,
        env_params,
        transitions: Transition,
        eval_params: EvalParams,
        eval_normalizers: EvalNormalizers,
        repr_trainstate: TrainState,
        pred_trainstate: TrainState,
    ) -> jnp.ndarray:
        sample_obs, sample_state, _ = self.env.reset(rng, env_params)
        obs_dim = sample_obs.shape[-1]

        rng, rng_latent, rng_inference, rng_pred = jax.random.split(rng, 4)
        if eval_params.method == Methods.gt_hip:
            return jnp.array([(sample_state.hip - eval_normalizers.hip_mean) / eval_normalizers.hip_std])[None]
        elif eval_params.method == Methods.zero_hip:
            return jnp.array([])
        elif eval_params.method == Methods.noisy_hip:
            real_hip = sample_state.hip
            uniform_hip = jax.random.choice(rng_latent, list(self.env.scheduler.possible_hips.values())[0])
            noise_cond = jax.random.uniform(rng_inference, shape=(1,)) < eval_params.config.offline.noisy_hip_rate
            noisy_hip = jnp.where(noise_cond, uniform_hip, real_hip)
            return jnp.array((noisy_hip - eval_normalizers.hip_mean) / eval_normalizers.hip_std)[None]
        if eval_params.config.offline.eval_mode == EvalMode.sampled_latent:
            matching_episode = self.sample_matching_episode_by_hip(
                transitions,
                sample_state.hip,
                rng_latent,
            )
            obs_augmentation = matching_episode.obs[..., obs_dim:][0][None]
        elif eval_params.config.offline.eval_mode == EvalMode.inferred_latent:
            matching_episode = self.sample_matching_episode_by_hip(
                transitions,
                sample_state.hip,
                rng_latent,
            )
            matching_episode_wo_latent = matching_episode.replace(
                obs=matching_episode.obs[..., :obs_dim],
                next_obs=matching_episode.next_obs[..., :obs_dim],
            )
            latent = eval_params.repr_trainer.inference_single_sample(repr_trainstate, matching_episode_wo_latent, rng_inference, eval_params.config)[None]
            latent_normalized = (latent - eval_normalizers.latent_mean) / eval_normalizers.latent_std
            obs_augmentation = latent_normalized
        elif eval_params.config.offline.eval_mode == EvalMode.predicted_latent:
            # actually infer current latent from previous latents
            context_transitions, target_transition = self.sample_matching_context_episodes_by_next_hip(
                transitions,
                sample_state.hip,
                eval_params.config.predictor.context_length,
                rng_latent,
            )
            context_transitions_wo_latent = context_transitions.replace(
                obs=context_transitions.obs[..., :obs_dim],
                next_obs=context_transitions.next_obs[..., :obs_dim],
            )

            rng_repr = jax.random.split(rng_inference, eval_params.config.predictor.context_length)
            context_latents = jax.vmap(
                eval_params.repr_trainer.inference_single_sample, in_axes=(None, 0, 0, None)
                )(repr_trainstate, context_transitions_wo_latent, rng_repr, eval_params.config)
            context_latents_normed = (context_latents - eval_normalizers.latent_mean) / eval_normalizers.latent_std
            pred_latent = eval_params.pred_trainer.infer_on_single_context(context_latents_normed, pred_trainstate, rng_pred)
            obs_augmentation = pred_latent
        return obs_augmentation

    @partial(jax.jit, static_argnames=('self', 'eval_params','record_transitions', 'record_hip_only'))
    def single_deployment(
            self,
            initial_env_param: ParameterizedEnvParams,
            train_state: TrainState,
            transitions: Transition,
            rng: chex.PRNGKey,
            eval_params: EvalParams,
            eval_normalizers: EvalNormalizers,
            repr_trainstate: TrainState,
            pred_trainstate: TrainState,
            record_transitions: bool = False,
            record_hip_only: bool = False,  # if recording transitions
    ) -> Tuple[jnp.ndarray, dict, Transition]:
        if record_hip_only:
            assert record_transitions, 'record_hip_only requires record_transitions to be True'

        env_params_list = [initial_env_param]
        env_params = initial_env_param
        jitted_sched_step = jax.jit(self.env_scheduler.step)
        for _ in range(eval_params.episodes_per_deployment - 1):
            subkey, rng = jax.random.split(rng)
            env_params = jitted_sched_step(env_params, subkey)
            env_params_list.append(env_params)
        env_params_stacked = jax.tree_map(lambda x, *y: jnp.stack([*y]), env_params_list[0], *env_params_list)

        # determine observation augmentation (i.e. gt-latent, zero, hip etc.)
        augmentation_start = time.time()
        obs_augmentation_list = []
        for episode_idx in range(eval_params.episodes_per_deployment):  # this could also be vmapped
            rng, subkey = jax.random.split(rng)
            obs_augmentation = self.determine_obs_augmentation(
                subkey,
                env_params_list[episode_idx],
                transitions,
                eval_params,
                eval_normalizers,
                repr_trainstate,
                pred_trainstate,
            )
            obs_augmentation_list.append(obs_augmentation)
        obs_augmentation = jnp.stack(obs_augmentation_list)
        print('augmentation-time', time.time() - augmentation_start)

        if eval_params.binarize_latent:
            raise RuntimeError('removed in training dataset')
            obs_augmentation = jnp.where(obs_augmentation > 0.0, 1.0, -1.0)

        rollout_start_time = time.time()
        subkeys = jax.random.split(rng, eval_params.episodes_per_deployment)
        rollout_partial = partial(
            self._rollout, train_state=train_state, eval_params=eval_params,
            eval_normalizers=eval_normalizers, record_transitions=record_transitions, record_hip_only=record_hip_only)
        episode_returns, _, info, transitions = jax.vmap(rollout_partial)(
            rng_input=subkeys,  
            env_params=jax.tree_map(lambda x: x[..., None], env_params_stacked),
            obs_augmentation=obs_augmentation
        )
        
        if record_transitions:
            transitions = jax.tree_map(lambda x: x[:,:,0], transitions)
        print('rollout-time', time.time() - rollout_start_time)
        return episode_returns, info, transitions

    @partial(jax.jit, static_argnames=('self', 'eval_params', 'record_transitions', 'record_hip_only'))
    def _rollout(self, rng_input, train_state, env_params, eval_params: EvalParams,
                    eval_normalizers: EvalNormalizers, obs_augmentation: Optional[jnp.ndarray] = None,
                    record_transitions: bool = False, record_hip_only: bool = False):
        rng_reset, rng_episode, rng_input = jax.random.split(rng_input, 3)
        obs, state, env_params = self.batch_reset(rng_reset[None], env_params)
        obs_dim = obs.shape[-1]

        def policy_step(state_input, _):
            """lax.scan compatible step transition in jax env."""
            obs, state, obs_augment, train_state, rng, cum_reward, valid_mask, env_params = state_input
            obs = (obs - eval_normalizers.obs_mean[:obs_dim]) / eval_normalizers.obs_std[:obs_dim]  # obs_mean also contains normalization for latent but that would be done twice

            if not eval_params.method == Methods.zero_hip:
                obs = jnp.concatenate([obs, obs_augment], axis=-1)

            rng, rng_step, rng_net = jax.random.split(rng, 3)
            action = self.select_action(train_state, obs, rng_net, exploration_noise=eval_params.do_exploration)

            next_o, next_s, reward, done, info, new_env_params = self.batch_step(
                    rng_step[None],
                    state,
                    action,
                    env_params
                )
            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            if record_transitions:
                transition = Transition(
                    obs=obs,
                    act=action,
                    rew=reward,
                    next_obs=next_o,
                    done=done,
                    state_before=state if not record_hip_only else state.hip,
                    hip=state.hip,
                    episode_step=jnp.zeros_like(reward),
                    deployment_step=jnp.zeros_like(reward),
                    deployment_id=jnp.zeros_like(reward),
                    env_params=env_params,
                )
            else:
                transition = None

            carry, y = [
                    next_o,
                    next_s,
                    obs_augment,
                    train_state,
                    rng,
                    new_cum_reward,
                    new_valid_mask,
                    new_env_params
                ], [transition, info]
            return carry, y

            # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
                policy_step,
                [
                    obs,
                    state,
                    obs_augmentation,
                    train_state,
                    rng_episode,
                    jnp.array([0.0]),
                    jnp.array([1.0]),
                    env_params
                ],
                (),
                self.env_params.max_steps_in_episode,
            )

        episode_returns = carry_out[-3]
        new_env_params = carry_out[-1]
        transitions, info = scan_out
        return episode_returns, new_env_params, info, transitions

    @partial(jax.jit, static_argnames=('self','num_train_envs', 'buffer_manager', 'random_actions', 'input_gt_hip'))
    def get_transition(
        self,
        train_state: TrainState,
        obs: chex.ArrayDevice,
        state: dict,
        buffer,
        rng: chex.PRNGKey,
        num_train_envs: int,
        random_actions: bool,
        buffer_manager: BufferManager,
        env_params,
        input_gt_hip: bool
    ):
        rng, rng_act = jax.random.split(rng)
        if random_actions:
            rng_act = jax.random.split(rng, num_train_envs)
            single_env_params = jax.tree_map(lambda x: x[0], env_params)
            action = jax.vmap(self.env.action_space(single_env_params).sample)(rng_act)
        else:
            action = self.select_action(
                train_state, obs, rng_act, exploration_noise=True
            )
        # exploration noise scaled by max action
        # print(action.shape)
        b_rng = jax.random.split(rng, num_train_envs)
        # Automatic env resetting in gymnax step!
        next_obs, next_state, reward, done, info, env_params = self.batch_step(
            b_rng, state, action, env_params
        )
        if input_gt_hip:
            next_obs = jnp.concatenate([next_obs, next_state.hip[..., None]], axis=-1)

        buffer = buffer_manager.append(
            buffer, obs, next_state.hip, action, reward, next_obs, done
        )
        return next_obs, next_state, buffer, env_params, info

    @partial(jax.jit, static_argnames=('self', 'num_train_envs', 'random_actions', 'buffer_manager', 'n_transitions', 'unroll', 'input_gt_hip'))
    def get_multiple_transitions(
        self,
        train_state: TrainState,
        obs: chex.ArrayDevice,
        state: dict,
        buffer,
        rng: chex.PRNGKey,
        num_train_envs: int,
        random_actions: bool,
        buffer_manager: BufferManager,
        env_params: ParameterizedEnvParams,
        n_transitions: int,
        input_gt_hip: bool,
        unroll: int = 1,
    ):
        # just uses the above function with jax.lax.scan and a small wrapper, to enable multiple steps to be jitted together
        def transition_step(state_input, _):
            obs, state, buffer, env_params, rng = state_input
            rng, subkey = jax.random.split(rng)

            obs, state, buffer, env_params, info = self.get_transition(
                train_state, obs, state, buffer, subkey, num_train_envs, random_actions, buffer_manager, env_params, input_gt_hip
            )
            return [obs, state, buffer, env_params, rng], None

        carry, _ = jax.lax.scan(
            transition_step,
            [
                obs,
                state,
                buffer,
                env_params,
                rng
            ],
            (),
            n_transitions,
            unroll=unroll,
        )
        next_obs, next_state, buffer, env_params, rng = carry
        return next_obs, next_state, buffer, env_params
    
