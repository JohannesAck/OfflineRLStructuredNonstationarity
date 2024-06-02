"""
Heavily modified by Johannes Ackermann initially based on utils/ppo.py in the gymnax_blines
"""

from collections import defaultdict
import os
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import wandb

from utils.BufferManager import BufferManager
from RLMethods import TD3Trainer
from utils.config import Methods, RLAlgorithm, TrainConfig
from utils.DeploymentDataset import Dataset, DeploymentDataset
from utils.helpers import get_env
from utils.RolloutManager import HiPRolloutManager


# @jax.disable_jit()
def train_online(rng, config: TrainConfig) -> Tuple[Any, Dataset]:
    """Training loop for TD3."""
    n_train_envs = config.online.num_train_envs
    env, env_params = get_env(config.env_name, config.env_kwargs)
    obs_dim = env.observation_space(env_params).shape[0]
    if config.online.input_gt_hip:
        obs_dim += 1
    act_dim = env.action_space(env_params).shape[0]
    max_action = env.action_space(env_params).high

    rng, subkey = jax.random.split(rng)
    trainer, online_trainstate = get_online_trainer(subkey, config, env, env_params)
        
    del env, env_params

    # Setup the rollout manager -> Collects data in vmapped-fashion over envs
    rollout_manager = HiPRolloutManager(
        config.env_name, config.env_kwargs, config.env_params, config.online.exploration_std,
        config.online.exploration_clip, config, trainer, input_gt_hip=config.online.input_gt_hip,
    )
    rollout_manager_eval = HiPRolloutManager(
        config.env_name_eval, config.env_kwargs_eval, config.env_params_eval, config.online.exploration_std,
        config.online.exploration_clip, config, trainer, input_gt_hip=config.online.input_gt_hip
    )

    buffer_manager = BufferManager(
        size=config.online.buffer_size + 1,
        num_envs=n_train_envs,
        action_size=rollout_manager.action_size,
        state_space=rollout_manager.observation_space,
    )


    buffer = buffer_manager.reset()

    rng, rng_step, rng_reset, rng_eval, rng_update, rng_batch = jax.random.split(rng, 6)
    obs, state, env_params = rollout_manager.batch_reset(
        jax.random.split(rng_reset, n_train_envs)
    )
    if config.online.input_gt_hip:
        obs = jnp.concatenate([obs, state.hip[..., None]], axis=-1)

    total_steps = 0
    num_total_epochs = int(config.online.num_train_steps // (n_train_envs * config.online.n_transitions_jit) + 1)

    t = tqdm.tqdm(total=config.online.num_train_steps, desc=f"{config.online.algorithm.name}:", leave=True, smoothing=0.05)
    for epoch in range(1, num_total_epochs + 1):
        obs, state, buffer, env_params = rollout_manager.get_multiple_transitions(
            jax.lax.stop_gradient(online_trainstate),  # not sure if this is necessary
            obs,
            state,
            buffer,
            rng_step,
            n_train_envs,
            total_steps < config.online.warmup_steps,
            buffer_manager,
            env_params,
            config.online.n_transitions_jit,
            input_gt_hip=config.online.input_gt_hip,
            unroll=min(8, config.online.n_transitions_jit),
        )

        total_steps += n_train_envs * config.online.n_transitions_jit
        t.update(n_train_envs * config.online.n_transitions_jit)
        if epoch > config.online.warmup_steps and (epoch % config.online.update_freq) == 0:
            n_updates_jit = config.online.n_updates_jit
            n_updates_looped = (n_train_envs * config.online.n_transitions_jit) // n_updates_jit
            assert n_train_envs % n_updates_jit == 0, 'ensure number of envs is divisible by number of jitted updates'
            assert config.online.policy_freq <= config.online.n_updates_jit, "Policy frequency > n_updates_jit leads to no updates to policy, ever"
            for _ in range(n_updates_looped):
                rng_update, rng = jax.random.split(rng)
                online_trainstate, metric_dict = trainer.sample_buff_and_update_n_times(
                    online_trainstate,
                    buffer=buffer,
                    num_existing_samples=buffer['_valid'],
                    rng=rng_update,
                    max_action=max_action,
                    action_dim=act_dim,
                    config=config
                )
            if epoch % (config.online.update_freq * config.loss_log_rate) == 0:
                metric_dict['online/step'] = total_steps
                wandb.log(metric_dict)

        if epoch - 1 == 0 or epoch > config.online.warmup_steps and (epoch - 1) % config.online.evaluate_every_epochs == 0:
            rng, rng_eval= jax.random.split(rng)
            eval_str = ''
            eval_dict = {}
            env_period = rollout_manager.env.scheduler.period
            if env_period is None:
                env_period = 1
            
            train_env_reward, info, _, _ = rollout_manager.batch_evaluate(
                rng_eval,
                online_trainstate,
                rollout_manager.init_env_params,
                do_exploration=False,
                num_envs=config.online.num_test_rollouts,
                episodes_per_deployment=env_period,
                method=Methods.gt_hip if config.online.input_gt_hip else Methods.zero_hip,
                config=config,
            )
            eval_dict['online/train_env_reward'] = train_env_reward.item()
            eval_str += f'r_tr: {train_env_reward:.2f} '
            info = jax.tree_map(lambda x: jnp.mean(x).item(), info)
            for key, val in info.items():
                if key.startswith('hip'):
                    eval_dict[f'online/info/train_env_{key}_rew'] = val
                else:
                    eval_dict[f'online/info/{key}'] = val
            
            eval_dict['online/step'] = total_steps
            wandb.log(eval_dict)
            t.set_description(eval_str)
            t.refresh()
        
        
        # save network at same rate as video
        if (epoch - 1 == 0 or epoch == num_total_epochs or epoch > config.online.warmup_steps and (epoch - 1) % config.online.video_every_epochs == 0):
            network_ckpt = jax.tree_map(lambda x: np.array(x), online_trainstate.actor.params)
            network_fp = os.path.join(config.run_dir, f'actor_ckpt_step_{total_steps}.npy')
            np.save(network_fp, network_ckpt)
            print(f'saved network checkpoint to {config.run_dir}')
            

    eval_reward = 0.0
    final_eval_multiplier = 5
    total_info_dict = defaultdict(float)
    for _ in tqdm.trange(final_eval_multiplier, desc='final_eval'):
        rng, rng_eval = jax.random.split(rng)
        env_period = rollout_manager_eval.env.scheduler.period
        if env_period is None:
            env_period = 1
        eval_reward_batch, info, _, _ = rollout_manager_eval.batch_evaluate(
            rng_eval,
            online_trainstate,
            rollout_manager_eval.init_env_params,
            do_exploration=False,
            num_envs=config.online.num_test_rollouts,
            episodes_per_deployment=env_period,
            method=Methods.gt_hip if config.online.input_gt_hip else Methods.zero_hip,
            config=config,
        )
        info = jax.tree_map(lambda x: jnp.mean(x).item(), info)
        for key, val in info.items():
            if key.startswith('hip'):
                total_info_dict[f'online/info/train_env_{key}_rew'] += val
            else:
                if not type(val) == dict:
                    total_info_dict[f'online/info/{key}'] += val

        eval_reward += eval_reward_batch.item()
    total_info_dict = jax.tree_map(lambda x: x / final_eval_multiplier, total_info_dict)
    eval_reward = eval_reward / final_eval_multiplier
    print(f'Final evaluation over {final_eval_multiplier * config.online.num_test_rollouts} rollouts on eval_env: {eval_reward}')
    log_dict = {
        'online/eval_env_reward': eval_reward,
        'online/final_eval_reward': eval_reward,
        'online/step': total_steps,
        **total_info_dict
    }
    wandb.log(log_dict)

    # collect dataset
    if not config.online.algorithm == RLAlgorithm.BraxPPO:
        del buffer, buffer_manager
    env_period = rollout_manager_eval.env_scheduler.period
    if env_period is None:
        env_period = 10
    episode_len = rollout_manager_eval.init_env_params.max_steps_in_episode
    rng, subkey = jax.random.split(rng)
    deployment_dataset = DeploymentDataset(
        n_deployments=config.online.n_deployments_record,
        n_episodes_deployment=env_period,
        n_timesteps_episode=episode_len,
        obs_shape=rollout_manager_eval.env.observation_space(rollout_manager_eval.init_env_params).shape,
        act_shape=rollout_manager_eval.env.action_space(rollout_manager_eval.init_env_params).shape,
        env_params=rollout_manager_eval.init_env_params,
        env_state=rollout_manager_eval.env.reset(subkey,)[1],
        record_hip_as_state=config.online.record_hip_as_state,
        on_cpu=True
    )
    ds = deployment_dataset.ds
    deployment_idx = 0
    for _ in tqdm.trange(config.online.n_deployments_record // config.online.traj_recording_parallel):
        rng, subkey = jax.random.split(rng)
        deployment_batch = rollout_manager_eval.batch_record_transitions(
            rng=subkey,
            train_state=online_trainstate,
            num_envs=config.online.traj_recording_parallel,
            starting_env_params=rollout_manager_eval.init_env_params,
            episodes_per_deployment=env_period,
            do_exploration=True,
            record_hip_only=config.online.record_hip_as_state,
            config=config,
            method=Methods.gt_hip if config.online.input_gt_hip else Methods.zero_hip
        )
        hip_name = rollout_manager_eval.env_scheduler.parameter_names[0]
        deployment_batch = deployment_batch.replace(
            obs=deployment_batch.obs[:,:,:,:-1] if config.online.input_gt_hip else deployment_batch.obs,
            deployment_id=jnp.tile(jnp.arange(deployment_idx, deployment_idx + config.online.traj_recording_parallel)[...,None,None], (1, env_period, episode_len)),
            hip=deployment_batch.env_params.__dict__[hip_name] if hip_name else jnp.zeros_like(deployment_batch.deployment_id),
        )
        ds = deployment_dataset.add_deployments(ds, deployment_batch, deployment_ids=np.arange(deployment_idx, deployment_idx + config.online.traj_recording_parallel))
        deployment_idx += config.online.traj_recording_parallel

    ds = jax.tree_map(lambda x: jax.lax.stop_gradient(x), ds)
    ds = jax.tree_map(lambda x: np.array(x) if not type(x) == int else x, ds)
    print(f'Generated dataset of {config.online.n_deployments_record} deployments with {env_period} episodes each')    
    
    network_ckpt = jax.tree_map(lambda x: np.array(x), online_trainstate.actor.params)
    return network_ckpt, ds

def get_online_trainer(rng, config, env, env_params):
    obs_dim = env.observation_space(env_params).shape[0]
    if config.online.input_gt_hip:
        obs_dim += 1
    act_dim = env.action_space(env_params).shape[0]
    max_action = env.action_space(env_params).high

    if config.online.algorithm == RLAlgorithm.TD3:
        trainer = TD3Trainer(config, action_space=env.action_space(env_params))
    else:
        raise NotImplementedError(f'Unknown algorithm for online training {config.online.algorithm}')
        
    rng, subkey = jax.random.split(rng)
    online_trainstate = trainer.get_models(
            obs_dim, act_dim, max_action,
            config, subkey,
        )
    
    return trainer, online_trainstate

