import os.path
import time
from typing import List, Optional
import einops

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from flax.training.train_state import TrainState

import wandb
from representation_models import PredictorTrainer, RepresentationTrainer
from representation_models.latent_predictor_model import visualize_embeddings_fakereal
from utils.BufferManager import BufferManager
from RLMethods import TD3BCTrainer
from utils.config import EvalMode, Methods, OfflineAlgorithm, RLAlgorithm, TrainConfig
from utils.DeploymentDataset import Dataset, MinimalTransition, Transition
from utils.helpers import get_env
from utils.RolloutManager import HiPRolloutManager
from utils.train_online import get_online_trainer
from utils.visualization import linear_probe_callback, visualize_embeddings



def train_offline(
    rng, 
    config: TrainConfig, 
    ds: Dataset,
    repr_trainer: Optional[RepresentationTrainer] = None,
    repr_trainstate: Optional[TrainState] = None,
    pred_trainer: Optional[PredictorTrainer] = None,
    pred_trainstate: Optional[TrainState] = None,
    ds_network_ckpt = None,
):
    """Offline Training Loop."""


    subsample_deployments = np.prod(ds.transition.obs.shape[:-1]) > 1e7 or ds.transition.obs.size > 1e8
    if subsample_deployments:
        ratio = 0.1
        subsampled_indices = jnp.arange(0, ds.n_deployments, step=int(1/ratio))
        ds = ds.replace(
            transition=jax.tree_map(lambda x: x[subsampled_indices], ds.transition),
            n_deployments=int(ratio * ds.n_deployments)
        )
        print('subsampled dataset')
        print(f'obs shape after {ds.transition.obs.shape}')


    env, env_params = get_env(config.env_name, config.env_kwargs)
    env_obs_dim = env.observation_space(env_params).shape[0]
    action_space = env.action_space(env_params)
    if config.method in [Methods.gt_hip, Methods.noisy_hip]:
        obs_dim = env_obs_dim + 1
    elif config.method == Methods.zero_hip:
        obs_dim = env_obs_dim
    else:
        obs_dim = ds.obs_shape

    rng, subkey = jax.random.split(rng)
    dspolicy_trainer, dspolicy_trainstate = None, None 

    offline_trainer, offline_train_state = get_offline_trainer(rng, config, action_space, obs_dim)

    # Setup the rollout manager -> Collects data in vmapped-fashion over envs
    rollout_manager_eval = HiPRolloutManager(
        config.env_name_eval, config.env_kwargs, config.env_params, 0.0, 0.0, config, offline_trainer
    )

    ds, rew_mean, rew_std, latent_mean, latent_std, latents_normed, buffer, hip_mean, hip_std, obs_mean, obs_std, inference_transitions = prepare_datasets(config, ds, env, env_obs_dim, rollout_manager_eval)


    visualize_latents = True
    if visualize_latents and not config.method in [Methods.gt_hip, Methods.zero_hip, Methods.noisy_hip]:  # used for debugging
        visualize_latents_for_debugging(env.observation_space(env_params).shape[0], buffer, inference_transitions, config)


    if config.offline.algorithm == OfflineAlgorithm.TD3BC:
        assert config.offline.policy_freq <= config.offline.n_updates_jit, "Policy frequency > num_envs leads to no updates to policy, ever"

    total_steps = 0
    log_steps, log_return = [], []
    num_total_its = int(config.offline.train_steps // config.offline.n_updates_jit + 1)
    t = tqdm.trange(1, config.offline.train_steps, desc=f"{config.offline.algorithm.name}", leave=True)
    for it in range(num_total_its):
        total_steps += config.offline.n_updates_jit
        t.update(config.offline.n_updates_jit)
        rng, rng_eval, rng_update = jax.random.split(rng, 3)
        metric_dict, offline_train_state = offline_trainer.sample_buff_and_update_n_times(
            offline_train_state,
            buffer=buffer,
            num_existing_samples=buffer['_valid'] if isinstance(buffer['_valid'], int) else buffer['_valid'].item(),
            rng=rng_update,
            action_dim=action_space.shape[0],
            max_action=action_space.high,
            config=config,
        )
        if it % 200 == 0:
            metric_dict['offline/step'] = total_steps
            wandb.log(metric_dict)


        if (it + 1) % config.offline.evaluate_every_epochs == 0:
            rng, rng_eval, rng_eval_2, rng_eval_3 = jax.random.split(rng, 4)
            env_period = rollout_manager_eval.env.scheduler.period
            if env_period is None:
                env_period = 10

            eval_start = time.time()
            assert config.vae.use_trans_encoding, 'Evaluation does not support context encoding currently'
            eval_reward, info, inferred_latents, eval_hips = rollout_manager_eval.batch_evaluate(
                rng_eval,
                offline_train_state,
                rollout_manager_eval.init_env_params,
                do_exploration=False,
                num_envs=config.offline.num_test_rollouts,
                episodes_per_deployment=env_period,
                obs_mean=obs_mean,
                obs_std=obs_std,
                transitions=inference_transitions,
                method=config.method,
                repr_trainstate=repr_trainstate,
                hip_mean=hip_mean,
                hip_std=hip_std,
                rew_mean=rew_mean,
                rew_std=rew_std,
                latent_mean=latent_mean,
                latent_std=latent_std,
                repr_trainer=repr_trainer,
                config=config,
                pred_trainer=pred_trainer,
                pred_trainstate=pred_trainstate,
                dspol_trainer=dspolicy_trainer,
                ds_pol_trainstate=dspolicy_trainstate
            )  

            eval_reward = eval_reward.item()
            print(time.time() - eval_start, 'seconds for eval')
            eval_dict = {
                'offline/eval_reward': eval_reward,
                'offline/step': total_steps,
            }
            info = jax.tree_map(lambda x: jnp.mean(x).item(), info)
            for key, val in info.items():
                if key in ['obs_history', 'last_act', 'last_contact_buffer']:
                    continue
                eval_dict[f'offline/info/{key}'] = val

            wandb.log(eval_dict)
            log_steps.append(total_steps)
            log_return.append(eval_reward)
            t.set_description(f"{config.offline.algorithm.name} R_te: {str(eval_reward)}")
            t.refresh()

        # save network at same rate as video
        if (it + 1) % config.offline.video_every_epochs == 0:
            network_ckpt = jax.tree_map(lambda x: np.array(x), offline_train_state.actor.params)
            network_fp = os.path.join(config.run_dir, f'actor_ckpt_step_{total_steps}.npy')
            np.save(network_fp, network_ckpt)
            print(f'saved network checkpoint to {config.run_dir}')
            
    # final eval:
    env_period = rollout_manager_eval.env.scheduler.period
    if env_period is None:
        env_period = 10
    eval_reward = 0.0
    final_eval_multiplier = 5
    for _ in tqdm.trange(final_eval_multiplier, desc='final_eval'):
        rng, rng_eval = jax.random.split(rng)
        eval_reward_batch, info, _, _ = rollout_manager_eval.batch_evaluate(
            rng_eval,
            offline_train_state,
            rollout_manager_eval.init_env_params,
            do_exploration=False,
            num_envs=config.offline.num_test_rollouts,
            episodes_per_deployment=env_period,
            obs_mean=obs_mean,
            obs_std=obs_std,
            transitions=inference_transitions,
            method=config.method,
            config=config,
            repr_trainer=repr_trainer,
            repr_trainstate=repr_trainstate,
            hip_mean=hip_mean,
            hip_std=hip_std,         
            rew_mean=rew_mean,
            rew_std=rew_std,
            latent_mean=latent_mean,
            latent_std=latent_std,
            pred_trainer=pred_trainer,
            pred_trainstate=pred_trainstate,
        )
            
        eval_reward += eval_reward_batch.item()
    
    eval_reward = eval_reward / final_eval_multiplier
    print(f'Final evaluation over {final_eval_multiplier * config.offline.num_test_rollouts} rollouts: {eval_reward}')
    final_eval_dict = {
        'offline/eval_reward': eval_reward,
        'offline/final_eval_reward': eval_reward,
        'offline/step': total_steps,
    }
    wandb.log(final_eval_dict)

    # save network at same rate as video
    network_ckpt = jax.tree_map(lambda x: np.array(x), offline_train_state.actor.params)
    network_fp = os.path.join(config.run_dir, f'actor_ckpt_step_final.npy')
    np.save(network_fp, network_ckpt)
    print(f'saved network checkpoint to {config.run_dir}')

    return (
        log_steps,
        log_return,
        offline_train_state.actor.params,
    )

def prepare_datasets(config, ds, env, env_obs_dim, rollout_manager_eval):
    rew_mean = ds.transition.rew.mean()
    rew_std = ds.transition.rew.std() + 1e-6

    if config.offline.normalize_latent:
        print('normalizing latent')
        latents = ds.transition.obs[..., env_obs_dim:]
        if config.offline.normalize_latent_by_dimension:
            latent_mean = latents.mean(axis=(0,1,2))
            latent_std = latents.std(axis=(0,1,2)) + 1e-6
        else:
            latent_mean = latents.mean((0,1,2))
            latent_std = latents.std((0,1,2)) + 1e-6
        latents_normed = (latents - latent_mean) / latent_std
        ds = ds.replace(
            transition=ds.transition.replace(
                obs = np.concatenate([ds.transition.obs[...,:env_obs_dim], latents_normed], axis=-1),
                next_obs = np.concatenate([ds.transition.next_obs[...,:env_obs_dim], latents_normed], axis=-1),
            )
        )
    else:
        latent_mean = 0.0
        latent_std = 1.0
    
    buffer_manager = BufferManager(
        size=2,  # unused in offline setting
        num_envs=1,
        action_size=rollout_manager_eval.action_size,
        state_space=rollout_manager_eval.observation_space,
    )
    buffer = buffer_manager.convert_dataset(ds)  # also gets rid of steps after done

    
    if config.method == Methods.gt_hip:
        print('adding ground-truth hip')
        hip = buffer['hips']
        hip_mean = hip.mean()
        hip_std = hip.std() + 1e-6
        hip = (hip - hip_mean) / hip_std
        assert len(rollout_manager_eval.observation_space.shape) == 1, 'next line will not work with 2D/3D observations'
        buffer['states'] = jnp.concatenate([buffer['states'][:,:rollout_manager_eval.observation_space.shape[0]], hip[...,None]], axis=-1)
        buffer['next_states'] = jnp.concatenate([buffer['next_states'][:,:rollout_manager_eval.observation_space.shape[0]], hip[...,None]], axis=-1)
    elif config.method == Methods.noisy_hip:
        print('adding noisy hip')
        hip = buffer['hips']
        hip_mean = hip.mean()
        hip_std = hip.std() + 1e-6
        random_hip = np.random.choice(list(env.scheduler.possible_hips.values())[0], hip.shape, replace=True)
        noise_cond = np.random.uniform(0.0, 1.0, hip.shape) < config.offline.noisy_hip_rate
        noisy_hip = np.where(noise_cond, random_hip, hip)
        noisy_hip = (noisy_hip - hip_mean) / hip_std
        assert len(rollout_manager_eval.observation_space.shape) == 1, 'next line will not work with 2D/3D observations'
        buffer['states'] = jnp.concatenate([buffer['states'][:,:rollout_manager_eval.observation_space.shape[0]], noisy_hip[...,None]], axis=-1)
        buffer['next_states'] = jnp.concatenate([buffer['next_states'][:,:rollout_manager_eval.observation_space.shape[0]], noisy_hip[...,None]], axis=-1)
    elif config.method == Methods.zero_hip:
        print('setting zero hip')
        buffer['states'] = buffer['states'][:,:rollout_manager_eval.observation_space.shape[0]]
        buffer['next_states'] = buffer['next_states'][:,:rollout_manager_eval.observation_space.shape[0]]
        hip_mean = 0.0
        hip_std = 1.0
    else:
        hip_mean = 0.0
        hip_std = 1.0

    
    # normalize states
    print('normalizing states   ')
    obs_mean = np.array(buffer['states'].mean(axis=0))
    obs_std = np.array(buffer['states'].std(axis=(0))) + 1e-6
    buffer['states'] = (buffer['states'] - obs_mean) / obs_std
    buffer['next_states'] = (buffer['next_states'] - obs_mean) / obs_std

    obs_mean = obs_mean[:env_obs_dim]
    obs_std = obs_std[:env_obs_dim]
    

    max_ds_size = int(5e6)
    if buffer['states'].shape[0] < max_ds_size:
        buffer = jax.tree_map(jnp.array, buffer)
    else:
        indices = np.random.choice(buffer['states'].shape[0], int(max_ds_size), replace=False)
        print('subsampling buffer')
        buffer = jax.tree_map(lambda x: jnp.array(x[indices]) if not type(x) == int else x, buffer)


    # dataset for inference of latent
    if ds.n_deployments > 250:
        # deploy_samples = np.random.choice(ds.n_deployments, size=250)
        valid_samples = np.arange(int(0.75 * ds.n_deployments), ds.n_deployments)
        deploy_samples = np.random.choice(valid_samples, size=250)
    else:
        deploy_samples = np.arange(ds.n_deployments)
    

    use_full_transition = False
    if config.method in [Methods.gt_hip, Methods.zero_hip, Methods.noisy_hip]:
        inference_transitions = None
    elif config.offline.eval_mode == EvalMode.sampled_latent:
        if use_full_transition:
            inference_transitions: Transition = jax.tree_map(lambda x: jnp.array(x[deploy_samples]), ds.transition)
        else:
            inference_transitions = MinimalTransition(
                obs=ds.transition.obs[deploy_samples],
                next_obs=ds.transition.next_obs[deploy_samples],
                act=ds.transition.act[deploy_samples],
                rew=ds.transition.rew[deploy_samples],
                done=ds.transition.done[deploy_samples],
                hip=ds.transition.hip[deploy_samples],
            )  # type: ignore
    else:
        if use_full_transition:
            inference_transitions: Transition = jax.tree_map(lambda x: x[deploy_samples], ds.transition)
            remove_latent = True
            if remove_latent:
                inference_transitions = inference_transitions.replace(
                    obs=(inference_transitions.obs[...,:env_obs_dim]  - obs_mean[:env_obs_dim]) / (obs_std[:env_obs_dim]),
                    next_obs = (inference_transitions.next_obs[...,:env_obs_dim] - obs_mean[:env_obs_dim]) / (obs_std[:env_obs_dim]),
                    rew = (inference_transitions.rew - rew_mean) / rew_std,
                )
            else:
                inference_transitions = inference_transitions.replace(
                    obs=(inference_transitions.obs  - obs_mean) / (obs_std),
                    next_obs = (inference_transitions.next_obs - obs_mean) / (obs_std),
                    rew = (inference_transitions.rew - rew_mean) / rew_std,
                )
            inference_transitions = jax.tree_map(jnp.array, inference_transitions)
        else:
            remove_latent = True
            if remove_latent:
                inference_transitions = MinimalTransition(
                    obs=(ds.transition.obs[deploy_samples][...,:env_obs_dim]  - obs_mean[:env_obs_dim]) / (obs_std[:env_obs_dim]),
                    next_obs=(ds.transition.next_obs[deploy_samples][...,:env_obs_dim] - obs_mean[:env_obs_dim]) / (obs_std[:env_obs_dim]),
                    act=ds.transition.act[deploy_samples],
                    rew=(ds.transition.rew[deploy_samples] - rew_mean) / rew_std,
                    done=ds.transition.done[deploy_samples],
                    hip=ds.transition.hip[deploy_samples],
                )  # type: ignore
            else:
                for _ in range(5):
                    print('LATENT NOT REMOVED!!!')  # only makes sense for debugging
                inference_transitions = MinimalTransition(
                    obs=(ds.transition.obs[deploy_samples]  - obs_mean) / (obs_std),
                    next_obs=(ds.transition.next_obs[deploy_samples] - obs_mean) / (obs_std),
                    act=ds.transition.act[deploy_samples],
                    rew=(ds.transition.rew[deploy_samples] - rew_mean) / rew_std,
                    done=ds.transition.done[deploy_samples],
                    hip=ds.transition.hip[deploy_samples],
                )  # type: ignore
            inference_transitions = jax.tree_map(jnp.array, inference_transitions)
    return ds, rew_mean, rew_std, latent_mean, latent_std, latents_normed, buffer, hip_mean, hip_std, obs_mean, obs_std, inference_transitions

def get_offline_trainer(rng, config: TrainConfig, action_space, obs_dim: int):
    act_dim = action_space.shape[0]
    max_action = action_space.high

    if config.offline.algorithm == OfflineAlgorithm.TD3BC:
        offline_trainer = TD3BCTrainer(config=config, action_space=action_space)
    else:
        raise ValueError(f'Unknown offline algorithm {config.offline.algorithm}')

    offline_train_state = offline_trainer.get_models(
        obs_dim, act_dim, max_action, config, rng
    )
    
    return offline_trainer, offline_train_state


def visualize_latents_for_debugging(obs_shape, buffer, inference_transitions, config:  TrainConfig):
    print('visualizing latents from training buffer')
    random_samples = np.random.choice(len(buffer['states']), size=1000, replace=False)
    vis_latents = buffer['states'][random_samples, obs_shape:]
    vis_hips = buffer['hips'][random_samples]
    pil_img_pca = visualize_embeddings(vis_latents, vis_hips, do_tsne=False)
    pil_img_tsne = visualize_embeddings(vis_latents, vis_hips, do_tsne=True)
    wandb.log({
            'offline/latent_pca_buff' : wandb.Image(pil_img_pca),
            'offline/latent_tsne_buff' : wandb.Image(pil_img_tsne)
        })
    if inference_transitions.obs.shape[-1] > obs_shape:
        print('visualizing latents from inference buffer')
        random_deployment = np.random.choice(inference_transitions.obs.shape[0], size=1000, replace=True)
        random_episode_steps = np.random.choice(inference_transitions.obs.shape[1], size=1000, replace=True)
        vis_latents_inf = inference_transitions.obs[random_deployment, random_episode_steps, 0,  obs_shape:]
        vis_hips_inf = inference_transitions.hip[random_deployment, random_episode_steps, 0]
        pil_img_pca = visualize_embeddings(vis_latents_inf, vis_hips_inf, do_tsne=False)
        pil_img_tsne = visualize_embeddings(vis_latents_inf, vis_hips_inf, do_tsne=True)
        wandb.log({
                'offline/latent_pca_inf' : wandb.Image(pil_img_pca),
                'offline/latent_tsne_inf' : wandb.Image(pil_img_tsne)
            })

        print('visualizing mixed')
        pil_img_pca = visualize_embeddings_fakereal(
                inferred_latents=vis_latents_inf, 
                true_latents=vis_latents, 
                hidden_parameters=vis_hips,
                do_tsne=False,
                hidden_parameters_inf=vis_hips_inf,
            )
        pil_img_tsne = visualize_embeddings_fakereal(
                inferred_latents=vis_latents_inf, 
                true_latents=vis_latents, 
                hidden_parameters=vis_hips,
                do_tsne=True,
                hidden_parameters_inf=vis_hips_inf,
            )
        wandb.log({
                'offline/latent_pca_mixed' : wandb.Image(pil_img_pca),
                'offline/latent_tsne_mixed' : wandb.Image(pil_img_tsne)
            })


