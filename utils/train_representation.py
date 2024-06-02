import pickle
import time
from functools import partial
from typing import Callable, List, Optional, Tuple
import tempfile

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
from flax.training.train_state import TrainState

import wandb
from representation_models import PredictorTrainer
from utils.DeploymentDataset import Dataset, MinimalTransition, Transition
from utils.config import TrainConfig, Methods
from representation_models import CPCTrainer, RepresentationTrainer
from utils.helpers import get_env


def train_representation(rng, config: TrainConfig, ds: Dataset):
    """Training loop for VAE."""
    rng, rng_init = jax.random.split(rng)
    
    repr_trainer, train_state = get_representation_model(config, rng_init)
    
    obs_mean = ds.transition.obs.mean(axis=(0,1,2))
    obs_std = ds.transition.obs.std(axis=(0,1,2)) + 1e-6
    rew_mean = ds.transition.rew.mean(axis=(0,1,2))
    rew_std = ds.transition.rew.std(axis=(0,1,2)) + 1e-6

    #set dones true after the first occurence per episode
    def all_done_after_first(episode_dones: chex.Array):  # overwrites all dones after the first positive one with true
        def f(done_encountered, done):
            done_encountered = jnp.where(done == 1.0, 1.0, done_encountered)
            return done_encountered, done_encountered
        _, dones_fixed = jax.lax.scan(f, 0.0, episode_dones)
        return dones_fixed
    done_fixed = jax.vmap(jax.vmap(all_done_after_first))(ds.transition.done).astype(bool)
    ds = ds.replace(
        transition=ds.transition.replace(
            done=done_fixed
        )
    )
    orig_n_episodes_deployment = ds.n_episodes_deployment
    # preserve dataset without normalizations etc.
    ds_orig_tempfile = tempfile.TemporaryFile()
    pickle.dump(ds, ds_orig_tempfile)

    # ds_orig = jax.tree_map(lambda x: np.array(x), ds)
    print(f'obs shape before {ds.transition.obs.shape}')

    # normalize states
    ds = ds.replace(
        transition=ds.transition.replace(
            obs=(ds.transition.obs - obs_mean) / obs_std,
            next_obs=(ds.transition.next_obs - obs_mean) / obs_std,
        )
    )
    
    # normalize rewards
    ds = ds.replace(
        transition=ds.transition.replace(
            rew=(ds.transition.rew - rew_mean) / rew_std,
        )
    )


    ds_train = ds
    subsample_deployments = True
    if subsample_deployments:
        deploy_ratio = 0.75
        subsampled_indices = jnp.arange(0, ds_train.n_deployments, step=int(1/deploy_ratio))
        ds_train = ds_train.replace(
            transition=jax.tree_map(lambda x: x[subsampled_indices], ds_train.transition),
            n_deployments=int(deploy_ratio * ds_train.n_deployments)
        )
        print('subsampled dataset')
        print(f'obs shape after {ds_train.transition.obs.shape}')
    
    subsample_timesteps = ds.n_timesteps_episode > 100
    if subsample_timesteps:
        ts_ratio = 0.1
        subsampled_timesteps = jnp.arange(0, ds_train.n_timesteps_episode, step=int(1/ts_ratio))
        ds_train = ds_train.replace(
            transition=jax.tree_map(lambda x: x[:,:,subsampled_timesteps], ds_train.transition),
            n_timesteps_episode=int(ts_ratio * ds_train.n_timesteps_episode)
        )
        print('subsampled dataset')
        print(f'obs shape after {ds_train.transition.obs.shape}')


    if config.vae.gt_hip_vae:
        hip = ds_train.transition.hip
        hip_mean = hip.mean(axis=(0, 1))
        hip_std = hip.std(axis=(0, 1)) + 1e-6
        hip = (hip - hip_mean) / hip_std
        ds_train.transition.obs = jnp.concatenate([ds_train.transition.obs, hip[...,None]], axis=-1)
        ds_train.transition.next_obs = jnp.concatenate([ds_train.transition.next_obs, hip[...,None]], axis=-1)
    else:
        hip_mean = 0.0
        hip_std = 1.0
    rng, subkey = jax.random.split(rng)
    

   
    total_steps = 0
    hipprobe_acc_test = 0.0
    log_steps, log_return = [], []

    num_total_epochs = int(config.vae.train_steps // config.vae.n_updates_jit + 1)
    
    repr_tqdm = tqdm.trange(1, num_total_epochs, desc="VAE", leave=True)

    use_full_transition = False
    if use_full_transition:
        ds_train = ds_train.replace(transition=jax.tree_map(jnp.array, ds.transition))
        print('USING FULL TRANSITION, USE ONLY FOR DEBUGGING')
        print('USING FULL TRANSITION, USE ONLY FOR DEBUGGING')
    else:
        # need those again later on
        ds_state_before = ds.transition.state_before
        ds_episode_step = ds.transition.episode_step
        ds_deployment_step = ds.transition.deployment_step
        ds_deployment_id = ds.transition.deployment_id
        ds_env_params = ds.transition.env_params

        ds_train = ds_train.replace(
            transition = MinimalTransition(
                    obs=jnp.array(ds_train.transition.obs),
                    act=jnp.array(ds_train.transition.act),
                    rew=jnp.array(ds_train.transition.rew),
                    next_obs=jnp.array(ds_train.transition.next_obs),
                    done=jnp.array(ds_train.transition.done),
                    hip=jnp.array(ds_train.transition.hip),
                )
        )

    for step in repr_tqdm:
        total_steps += config.vae.n_updates_jit

        rng, subkey = jax.random.split(rng)


        metric_dict, train_state = repr_trainer.sample_and_update(
            train_state,
            dataset=ds_train,
            rng=subkey,
            config=config
        )
        if jnp.isnan(metric_dict['repr/total_loss']):
            print(f'nan at step {step}')

        
        if step % 100 == 0:
            metric_dict['repr/step'] = total_steps
            wandb.log(metric_dict)
            if config.method in [Methods.cpc]:
                repr_tqdm.set_description(f"{config.method.name} L:{metric_dict['repr/total_loss']:.3f} Recall:{metric_dict['repr/cpc_recall']:.2f} Probe:{hipprobe_acc_test:.2f}")
            else:
                repr_tqdm.set_description(f"{config.method.name}")
            repr_tqdm.refresh()

        if (step - 1) % (config.vae.evaluate_every_steps) == 0:
            rng, subkey = jax.random.split(rng)
            evaluation_dict = repr_trainer.eval(
                train_state,
                ds,
                subkey,
                config,
            )

            if config.method == Methods.cpc:
                if config.vae.use_trans_encoding:
                    hipprobe_acc_test = evaluation_dict['repr/transhipprobe_acc_test']
                else:
                    hipprobe_acc_test = evaluation_dict['repr/contexthipprobe_acc_test']
            else:
                raise ValueError(f'unknown method {config.method}')
            evaluation_dict['repr/step'] = total_steps
            wandb.log(evaluation_dict)

    # infer latents for full dataset
    rng, subkey = jax.random.split(rng)

    if not use_full_transition:
        # add unused information back in for inference debugging
        maybe_subsample = lambda x: jax.tree_map(lambda y: y[:,:,subsampled_timesteps], x) if subsample_timesteps else x
        ds = ds.replace(  
            transition = Transition(
                    obs=ds.transition.obs,
                    act=ds.transition.act,
                    rew=ds.transition.rew,
                    next_obs=ds.transition.next_obs,
                    done=ds.transition.done,
                    hip=ds.transition.hip,
                    state_before=ds_state_before,
                    episode_step=ds_episode_step,
                    deployment_step=ds_deployment_step,
                    deployment_id=ds_deployment_id,
                    env_params=ds_env_params,
                )
        )

    if subsample_timesteps:
        ds_inf = ds.replace(
            transition=jax.tree_map(lambda x: x[:,:,subsampled_timesteps], ds.transition),
            n_timesteps_episode=int(ts_ratio * ds.n_timesteps_episode)
        )
        print('subsampled dataset')
        print(f'obs shape after {ds_inf.transition.obs.shape}')
    else:
        ds_inf = ds

    latents, valid_deploystep_mask = repr_trainer.inference_full_dataset(
        train_state,
        ds_inf,
        orig_n_episodes_deployment,
        subkey,
        config,
    )

    del ds_state_before, ds_episode_step, ds_deployment_step, ds_deployment_id, ds_env_params
    del ds

    ds_orig_tempfile.seek(0)
    ds_orig = pickle.load(ds_orig_tempfile)
    ds_orig_tempfile.close()
    # train latent predictor
    print('Starting predictor training')
    rng, subkey = jax.random.split(rng)
    predictor_train_state = train_predictor(rng, config, ds_orig, latents)


    # augment dataset
    print('appending inferred context to dataset') 
    latents_tiled = np.tile(latents[:,:,None,:], [1,1, ds_orig.n_timesteps_episode, 1])



    valid_deploystep_mask = np.array(valid_deploystep_mask).astype(bool)
    if np.all(valid_deploystep_mask):
        transitions_valid = ds_orig.transition
    else:
        transitions_valid = jax.tree_map(lambda x: x[:,valid_deploystep_mask], ds_orig.transition)

    transitions_augmented = transitions_valid.replace(
        obs=np.concatenate([transitions_valid.obs, latents_tiled], axis=-1),
        next_obs=np.concatenate([transitions_valid.next_obs, latents_tiled], axis=-1),
    )

    ds_augmented = ds_orig.replace(
        transition=transitions_augmented,
        obs_shape = transitions_augmented.obs.shape[-1],
        n_episodes_deployment = transitions_augmented.obs.shape[1],
    )
    ds_augmented = jax.tree_map(lambda x: x if type(x) == np.ndarray else np.array(x), ds_augmented)
        
    return (
        log_steps,
        log_return,
        train_state,
        predictor_train_state,
        ds_augmented,
    )

def train_predictor(rng: chex.PRNGKey, config: TrainConfig, ds_orig: Dataset, latents: chex.Array) -> TrainState:
    predictor_trainer = PredictorTrainer()
    rng, rng_init = jax.random.split(rng)
    _, predictor_train_state = predictor_trainer.get_model(rng_init, config)
        
    normalized_latent = (latents - latents.mean(axis=(0,1))) / (latents.std(axis=(0,1)) + 1e-6)
    train_ratio = 0.9
    train_latents = normalized_latent[:int(train_ratio * latents.shape[0])]
    test_latents = normalized_latent[int(train_ratio * latents.shape[0]):]
    train_hips = ds_orig.transition.hip[:int(train_ratio * latents.shape[0]),:,0]
    test_hips = ds_orig.transition.hip[int(train_ratio * latents.shape[0]):,:,0]

    pred_tqdm = tqdm.trange(config.predictor.training_epochs, desc='Predictor', leave=True)
    for pred_epoch_idx in pred_tqdm:
        rng, subkey = jax.random.split(rng)
        predictor_train_state, train_loss = predictor_trainer.train_epoch(
                train_latents,
                predictor_train_state,
                subkey,
                config,
            )
        rng, subkey = jax.random.split(rng)
            
        if pred_epoch_idx % config.predictor.evaluate_every_epoch == 0:
            test_loss, test_mape = predictor_trainer.evaluate(
                    test_latents,
                    predictor_train_state,
                    subkey,
                    config,
                )
            rng, subkey = jax.random.split(rng)
            vis_dict = predictor_trainer.visualize_result(
                    test_latents,
                    test_hips,
                    predictor_train_state,
                    subkey,
                    config,
                )
            vis_dict['pred/step'] = pred_epoch_idx
            wandb.log({
                    'pred/train_loss': train_loss,
                    'pred/test_loss': test_loss,
                    'pred/test_mape': test_mape,
                    'pred/step': pred_epoch_idx
                    }
                )
            wandb.log(vis_dict)
            pred_tqdm.set_description(f'Predictor L_tr={train_loss:.3f}, L_te={test_loss:.3f} L_MAPE= {test_mape:.3f}')
    return predictor_train_state

def get_representation_model(
        config, 
        rng_init, 
        for_inference: bool = False
    ) -> Tuple[RepresentationTrainer, TrainState]:
    if config.method == Methods.cpc:
        trainer = CPCTrainer()
        model, train_state = trainer.get_model(rng_init, config)
        if for_inference:
            train_state = train_state.replace(
                apply_fn=trainer.apply_fn_encoder,
            )
    else:
        raise ValueError(f'Unknown method: {config.method}')
    
    return trainer, train_state





