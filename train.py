import dataclasses
import os
from sys import argv
import pickle
import random

import jax
import wandb
import pyrallis
import numpy as np


from utils.train_online import train_online
from utils.train_offline import train_offline
from utils.train_representation import get_representation_model, train_representation
from representation_models import PredictorTrainer
from utils.config import TrainConfig, Methods

def human_readable_size(size, decimal_places=2):  
    # I took this from stackoverflow somewhere
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
        if size < 1024.0 or unit == 'PiB':
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


def wandb_init(config: TrainConfig) -> str:
    run = wandb.init(
        config=dataclasses.asdict(config),
        project=config.project,
        group=config.group,
        name=config.name,
        id=config.run_id,
    )
    wandb.run.save()
    
    wandb.define_metric("online/step")
    wandb.define_metric("online/*", step_metric="online/step")
    wandb.define_metric("repr/step")
    wandb.define_metric("repr/*", step_metric="repr/step")
    wandb.define_metric("pred/step")
    wandb.define_metric("pred/*", step_metric="pred/step")
    wandb.define_metric("offline/step")
    wandb.define_metric("offline/*", step_metric="offline/step")

    wandb.run.log_code(".") 
    return run.dir

@pyrallis.wrap()
def pyrallis_main(config: TrainConfig):
    return main(config)

# @jax.disable_jit()
def main(config: TrainConfig):
    main_hip(config)

def main_hip(config: TrainConfig):
    """
    Run experiment in three stages:
    1) Online RL to generate Dataset
    2) VAE training
    3) Offline RL with VAE latent added to observation
    """
    wandb_run_dir = wandb_init(config)
    config.run_dir = wandb_run_dir
    rng = jax.random.PRNGKey(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    print('seed', config.seed)
    # Setup the model architecture
    print(config)

    base_name = f"{config.env_name_eval}{config.online.target_reward if config.online.target_reward is not None else ''}"
    dataset_fp = base_name + "_dataset.pkl"
    datasetpolicy_fp = base_name + "_datasetpolicy.pkl"
    force_train_online = False
    cached_dataset_fp = os.path.join('/local_storage/', dataset_fp)
    cached_datasetpolicy_fp = os.path.join('/local_storage/', datasetpolicy_fp)
    if os.path.exists(cached_dataset_fp):  # check for cached in node's local storage
        dataset_fp = cached_dataset_fp
        datasetpolicy_fp = cached_datasetpolicy_fp
    
    latent_dataset_fp = base_name + f"_dataset_latentadded{config.method.name}{config.vae.latent_dim}.pkl"
    representation_fp = base_name + f"_repr_{config.method.name}{config.vae.latent_dim}.pkl"
    predictor_fp = base_name + f"_pred_{config.method.name}{config.vae.latent_dim}.pkl"
    cached_latent_dataset_fp = os.path.join('/local_storage/', latent_dataset_fp)
    if os.path.exists(cached_latent_dataset_fp):  # check for cached in node's local storage
        latent_dataset_fp = cached_latent_dataset_fp
    cached_representation_fp = os.path.join('/local_storage/', representation_fp)
    if os.path.exists(cached_representation_fp):  # check for cached in node's local storage
        representation_fp = cached_representation_fp
    cached_predictor_fp = os.path.join('/local_storage/', predictor_fp)
    if os.path.exists(cached_predictor_fp):  # check for cached in node's local storage
        predictor_fp = cached_predictor_fp

    # train online and generate dataset
    rng, subkey  = jax.random.split(rng)
    repr_trainstate = repr_trainer = None
    pred_trainstate = pred_trainer = None
    if os.path.exists(latent_dataset_fp) and config.load_repr and not force_train_online:
        print('loading dataset from ', latent_dataset_fp)
        with open(latent_dataset_fp, "rb") as f:
            dataset = pickle.load(f)
            print('loaded')
        ds_network_ckpt = None
        print('loading representation trainstate from ', representation_fp)
        with open(representation_fp, "rb") as f:
            repr_params = pickle.load(f)
            print('loaded')
        repr_trainer, repr_trainstate = get_representation_model(config, subkey, for_inference=True)
        repr_trainstate = repr_trainstate.replace(params=repr_params)
        print('loading predictor trainstate from ', predictor_fp)
        with open(predictor_fp, "rb") as f:
            pred_params = pickle.load(f)
            print('loaded')
        pred_trainer = PredictorTrainer()
        _, pred_trainstate = pred_trainer.get_model(subkey, config)
        pred_trainstate = pred_trainstate.replace(params=pred_params)

    elif os.path.exists(dataset_fp) and not force_train_online:
        print('loading dataset from ', dataset_fp)
        config.load_repr = False
        with open(dataset_fp, "rb") as f:
            dataset = pickle.load(f)
            print('loaded')
        print('loading dataset policy from ', datasetpolicy_fp)
        ds_network_ckpt = None
    else:
        print('generating dataset')
        config.load_repr = False
        ds_network_ckpt, dataset = train_online(subkey, config)
        save_object(dataset_fp, cached_dataset_fp, dataset)
        save_object(datasetpolicy_fp, cached_datasetpolicy_fp, ds_network_ckpt)

    # train vae and relabel dataset
    rng, subkey  = jax.random.split(rng)
    if not config.method in [Methods.zero_hip, Methods.gt_hip, Methods.noisy_hip] and not config.load_repr:
        log_steps, log_return, repr_trainstate, pred_trainstate, dataset = train_representation(subkey, config, dataset)
        repr_params = repr_trainstate.params
        pred_params = pred_trainstate.params
        if not os.path.exists('/local_storage/'):
            save_object(latent_dataset_fp, cached_latent_dataset_fp, dataset)
            save_object(representation_fp, cached_representation_fp, repr_params)
            save_object(predictor_fp, cached_predictor_fp, pred_params)

        repr_trainer, repr_trainstate = get_representation_model(config, subkey, for_inference=True)
        repr_trainstate = repr_trainstate.replace(params=repr_params)

        rng, subkey  = jax.random.split(rng)
        pred_trainer = PredictorTrainer()
        _, pred_trainstate = pred_trainer.get_model(subkey, config)
        pred_trainstate = pred_trainstate.replace(params=pred_params)

    # train offline
    if not config.vae_only:
        rng, subkey  = jax.random.split(rng)
        log_steps, log_return, ds_network_ckpt = train_offline(
            subkey, config, dataset, repr_trainer, repr_trainstate, pred_trainer, pred_trainstate, ds_network_ckpt
        )
    return log_steps, log_return

def save_object(local_fp, cache_fp, obj, overwrite: bool = False):
    if overwrite or not os.path.exists(cache_fp):
        if os.path.exists('/local_storage/'):
            print(f'saving to {cache_fp} ')
            with open(cache_fp, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'saved object with size {human_readable_size(os.path.getsize(cache_fp))} to', cache_fp)
        print(f'saving to {local_fp}')
    if overwrite or not os.path.exists(local_fp):
        with open(local_fp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'saved object with size {human_readable_size(os.path.getsize(local_fp))} to', local_fp)
    else:
        print(f'skipping save, file already exists {local_fp}')

class bcolors:  #https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_used_config_path():
    config_element = [s for s in argv if s.startswith('--config_path=')]
    if config_element:
        config_element = config_element[0]
        config_path = config_element.split('=')[1]
        print(f'{bcolors.OKCYAN}{bcolors.BOLD} using config path', bcolors.UNDERLINE + config_path + bcolors.ENDC)
    else:
        print('no config path specified, using default config')


if __name__ == "__main__":
    print_used_config_path()
    pyrallis_main()
