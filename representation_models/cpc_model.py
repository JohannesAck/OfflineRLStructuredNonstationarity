from collections import defaultdict
from functools import partial
import time
from typing import Any, Callable, Optional, Tuple

import chex
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import wandb
from utils.DeploymentDataset import Dataset, DeploymentDataset, MinimalTransition, Transition
from representation_models.RepresentationTrainer import RepresentationTrainer

from utils.config import TrainConfig
from utils.helpers import get_env
from utils.visualization import linear_probe_callback, visualize_embeddings



def default_mlp_init(scale=0.05):
    return nn.initializers.uniform(scale)


class Encoder(nn.Module):
    num_hidden_units: int
    num_hidden_layers: int
    encoding_dim: int
    model_name: str = "encoder"

    @nn.compact
    def __call__(self, obs, act, rew, next_obs):
        x = jnp.concatenate([obs, act, rew, next_obs], axis=-1)
        x = nn.relu(
            nn.Dense(
                self.num_hidden_units,
                name=self.model_name + "_fc_1",
                bias_init=default_mlp_init(),
            )(x)
        )
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x = nn.relu(
                nn.Dense(
                    self.num_hidden_units,
                    name=self.model_name + f"_fc_{i+1}",
                    bias_init=default_mlp_init(),
                )(x)
            )
        encoding = nn.Dense(
            self.encoding_dim,
            name=self.model_name + "_encoding_out",
            bias_init=default_mlp_init(),
        )(x)
        return encoding


class ContrastiveMLP(nn.Module):
    num_hidden_units: int
    num_hidden_layers: int
    model_name: str = "constrastiveMLP"

    @nn.compact
    def __call__(self, context, sample):
        x = jnp.concatenate([context, sample], axis=-1)
        x = nn.elu(
            nn.Dense(
                self.num_hidden_units,
                name=self.model_name + "_fc_1",
                bias_init=default_mlp_init(),
            )(x)
        )
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x = nn.elu(
                nn.Dense(
                    self.num_hidden_units,
                    name=self.model_name + f"_fc_{i+1}",
                    bias_init=default_mlp_init(),
                )(x)
            )
        out = nn.Dense(
            1,
            name=self.model_name + "_fc_out",
            bias_init=default_mlp_init(),
        )(x)
        return out


class CPCModel(nn.Module):
    """
    CPC-Model

    """
    obs_dim: int
    act_dim: int
    num_hidden_units: int
    encoder_hidden_layers: int
    mlp_hidden_layers: int
    recurrent_state_dim: int
    trans_encoding_dim: int
    model_name: str = "cpc_model"
    rnn_class = nn.GRUCell

    def setup(self) -> None:
        self.encoder = Encoder(
            num_hidden_units=self.num_hidden_units,
            num_hidden_layers=self.encoder_hidden_layers,
            encoding_dim=self.trans_encoding_dim,
        )

        # lstm for hidden state recurrence
        self.rnn_cell = self.rnn_class(features=self.recurrent_state_dim, name=self.model_name + 'rnn_cell')

        self.mlp_head = ContrastiveMLP(
            num_hidden_units=self.num_hidden_units,
            num_hidden_layers=self.mlp_hidden_layers
        )

    def apply_encoder(self, obs, act, rew, next_obs):
        # necessary to call this on its own.
        return self.encoder(obs, act, rew, next_obs)

    def apply_mlp(self, context, transition_encoding):
        return self.mlp_head(context, transition_encoding)
    
    def apply_encoder_rnn(self, obs, act, rew, next_obs, prev_hidden_state) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Inference for a single timestep. Use jax.lax.scan to run over a trajectory.
        """
        encoding = self.apply_encoder(obs, act, rew, next_obs)
        
        new_hidden_state, rnn_out = self.rnn_cell(prev_hidden_state, encoding)
        
        # mlp_head will be applied outside the call function in the training function, I guess
        return new_hidden_state, rnn_out, encoding
    
    def __call__(self, obs, act, rew, next_obs, prev_hidden_state) -> Tuple[jax.Array, jax.Array, jax.Array]:
        # not really used, but needed for init to run through all modules
        encoding = self.apply_encoder(obs, act, rew, next_obs)
        
        context, rnn_out = self.rnn_cell(prev_hidden_state, encoding)

        prediction = self.apply_mlp(context, encoding)
        return context, encoding, prediction
        

class CPCTrainer(RepresentationTrainer):
    
    def __init__(self):
        self.apply_fn_encoder: Callable = None
        self.apply_fn_encoder_rnn: Callable = None
        self.apply_fn_mlp: Callable = None
        self.initial_rnn_state: jax.Array = None

    def get_model(
            self,
            rng, 
            config: TrainConfig, 
            obs_dim: Optional[int] = None, 
            action_dim: Optional[int] = None
        ) -> Tuple[CPCModel, TrainState]:
        """Instantiate a model according to obs shape of environment."""

        if obs_dim is None or action_dim is None:
            env, env_params = get_env(config.env_name, config.env_kwargs)
            if obs_dim is None:
                obs_dim = env.observation_space(env_params).shape[0]
            if action_dim is None:
                action_dim = env.action_space(env_params).shape[0]
            del env, env_params
        
        model = CPCModel(
            obs_dim=obs_dim,
            act_dim=action_dim,
            num_hidden_units=config.vae.num_hidden_units,
            encoder_hidden_layers=config.vae.encoder_hidden_layers,
            trans_encoding_dim=config.vae.cpc_trans_encoding_dim,
            mlp_hidden_layers=config.vae.decoder_hidden_layers,
            recurrent_state_dim=config.vae.cpc_context_encoding_dim,
        )
        carry_init = jnp.zeros((config.vae.latent_dim))

        rng, subkey = jax.random.split(rng)
        params = model.init(subkey, jnp.zeros(obs_dim), jnp.zeros(action_dim), jnp.zeros(1), jnp.zeros(obs_dim), carry_init)
        
        self.initial_rnn_state = jnp.zeros((config.vae.latent_dim))
        self.apply_fn_encoder = partial(model.apply, method=model.apply_encoder)
        self.apply_fn_encoder_rnn = partial(model.apply, method=model.apply_encoder_rnn)
        self.apply_fn_mlp = partial(model.apply, method=model.apply_mlp)
        train_state = TrainState.create(
            apply_fn=self.apply_fn_encoder_rnn,
            params=params,
            tx=optax.adam(config.vae.lr),
        )

        return model, train_state

    def sample_and_update(
        self,
        train_state: TrainState,
        dataset: Dataset,
        rng: chex.PRNGKey,
        config: TrainConfig,
    ) -> Tuple[dict, TrainState]:
        
        assert config.vae.n_updates_jit == 1

        metric_dict, train_state = sample_buff_and_update_n_times_cpc(
            train_state,
            self.apply_fn_encoder,
            self.apply_fn_mlp,
            dataset.transition,
            self.initial_rnn_state,
            rng,
            dataset.n_deployments,
            dataset.n_episodes_deployment,
            dataset.n_timesteps_episode,
            config.vae.context_len_cpc,
            config.vae.cpc_n_negative,
            config.vae.cpc_offset,
            config.vae.batch_size,
        )
        return metric_dict, train_state

    def eval(
        self,
        train_state: TrainState,
        dataset: Dataset,
        rng: chex.PRNGKey,
        config: TrainConfig,
    ):
        """
        Evaluates CPC learned representations on a sample of `n_eval_episodes` episodes from the dataset.
        """
        evaluation_dict = {}

        rng, subkey = jax.random.split(rng)
        # shape is [deployment, context_step, sampled_timestep, latent_dim]
        n_eval_deployments = config.vae.n_evaluation_episodes // dataset.n_episodes_deployment
        context_trans, trans_encs, context_encs = inference_on_evaluation_trajectories_cpc(
                        train_state,
                        dataset,
                        self.initial_rnn_state,
                        n_eval_deployments,
                        config.vae.context_len_cpc,
                        config.vae.cpc_offset,
                        rng=subkey
                    )

        ### transition encoding evaluation
        # average across the different timesteps sampled for each episode, but only valid use valid ones, i.e. not after the done flag was set
        valid_trans_enc_filter = ~context_trans.done

        # remove invalid transitions that were sampled after the done flag was set
        def reduce_mean_masked(x, mask, hips):
            mask_sum = jnp.sum(mask)
            mask_bc = jnp.broadcast_to(mask[:, None], x.shape)
            mean_x = jnp.sum(x * mask_bc, 0) / mask_sum  # this is expected to make nans, which are handled later
            mean_hip = jnp.sum(hips * mask) / mask_sum  # this is just a sanity check
            return mean_x, mean_hip
        trans_encs_mean, mean_trans_hip = jax.vmap(jax.vmap(reduce_mean_masked))(trans_encs, valid_trans_enc_filter, context_trans.hip)

        # flatten and remove nans
        trans_encs_mean = trans_encs_mean.reshape(-1, trans_encs_mean.shape[-1])
        mean_trans_hip = mean_trans_hip.reshape(-1, 1)
        isnan = np.any(np.isnan(trans_encs_mean), 1)
        trans_encs_mean = trans_encs_mean[~isnan]
        mean_trans_hip = mean_trans_hip[~isnan]

        pil_img_pca = visualize_embeddings(trans_encs_mean, mean_trans_hip, do_tsne=False)
        evaluation_dict['repr/transenc_pca'] = wandb.Image(pil_img_pca)

        rng, subkey = jax.random.split(rng)
        idx_tsne = jax.random.randint(subkey, (config.vae.n_tsne_evaluation_episodes,), 0, trans_encs_mean.shape[0])
        pil_img_tsne = visualize_embeddings(trans_encs_mean[idx_tsne], mean_trans_hip[idx_tsne], do_tsne=True)
        evaluation_dict['repr/transenc_tsne'] = wandb.Image(pil_img_tsne)

        hipprobe_acc_train, hipprobe_acc_test = linear_probe_callback(trans_encs_mean, mean_trans_hip, len(jnp.unique(mean_trans_hip)) < 10)
        evaluation_dict['repr/transhipprobe_acc_train'] = hipprobe_acc_train
        evaluation_dict['repr/transhipprobe_acc_test'] = hipprobe_acc_test

        ### context encoding evaluation
        # only use last hiddent_state, the previous ones are not valid context encodings:
        valid_context_enc_filter = ~jnp.any(context_trans.done, axis=1)
        context_encs_mean, mean_context_hip = jax.vmap(reduce_mean_masked)(context_encs[:,-1], valid_context_enc_filter, context_trans.hip[:,-1])
        # flatten and remove nans
        context_encs_mean = context_encs_mean.reshape(-1, context_encs_mean.shape[-1])
        mean_context_hip = mean_context_hip.reshape(-1, 1)
        isnan = np.any(np.isnan(context_encs_mean), 1)
        context_encs_mean = context_encs_mean[~isnan]
        mean_context_hip = mean_context_hip[~isnan]


        rng, subkey = jax.random.split(rng)
        # flatten = lambda x: x.reshape(-1, x.shape[-1])
        pil_img_pca = visualize_embeddings(context_encs_mean, mean_context_hip, do_tsne=False)
        evaluation_dict['repr/contextenc_pca'] = wandb.Image(pil_img_pca)
        rng, subkey = jax.random.split(rng)
        idx_tsne = jax.random.randint(subkey, (config.vae.n_tsne_evaluation_episodes,), 0, context_encs_mean.shape[0])
        pil_img_tsne = visualize_embeddings(context_encs_mean[idx_tsne], mean_context_hip[idx_tsne], do_tsne=True)
        evaluation_dict['repr/contextenc_tsne'] = wandb.Image(pil_img_tsne)

        hipprobe_acc_train, hipprobe_acc_test = linear_probe_callback(context_encs_mean, mean_context_hip, len(jnp.unique(mean_context_hip)) < 10)
        evaluation_dict['repr/contexthipprobe_acc_train'] = hipprobe_acc_train
        evaluation_dict['repr/contexthipprobe_acc_test'] = hipprobe_acc_test


        return evaluation_dict
       
    @partial(jax.jit, static_argnames=('self', 'config'))
    def inference_single_sample(
        self,
        train_state: TrainState,
        input_transitions: Transition,
        rng: chex.PRNGKey,
        config: TrainConfig,
    ) -> jax.Array:
        """
        Runs inference on a single episode, or multiple episodes depending on the model.
        Input is a list of input transitions, the shape depends on the method
        """
        if config.vae.use_trans_encoding:
            trans_encs = self.apply_fn_encoder(
                train_state.params, 
                input_transitions.obs,
                input_transitions.act,
                input_transitions.rew[..., None],
                input_transitions.next_obs,
            )
            def reduce_mean_masked(x, mask):
                mask_sum = jnp.sum(mask)
                mask_bc = jnp.broadcast_to(mask, x.shape)
                mean_x = jnp.sum(x * mask_bc, 0) / mask_sum
                return mean_x  
            latent = reduce_mean_masked(trans_encs, ~input_transitions.done[..., None])
        else:
            raise NotImplementedError('not implemented currently')

        return latent
    
    def inference_full_dataset(
        self,
        train_state: TrainState,
        dataset: Dataset,
        orig_n_episodes_deployment: int,
        rng: chex.PRNGKey,
        config: TrainConfig,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Runs inference on the full dataset to augment the states.
        """
        rng, subkey = jax.random.split(rng)

        if config.vae.use_trans_encoding:
            
            trans_encs, deployment_ids, deployment_steps = inference_on_full_dataset_transenc(
                train_state,
                self.apply_fn_encoder,
                dataset,
                rng=subkey,
            )
            latents = trans_encs
            validity_mask = jnp.ones_like(trans_encs)

        else:  # use context encoding
            raise ValueError('not supported')

        valid_deploystep_mask = validity_mask[0,:,0]
        return latents, valid_deploystep_mask



def get_loss(
    params: flax.core.frozen_dict.FrozenDict,
    encode_rnn_apply_function: Callable[...,Any],
    encode_apply_function: Callable[...,Any],
    mlp_apply_function: Callable[...,Any],
    context_transitions: Transition,
    positive_transitions: Transition,
    initial_hidden_state: Tuple[jnp.ndarray],
    n_negative: int,
    rng: chex.PRNGKey
):
    """Compute loss for CPC."""

    batch_size = context_transitions.obs.shape[0]  # this might need to be replaced by a staticarg for jit
    cpc_context = context_transitions.obs.shape[1]  # this might need to be replaced by a staticarg for jit

    _, hidden_states = inference_encoder_and_gru(  # jitted
        params,
        encode_rnn_apply_function,
        context_transitions.obs,
        context_transitions.act,
        context_transitions.rew[:,:, None],
        context_transitions.next_obs,
        initial_hidden_state,
        cpc_context
    )
    context = hidden_states[:,-1]
    
    encoded_positives = jax.jit(encode_apply_function)(
        params,
        positive_transitions.obs,
        positive_transitions.act,
        positive_transitions.rew[..., None],
        positive_transitions.next_obs
    )

    # shuffle positives to become negatives
    # just positive is fine because they're already shuffled randomly
    offset = jax.random.randint(rng, (n_negative,), 1, batch_size)
    offset_indices = jnp.tile(jnp.arange(batch_size)[:, None], [1, n_negative]) + jnp.tile(offset, [batch_size, 1])
    offset_indices = offset_indices % batch_size
    encoded_negatives = encoded_positives[offset_indices]

    positive_mlp_predictions = jax.jit(mlp_apply_function)(params, context, encoded_positives)
    
    context_rep = jnp.tile(context[:, None, :], [1, n_negative, 1])
    negative_mlp_predictions = jax.jit(mlp_apply_function)(params, context_rep, encoded_negatives)

    loss = optax.sigmoid_binary_cross_entropy(
        logits=jnp.concatenate([positive_mlp_predictions.flatten(), negative_mlp_predictions.flatten()]),
        labels=jnp.concatenate(
            [
                jnp.ones((batch_size,)),
                jnp.zeros((batch_size * n_negative,))
            ]
        )
    )
    # if any of the context or positive transitions are already done, the whole loss is invalid
    # if negative transitions are already done it might be okay to still use the loss, as it'll be from a different hip than it should be
    loss_invalid_mask = jnp.any(context_transitions.done, axis=-1) | positive_transitions.done
    loss_invalid_mask = jnp.tile(loss_invalid_mask, n_negative+1)
    loss = loss * (1.0 - loss_invalid_mask)
    loss = loss.mean() / ((1.0 - loss_invalid_mask).mean() + 1e-8)  # normalize by the number of valid losses
    # breakpoint_if_nonfinite(loss)

    accuracy = (positive_mlp_predictions > 0.0).sum() + (negative_mlp_predictions < 0.0).sum()
    accuracy = accuracy / (batch_size * (n_negative + 1))
    recall = (positive_mlp_predictions > 0.0).mean()
    return loss, (accuracy, recall)

@partial(jax.jit, static_argnums=(1,2,5))

def update(train_state, apply_fn_encoder, apply_fn_mlp, context_transitions, positive_transitions, n_negative, initial_rnn_state, subkey):

    loss_fn = jax.value_and_grad(get_loss, has_aux=True)
    loss, grads = loss_fn(
            train_state.params,
            train_state.apply_fn,
            apply_fn_encoder,
            apply_fn_mlp,
            context_transitions,
            positive_transitions,
            initial_rnn_state,
            n_negative,
            subkey,
        )
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss, grads


def get_cpc_loss_negatives(
    params: flax.core.frozen_dict.FrozenDict,
    encode_rnn_apply_function: Callable[...,Any],
    encode_apply_function: Callable[...,Any],
    mlp_apply_function: Callable[...,Any],
    context_transitions: Transition,
    positive_transitions: Transition,
    negative_transitions: Transition,
    initial_hidden_state: Tuple[jnp.ndarray],
    n_negative: int,
    rng: chex.PRNGKey
):
    """Compute loss for CPC with real negative samples."""

    batch_size = context_transitions.obs.shape[0]  # this might need to be replaced by a staticarg for jit
    cpc_context = context_transitions.obs.shape[1]  # this might need to be replaced by a staticarg for jit

    _, hidden_states = inference_encoder_and_gru(  # jitted
        params,
        encode_rnn_apply_function,
        context_transitions.obs,
        context_transitions.act,
        context_transitions.rew[:,:, None],
        context_transitions.next_obs,
        initial_hidden_state,
        cpc_context
    )
    context = hidden_states[:,-1]
    
    encoded_positives = jax.jit(encode_apply_function)(
        params,
        positive_transitions.obs,
        positive_transitions.act,
        positive_transitions.rew[..., None],
        positive_transitions.next_obs
    )

    encoded_negatives = jax.jit(encode_apply_function)(
        params,
        negative_transitions.obs,
        negative_transitions.act,
        negative_transitions.rew[..., None],
        negative_transitions.next_obs
    )

    positive_mlp_predictions = jax.jit(mlp_apply_function)(params, context, encoded_positives)
    
    context_rep = jnp.tile(context[:, None, :], [1, n_negative, 1])
    negative_mlp_predictions = jax.jit(mlp_apply_function)(params, context_rep, encoded_negatives)

    loss = optax.sigmoid_binary_cross_entropy(
        logits=jnp.concatenate([positive_mlp_predictions.flatten(), negative_mlp_predictions.flatten()]),
        labels=jnp.concatenate(
            [
                jnp.ones((batch_size,)),
                jnp.zeros((batch_size * n_negative,))
            ]
        )
    )
    # if any of the context or positive transitions are already done, the whole loss is invalid
    # if negative transitions are already done it might be okay to still use the loss, as it'll be from a different hip than it should be
    loss_invalid_mask = jnp.any(context_transitions.done, axis=-1) | positive_transitions.done
    loss_invalid_mask = jnp.tile(loss_invalid_mask, n_negative+1)
    loss = loss * (1.0 - loss_invalid_mask)
    loss = loss.mean() / (1.0 - loss_invalid_mask).mean()  # normalize by the number of valid losses
    # breakpoint_if_nonfinite(loss)

    accuracy = (positive_mlp_predictions > 0.0).sum() + (negative_mlp_predictions < 0.0).sum()
    accuracy = accuracy / (batch_size * (n_negative + 1))
    recall = (positive_mlp_predictions > 0.0).mean()
    return loss, (accuracy, recall)


def update_cpc_negatives(
        train_state_cpc, 
        apply_fn_encoder, 
        apply_fn_mlp, 
        context_transitions, 
        positive_transitions, 
        negative_transitions,
        n_negative, 
        initial_rnn_state,
        subkey
    ):
    loss = get_cpc_loss_negatives( # debugging
        train_state_cpc.params,
        train_state_cpc.apply_fn,
        apply_fn_encoder,
        apply_fn_mlp,
        context_transitions,
        positive_transitions,
        negative_transitions,
        initial_rnn_state,
        n_negative,
        subkey,
    )

    loss_fn = jax.value_and_grad(get_cpc_loss_negatives, has_aux=True)
    loss, grads = loss_fn(
            train_state_cpc.params,
            train_state_cpc.apply_fn,
            apply_fn_encoder,
            apply_fn_mlp,
            context_transitions,
            positive_transitions,
            negative_transitions,
            initial_rnn_state,
            n_negative,
            subkey,
        )
    train_state_cpc = train_state_cpc.apply_gradients(grads=grads)
    return train_state_cpc, loss, grads

# @jax.disable_jit()
@partial(jax.jit, static_argnames=('n_deployments', 'episodes_per_deployment','steps_per_episode','batch_size',
                                    'context_len', 'prediction_offset', 'n_negative', 'apply_fn_encoder', 'apply_fn_mlp'))

def sample_buff_and_update_n_times_cpc(
    train_state_cpc: TrainState,
    apply_fn_encoder,
    apply_fn_mlp,
    minimal_transition: MinimalTransition,
    initial_rnn_state: jnp.ndarray,
    rng: chex.PRNGKey,
    n_deployments: int,
    episodes_per_deployment: int,
    steps_per_episode: int,
    context_len: int,
    n_negative: int,
    prediction_offset: int,
    batch_size: int,
):
    """
    Version of the above function for recurrent VAEs.
    """
    avg_metrics_dict = defaultdict(int)
    
    rng, subkey = jax.random.split(rng)
    context_transitions, positive_transitions = DeploymentDataset.sample_contrastive_batch_faster(
        minimal_transition, batch_size, context_len, prediction_offset,
        n_deployments, episodes_per_deployment, steps_per_episode, subkey, replace=True
    )
    rng, subkey = jax.random.split(rng)
    train_state_cpc, loss, grads = update(
        train_state_cpc, 
        apply_fn_encoder, 
        apply_fn_mlp, 
        context_transitions, 
        positive_transitions, 
        n_negative, 
        initial_rnn_state, 
        subkey
    )


    avg_metrics_dict["repr/repr_grad_norm"] = jnp.mean(jnp.array(jax.tree_util.tree_flatten(jax.tree_map(jnp.linalg.norm, grads))[0]))
    avg_metrics_dict["repr/total_loss"] = loss[0]
    avg_metrics_dict["repr/cpc_accuracy"] = loss[1][0]
    avg_metrics_dict["repr/cpc_recall"] = loss[1][1]


    return avg_metrics_dict, train_state_cpc


@partial(jax.jit, static_argnames=('encoder_rnn_apply_fn', 'cpc_context'))
def inference_encoder_and_gru(params, encoder_rnn_apply_fn, obs_batch, act_batch, rew_batch, next_obs_batch, initial_hidden_state, cpc_context):
    hidden_state = initial_hidden_state

    def body_fn(carry, x):
        hidden_state = carry
        new_hidden_state, rnn_out, encoding = encoder_rnn_apply_fn(params, *x, hidden_state)
        return new_hidden_state, (new_hidden_state, rnn_out, encoding)

    x = (obs_batch, act_batch, rew_batch, next_obs_batch)
    x = jax.tree_map(lambda a: jnp.swapaxes(a, 0, 1), x)
    hidden_state_tiled = jax.tree_map(lambda x: jnp.tile(x, [obs_batch.shape[0], 1]), hidden_state)
    
    final_carry, scan_out = jax.lax.scan(body_fn, hidden_state_tiled, x, unroll=min(16, cpc_context))
    
    scan_out = jax.tree_map(lambda a: jnp.swapaxes(a, 0, 1), scan_out)
    hidden_states, rnn_outs, encodings = scan_out  # for GRU hidden_states and rnn_outs are the same
    
    return encodings, hidden_states


def batched_inference(train_state, encoder_rnn_apply_fn, context_trans, initial_hidden_state, cpc_context, batch_size):
    n_batch = np.ceil(context_trans.obs.shape[0] / batch_size).astype(int)
    trans_encs = []
    context_encs = []
    for idx_batch in tqdm.trange(n_batch, desc='VAE Inference', disable=n_batch <= 30):

        trans_encs_batch, context_encs_batch = jax.jit(jax.vmap(inference_encoder_and_gru, in_axes=(None, None, 0, 0, 0, 0, None, None)), static_argnums=(1,7))(
            jax.lax.stop_gradient(train_state.params),
            encoder_rnn_apply_fn,
            context_trans.obs[idx_batch * batch_size: (idx_batch + 1) * batch_size],
            context_trans.act[idx_batch * batch_size: (idx_batch + 1) * batch_size],
            context_trans.rew[idx_batch * batch_size: (idx_batch + 1) * batch_size][..., None],
            context_trans.next_obs[idx_batch * batch_size: (idx_batch + 1) * batch_size],
            initial_hidden_state,  # this might break if switching to an LSTM
            cpc_context
        ) 

        context_encs_batch = jax.device_put(context_encs_batch, jax.devices('cpu')[0])
        trans_encs_batch = jax.device_put(trans_encs_batch, jax.devices('cpu')[0])
        context_encs.append(context_encs_batch)
        trans_encs.append(trans_encs_batch)

    trans_encs = jnp.concatenate(trans_encs, axis=0)
    context_encs = jnp.concatenate(context_encs, axis=0)
    return trans_encs,context_encs


def inference_on_evaluation_trajectories_cpc(
        train_state: TrainState,
        dataset: Dataset,
        initial_hidden_state: Tuple[jnp.ndarray, jnp.ndarray],
        n_deploy_eval: int,
        cpc_context: int,
        cpc_offset: int,
        rng: chex.PRNGKey,
    ) -> Tuple[Transition, chex.Array, chex.Array]:
    """
    Get embeddings of each transition and the representation of the RNN for 
    almost each trajectory in the buffer.
    """
    rng, subkey = jax.random.split(rng)

    # inference on a random subsample of the buffer
    n_timestep_eval = min(16, dataset.n_timesteps_episode)
    n_deploystep_samples = min(16, dataset.n_episodes_deployment - cpc_context - cpc_offset + 1)
    context_trans, _ = DeploymentDataset.sample_contrastive_evaluation_batch_cpc(
        dataset, n_deploy_eval, cpc_context, cpc_offset, dataset.n_deployments, dataset.n_episodes_deployment, dataset.n_timesteps_episode, n_timestep_eval, n_deploystep_samples, subkey
    )

    if context_trans.obs.shape[1] < 16:
        batch_size = min(1024, context_trans.obs.shape[0])
    else:
        batch_size = min(64, context_trans.obs.shape[0])

    trans_encs, context_encs = batched_inference(train_state, train_state.apply_fn, context_trans, initial_hidden_state, cpc_context, batch_size)

    return context_trans, trans_encs, context_encs


def inference_on_full_dataset(
        train_state: TrainState,
        dataset: Dataset,
        initial_hidden_state: Tuple[jnp.ndarray, jnp.ndarray],
        cpc_context: int,
        cpc_offset: int,
        rng: chex.PRNGKey,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Get embeddings of each transition and the representation of the RNN for 
    almost each trajectory in the buffer.
    """

    # jittable!
    def sample_and_inference_batch(train_state, encoder_rnn_apply_fn, dataset, deployment_ids, initial_hidden_state, episodes_per_deployment, steps_per_episode, cpc_context, cpc_offset, subkey):
        context_trans, positive_trans = DeploymentDataset.sample_given_deployments_for_cpc(
            dataset, deployment_ids, cpc_context, cpc_offset,
            episodes_per_deployment, steps_per_episode, steps_per_episode, episodes_per_deployment, subkey 
        )
        # reorder form [deployment_step_sample, deployment_id,...] to [deployment_id, deployment_step_sample,...]
        context_trans = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), context_trans)
        positive_trans = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), positive_trans)

        # flatten for inference
        previous_shape = context_trans.obs.shape[:2]
        context_trans = jax.tree_map(jnp.concatenate, context_trans)
        positive_trans = jax.tree_map(jnp.concatenate, positive_trans)

        trans_encs_batch, context_encs_batch = jax.vmap(inference_encoder_and_gru, in_axes=(None, None, 0, 0, 0, 0, None, None))(
            jax.lax.stop_gradient(train_state.params),
            encoder_rnn_apply_fn,
            context_trans.obs,
            context_trans.act,
            context_trans.rew[..., None],
            context_trans.next_obs,
            initial_hidden_state,  # this might break if switching to an LSTM
            cpc_context
        )
        # restore to [deployment_id, deploymen_step_sample]
        trans_encs_batch = jnp.reshape(trans_encs_batch, [previous_shape[0], previous_shape[1], cpc_context, steps_per_episode, -1])
        context_encs_batch = jnp.reshape(context_encs_batch, [previous_shape[0], previous_shape[1], cpc_context, steps_per_episode, -1])
        
        # deployment_id and step for debugging mostly
        deployment_ids = jnp.reshape(context_trans.deployment_id, [previous_shape[0], previous_shape[1], cpc_context, steps_per_episode, -1])
        deployment_step = jnp.reshape(context_trans.deployment_step, [previous_shape[0], previous_shape[1], cpc_context, steps_per_episode, -1])
        dones_batch = jnp.reshape(context_trans.done, [previous_shape[0], previous_shape[1], cpc_context, steps_per_episode, -1])
        return deployment_ids, deployment_step, trans_encs_batch, context_encs_batch, dones_batch
    
    # jitted_sample_inference = sample_and_inference_batch
    jitted_sample_inference = jax.jit(sample_and_inference_batch, static_argnums=(1, 5, 6, 7, 8))

    rng, subkey = jax.random.split(rng)

    if dataset.transition.obs.shape[1] < 16:  # if deployments are small we can use bigger batches (batch_size given in number of deployments)
        batch_size = min(1024, dataset.n_deployments)
    else:
        batch_size = min(64, dataset.n_deployments)
    
    n_batch = np.ceil(dataset.n_deployments / batch_size).astype(int)
    trans_encs = []
    context_encs = []
    deployment_ids = []
    deployment_steps = []
    for idx_batch in tqdm.trange(n_batch, desc='CPC final inference'):
        if (idx_batch + 1) * batch_size > dataset.n_deployments:
            deployment_ids_batch_sample = jnp.arange(idx_batch * batch_size, dataset.n_deployments, dtype=jnp.int32)
        else:
            deployment_ids_batch_sample = jnp.arange(idx_batch * batch_size, (idx_batch + 1) * batch_size, dtype=jnp.int32)
        deployment_ids_batch, deployment_steps_batch, trans_encs_batch, context_encs_batch, dones_batch = jitted_sample_inference(
            train_state, train_state.apply_fn, dataset, deployment_ids_batch_sample, initial_hidden_state, 
            dataset.n_episodes_deployment, dataset.n_timesteps_episode,cpc_context, cpc_offset, subkey
        )

        # average over timestep_samples before done
        def reduce_mean_masked(x, mask):
            mask_sum = jnp.sum(mask)
            mask_bc = jnp.broadcast_to(mask, x.shape)
            mean_x = jnp.sum(x * mask_bc, 0) / mask_sum
            return mean_x  
        context_encs_batch_mean = jax.vmap(jax.vmap(reduce_mean_masked))(context_encs_batch[:,:,-1], ~dones_batch[:,:,-1,:])
        trans_encs_batch_mean = jax.vmap(jax.vmap(jax.vmap(reduce_mean_masked)))(trans_encs_batch, ~dones_batch)
        deployment_ids_batch_mean = jax.vmap(jax.vmap(jax.vmap(reduce_mean_masked)))(deployment_ids_batch, ~dones_batch)
        deployment_steps_batch_mean = jax.vmap(jax.vmap(jax.vmap(reduce_mean_masked)))(deployment_steps_batch, ~dones_batch)

        assert context_encs_batch_mean.shape[-1] == context_encs_batch.shape[-1]
        assert trans_encs_batch_mean.shape[-1] == trans_encs_batch_mean.shape[-1]
        
        context_encs_batch_mean = jax.device_put(context_encs_batch_mean, jax.devices('cpu')[0])
        trans_encs_batch_mean = jax.device_put(trans_encs_batch_mean, jax.devices('cpu')[0])
        deployment_ids_batch_mean = jax.device_put(deployment_ids_batch_mean, jax.devices('cpu')[0])
        deployment_steps_batch_mean = jax.device_put(deployment_steps_batch_mean, jax.devices('cpu')[0])
        context_encs.append(context_encs_batch_mean)
        trans_encs.append(trans_encs_batch_mean)
        deployment_ids.append(deployment_ids_batch_mean)
        deployment_steps.append(deployment_steps_batch_mean)

    trans_encs = jnp.concatenate(trans_encs, axis=0)
    context_encs = jnp.concatenate(context_encs, axis=0)
    deployment_ids = jnp.concatenate(deployment_ids, axis=0)
    deployment_steps = jnp.concatenate(deployment_steps, axis=0)


    return trans_encs, context_encs, deployment_ids, deployment_steps


def inference_on_full_dataset_transenc(
        train_state: TrainState,
        encoder_apply_fn: Callable,
        dataset: Dataset,
        rng: chex.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """
    Get embeddings of each transition and the representation of the RNN for 
    almost each trajectory in the buffer.
    """

    # jittable!
    def sample_and_inference_batch(train_state, encoder_apply_fn, dataset: Dataset, deployment_ids):
        transitions = jax.tree_map(lambda x: x[deployment_ids], dataset.transition)
        trans_encs_batch = encoder_apply_fn(
            jax.lax.stop_gradient(train_state.params),
            transitions.obs,
            transitions.act,
            transitions.rew[..., None],
            transitions.next_obs,
        )
        return transitions.deployment_id, transitions.deployment_step, trans_encs_batch, transitions.done
    
    # jitted_sample_inference = sample_and_inference_batch
    jitted_sample_inference = jax.jit(sample_and_inference_batch, static_argnums=(1))

    rng, subkey = jax.random.split(rng)

    if dataset.transition.obs.shape[-1] > 300:
        batch_size = min(16, dataset.n_deployments)
    elif dataset.transition.obs.shape[1] < 16:  # if deployments are small we can use bigger batches (batch_size given in number of deployments)
        batch_size = min(1024, dataset.n_deployments)
    else:
        batch_size = min(64, dataset.n_deployments)
    
    n_batch = np.ceil(dataset.n_deployments / batch_size).astype(int)
    trans_encs = []
    deployment_ids = []
    deployment_steps = []
    for idx_batch in tqdm.trange(n_batch, desc='CPC final inference'):
        if (idx_batch + 1) * batch_size > dataset.n_deployments:
            deployment_ids_batch_sample = jnp.arange(idx_batch * batch_size, dataset.n_deployments, dtype=jnp.int32)
        else:
            deployment_ids_batch_sample = jnp.arange(idx_batch * batch_size, (idx_batch + 1) * batch_size, dtype=jnp.int32)
        deployment_ids_batch, deployment_steps_batch, trans_encs_batch, dones_batch = jitted_sample_inference(
            train_state, encoder_apply_fn, dataset, deployment_ids_batch_sample,
        )
        # average over timestep_samples before done

        def reduce_mean_masked(x, mask):
            mask_sum = jnp.sum(mask)
            mask_bc = jnp.broadcast_to(mask, x.shape)
            mean_x = jnp.sum(x * mask_bc, 0) / mask_sum
            return mean_x  
        trans_encs_batch_mean = jax.vmap(jax.vmap(reduce_mean_masked))(trans_encs_batch, ~dones_batch[..., None])
        deployment_ids_batch_mean = jax.vmap(jax.vmap(reduce_mean_masked))(deployment_ids_batch[..., None], ~dones_batch[..., None])
        deployment_steps_batch_mean = jax.vmap(jax.vmap(reduce_mean_masked))(deployment_steps_batch[..., None], ~dones_batch[..., None])

        assert trans_encs_batch_mean.shape[-1] == trans_encs_batch_mean.shape[-1]
        
        trans_encs_batch_mean = jax.device_put(trans_encs_batch_mean, jax.devices('cpu')[0])
        deployment_ids_batch_mean = jax.device_put(deployment_ids_batch_mean, jax.devices('cpu')[0])
        deployment_steps_batch_mean = jax.device_put(deployment_steps_batch_mean, jax.devices('cpu')[0])
        trans_encs.append(trans_encs_batch_mean)
        deployment_ids.append(deployment_ids_batch_mean)
        deployment_steps.append(deployment_steps_batch_mean)

    trans_encs = jnp.concatenate(trans_encs, axis=0)
    deployment_ids = jnp.concatenate(deployment_ids, axis=0)
    deployment_steps = jnp.concatenate(deployment_steps, axis=0)


    return trans_encs, deployment_ids, deployment_steps

