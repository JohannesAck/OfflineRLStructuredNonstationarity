

from functools import partial
from typing import Any, Callable, Optional, Tuple
import PIL.Image
import chex
import einops
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
import optax
import sklearn.preprocessing, sklearn.linear_model, sklearn.metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tqdm
import wandb

from utils.config import TrainConfig
from utils.visualization import fig2img

class LatentPredictorModel(nn.Module):
    latent_dim: int
    recurrent_dim: int
    num_hidden_units: int
    num_hidden_layers: int
    model_name: str = 'latent_predictor'

    def setup(self) -> None:
        self.fc = [
            nn.Dense(features=self.num_hidden_units, name=self.model_name + '_fc_0')
            for _ in range(self.num_hidden_layers)
        ]
        self.scan_cell = nn.transforms.scan(
            nn.GRUCell, 
            variable_broadcast='params', 
            split_rngs={'params': False},
            in_axes=1, out_axes=1
        )(features=self.recurrent_dim, name=self.model_name + '_rnn_cell')
        self.output_layer = nn.Dense(features=self.latent_dim, name=self.model_name + '_output_layer') 

    @nn.compact
    def __call__(self, latent, rng) -> Any:
        x = latent
        for hidden_layer in self.fc:
            x = nn.relu(hidden_layer(x))

        carry_in = self.scan_cell.initialize_carry(rng, x[:, 0].shape)
        carry, outputs = self.scan_cell(carry_in, x)
        x = self.output_layer(carry) 
        # actually maybe these are the same for GRUs
        return x

class PredictorTrainer(object):
    def get_model(
            self, 
            rng: chex.PRNGKey,
            config: TrainConfig,
        ) -> Tuple[LatentPredictorModel, TrainState]:
        model = LatentPredictorModel(
            latent_dim=config.vae.latent_dim,
            recurrent_dim=config.predictor.recurrent_dim,
            num_hidden_units=config.predictor.num_hidden_units,
            num_hidden_layers=config.predictor.num_hidden_layers
        )

        rng, subkey = jax.random.split(rng)
        seqlen = config.vae.recenc_vae_seqlen
        bs = config.vae.batch_size
        params = model.init(subkey, jnp.zeros((bs, seqlen, config.vae.latent_dim)), rng=subkey)
        
        train_state = TrainState.create(
                apply_fn=model.apply,
                params=params,
                tx=optax.adam(config.predictor.lr),
        )

        return model, train_state

    def train_epoch(
            self, 
            train_latents: jnp.ndarray,
            train_state: TrainState,
            rng: chex.PRNGKey,
            config: TrainConfig,
        ) -> Tuple[TrainState, float]:
        """
        Train the model for one epoch.
        Returns the updated train_state and the mean training loss.
        """

        batch_num = np.ceil(train_latents.shape[0] / config.predictor.batch_size).astype(int).item()

        n_deployments = train_latents.shape[0]
        n_episodes_per_deployment = train_latents.shape[1]

        
        mean_loss = 0.0
        for batch_idx in range(batch_num):
            rng, subkey = jax.random.split(rng)
            train_state, loss = sample_batch_and_update(
                train_state,
                train_latents,
                config.predictor.batch_size, 
                config.predictor.context_length,
                n_deployments,
                n_episodes_per_deployment,
                subkey
            )
            mean_loss += loss
        
        mean_loss /= batch_num
        mean_loss = mean_loss.item()
        return train_state, mean_loss

    def evaluate(
        self,
        test_latents: jnp.ndarray,
        train_state: TrainState,
        rng: chex.PRNGKey,
        config: TrainConfig,
    ) -> Tuple[float, float]:
        """
        Evaluate the model on the test set.
        Returns the mean test loss.
        """
        n_deployments = test_latents.shape[0]
        n_episodes_per_deployment = test_latents.shape[1]
        batch_num = np.ceil(test_latents.shape[0] / config.predictor.batch_size).astype(int).item()

        mean_loss = 0.0
        mape = 0.0
        for batch_idx in range(batch_num):
            rng, rng_batch = jax.random.split(rng)
            batch = sample_batch(
                test_latents,
                config.predictor.batch_size,
                config.predictor.context_length + 1,
                n_deployments=n_deployments,
                n_episodes_per_deployment=n_episodes_per_deployment,
                rng=rng_batch,
            )
            loss = get_mse_loss(train_state.params, train_state.apply_fn, batch[:, :-1], batch[:, -1], rng_batch)
            mape += jnp.mean(jnp.abs((train_state.apply_fn(train_state.params, batch[:, :-1], rng_batch) - batch[:, -1]) / batch[:, -1]))
            mean_loss += loss
        
        mean_loss /= batch_num
        mape /= batch_num
        return mean_loss.item(), mape.item()

    @partial(jax.jit, static_argnames=('self'))
    def infer_on_single_context(
        self,
        context_latents: jnp.ndarray,
        train_state: TrainState,
        rng: chex.PRNGKey,
    ) -> jnp.ndarray:
        """
        Infer the next latent given a context.
        """
        pred_next_latent = train_state.apply_fn(train_state.params, context_latents[None], rng)
        return pred_next_latent
    
    def inference_full_dataset(
            self,
            input_latents: jnp.ndarray,
            train_state: TrainState,
            rng: chex.PRNGKey,
            config: TrainConfig,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Infer on full dataset
        """
        n_deployments = input_latents.shape[0]
        n_episodes_per_deployment = input_latents.shape[1]
        batch_size = config.predictor.batch_size
        batch_num = np.ceil(input_latents.shape[0] / batch_size).astype(int).item()

        inferred_latent_steps = jnp.arange(config.predictor.context_length, n_episodes_per_deployment)

        def get_batch_element(deployment_idx, inferred_step):
            return jax.lax.dynamic_slice_in_dim(
                input_latents[deployment_idx], inferred_step - config.predictor.context_length, config.predictor.context_length
            )
        vmapped_over_inferred_steps = jax.vmap(get_batch_element, in_axes=(None, 0))
        vmapped_over_deploys = jax.vmap(vmapped_over_inferred_steps, in_axes=(0, None))
        batched_inference = jax.jit(jax.vmap(train_state.apply_fn, in_axes=(None, 0, 0)))

        pred_latent_list = []
        for batch_idx in tqdm.trange(batch_num, desc='predictor inference'):
            deployment_idxs = jnp.arange(batch_idx * batch_size, min((batch_idx + 1) * batch_size, n_deployments))

            batch = vmapped_over_deploys(deployment_idxs, inferred_latent_steps)
            
            rng, subkey = jax.random.split(rng)
            rng_inf = jax.random.split(subkey, deployment_idxs.shape[0])
            pred_next_latent = batched_inference(train_state.params, batch, rng_inf)
            
            expected_next_latent = input_latents[deployment_idxs][:,inferred_latent_steps]
            mse = jnp.mean(jnp.square(pred_next_latent - expected_next_latent))  # for debugging  
            
            pred_latent_list.append(pred_next_latent)
        pred_latents = jnp.concatenate(pred_latent_list, axis=0)

        valid_deploystep_mask = jnp.ones(input_latents.shape[1])
        valid_deploystep_mask = valid_deploystep_mask.at[:config.predictor.context_length].set(0)

        total_mse = np.mean(np.abs((input_latents[:,valid_deploystep_mask.astype(bool)] - pred_latents)))  # debugging
        return pred_latents, valid_deploystep_mask


    def visualize_result(
        self,
        test_latents: jnp.ndarray,
        test_hips: jnp.ndarray,
        train_state: TrainState,
        rng: chex.PRNGKey,
        config: TrainConfig,
    ) -> dict:
        n_deployments = test_latents.shape[0]
        n_episodes_per_deployment = test_latents.shape[1]

        # slice up test test_latents into contexts sequences
        start_times = jnp.arange(n_episodes_per_deployment - config.predictor.context_length)
        def get_single_sequence(deployment, hips_deploy, start_time):
            context = jax.lax.dynamic_slice_in_dim(deployment, start_time, config.predictor.context_length)
            context_hips = jax.lax.dynamic_slice_in_dim(hips_deploy, start_time, config.predictor.context_length)
            target = jax.lax.dynamic_slice_in_dim(deployment, start_time + config.predictor.context_length, 1)
            target_hip = jax.lax.dynamic_slice_in_dim(hips_deploy, start_time + config.predictor.context_length, 1)
            return context, context_hips, target, target_hip    
        
        vmapped_on_starts = jax.vmap(get_single_sequence, in_axes=(None, None, 0))
        vmapped_on_deploy = jax.vmap(vmapped_on_starts, in_axes=(0, 0, None))
        contexts, context_hips, targets, target_hips = vmapped_on_deploy(test_latents, test_hips, start_times)
        contexts_flat = contexts.reshape(-1, config.predictor.context_length, config.vae.latent_dim)
        targets_flat = targets.reshape(-1, 1, config.vae.latent_dim)
        target_hips_flat = target_hips.reshape(-1, 1, 1)

        # infer next latent given context
        rng_infer = jax.random.split(rng, contexts_flat.shape[0])
        inferred_next_latents = jax.vmap(self.infer_on_single_context, in_axes=(0, None, 0))(contexts_flat, train_state, rng_infer)

        # mse = jnp.mean(jnp.square(inferred_next_latents - targets_flat)).item()
        # print(f'mse in eval {mse:.3f}')  # just as a sanity check basically
        
        # evaluation_dict = {'pred/mse': mse}
        evaluation_dict = {}

        inferred_next_latents = np.array(inferred_next_latents[:, 0])
        targets_flat = np.array(targets_flat[:, 0])
        target_hips_flat = np.array(target_hips_flat[:, 0])

        probe_acc_train, probe_acc_test = get_linear_probeacc_realtrain_faketest(
            train_latents=contexts_flat,
            test_latents=inferred_next_latents,
            train_hips=context_hips,
            test_hips=target_hips_flat,
        )
        evaluation_dict['pred/probe_acc_inf'] = probe_acc_train
        evaluation_dict['pred/probe_acc_pred'] = probe_acc_test
        
        visualization_idxes = np.random.choice(len(inferred_next_latents), size=300, replace=False)
        pil_img_pca = visualize_embeddings_fakereal(
            inferred_next_latents[visualization_idxes],
            targets_flat[visualization_idxes],
            target_hips_flat[visualization_idxes],
            do_tsne=False
        )
        evaluation_dict['pred/vis_pca'] = wandb.Image(pil_img_pca)

        pil_img_tsne = visualize_embeddings_fakereal(
            inferred_next_latents[visualization_idxes],
            targets_flat[visualization_idxes],
            target_hips_flat[visualization_idxes],
            do_tsne=True
        )
        evaluation_dict['pred/vis_tsne'] = wandb.Image(pil_img_tsne)
        
        return evaluation_dict

def get_linear_probeacc_realtrain_faketest(
        train_latents: jnp.ndarray,
        test_latents: jnp.ndarray,
        train_hips: jnp.ndarray,
        test_hips: jnp.ndarray,
) -> Tuple[float, float]:

    # convert labels from float to int
    train_latents_flat = einops.rearrange(train_latents, 'n m d -> (n m) d')
    train_hips_flat = train_hips.reshape(-1, train_latents.shape[1], 1)
    train_hips_flat = einops.rearrange(train_hips_flat, 'n m 1 -> (n m) 1')
    
    unique_labels = np.unique(train_hips_flat)
    classification_labels = np.zeros(train_hips_flat.shape, dtype=np.int32)
    for idx, label_it in enumerate(unique_labels):
        classification_labels[train_hips_flat == label_it] = idx
    labels_train = classification_labels

    classification_labels = np.zeros(test_hips.shape, dtype=np.int32)
    for idx, label_it in enumerate(unique_labels):
        classification_labels[test_hips == label_it] = idx
    labels_test = classification_labels

    normalizer = sklearn.preprocessing.Normalizer().fit(train_latents_flat)
    features_train = normalizer.transform(train_latents_flat)
    features_test = normalizer.transform(test_latents)

    linear_model = sklearn.linear_model.LogisticRegression(max_iter=int(1e4))

    linear_model.fit(features_train, labels_train[:, 0])
    train_accuracy = linear_model.score(features_train, labels_train)
    test_accuracy = linear_model.score(features_test, labels_test)

    return train_accuracy, test_accuracy


def get_mse_loss(
    params: chex.ArrayTree,
    apply_fn: Callable,
    context_latents: jnp.ndarray, 
    next_latent_target: jnp.ndarray, 
    rng: chex.PRNGKey
) -> jnp.ndarray:
    pred_next_latent = apply_fn(params, context_latents, rng)
    return jnp.mean(jnp.square(pred_next_latent - next_latent_target))


@partial(jax.jit, static_argnames=('batch_size', 'context_length', 'n_deployments', 'n_episodes_per_deployment'))
def sample_batch_and_update(
        train_state: TrainState,
        train_latents: jnp.ndarray,
        batch_size: int, 
        context_length: int,
        n_deployments: int,
        n_episodes_per_deployment: int,
        rng: chex.PRNGKey
) -> Tuple[TrainState, float]:
    rng_batch, rng_update = jax.random.split(rng)
    batch = sample_batch(
        train_latents,
        batch_size,
        context_length + 1,
        n_deployments=n_deployments,
        n_episodes_per_deployment=n_episodes_per_deployment,
        rng=rng_batch,
    )
    loss, grad = jax.value_and_grad(get_mse_loss)(train_state.params, train_state.apply_fn, batch[:, :-1], batch[:, -1], rng_update)
    train_state = train_state.apply_gradients(grads=grad)
    return train_state, loss


@partial(jax.jit, static_argnums=(1,2,3,4))
def sample_batch(
    train_latents: jnp.ndarray,
    batch_size: int,
    sequence_length: int,
    n_deployments: int,
    n_episodes_per_deployment: int,
    rng: chex.PRNGKey,
) -> jnp.ndarray:
    rng_deploy, rng_start = jax.random.split(rng)
    deployment_idxs = jax.random.randint(rng_deploy, (batch_size,), 0, n_deployments)
    start_steps = jax.random.randint(
        rng_start, (batch_size,), 
        0, n_episodes_per_deployment- sequence_length + 1
    )

    def get_batch_element(deployment_idx, start_step):
        return jax.lax.dynamic_slice_in_dim(train_latents[deployment_idx], start_step, sequence_length)
    
    batch = jax.vmap(get_batch_element)(deployment_idxs, start_steps)
    return batch


def visualize_embeddings_fakereal(
    inferred_latents: np.ndarray, 
    true_latents: np.ndarray, 
    hidden_parameters: np.ndarray,
    do_tsne: bool,
    hidden_parameters_inf: Optional[np.ndarray] = None,
) -> PIL.Image.Image:
    
    if hidden_parameters_inf is None:
        hidden_parameters_inf = hidden_parameters
    
    concatenated_latents = np.concatenate([inferred_latents, true_latents], axis=0)

    latent_dim = concatenated_latents.shape[-1]
    if latent_dim > 2:
        if do_tsne:
            tsne = TSNE()
            points = tsne.fit_transform(concatenated_latents)
        else:
            pca = PCA(n_components=2)
            points = pca.fit_transform(concatenated_latents)
    else:
        if latent_dim == 1:
            points = np.zeros([concatenated_latents.shape[0], 2])
            points[:, 0] = concatenated_latents.flatten()
        else:  # i.e. latent_dim == 2
            points = concatenated_latents
    
    inferred_points = points[:inferred_latents.shape[0]]
    true_points = points[inferred_latents.shape[0]:]
    # plt.figure(figsize=(4, 3))
    ps_inf = plt.scatter(inferred_points[:, 0], inferred_points[:, 1], 
                         c=hidden_parameters_inf, cmap='nipy_spectral', alpha=0.4, marker='o')
    ps_true = plt.scatter(true_points[:, 0], true_points[:, 1],
                            c=hidden_parameters, cmap='nipy_spectral', alpha=0.4, marker='x')
    if len(np.unique(hidden_parameters)) < 10:
        plt.colorbar(ps_inf, ticks=np.unique(hidden_parameters))
    else:
        plt.colorbar(ps_inf)
    plt.gca().set_aspect('equal', adjustable='datalim')
    fig = plt.gcf()
    pil_img = fig2img(fig)
    plt.clf()
    return pil_img
