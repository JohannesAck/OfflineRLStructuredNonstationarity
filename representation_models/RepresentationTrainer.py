

from typing import Optional, Tuple
import chex
import flax
from flax.training.train_state import TrainState
import jax
from utils.DeploymentDataset import Dataset, Transition

from utils.config import TrainConfig


class RepresentationTrainer(object):

    def get_model(
        self,
        rng: chex.PRNGKey, 
        config: TrainConfig,
        obs_dim: Optional[int] = None, 
        action_dim: Optional[int] = None
    ) -> Tuple[flax.linen.Module, TrainState]:
        raise NotImplementedError()
    
    # @staticmethod
    # def get_loss():
    #     raise NotImplementedError()


    # @staticmethod
    # def update():
    #     raise NotImplementedError()
    
    @staticmethod
    def sample_buff_and_update_n_times():
        raise NotImplementedError()
    
    def sample_and_update(
        self,
        train_state: TrainState,
        dataset: Dataset,
        rng: chex.PRNGKey,
        config: TrainConfig,
    ) -> Tuple[dict, TrainState]:
        """
        Samples a batch from the dataset and updates the model.
        Returns a dictionary of metrics and the updated train state.
        """
        raise NotImplementedError()
    
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
        raise NotImplementedError()
    
    def predict_next_latent(
        self,
        train_state: TrainState,
        input_transitions: Transition,
        rng: chex.PRNGKey,
        config: TrainConfig,
    ) -> jax.Array:
        """
        Predicts the next latent, given the last episode or last multiple episodes depending on the method
        """
        raise NotImplementedError()
    
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
        `dataset` has the same normalizations as the the dataset used for training.
        `ds_orig` has the original states, without any normalizations.

        Returns latents for each episode, and a validity-deploystep-mask for episodes that can not be inferred, i.e. at the start 
        of a deployment the VRNN needs multiple episodes as context.
        """
        raise NotImplementedError()
    
    def eval(
        self,
        train_state: TrainState,
        dataset: Dataset,
        rng: chex.PRNGKey,
        config: TrainConfig,
    ) -> dict:
        """
        Evaluates the model on a sample of episodes from the dataset.
        Returns a dict of metrics, which can be floats or wandb.images.
        """
        raise NotImplementedError()