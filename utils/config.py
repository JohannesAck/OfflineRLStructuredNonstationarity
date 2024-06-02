from dataclasses import dataclass, field
import datetime
from enum import Enum
import random
from typing import Optional, Union
import uuid
import pyrallis


def parse_float_or_none(s: str) -> Optional[float]:
    if isinstance(s, int):
        return float(s)
    if isinstance(s, float):
        return s
    elif s.lower() == 'none':
        return None
    else:
        return float(s)
pyrallis.decode.register(Optional[float], parse_float_or_none)

def parse_int_or_none(s: str) -> Optional[float]:
    if isinstance(s, int):
        return int(s)
    if isinstance(s, float) or '.' in s:
        raise ValueError(f'float {s} cannot be parsed as int')
    elif s.lower() == 'none':
        return None
    else:
        return int(s)
pyrallis.decode.register(Optional[int], parse_int_or_none)

@dataclass
class NetworkConfig:
    num_hidden_units: int = 128
    num_hidden_layers: int = 2
    use_layer_norm: bool = False
    activation: str = 'relu'

@dataclass
class NetworkConfigLN: # to set the default value for offline rl
    num_hidden_units: int = 128
    num_hidden_layers: int = 2
    use_layer_norm: bool = True
    activation: str = 'relu'

class Methods(Enum):
    zero_hip: str = 'zero_hip'
    noisy_hip: str = 'noisy_hip'
    gt_hip: str = 'gt-hip'
    cpc: str = 'cpc'

class EvalMode(Enum):
    sampled_latent: str = 'sampled_latent'  # sample an episode with the same hip as the current episode and use latent from it (representation model having been run during the representation training, ds-augmentation)
    inferred_latent: str = 'inferred_latent'  # sample an episode with the same hip as the current episode and run representation model on it
    predicted_latent: str = 'predicted_latent'  # sample a context of context-length episodes, run representation model on each and predict the next latent using the predictor

class RLAlgorithm(Enum):
    TD3: str = 'TD3'

@dataclass
class OnlineRLConfig:
    algorithm: RLAlgorithm = RLAlgorithm.TD3

    num_train_steps: int =  8000
    max_grad_norm: float = 1.0  # Global norm to clip gradients by
    gamma: float = 0.99  # Discount factor
    buffer_size: int = 1000000 # "buffer_size"
    lr: float = 1e-03 
    update_freq: int = 1  # update policy after each n rollouts
    # using n_envs == 1 and policy_freq == 2 would lead to never updating the policy
    polyak: float = 0.995  # target network polyak
    batch_size: int = 256  # batch size per update  (not times envs or something like that)
    warmup_steps: int = 50  # times number of envs

    target_reward: Optional[float] = None  # stop training if target reward is exceeded in evaluation env and generate dataset

    input_gt_hip: bool = False  # Input GT-Hip while training online agent

    # performance setup
    num_train_envs: int = 32  # Number of parallel env workers
    n_updates_jit: int = 8
    n_transitions_jit: int = 2

    # network setup
    actor_network_config: NetworkConfig = field(default_factory=NetworkConfig)
    critic_network_config: NetworkConfig = field(default_factory=NetworkConfig)

    # TD3 specific    
    policy_freq: int = 2  
    exploration_std: float = 0.1
    exploration_clip: float = 0.1
    policy_noise_std: float = 0.1  # noise during update
    policy_noise_clip: float = 0.1  # noise during update

    # evaluation
    evaluate_every_epochs: int = 100
    num_test_rollouts: int = 164
    video_every_epochs: int = 10000

    # dataset generation
    traj_recording_parallel: int = 1000
    n_deployments_record: int = 1000
    record_hip_as_state: bool = True

    load_policy_path: Optional[str] = None

    def __hash__(self):
        return hash(self.__repr__())  # not sure if this is a great idea but should work. Anyways don't change config after post_init please.

@dataclass
class LatentPredictorConfig:  # config for the next-latent predictor
    num_hidden_units: int = 128
    num_hidden_layers: int = 1
    recurrent_dim: int = 32
    
    training_epochs: int = 100
    lr: float = 1e-03
    batch_size: int = 256
    context_length: int = 5
    evaluate_every_epoch: int = 30
    
    def __hash__(self):
        return hash(self.__repr__())  # not sure if this is a great idea but should work. Anyways don't change config after post_init please.


@dataclass
class VAEConfig:
    vae_beta: float = 1e-6
    latent_dim: int = 8  # for CPC this is the dimension of the GRU hidden state
    
    encoder_hidden_layers: int = 2
    decoder_hidden_layers: int = 2  # in case of CPC, this is the number of layers in the MLP head
    num_hidden_units: int = 128
    
    # VAE
    latent_in_second_layer: bool = True
    two_transition_vae: bool = True
    use_state_loss: bool = True
    use_reward_loss: bool = True

    # CPC    
    context_len_cpc: int = 5
    cpc_offset: int = 1  # distance from context to predicted sample. 1 = immediately next sample
    cpc_n_negative: int = 10
    cpc_trans_encoding_dim: int = 8
    cpc_context_encoding_dim: int = 32
    use_trans_encoding: bool = True  # if false uses the context encoding as latent
    predict_next_using_context: bool = False  # if true, we predict the next sample using the context, otherwise we predict the next sample using the transencoding
    
    recurrent_state_dim: int = 32

    # Recenv VAE
    recenc_vae_seqlen: int = 10 # sequence length for recurrent encoder

    # baselines
    zero_latent_vae: bool = False  # basically runs decoders without latent, to see if the latent is actually needed
    gt_hip_vae: bool = False  # uses the ground truth hip as latent in the decoders to see if the latent improves performance

    lr: float = 3e-04
    train_steps: int = 1000000
    batch_size: int = 512
    
    n_updates_jit: int = 1

    evaluate_every_steps: int = 1000
    n_evaluation_episodes: int = 1000
    n_tsne_evaluation_episodes: int = 300
    do_mlp_reward_probe: bool = False

    def __hash__(self):
        return hash(self.__repr__())  # not sure if this is a great idea but should work. Anyways don't change config after post_init please.


class OfflineAlgorithm(Enum):
    TD3BC: str = 'TD3BC'

@dataclass
class OfflineRLConfig:
    algorithm: OfflineAlgorithm = OfflineAlgorithm.TD3BC

    train_single_hip: Optional[float] = None

    max_grad_norm: float = 1.0  # Global norm to clip gradients by
    gamma: float = 0.99  # Discount factor
    buffer_size: int = 1000000 
    actor_lr: float = 3e-04
    critic_lr: float = 3e-04
    # using n_envs == 1 and policy_freq == 2 would lead to never updating the policy
    polyak: float = 0.995  # target network polyak
    batch_size: int = 128  # batch size per update  (not times envs or something like that)

    # performance setup
    n_updates_jit: int = 8

    train_with_predictor_output: bool = False  # if true, we augment the latent with the output of the predictor rather than the representation model

    # network setup
    network_config:  NetworkConfigLN = field(default_factory=NetworkConfigLN)

    #TD3BC specific
    td3_alpha: Optional[float] = 2.5  # None means no BC
    policy_freq: int = 2  # how often update policy compared to critic. This might be a bit off, as we do it per batch of 32 envs, so for example using 3 will be inaccurate
    td3_policy_noise_std: float = 0.0  # noise during update
    td3_policy_noise_clip: float = 0.0  # noise during update
  
    eval_mode: EvalMode = EvalMode.predicted_latent

    # for baseline
    noisy_hip_rate: float = 0.1  # rate of uniformly picking a random hip instead of the ground truth hip
    
    normalize_latent: bool = True
    normalize_latent_by_dimension: bool = True  # if true, we normalize each dimension of the latent separately

    train_steps: int = 1000000

    evaluate_every_epochs: int = 1000
    num_test_rollouts: int = 1000
    video_every_epochs: int = 10000
   
    def __hash__(self):
        return hash(self.__repr__())  # not sure if this is a great idea but should work. Anyways don't change config after post_init please.


@dataclass
class TrainConfig:
    # wandb logging
    project: str = "OfflineNonstationary"
    group: str = "Group"
    name: str = "Debug"

    d4rl: bool = False
    
    env_name: str = "XY-const-v1"  # XY-const-v1
    env_kwargs: dict = field(default_factory=dict)
    env_params: dict = field(default_factory=dict)
    env_name_eval: str = "XY-uniform-v1"  # XY-const-v1
    env_kwargs_eval: dict = field(default_factory=dict)
    env_params_eval: dict = field(default_factory=dict)

    online: OnlineRLConfig = field(default_factory=OnlineRLConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    offline: OfflineRLConfig = field(default_factory=OfflineRLConfig)
    predictor: LatentPredictorConfig = field(default_factory=LatentPredictorConfig)

    run_dir: Optional[str] = None
    seed: int = -1
    eval_seed: int = -1

    loss_log_rate: int = 5
    vae_only: bool = False
    load_repr: bool = False

    intermediate_exp: bool = False  # use intermediate-expert mixed dataset

    method: Methods = Methods.cpc

    def __hash__(self):
        return hash(self.__repr__())  # not sure if this is a great idea but should work. Anyways don't change config after post_init please.


    def __post_init__(self):
        self.run_id = str(uuid.uuid4())
        if self.d4rl:
            self.name = f"{self.name}-{self.env_name}-"\
                f"-{self.offline.algorithm.name}-{datetime.datetime.now().isoformat().split('.')[0]}-{self.run_id[:8]}"
        else:
            self.name = f"{self.name}-{self.env_name_eval}-{self.online.target_reward if self.online.target_reward is not None else ''}"\
                f"-{self.method.name}-{datetime.datetime.now().isoformat().split('.')[0]}-{self.run_id[:8]}"
        
        if self.method == Methods.cpc:
            if self.vae.use_trans_encoding:
                self.vae.latent_dim = self.vae.cpc_trans_encoding_dim
            else:
                self.vae.latent_dim = self.vae.cpc_context_encoding_dim
            print(f'overwriting latent dim to {self.vae.latent_dim} because using CPC')
        assert not (self.vae_only and self.method in [Methods.zero_hip, Methods.gt_hip]), "can't use vae only with zero or gt hip"
        assert not (self.load_repr and self.method in [Methods.zero_hip, Methods.gt_hip]), "can't use load repr with zero or gt hip"

        if self.seed == -1:
            self.seed = random.randint(0, 2 ** 32 - 1)
        if self.eval_seed == -1:
            self.eval_seed = random.randint(0, 2 ** 32 - 1)

