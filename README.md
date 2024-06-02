# OfflineRLNonstationarity
Implementation for our method called COSPA from the RLC 2024 paper "Offline Reinforcement Learning from Datasets with Structured Non-Stationarity".

See [arXiv](https://arxiv.org/abs/2405.14114) and [project page](https://sites.google.com/view/offlinerl-nonstationarity).

### Setup
We recommend to use Docker with the provided Dockerfile in `.devcontainer`, or to just use the whole devcontainer in VS code.
Alternatively you can also use pip to install the required packages from `requirements/requirements.txt`

The datasets can be downloaded from Huggingface here: https://huggingface.co/datasets/johannesack/OfflineRLStructuredNonstationary
Datasets should be placed in the same directory as `train.py` (which should also be your working directory).

Experiments can be started with, for example
```
python train.py --config_path=agents/XY-evolvediscretelong-v3/td3_cpc.yaml
```

Results are logged to wandb, so a wandb authorization is requested. To avoid this simply use

```
WANDB_MODE=offline python train.py --config_path=agents/XY-evolvediscretelong-v3/td3_cpc.yaml
```

If there are any issues, feel free to open an issue or send me an E-Mail!

This was my first JAX project, so there are a bunch of things that could be written more beautifully, but it works and it's quite fast!

### Acknowledgements
* This repository was initially based on the great [gymnax-blines](https://github.com/RobertTLange/gymnax-blines), but has since been heavily modified.

* The Ant and Barkour environments are modified from the [brax](https://github.com/google/brax) environments.

* The RL algorithm is based on my previous project about [Task Clustering in RL](https://github.com/JohannesAck/EMTaskClustering).
