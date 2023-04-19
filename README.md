<!-- LOGDIR
python3.7
libtorchbeast
crafter
gym_minigrid -->

# Structured Exploration with Achievements (SEA)

This is the official repository for our paper ["Learning Achievement Structure for Structured Exploration in Domains with Sparse Reward"](https://openreview.net/forum?id=NDWl9qcUpvy). The code is based on the [torchbeast](https://github.com/facebookresearch/torchbeast) IMPALA baseline implementation in [NetHack 2021 NeurIPS Challenge](http://gitlab.aicrowd.com/nethack/neurips-2021-the-nethack-challenge).



## Setup

Our code is tested on Python 3.7.16 and PyTorch 1.13.1. For other dependencies:

```shell
pip install -r requirements.txt
```

### libtorchbeast
We use a custom version of libtorchbeast. You can clone this [fork](https://github.com/footoredo/torchbeast) locally and use the following steps ("Installing PolyBeast" section in their README) to install.

```shell
cd torchbeast  # Your local torchbeast repo dir
pip install -r requirements.txt
git submodule update --init --recursive
pip install nest/
python setup.py install
```

### Crafter
We use a custom version of Crafter. You can clone this [fork](https://github.com/footoredo/crafter) locally follow the instructions there to install.

## Settings

- Replace the content in `LOGDIR` file with your designated log dir.
- If you want to use wandb, fill in your wandb config in `train_all.sh`.

## Quickstart

To replicate our experiments on Crafter in our paper, run the following command:
```shell
./train_all.sh $expr_name
```

## Running each stage

### Data collection policy

To train initial data collection policy with IMPAPA, run the following command:
```shell
./train.sh crafter run $expr_name \
    total_steps=2e8
```

### Achievement representation learning

To collect data with data collection policy and learn achievement representations, run the following command:
```shell
./train.sh crafter pred $expr_name \
    actor_load_dir="$run_savedir/checkpoint.tar" \   # replace this with data collection policy
    total_steps=0.5e8 \
    contrast_step_limit=1e6                          # this parameter limits the environment interaction steps
```

### Achievement clustering

To do automatic clustering of achievements with learned achievement representations, run the following command:
```shell
./train.sh crafter clustering $expr_name \
    actor_load_dir="$run_savedir/checkpoint.tar" \      # replace this with data collection policy
    pred_model_load_dir="$pred_savedir/checkpoint.tar"  # replace this with representation model
```

### Sub-policy learning

To train sub-policies that reach each known achievement and explore for new achievements, run the following command:
```shell
./train.sh crafter mo $expr_name \
    num_objectives=$num_clusters \  # this is given in the last step
    cluster_load_dir="$cluster_savedir/cluster.data" \
    cluster_pred_model_load_dir="$pred_savedir" \
    causal_graph_load_path="$cluster_savedir/graph.data" \
    include_new_tasks=True \  # set this to False if you don't want to train exploration policy
    total_steps=3e8
```

## Citation

```BibTex
@inproceedings{
    zhou2023learning,
    title={Learning Achievement Structure for Structured Exploration in Domains with Sparse Reward},
    author={Zihan Zhou and Animesh Garg},
    booktitle={The Eleventh International Conference on Learning Representations },
    year={2023},
    url={https://openreview.net/forum?id=NDWl9qcUpvy}
}
```