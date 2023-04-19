#!/bin/bash

expr_name=$1
use_wandb="False"  # Replace this with True if you want to use wandb
wandb_project="project"
wandb_entity="username"

# Training data collecting policy with IMPALA
printf "n" | ./train.sh crafter run $expr_name \
    use_crafter_monitor=True \
    total_steps=2e8 \
    wandb=$use_wandb \
    project=$wandb_project \
    entity=$wandb_entity \
    group=run
echo Data collecting policy done.
run_savedir=`python read_results.py savedir`

# 
printf "n" | ./train.sh crafter pred $expr_name \
    actor_load_dir="$run_savedir/checkpoint.tar" \
    total_steps=0.5e8 \
    contrast_step_limit=1e6 \
    wandb=$use_wandb \
    project=$wandb_project \
    entity=$wandb_entity \
    group=pred
echo Achievement learning done.
pred_savedir=`python read_results.py savedir`

printf "n" | ./train.sh crafter clustering $expr_name \
    actor_load_dir="$run_savedir/checkpoint.tar" \
    pred_model_load_dir="$pred_savedir/checkpoint.tar"
echo Achievement clustering done.
cluster_savedir=`python read_results.py savedir`
num_clusters=`python read_results.py num_clusters`

printf "n" | ./train.sh crafter mo $expr_name \
    num_objectives=$num_clusters \
    cluster_load_dir="$cluster_savedir/cluster.data" \
    cluster_pred_model_load_dir="$pred_savedir" \
    causal_graph_load_path="$cluster_savedir/graph.data" \
    include_new_tasks=True \
    use_crafter_monitor=True \
    total_steps=3e8 \
    wandb=$use_wandb \
    project=$wandb_project \
    entity=$wandb_entity \
    group=mo
echo Sub-policy training done.
mo_savedir=`python read_results.py savedir`
echo $mo_savedir
