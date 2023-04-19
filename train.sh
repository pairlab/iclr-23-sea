#!/bin/bash

kill -9 $(pgrep -f "torchbeast.polyhydra")

python_exe="python"
root=`cat LOGDIR`
env=$1
task=$2
name=$3
env_names=(${env//\// })
base_env=${env_names[0]}
if [ -z ${env_names[1]} ]
then
    group=$task
else
    group=${env_names[1]}-${task}
fi
savedir=$root/$env/$task/$name
shift 3
if [ -d "$savedir" ]; then
    read -p "$savedir exists, resume? (y/n, y to resume, n to replace) " -n 1 -r
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        echo "Resume training..."
    elif [[ $REPLY =~ ^[Nn]$ ]]
    then
        rm -rf $savedir
    else
        exit 0
    fi
fi

$python_exe -m torchbeast.polyhydra env_config=${base_env} task=${task} \
    data_root=$root \
    env=${env//\//-} \
    savedir=$savedir \
    seed=$RANDOM \
    wandb_name=$name \
    $@ # > torchbeast/output.log

echo train.sh Done