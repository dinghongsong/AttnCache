#!/bin/bash
# export TASK_NAME='STS16'

python train_embedding_model.py \
    --epoch 2 \
    --batchsize 64 \
    --task_name $1 \
    $2 \
    --dataset_dir /home/sdh/MetaEOL/MetaEOL/database/Llama-3.2-3B-Instruct \

