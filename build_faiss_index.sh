#!/bin/bash

python build_faiss_index.py \
    --epoch 2 \
    --task_name $1 \
    --dataset_dir /home/sdh/MetaEOL/MetaEOL/database/Llama-3.2-3B-Instruct \

