#!/bin/bash

for replace_layer in 2; do
    python evaluation.py \
        --model_name_or_path "meta-llama/Llama-3.2-3B-Instruct" \
        --mode test \
        --task_set sts \
        --prompt_method "prompteol"\
        --is_LazyFormer \
        --replace_layer $replace_layer \
        >LazyFormer_replace_layer"$replace_layer".log

done 