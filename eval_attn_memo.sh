#!/bin/bash

export threshold=0.9999
export replace_layer=16

python evaluation.py \
    --model_name_or_path "meta-llama/Llama-3.2-3B-Instruct" \
    --mode test \
    --task_set sts \
    --prompt_method "prompteol"\
    --is_attn_memo \
    --threshold $threshold \
    --replace_layer $replace_layer \
    > attn_memo_threshold"$threshold"_replace_layer"$replace_layer".log

