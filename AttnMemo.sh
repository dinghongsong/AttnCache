#!/bin/bash
for threshold in 0.9999 0.999; do 
    python evaluation.py \
            --model_name_or_path "meta-llama/Llama-3.2-3B-Instruct" \
            --mode test \
            --task_set sts \
            --prompt_method "prompteol"\
            --is_attn_memo \
            --threshold 0.9999 \
            --batch_size 128 \
            > AttnMemo_threshold_"$threshold".log
done 