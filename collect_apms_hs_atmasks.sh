#!/bin/bash
# export TASK_NAME='STS16'

for task_name in "STS12" "STS13"  "STS14"  "STS15"   "SICKRelatedness" "STSBenchmark"; do
    python evaluation.py \
        --model_name_or_path "meta-llama/Llama-3.2-3B-Instruct" \
        --mode test \
        --task_set sts \
        --prompt_method "prompteol"\
        --task_name $task_name \
        --collect_hiddenstates_apms
done 