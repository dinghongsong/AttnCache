# Code Implementation for AttnCache
## Conda Environment

```
conda env create -f environment.yml
conda activate AttnCache
```

## Download data

``` sh

bash download_dataset.sh

```
## Preprocess

### Collect Hidden States and Attention Maps 

```
python evaluation.py \
        --model_name_or_path mistralai/Mistral-7B-v0.1 \
        --task_set sts \
        <!-- --task_name STS16 \ -->
        --collect_hiddenstates_apms \
        --save_dir /home/sdh/AttnCache/AttnCache/database/ \
        --device cpu
```

### Train Feature Projector and Build Index DB

```
python train_fp_and_build_db.py \
        --model_name_or_path meta-llama/Llama-3.1-8B \
        --epoch 2 \
        --batchsize 64 \
        --task_name STS14 \
        --is_attn_cache \
        --save_dir /home/sdh/AttnCache/AttnCache/database/
```

## Running Script
```
python evaluation.py  \
        --model_name_or_path meta-llama/Llama-3.1-8B \
        --task_set sts \
        --task_name STS16 \
        --is_attn_memo\
        --batch_size 1 \
        --save_dir /home/sdh/AttnCache/AttnCache/database/ \
        --device cpu
```    
The argument `task_set` can also be set to `transfer`. Similarly, the argument `prompt_method` can also be set to `metaeol`.

## Acknowledgement

Our code is developed upon [PromptEOL](https://github.com/kongds/scaling_sentemb). We thank the authors of PromptEOL for their great efforts.