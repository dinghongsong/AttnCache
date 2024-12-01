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

### Collect Attention Maps and Hidden States

### Train Feature Projector and Build Index DB

## Running Script
```
python evaluation.py --model_name_or_path "mistralai/Mistral-7B-v0.1" --mode test --task_set sts --prompt_method prompteol
```
The argument `task_set` can also be set to `transfer`. Similarly, the argument `prompt_method` can also be set to `metaeol`.

## Acknowledgement

Our code is developed upon [PromptEOL](https://github.com/kongds/scaling_sentemb). We thank the authors of PromptEOL for their great efforts.