from transformers import AutoTokenizer, AutoConfig
import torch
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
import os
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
import random
import argparse
from statistics import mode
import torch 
import numpy as np
import pandas as pd
from models.utils import VecDB, Emb, LatencyCollector, register_forward_latency_collector, parse_args

from categories import subcategories, categories
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from models.modeling_qwen2_moe import Qwen2MoeForCausalLM
from models.modeling_deepseek import DeepseekForCausalLM
from models.modeling_llama import LlamaForCausalLM

def load_model_and_tokenizer(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"  
        )
           

    if args.model_path == "deepseek-ai/deepseek-moe-16b-chat":

        model = DeepseekForCausalLM.from_pretrained(args.model_path, quantization_config=bnb_config,trust_remote_code=True )
    
    elif args.model_path == "meta-llama/Llama-3.1-8B-Instruct":

        model = LlamaForCausalLM.from_pretrained(args.model_path, quantization_config=bnb_config)

    # elif args.model_path == "Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4":                
    #     model = Qwen2MoeForCausalLM.from_pretrained(args.model_path, torch_dtype="auto")
    
    elif args.model_path == "Qwen/Qwen1.5-MoE-A2.7B-Chat":
     
        model = Qwen2MoeForCausalLM.from_pretrained(args.model_path, quantization_config=bnb_config)


    else:
        model = LlamaForCausalLM.from_pretrained(args.model_path,  torch_dtype=torch.bfloat16)

    model.to(torch.device(args.device))
    model.eval()

    return model, tokenizer, config

choices = ["A", "B", "C", "D"]
def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt




if __name__ == "__main__":

    # =================== model ===================

    args = parse_args()
    model, tokenizer, config = load_model_and_tokenizer(args)

   # ==================== MMLU Dataset ===================
    
    subcats = defaultdict(list)
    for k, v_list in subcategories.items():
        for v in v_list:
            subcats[v].append(k)

    subcategory = subcats[f'{args.subcategory}']

    # =================== collect hidden_states & apms & attn masks ===================

    cnt = 0
    max_len = 0
    length = []
    for subject in subcategory:
        if cnt >= 100:
                break

        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        train_prompt = gen_prompt(dev_df, subject, k=args.ntrain) # k-shot

        for i in range(test_df.shape[0]):
            cnt += 1
            if cnt >= 100:
                break
            prompt_end = format_example(test_df, i, include_answer=False)
            prompt = train_prompt + prompt_end
            inputs = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt", max_length=args.max_length).to(args.device)
            
            non_pad_token_count = inputs['attention_mask'].sum().item()
            max_len = max(max_len, non_pad_token_count)
            length.append(non_pad_token_count)

            with torch.no_grad():
                outputs = model(**inputs, collect_hiddenstates_apms=True, save_dir=args.save_dir)
        
    print(f"Collection {args.subcategory} {cnt} sentences, shots: {args.ntrain}, max_len: {max_len}, avg_len: { sum(length) / len(length)}")
            





