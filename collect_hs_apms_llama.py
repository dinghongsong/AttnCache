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
from models.modeling_llama import LlamaForCausalLM
from categories import subcategories, categories
from collections import defaultdict

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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--save-dir", type=str, default="Llama3_8b_DB")

    # =========================== MMLU Dataset ===========================
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--ntrain", "-k", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--subcategory", type=str, choices=['math', 'health', 'physics', 'business', 'biology', 
                                                        'chemistry', 'computer science', 'economics', 
                                                        'engineering', 'philosophy', 'other', 'history', 
                                                        'geography', 'politics', 'psychology', 'culture', 'law'], 
                        default='math', help="17 subcategories in MMLU")


    return parser.parse_args()


if __name__ == "__main__":

    # =================== model ===================

    args = parse_args()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device =  torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    config = AutoConfig.from_pretrained(args.model_path)

    model = LlamaForCausalLM.from_pretrained(args.model_path)
    model.to(device)
    model.eval()  
   
   # ==================== MMLU Dataset ===================
    
    subcats = defaultdict(list)
    for k, v_list in subcategories.items():
        for v in v_list:
            subcats[v].append(k)

    subcategory = subcats[f'{args.subcategory}']

    # =================== collect hidden_states & apms & attn masks ===================

    cnt = 0
    for subject in subcategory:

        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        train_prompt = gen_prompt(dev_df, subject, k=args.ntrain) # k-shot

        for i in range(test_df.shape[0]):
            cnt += 1
            prompt_end = format_example(test_df, i, include_answer=False)
            prompt = train_prompt + prompt_end
            inputs = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt", max_length=args.max_length).to(device)

            with torch.no_grad():
                outputs = model(**inputs, collect_hiddenstates_apms=True, save_dir=args.save_dir)
        
    print(f"Collection {args.subcategory} {cnt} sentences success!")
            





