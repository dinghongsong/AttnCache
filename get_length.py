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
from models.utils import VecDB, Emb, LatencyCollector, register_forward_latency_collector, parse_args

from categories import subcategories, categories
from collections import defaultdict

def load_model_and_tokenizer(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    config = AutoConfig.from_pretrained(args.model_path)

    # model = LlamaForCausalLM.from_pretrained(args.model_path,  torch_dtype=torch.bfloat16)
    # model.to(torch.device(args.device))
    # model.eval()

    return None, tokenizer, config

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
    
    for subc in  ['math']:#, 'health', 'physics', 'business', 'biology', 
                                                    # 'chemistry', 'computer science', 'economics', 
                                                    # 'engineering', 'philosophy', 'other', 'history', 
                                                    # 'geography', 'politics', 'psychology', 'culture', 'law']:

        subcategory = subcats[subc] #- ['abstract_algebra']
        # subcategory = ['elementary_mathematics']
        # =================== collect hidden_states & apms & attn masks ===================

        
        for k in range(6):
            cnt = 0
            max_len = 0
            length = []
            for subject in subcategory:
                # if subject == 'abstract_algebra':
                #     continue

                val_df = pd.read_csv(
                        os.path.join(args.data_dir, "val", subject + "_val.csv"), header=None
                    )
                
                # test_df = pd.read_csv(
                # os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
                # )
                
                dev_df = pd.read_csv(
                    os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
                )[: k]
                train_prompt = gen_prompt(dev_df, subject, k=k) # k-shot
                # print(train_prompt)

                for i in range(val_df.shape[0]):
                # for i in range(test_df.shape[0]):
                    cnt += 1
                    prompt_end = format_example(val_df, i, include_answer=False)
                    # prompt_end = format_example(test_df, i, include_answer=False)
                    # print(prompt_end)
                    prompt = train_prompt + prompt_end
                    inputs = tokenizer(prompt, truncation=True, return_tensors="pt").to(args.device)
                    
                    non_pad_token_count = inputs['attention_mask'].sum().item()
                    max_len = max(max_len, non_pad_token_count)
                    length.append(non_pad_token_count)

                    # with torch.no_grad():
                    #     outputs = model(**inputs, collect_hiddenstates_apms=True, save_dir=args.save_dir)
                
            print(f"{subc} {cnt} sentences, shots: {k}, max_len: {max_len}, avg_len: { round(sum(length) / len(length), 2)}")



# engineering 16 sentences, shots: 0, max_len: 100, avg_len: 66.31
# engineering 16 sentences, shots: 1, max_len: 159, avg_len: 125.31
# engineering 16 sentences, shots: 2, max_len: 221, avg_len: 187.31
# engineering 16 sentences, shots: 3, max_len: 304, avg_len: 270.31
# engineering 16 sentences, shots: 4, max_len: 369, avg_len: 335.31
# engineering 16 sentences, shots: 5, max_len: 457, avg_len: 423.31


# math 1064 sentences, shots: 0, max_len: 373, avg_len: 93.85
# math 1064 sentences, shots: 1, max_len: 483, avg_len: 178.86
# math 1064 sentences, shots: 2, max_len: 613, avg_len: 284.79
# math 1064 sentences, shots: 3, max_len: 784, avg_len: 393.12
# math 1064 sentences, shots: 4, max_len: 954, avg_len: 490.28
# math 1064 sentences, shots: 5, max_len: 1019, avg_len: 560.28

# math 115 sentences, shots: 0, max_len: 189, avg_len: 92.89
# math 115 sentences, shots: 1, max_len: 299, avg_len: 177.73
# math 115 sentences, shots: 2, max_len: 429, avg_len: 283.67
# math 115 sentences, shots: 3, max_len: 600, avg_len: 391.7
# math 115 sentences, shots: 4, max_len: 770, avg_len: 488.55
# math 115 sentences, shots: 5, max_len: 835, avg_len: 558.51


# health 181 sentences, shots: 0, max_len: 331, avg_len: 96.73
# health 181 sentences, shots: 1, max_len: 606, avg_len: 198.45
# health 181 sentences, shots: 2, max_len: 743, avg_len: 281.73
# health 181 sentences, shots: 3, max_len: 882, avg_len: 367.76
# health 181 sentences, shots: 4, max_len: 1105, avg_len: 459.36
# health 181 sentences, shots: 5, max_len: 1209, avg_len: 520.92
# physics 70 sentences, shots: 0, max_len: 204, avg_len: 84.6
# physics 70 sentences, shots: 1, max_len: 267, avg_len: 146.7
# physics 70 sentences, shots: 2, max_len: 389, avg_len: 232.8
# physics 70 sentences, shots: 3, max_len: 484, avg_len: 306.11
# physics 70 sentences, shots: 4, max_len: 559, avg_len: 378.57
# physics 70 sentences, shots: 5, max_len: 637, avg_len: 444.4
# business 47 sentences, shots: 0, max_len: 119, avg_len: 70.28
# business 47 sentences, shots: 1, max_len: 199, avg_len: 122.51
# business 47 sentences, shots: 2, max_len: 249, avg_len: 169.64
# business 47 sentences, shots: 3, max_len: 346, avg_len: 245.21
# business 47 sentences, shots: 4, max_len: 473, avg_len: 326.85
# business 47 sentences, shots: 5, max_len: 583, avg_len: 395.34
# biology 48 sentences, shots: 0, max_len: 156, avg_len: 91.12
# biology 48 sentences, shots: 1, max_len: 228, avg_len: 144.79
# biology 48 sentences, shots: 2, max_len: 322, avg_len: 244.12
# biology 48 sentences, shots: 3, max_len: 415, avg_len: 349.79
# biology 48 sentences, shots: 4, max_len: 464, avg_len: 398.79
# biology 48 sentences, shots: 5, max_len: 559, avg_len: 481.46
# chemistry 30 sentences, shots: 0, max_len: 180, avg_len: 107.63
# chemistry 30 sentences, shots: 1, max_len: 272, avg_len: 162.97
# chemistry 30 sentences, shots: 2, max_len: 358, avg_len: 215.97
# chemistry 30 sentences, shots: 3, max_len: 415, avg_len: 275.17
# chemistry 30 sentences, shots: 4, max_len: 506, avg_len: 371.3
# chemistry 30 sentences, shots: 5, max_len: 619, avg_len: 505.57
# computer science 42 sentences, shots: 0, max_len: 224, avg_len: 104.95
# computer science 42 sentences, shots: 1, max_len: 298, avg_len: 192.48
# computer science 42 sentences, shots: 2, max_len: 474, avg_len: 310.86
# computer science 42 sentences, shots: 3, max_len: 560, avg_len: 406.52
# computer science 42 sentences, shots: 4, max_len: 672, avg_len: 496.83
# computer science 42 sentences, shots: 5, max_len: 927, avg_len: 661.19
# economics 81 sentences, shots: 0, max_len: 183, avg_len: 85.46
# economics 81 sentences, shots: 1, max_len: 219, avg_len: 160.96
# economics 81 sentences, shots: 2, max_len: 299, avg_len: 216.69
# economics 81 sentences, shots: 3, max_len: 464, avg_len: 272.11
# economics 81 sentences, shots: 4, max_len: 532, avg_len: 332.69
# economics 81 sentences, shots: 5, max_len: 668, avg_len: 396.93
# engineering 16 sentences, shots: 0, max_len: 100, avg_len: 66.31
# engineering 16 sentences, shots: 1, max_len: 159, avg_len: 125.31
# engineering 16 sentences, shots: 2, max_len: 221, avg_len: 187.31
# engineering 16 sentences, shots: 3, max_len: 304, avg_len: 270.31
# engineering 16 sentences, shots: 4, max_len: 369, avg_len: 335.31
# engineering 16 sentences, shots: 5, max_len: 457, avg_len: 423.31
# philosophy 223 sentences, shots: 0, max_len: 174, avg_len: 99.69
# philosophy 223 sentences, shots: 1, max_len: 259, avg_len: 185.22
# philosophy 223 sentences, shots: 2, max_len: 363, avg_len: 260.96
# philosophy 223 sentences, shots: 3, max_len: 468, avg_len: 341.47
# philosophy 223 sentences, shots: 4, max_len: 575, avg_len: 427.52
# philosophy 223 sentences, shots: 5, max_len: 673, avg_len: 507.84
# other 127 sentences, shots: 0, max_len: 202, avg_len: 74.64
# other 127 sentences, shots: 1, max_len: 306, avg_len: 129.16
# other 127 sentences, shots: 2, max_len: 364, avg_len: 176.29
# other 127 sentences, shots: 3, max_len: 520, avg_len: 257.06
# other 127 sentences, shots: 4, max_len: 675, avg_len: 317.28
# other 127 sentences, shots: 5, max_len: 731, avg_len: 378.22
# history 101 sentences, shots: 0, max_len: 764, avg_len: 264.57
# history 101 sentences, shots: 1, max_len: 966, avg_len: 476.03
# history 101 sentences, shots: 2, max_len: 1492, avg_len: 771.46
# history 101 sentences, shots: 3, max_len: 1953, avg_len: 1038.62
# history 101 sentences, shots: 4, max_len: 2451, avg_len: 1280.72
# history 101 sentences, shots: 5, max_len: 2939, avg_len: 1520.46
# geography 22 sentences, shots: 0, max_len: 91, avg_len: 61.0
# geography 22 sentences, shots: 1, max_len: 152, avg_len: 122.0
# geography 22 sentences, shots: 2, max_len: 221, avg_len: 191.0
# geography 22 sentences, shots: 3, max_len: 297, avg_len: 267.0
# geography 22 sentences, shots: 4, max_len: 341, avg_len: 311.0
# geography 22 sentences, shots: 5, max_len: 382, avg_len: 352.0
# politics 71 sentences, shots: 0, max_len: 347, avg_len: 116.83
# politics 71 sentences, shots: 1, max_len: 620, avg_len: 268.69
# politics 71 sentences, shots: 2, max_len: 722, avg_len: 346.42
# politics 71 sentences, shots: 3, max_len: 856, avg_len: 437.2
# politics 71 sentences, shots: 4, max_len: 1072, avg_len: 561.77
# politics 71 sentences, shots: 5, max_len: 1326, avg_len: 700.45
# psychology 129 sentences, shots: 0, max_len: 222, avg_len: 90.09
# psychology 129 sentences, shots: 1, max_len: 405, avg_len: 216.81
# psychology 129 sentences, shots: 2, max_len: 478, avg_len: 298.18
# psychology 129 sentences, shots: 3, max_len: 534, avg_len: 386.74
# psychology 129 sentences, shots: 4, max_len: 595, avg_len: 451.92
# psychology 129 sentences, shots: 5, max_len: 689, avg_len: 525.46
# culture 34 sentences, shots: 0, max_len: 103, avg_len: 76.65
# culture 34 sentences, shots: 1, max_len: 152, avg_len: 123.18
# culture 34 sentences, shots: 2, max_len: 203, avg_len: 173.29
# culture 34 sentences, shots: 3, max_len: 296, avg_len: 260.47
# culture 34 sentences, shots: 4, max_len: 367, avg_len: 323.35
# culture 34 sentences, shots: 5, max_len: 449, avg_len: 389.47
# law 194 sentences, shots: 0, max_len: 723, avg_len: 240.52
# law 194 sentences, shots: 1, max_len: 1016, avg_len: 507.48
# law 194 sentences, shots: 2, max_len: 1206, avg_len: 684.28
# law 194 sentences, shots: 3, max_len: 1278, avg_len: 756.7
# law 194 sentences, shots: 4, max_len: 1711, avg_len: 1146.26
# law 194 sentences, shots: 5, max_len: 2078, avg_len: 1479.33

# val_df
# Collection math 115 sentences,  max_len: 189, avg_len: 92.89
# Collection health 181 sentences,  max_len: 331, avg_len: 96.73
# Collection physics 70 sentences,  max_len: 204, avg_len: 84.6
# Collection business 47 sentences,  max_len: 119, avg_len: 70.28
# Collection biology 48 sentences,  max_len: 156, avg_len: 91.12
# Collection chemistry 30 sentences,  max_len: 180, avg_len: 107.63
# Collection computer science 42 sentences,  max_len: 224, avg_len: 104.95
# Collection economics 81 sentences,  max_len: 183, avg_len: 85.46
# Collection engineering 16 sentences,  max_len: 100, avg_len: 66.31
# Collection philosophy 223 sentences,  max_len: 174, avg_len: 99.69
# Collection other 127 sentences,  max_len: 202, avg_len: 74.64
# Collection history 101 sentences,  max_len: 764, avg_len: 264.57
# Collection geography 22 sentences,  max_len: 91, avg_len: 61.0
# Collection politics 71 sentences,  max_len: 347, avg_len: 116.83
# Collection psychology 129 sentences,  max_len: 222, avg_len: 90.09
# Collection culture 34 sentences,  max_len: 103, avg_len: 76.65
# Collection law 194 sentences,  max_len: 723, avg_len: 240.52

# test_df
# Collection math 1064 sentences,  max_len: 373, avg_len: 93.85
# Collection health 1640 sentences,  max_len: 986, avg_len: 99.69
# Collection physics 640 sentences,  max_len: 275, avg_len: 84.53
# Collection business 437 sentences,  max_len: 153, avg_len: 71.17
# Collection biology 454 sentences,  max_len: 217, avg_len: 93.16
# Collection chemistry 303 sentences,  max_len: 208, avg_len: 95.29
# Collection computer science 412 sentences,  max_len: 409, avg_len: 104.08
# Collection economics 742 sentences,  max_len: 210, avg_len: 88.63
# Collection engineering 145 sentences,  max_len: 106, avg_len: 61.65
# Collection philosophy 2012 sentences,  max_len: 253, avg_len: 97.59
# Collection other 1165 sentences,  max_len: 272, avg_len: 74.72
# Collection history 930 sentences,  max_len: 760, avg_len: 254.23
# Collection geography 198 sentences,  max_len: 147, avg_len: 64.05
# Collection politics 648 sentences,  max_len: 419, avg_len: 113.82
# Collection psychology 1157 sentences,  max_len: 266, avg_len: 85.58
# Collection culture 332 sentences,  max_len: 205, avg_len: 78.58
# Collection law 1763 sentences,  max_len: 745, avg_len: 243.44