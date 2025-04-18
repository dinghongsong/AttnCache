from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
import argparse
from statistics import mode
import torch 
import numpy as np
import os
from models.utils import VecDB, Emb, LatencyCollector, register_forward_latency_collector
import pickle
from sklearn.metrics import accuracy_score
import pandas as pd
from models.modeling_llama import LlamaForCausalLM
from categories import subcategories, categories
from collections import defaultdict
import termcolor

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

def format_all_examples(df, include_answer=True):
    k = df.shape[1] - 2
    formatted = []
    for idx in range(len(df)):
        prompt = df.iloc[idx, 0]
        for j in range(k):
            prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(df.iloc[idx, k + 1])
        formatted.append(prompt)
    return formatted



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

    # parser.add_argument("--model-path", type=str, default="meta-llama/Llama-3.2-3B-Instruct") 
    parser.add_argument("--model-path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")   
    parser.add_argument("--save-dir", type=str, default="Llama3_8b_DB")
    parser.add_argument("--threshold", type=float, default=0.95)

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

def retrieve(data):
    inputs = tokenizer(data, padding="max_length", truncation=True, max_length=args.max_length, return_tensors="pt")
    inputs = inputs.to(device)
    
    x = model.model.embed_tokens(inputs["input_ids"])
    x = model.model.layers[0].input_layernorm(x)

    feature_vector = feature_projector.embed(x.cpu().detach().numpy())
    sims, idx_list = vecDB.search(feature_vector)

    reuse_tensor_index = np.flatnonzero(1 - sims >= args.threshold)
    hitted_records = idx_list[reuse_tensor_index]
    compute_tensor_index = np.flatnonzero(1 - sims < args.threshold)

    return reuse_tensor_index, hitted_records, compute_tensor_index, inputs



def evaluate(reuse_tensor_index, hitted_records, compute_tensor_index, inputs):
    
    with torch.no_grad():
        total_tensor_index = np.concatenate((reuse_tensor_index, compute_tensor_index), axis=0)
        attention_cache = torch.empty((config.num_hidden_layers, len(total_tensor_index), config.num_attention_heads, args.max_length, args.max_length))                             
        if len(reuse_tensor_index) != 0:
            print(f"=========== hit {len(reuse_tensor_index)} APMs")
            for layer_idx in range(config.num_hidden_layers):
                for idx, record in zip(reuse_tensor_index, hitted_records):
                    with open(f"{args.database_path}/APMsDB/{record[0]}.pickle", "rb") as file:
                        attn_weights = pickle.load(file)
                        attention_cache[layer_idx][idx] = torch.from_numpy(attn_weights)

                hitted_records += 1  # index of next layer hitted apms
        else:
            print(f"=========== no hit APMs")

        attention_cache = attention_cache.to(device)
        self_attn_latency_collector = LatencyCollector()
        register_forward_latency_collector(self_attn_latency_collector, model.model.layers[-1].self_attn)
        e2e_latency_collector = LatencyCollector()
        register_forward_latency_collector(e2e_latency_collector, model.model)
        
        ttft_list = []
        for _ in range(11):

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
    
            outputs = model(**inputs, attention_cache=attention_cache, compute_tensor_index=compute_tensor_index)

            end.record()
            torch.cuda.synchronize()
            inference_time = start.elapsed_time(end)
            # print(termcolor.colored(f'Prefill latency: {inference_time:.2f} ms', 'yellow'))
            ttft_list.append(inference_time)
            
    return outputs, self_attn_latency_collector, e2e_latency_collector, ttft_list


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


    # =================== feature_projector & vecDB ===================

    feature_projector_save_path = f'{args.database_path}/Embedding_models/mlp_model_attn_cache-epoch3.pth'
    feature_projector = Emb(f"{feature_projector_save_path}")
    vecDB_save_path =  f"{args.database_path}/VectorDB/attn_cache_epoch-3_vectors.faiss"
    vecDB = VecDB().load(f"{vecDB_save_path}") 
    
    
    # =================== MMLU Dataset ===================
   
    subcats = defaultdict(list)
    for k, v_list in subcategories.items():
        for v in v_list:
            subcats[v].append(k)

    subcategory = subcats[f'{args.subcategory}']

    # =================== evaluation  ===================
    
    all_test_df = []
    for subject in subcategory:
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )
        all_test_df.append(test_df)
    all_test_df = pd.concat(all_test_df, axis=0, ignore_index=True)

    cors = []
    cnt = 0
    hit = 0
    for subject in subcategory:
        
        val_df = pd.read_csv(
            os.path.join(args.data_dir, "val", subject + "_val.csv"), header=None
        )
        dev_df = pd.read_csv(
                os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
            )[: args.ntrain]
        train_prompt = gen_prompt(dev_df, subject, k=args.ntrain) # k-shot

        for i in range(val_df.shape[0]):
            cnt += 1
            label = val_df.iloc[i, val_df.shape[1] - 1]
            prompt_end = format_example(val_df, i, include_answer=False)
            prompt = train_prompt + prompt_end
            reuse_tensor_index, hitted_records, compute_tensor_index, inputs = retrieve(prompt)

            if hitted_records.size:
                hit += 1
                print('-' * 90)
                s1 = val_df.iloc[i, 0]
                t1 = tokenizer.tokenize(s1)
                print("current idx in val_df: ", i)
                print("sentence 1 in val_df: ", s1)
                print("sentence 1 length: ", len(t1))

                idx = int(hitted_records[0][0] // config.num_hidden_layers)
                s2 = all_test_df.iloc[idx, 0]
                t2 = tokenizer.tokenize(s2)
                print("reuse idx in all_test_df: ", idx)
                print("sentence 2 in all_test_df: ", s2)
                print("sentence 2 length: ", len(t2))



                # idx = int(hitted_records[0][0] // config.num_hidden_layers)
                # s1 = all_test_df.iloc[idx, 0]
                # s2 = val_df.iloc[i, 0]
                # t1 = tokenizer.tokenize(s1)
                # t2 = tokenizer.tokenize(s2)
                # print("reuse idx in all_test_df: ", idx)
                # print("sentence 1 in all_test_df: ", s1)
                # print("sentence 1 lenth: ", len(t1))

                # print("current idx in val_df: ", i)
                # print("sentence 2 in val_df: ", s2)
                # print("sentence 2 lenth: ", len(t2))
                
                # =================== USE AttnCache ===================

                outputs, self_attn_latency_collector, e2e_latency_collector, ttft_list = evaluate(reuse_tensor_index, hitted_records, compute_tensor_index, inputs)

                self_attn_latency = self_attn_latency_collector.latency_list[1:]
                self_attn_average_time = np.mean(self_attn_latency) * 1000
                print(f"self-attn with_attn_cache average time: {self_attn_average_time} ms")

                e2e_latency = e2e_latency_collector.latency_list[1:]
                e2e_latency_average_time = np.mean(e2e_latency) * 1000
                print(f"end to end with_attn_cache average time: {e2e_latency_average_time} ms")

                ttft_average_time = np.mean(ttft_list[1:])
                print(f"time to first token average time: {ttft_average_time} ms")

                logits = outputs.logits[:, -1, :][0]
                probs = (
                torch.nn.functional.softmax(
                    torch.tensor(
                        [
                            logits[tokenizer("A").input_ids[1]],
                            logits[tokenizer("B").input_ids[1]],
                            logits[tokenizer("C").input_ids[1]],
                            logits[tokenizer("D").input_ids[1]],
                        ]
                    ),
                        dim=0,
                    ).detach().cpu().numpy()
                )
                pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
                cor = pred == label
                cors.append(cor)

                # =================== without AttnCache ===================
                
                
                self_attn_latency_collector = LatencyCollector()
                register_forward_latency_collector(self_attn_latency_collector, model.model.layers[-1].self_attn)
                e2e_latency_collector = LatencyCollector()
                register_forward_latency_collector(e2e_latency_collector, model.model)
                
                ttft_list = []
                for _ in range(11):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()            
                    outputs1 = model(**inputs, attention_cache=None, compute_tensor_index=None)
                    end.record()
                    torch.cuda.synchronize()
                    inference_time = start.elapsed_time(end)
                    ttft_list.append(inference_time)

                self_attn_latency = self_attn_latency_collector.latency_list[1:]
                self_attn_average_time_wo = np.mean(self_attn_latency) * 1000
                print(f"self-attn without_attn_cache average time: {self_attn_average_time_wo} ms")

                e2e_latency = e2e_latency_collector.latency_list[1:]
                e2e_latency_average_time_wo = np.mean(e2e_latency) * 1000
                print(f"end to end without_attn_cache average time: {e2e_latency_average_time_wo} ms")

                

                ttft_average_time = np.mean(ttft_list[1:])
                print(f"time to first token average time: {ttft_average_time} ms")

                print("self-att average speedup: ", self_attn_average_time_wo / self_attn_average_time)
                print("end to end average speedup: ", e2e_latency_average_time_wo / e2e_latency_average_time)

            else:
                outputs = model(**inputs, attention_cache=None, compute_tensor_index=None)

                logits = outputs.logits[:, -1, :][0]
                probs = (
                torch.nn.functional.softmax(
                    torch.tensor(
                        [
                            logits[tokenizer("A").input_ids[1]],
                            logits[tokenizer("B").input_ids[1]],
                            logits[tokenizer("C").input_ids[1]],
                            logits[tokenizer("D").input_ids[1]],
                        ]
                    ),
                        dim=0,
                    ).detach().cpu().numpy()
                )
                pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
                cor = pred == label
                cors.append(cor)
    
    print('-' * 90)    
    print("threshold: ", args.threshold, "hit: ", hit, "cnt :", cnt, "ratio: ", hit / cnt)        
    print("Average accuracy {:.3f} - {}".format(np.mean(cors), subcategory))










