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
from models.utils import VecDB, Emb, LatencyCollector, register_forward_latency_collector, parse_args
import pickle
from sklearn.metrics import accuracy_score
import pandas as pd
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

    model.to(torch.device(device))
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



def retrieve(data):
    inputs = tokenizer(data, padding="max_length", truncation=True, max_length=args.max_length, return_tensors="pt")
    inputs = inputs.to(device)
    
    x = model.model.embed_tokens(inputs["input_ids"])
    x = model.model.layers[0].input_layernorm(x)
    x = x.to(args.device)
    feature_vector =  feature_projector.embed(x).cpu().detach()
    feature_vector = feature_vector.to(torch.float32)#.numpy()
    sims, idx_list = vecDB.search(feature_vector)

    reuse_tensor_index = np.flatnonzero(1 - sims >= args.threshold)
    hitted_records = idx_list[reuse_tensor_index]
    compute_tensor_index = np.flatnonzero(1 - sims < args.threshold)

    return reuse_tensor_index, hitted_records, compute_tensor_index, inputs



def evaluate(reuse_tensor_index, hitted_records, compute_tensor_index, inputs):
    
    with torch.no_grad():
        total_tensor_index = np.concatenate((reuse_tensor_index, compute_tensor_index), axis=0)
        attention_cache = torch.empty((config.num_hidden_layers, len(total_tensor_index), config.num_attention_heads, args.max_length, args.max_length),dtype=dtype)                             
        if len(reuse_tensor_index) != 0:
            print(f"=========== hit {len(reuse_tensor_index)} APMs")
            for layer_idx in range(config.num_hidden_layers):
                for idx, record in zip(reuse_tensor_index, hitted_records):
                    x_loaded = torch.load(f"{args.save_dir}/States/{record[0]}.pt", map_location='cuda')
                    attn_weights = x_loaded["attn_weights"]
                    attention_cache[layer_idx][idx] = attn_weights

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

    if args.model_path == "deepseek-ai/deepseek-moe-16b-chat" \
       or args.model_path == "Qwen/Qwen1.5-MoE-A2.7B-Chat" \
        or args.model_path == "meta-llama/Llama-3.1-8B-Instruct":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
    
    if args.model_path == "Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4" or args.model_path =="Qwen/Qwen1.5-MoE-A2.7B-Chat":
        ans_idx = 0
    else:
        ans_idx = 1
    # ans_idx = 0
    
    device = torch.device('cuda')
    # device = torch.device('cpu')

    model, tokenizer, config = load_model_and_tokenizer(args)

    # =================== MMLU Dataset ===================
   
    subcats = defaultdict(list)
    for k, v_list in subcategories.items():
        for v in v_list:
            subcats[v].append(k)

    subcategory = subcats[f'{args.subcategory}']

    # =================== feature_projector & vecDB ===================

    feature_projector = Emb(args)
    # vecDB_save_path =  f"{args.save_dir}/VectorDB/attn_cache_epoch-3_vectors.faiss"
    vecDB = VecDB().load(args.vec_db_save_path) 
    
    
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
    attn_speedups = []
    e2e_speedups = []
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
            # if None:
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

                
                # =================== USE AttnCache ===================

                outputs, self_attn_latency_collector, e2e_latency_collector, ttft_list = evaluate(reuse_tensor_index, hitted_records, compute_tensor_index, inputs)

                self_attn_latency = self_attn_latency_collector.latency_list[1:] # drop first runing
                self_attn_average_time = np.mean(self_attn_latency) * 1000
                # print(f"self-attn average time: {self_attn_average_time} ms")

                e2e_latency = e2e_latency_collector.latency_list[1:]
                e2e_latency_average_time = np.mean(e2e_latency) * 1000
                # print(f"end to end average time: {e2e_latency_average_time} ms")

                ttft_average_time = np.mean(ttft_list[1:])
                # print(f"TTFT time: {ttft_average_time} ms")

                logits = outputs.logits[:, -1, :][0]
                a = tokenizer("A")
                b = tokenizer("A").input_ids
                c = tokenizer("A").input_ids[ans_idx]

                a1 = tokenizer("B")
                b1 = tokenizer("B").input_ids
                c1 = tokenizer("B").input_ids[ans_idx]

                a2 = tokenizer("C")
                b2 = tokenizer("C").input_ids
                c2 = tokenizer("C").input_ids[ans_idx]

                a3 = tokenizer("D")
                b3 = tokenizer("D").input_ids
                c3 = tokenizer("D").input_ids[ans_idx]

                probs = (
                torch.nn.functional.softmax(
                    torch.tensor(
                        [
                            logits[tokenizer("A").input_ids[ans_idx]],
                            logits[tokenizer("B").input_ids[ans_idx]],
                            logits[tokenizer("C").input_ids[ans_idx]],
                            logits[tokenizer("D").input_ids[ans_idx]],
                        ]
                    ),
                        dim=0,
                    ).detach().cpu().numpy()
                )
                pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
                cor = pred == label
                cors.append(cor)

                # =================== without AttnCache ===================
                
                # print("=================== without AttnCache ===================")
                self_attn_latency_collector = LatencyCollector()
                register_forward_latency_collector(self_attn_latency_collector, model.model.layers[-1].self_attn)
                e2e_latency_collector = LatencyCollector()
                register_forward_latency_collector(e2e_latency_collector, model.model)
                
                ttft_list = []
                with torch.no_grad():
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
                # print(f"self-attn average time: {self_attn_average_time_wo} ms")

                e2e_latency = e2e_latency_collector.latency_list[1:]
                e2e_latency_average_time_wo = np.mean(e2e_latency) * 1000
                # print(f"end to end average time: {e2e_latency_average_time_wo} ms")

                

                ttft_average_time = np.mean(ttft_list[1:])
                # print(f"TTFT time: {ttft_average_time} ms")
                
                print("=================== Speedup ===================")
                attn_speedup = self_attn_average_time_wo / self_attn_average_time
                e2e_speedup = e2e_latency_average_time_wo / e2e_latency_average_time
                attn_speedups.append(attn_speedup)
                e2e_speedups.append(e2e_speedup)
                print("self-att average speedup: ", attn_speedup)
                print("end to end average speedup: ", e2e_speedup)
                     # cpu: /2.92 - e2e_avg_speedup: 1.57
            else:    # gpu: 3.01 - e2e_avg_speedup: 1.63 / 3.7 - e2e_avg_speedup: 2.06
                with torch.no_grad():
                    outputs = model(**inputs, attention_cache=None, compute_tensor_index=None)

                logits = outputs.logits[:, -1, :][0]
                probs = (
                torch.nn.functional.softmax(
                    torch.tensor(
                        [
                            logits[tokenizer("A").input_ids[ans_idx]],
                            logits[tokenizer("B").input_ids[ans_idx]],
                            logits[tokenizer("C").input_ids[ans_idx]],
                            logits[tokenizer("D").input_ids[ans_idx]],
                        ]
                    ),
                        dim=0,
                    ).detach().cpu().numpy()
                )
                pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
                cor = pred == label
                cors.append(cor)
    
    print('-' * 90)   
    print("device: ", device, "n-shot: ", args.ntrain, "threshold: ", args.threshold, "hit: ", hit, "cnt :", cnt, "ratio: ", hit / cnt)        
    attn_avg_speedup = round(np.mean(attn_speedups) ,2)
    e2e_avg_speedup = round(np.mean(e2e_speedups),2)
    avg_acc = round(np.mean(cors) * 100,2)
    print("Model: ", args.model_path )
    
    print("Average accuracy {} - {}".format(avg_acc, subcategory))

    print("attn_avg_speedup: {} - e2e_avg_speedup: {}".format(attn_avg_speedup, e2e_avg_speedup))
    # print(cors)
