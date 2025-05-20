import faiss
import numpy as np 
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset 
import time 
import os
from tqdm import tqdm
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
# from .modeling_llama import LlamaForCausalLM

class LinearNet(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()


        if args.model_path == "meta-llama/Llama-3.2-3B-Instruct":
            dtype = torch.bfloat16

            # if args.ntrain == 5: #engineering
            #     self.fc1 = nn.Linear(175104, 128).to(dtype) #5 shot
            # elif args.ntrain == 4:
            #     self.fc1 = nn.Linear(141312, 128).to(dtype) #4 shot
            # elif args.ntrain == 3:
            #     self.fc1 = nn.Linear(116736, 128).to(dtype) #3 shot
            # elif args.ntrain == 2:
            #     self.fc1 = nn.Linear(168960, 128).to(dtype) #2 shot
            # elif args.ntrain == 1:
            #     self.fc1 = nn.Linear(121344, 128).to(dtype) #1 shot
            # elif args.ntrain == 0:
            #     self.fc1 = nn.Linear(76800, 128).to(dtype) # 0 shot

            if args.ntrain == 5: #math
                self.fc1 = nn.Linear(159744, 128).to(dtype) #5 shot
            elif args.ntrain == 4:
                self.fc1 = nn.Linear(147456, 128).to(dtype) #4 shot
            elif args.ntrain == 3:
                self.fc1 = nn.Linear(115200, 128).to(dtype) #3 shot
            elif args.ntrain == 2:
                self.fc1 = nn.Linear(164352, 128).to(dtype) #2 shot
            elif args.ntrain == 1:
                self.fc1 = nn.Linear(113664, 128).to(dtype) #1 shot
            elif args.ntrain == 0:
                self.fc1 = nn.Linear(144384, 128).to(dtype) # 0 shot
        
        if args.model_path == "meta-llama/Llama-3.1-8B-Instruct":
            dtype = torch.float16

            if args.ntrain == 5: #math
                self.fc1 = nn.Linear(159744, 128).to(dtype) #5 shot
            elif args.ntrain == 4:
                self.fc1 = nn.Linear(147456, 128).to(dtype) #4 shot
            elif args.ntrain == 3:
                self.fc1 = nn.Linear(115200, 128).to(dtype) #3 shot
            elif args.ntrain == 2:
                self.fc1 = nn.Linear(164352, 128).to(dtype) #2 shot
            elif args.ntrain == 1:
                self.fc1 = nn.Linear(113664, 128).to(dtype) #1 shot
            elif args.ntrain == 0:
                self.fc1 = nn.Linear(192512, 128).to(dtype) # 0 shot
        
        if args.model_path == "deepseek-ai/deepseek-moe-16b-chat":
            dtype = torch.float16
            if args.ntrain == 5: #math
                self.fc1 = nn.Linear(159744, 128).to(dtype) #5 shot
            elif args.ntrain == 4:
                self.fc1 = nn.Linear(147456, 128).to(dtype) #4 shot
            elif args.ntrain == 3:
                self.fc1 = nn.Linear(115200, 128).to(dtype) #3 shot
            elif args.ntrain == 2:
                self.fc1 = nn.Linear(164352, 128).to(dtype) #2 shot
            elif args.ntrain == 1:
                self.fc1 = nn.Linear(113664, 128).to(dtype) #1 shot
            elif args.ntrain == 0:
                self.fc1 = nn.Linear(96256, 128).to(dtype) # 0 shot
            
        if args.model_path ==  "Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4":
            dtype = torch.float16
            if args.ntrain == 5: #math
                self.fc1 = nn.Linear(159744, 128).to(dtype) #5 shot
            elif args.ntrain == 4:
                self.fc1 = nn.Linear(147456, 128).to(dtype) #4 shot
            elif args.ntrain == 3:
                self.fc1 = nn.Linear(115200, 128).to(dtype) #3 shot
            elif args.ntrain == 2:
                self.fc1 = nn.Linear(164352, 128).to(dtype) #2 shot
            elif args.ntrain == 1:
                self.fc1 = nn.Linear(113664, 128).to(dtype) #1 shot
            elif args.ntrain == 0:
                self.fc1 = nn.Linear(96256, 128).to(dtype) # 0 shot
            
        if args.model_path ==  "Qwen/Qwen1.5-MoE-A2.7B-Chat":
            dtype = torch.float16
            if args.ntrain == 5: #math
                self.fc1 = nn.Linear(159744, 128).to(dtype) #5 shot
            elif args.ntrain == 4:
                self.fc1 = nn.Linear(147456, 128).to(dtype) #4 shot
            elif args.ntrain == 3:
                self.fc1 = nn.Linear(115200, 128).to(dtype) #3 shot
            elif args.ntrain == 2:
                self.fc1 = nn.Linear(164352, 128).to(dtype) #2 shot
            elif args.ntrain == 1:
                self.fc1 = nn.Linear(113664, 128).to(dtype) #1 shot
            elif args.ntrain == 0:
                self.fc1 = nn.Linear(96256, 128).to(dtype) # 0 shot

        self.fc2 = nn.Linear(128, 128).to(dtype)
        self.bn1 = nn.BatchNorm2d(1).to(dtype)
        self.bn2 = nn.BatchNorm1d(128).to(dtype)
    
    def forward_once(self, x):
        x = torch.unsqueeze(x,1)
        x = F.max_pool2d(x, 2)
        while x.shape[2] > 128:
            x = F.max_pool2d(x, kernel_size=(2, 1))  
        x = self.bn1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.fc2(x)
        return x

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2

class Emb():
    def __init__(self, args) -> None:
        self.device = args.device

        self.emb = LinearNet(args).to(self.device)#.to_empty('cpu')#.to(device)
        self.emb.load_state_dict(torch.load(args.feature_projector_save_path, map_location=self.device))
        self.emb.eval()
    def embed(self, inputs):
        """
        return a numpy obj
        """
        return self.emb.forward_once(inputs)
    
class VecDB():
    def __init__(self, d=128, nlist=128, m=8, bit=8) -> None:
 
        quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFFlat(quantizer, d, nlist)

    def train_index(self, embeddings):
        assert not self.index.is_trained
        self.index.train(embeddings)
        assert self.index.is_trained

    def add(self, embedding):
        self.index.add(embedding)

    def search(self, embedding, k=1, nprob = 128):
        self.index.nprob = nprob
        D, I = self.index.search(embedding, k)
        return D, I
    
    def load(self, path):
        self.index = faiss.read_index(path)
        return self
    
    def save(self, save_path):
        faiss.write_index(self.index, save_path)
        

class HiddenStatesAMPsDataset(Dataset):
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir

    def __len__(self):
        return len([name for name in os.listdir(f"{self.data_dir}/States")])

    def __getitem__(self, index): 

        x_loaded = torch.load(f"{self.data_dir}/States/{index}.pt", map_location='cuda')
        
        return {"hiddenstates":x_loaded["hidden_states"], "amps": x_loaded["attn_weights"],
                 "attn_mask":x_loaded["attention_mask"]}
    

class LatencyCollector:
    def __init__(self):
        self.start = None
        self.latency_list = []

    def pre_hook(self, *args):
        self.start = time.time()

    def hook(self, *args):
        self.latency_list.append(time.time() - self.start)


def register_forward_latency_collector(latency_collector, model):
    model.register_forward_pre_hook(latency_collector.pre_hook)
    model.register_forward_hook(latency_collector.hook)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="meta-llama/Llama-3.2-3B-Instruct")     
    # parser.add_argument("--model-path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")     
    # parser.add_argument("--model-path", type=str, default="deepseek-ai/deepseek-moe-16b-chat")  
    # parser.add_argument("--model-path", type=str, default="Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4")  
    # parser.add_argument("--model-path", type=str, default="Qwen/Qwen1.5-MoE-A2.7B-Chat")  

    parser.add_argument("--data_dir", "-d", type=str, default="data")
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default=default_device)

    # =========================== MMLU Dataset ===========================
    parser.add_argument("--threshold", type=float, default=0.995)

    parser.add_argument("--ntrain", "-k", type=int, default=0)
    parser.add_argument("--subcategory", type=str, choices=['math', 'health', 'physics', 'business', 'biology', 
                                                    'chemistry', 'computer science', 'economics', 
                                                    'engineering', 'philosophy', 'other', 'history', 
                                                    'geography', 'politics', 'psychology', 'culture', 'law'], 
                    default='math', help="17 subcategories in MMLU")
    
  

    # ======================== Feature Projector Training ===========================
    
    parser.add_argument('--epoch', '-e', type=int, default=3, help='Number of epoch to train')
    parser.add_argument('--batchsize', '-b', type=int, default=32, help='Number of images in each mini-batch')
    
    args = parser.parse_args()

    if args.ntrain == 0: #math
        args.max_length = 189
    elif args.ntrain == 1:
        args.max_length = 299
    elif args.ntrain == 2:
        args.max_length = 429
    elif args.ntrain == 3:
        args.max_length = 600
    elif args.ntrain == 4:
        args.max_length = 770
    elif args.ntrain == 5:
        args.max_length = 835

    args.save_dir = os.path.join("DB", args.model_path, args.subcategory, f'{args.ntrain}_shots')
    args.feature_projector_save_path = os.path.join(args.save_dir, f'Embedding_models/mlp_model_attn_cache-epoch{args.epoch}.pth')
    args.vec_db_save_path = os.path.join(args.save_dir, f'VectorDB/attn_cache_epoch-{args.epoch}_vectors.faiss')
    


    return args
