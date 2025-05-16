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
    def __init__(self,model_name) -> None:
        super().__init__()


        # bert
        # self.fc1 = nn.Linear(12288, 128) # 7b 8b
        # llama
        if model_name == "meta-llama/Llama-3.1-8B-Instruct":
            self.fc1 = nn.Linear(262144, 128).to(torch.bfloat16) #llama3-8b
        if model_name == "meta-llama/Llama-3.2-3B-Instruct":
            self.fc1 = nn.Linear(196608, 128).to(torch.bfloat16) #llama3-8b


        self.fc2 = nn.Linear(128, 128).to(torch.bfloat16)
        self.bn1 = nn.BatchNorm2d(1).to(torch.bfloat16)
        self.bn2 = nn.BatchNorm1d(128).to(torch.bfloat16)
    
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
    def __init__(self, model_dir, model_name, device) -> None:
        self.device = device
        # self.emb = LinearNet(model_name).to_empty(device='cpu')#.to_empty('cpu')#.to(device)
        # self.emb.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
        # self.emb.eval()

        self.emb = LinearNet(model_name).to(self.device)#.to_empty('cpu')#.to(device)
        self.emb.load_state_dict(torch.load(model_dir, map_location=self.device))
        self.emb.eval()
    def embed(self, inputs):
        """
        return a numpy obj
        """
        # return self.emb.forward_once(torch.from_numpy(inputs)).detach().numpy()
        return self.emb.forward_once(inputs)
    
class VecDB():
    def __init__(self, d=128, nlist=128, m=8, bit=8) -> None:
 
        quantizer = faiss.IndexFlatL2(d)
        # self.index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bit)
        self.index = faiss.IndexIVFFlat(quantizer, d, nlist)
        # self.index = faiss.IndexFlatL2(d) 
    def train_index(self, embeddings):
        assert not self.index.is_trained
        self.index.train(embeddings)
        assert self.index.is_trained

    def add(self, embedding):
        self.index.add(embedding)

    def search(self, embedding, k=1, nprob = 128):
        self.index.nprob = nprob
        D, I = self.index.search(embedding, k)#I为每个待检索query最相似TopK的索引list，D为其对应的距离
        return D, I
    
    def load(self, path):
        self.index = faiss.read_index(path)
        return self
    
    def save(self, save_path):#mlp_model_attn_mask-epoch{epoch}.pth
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

    # parser.add_argument("--model-path", type=str, default="meta-llama/Llama-3.2-3B-Instruct") 
    # parser.add_argument("--save-dir", type=str, default="LlamaDB")

    parser.add_argument("--model-path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")   
    
    
    parser.add_argument("--threshold", type=float, default=0.995)

    # =========================== MMLU Dataset ===========================
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--ntrain", "-k", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--subcategory", type=str, choices=['math', 'health', 'physics', 'business', 'biology', 
                                                    'chemistry', 'computer science', 'economics', 
                                                    'engineering', 'philosophy', 'other', 'history', 
                                                    'geography', 'politics', 'psychology', 'culture', 'law'], 
                    default='math', help="17 subcategories in MMLU")
    
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default=default_device)

    # ======================== Feature Projector Training ===========================
    
    parser.add_argument('--epoch', '-e', type=int, default=3, help='Number of epoch to train')
    parser.add_argument('--batchsize', '-b', type=int, default=32, help='Number of images in each mini-batch')
    
    args = parser.parse_args()
    args.save_dir = os.path.join("DB", args.model_path, f'{args.ntrain}_shots')
    args.feature_projector_save_path = os.path.join(args.save_dir, f'Embedding_models/mlp_model_attn_cache-epoch{args.epoch}.pth')
    args.vec_db_save_path = os.path.join(args.save_dir, f'VectorDB/attn_cache_epoch-{args.epoch}_vectors.faiss')
    


    return args

