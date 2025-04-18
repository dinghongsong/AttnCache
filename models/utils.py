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



class LinearNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()


        # bert
        # self.fc1 = nn.Linear(12288, 128) # 7b 8b
        # llama
        self.fc1 = nn.Linear(262144, 128) #llama3-8b


        self.fc2 = nn.Linear(128, 128)
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm1d(128)
    
    def forward_once(self, x):
        x = torch.unsqueeze(x,1)
        x = F.max_pool2d(x, 2)
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
    def __init__(self, model_dir) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emb = LinearNet().to_empty(device='cpu')#.to_empty('cpu')#.to(device)
        self.emb.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
        self.emb.eval()
    def embed(self, inputs):
        """
        return a numpy obj
        """
        return self.emb.forward_once(torch.from_numpy(inputs)).detach().numpy()
        # return self.emb.forward_once(inputs)
    
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
        

class HiddenStatesAPMsDataset(Dataset):
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
        self.hiddenstates_dir = self.data_dir + '/HiddenStatesDB'
        self.apms_dir = self.data_dir + '/APMsDB'
        self.attn_mask_dir = self.data_dir + '/AttnMask'

    def __len__(self):
        return len([name for name in os.listdir(self.apms_dir)])

    def __getitem__(self, index): 
        with open(self.hiddenstates_dir + "/" + str(index) + ".pickle", 'rb') as f: 
            hiddenstates = pickle.load(f)
        with open(self.apms_dir + "/" + str(index) + ".pickle", 'rb') as f: 
            apms = pickle.load(f)
        with open(self.attn_mask_dir + "/" + str(index) + ".pickle", 'rb') as f: 
            attn_mask = pickle.load(f)
        
        return {"hiddenstates":hiddenstates, "apms": apms, "attn_mask":attn_mask}
    

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
