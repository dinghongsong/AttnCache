import argparse
from statistics import mode
import torch 
import numpy as np
import time 
from torch.autograd import Variable
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
from itertools import combinations
import torch
from models.utils import LinearNet, HiddenStatesAMPsDataset, VecDB, Emb
from models.utils import VecDB, Emb, LatencyCollector, register_forward_latency_collector, parse_args

import os
from tqdm import tqdm


def create_pairs(inputs, args):
    x0_data = []
    x1_data = []
    labels = []
    data = list(combinations(inputs['hiddenstates'], 2))
    sample = list(combinations(inputs['amps'], 2)) 
    attn_masks = list(combinations(inputs['attn_mask'], 2)) 
    for (x0, x1), (amp0, amp1), (attn_mask0, attn_mask1) in zip(data, sample, attn_masks):

        batch_size, num_head, seq_len = amp0.shape[0], amp0.shape[1], amp0.shape[2]
        amp_diff = torch.abs(amp0-amp1)

        
        length0 = torch.sum(attn_mask0[0,0,-1, :] == 0.0)
        length1 = torch.sum(attn_mask1[0,0,-1, :] == 0.0)
        
        label = torch.sum(amp_diff, axis=tuple(range(1, amp_diff.ndim))) / seq_len  / num_head / 2 + torch.abs(length0 - length1)
        # print("label: ", label)
        x0_data.append(x0)
        x1_data.append(x1)
        labels.append(label)
    x0_data = torch.cat(x0_data, dim=0)
    x1_data = torch.cat(x1_data, dim=0)
    labels = torch.cat(labels, dim=0)

    return x0_data, x1_data, labels

def train_feature_projector(args):
    model = LinearNet(args.model_path).to(args.device)
    learning_rate = 0.1
    momentum = 0.9
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    datasets = HiddenStatesAMPsDataset(args.save_dir)
    pwdist = torch.nn.PairwiseDistance()
    loss_fn = torch.nn.SmoothL1Loss()
    train_loader = DataLoader(datasets, batch_size=args.batchsize, shuffle=True)
    
    for epoch in range(1, args.epoch + 1):
        train_loss = []
        model.train() 
        start = time.time()
        for batch_idx, inputs in enumerate(train_loader):
            x0, x1, labels = create_pairs(inputs, args)
            x0, x1, labels = x0.to(args.device), x1.to(args.device), labels.to(args.device)
            x0, x1, labels = Variable(x0), Variable(x1), Variable(labels)
            output1, output2 = model(x0, x1)
            y_hat = pwdist(output1, output2)
            loss = loss_fn(y_hat, labels)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print("Loss: {}, batch:{}, Epoch:{}, Time/batch = {}".format(loss.item(), batch_idx, epoch, time.time()-start))
                torch.save(model.state_dict(), args.feature_projector_save_path)
         
      
    torch.save(model.state_dict(), args.feature_projector_save_path)
    print(f"Save Feature Projector to {args.feature_projector_save_path}")
    return model



def build_index_database(model, args):
    print("Build and Save faiss index ...")
    vecdb = VecDB()
    files_num = len(os.listdir(f"{args.save_dir}/States"))
    with torch.no_grad():
        for i in tqdm(range(files_num)):
            x_loaded = torch.load(f"{args.save_dir}/States/{i}.pt", map_location='cuda')
            hidden = x_loaded["hidden_states"]
            if i == 0:
                
                layer_tensor =  model.embed(hidden).cpu()
    
            else:
                tmp = model.embed(hidden).cpu()
                layer_tensor = torch.vstack((layer_tensor, tmp))
                
    layer_tensor = layer_tensor.to(torch.float32)
    vecdb.train_index(layer_tensor)
    vecdb.add(layer_tensor)
    vecdb.save(args.vec_db_save_path)
    print("Build and Save faiss index success!")


if __name__ == "__main__" :

    args = parse_args()
    train_feature_projector(args)

    feature_projector = Emb(args.feature_projector_save_path, args.model_path, args.device)
    build_index_database(feature_projector, args)
    
    #0.287

    

