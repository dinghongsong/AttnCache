import argparse
from statistics import mode
import torch 
import numpy as np
import time 
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
from itertools import combinations
import torch
from models.utils import LinearNet, HiddenStatesAPMsDataset, VecDB
import os
import tqdm
import pickle

def create_pairs(inputs, args):
    x0_data = []
    x1_data = []
    labels = []
    data = list(combinations(inputs['hiddenstates'], 2))
    sample = list(combinations(inputs['apms'], 2)) 
    attn_masks = list(combinations(inputs['attn_mask'], 2)) 
    for (x0, x1), (apm0, apm1), (attn_mask0, attn_mask1) in zip(data, sample, attn_masks):

        apm0 = apm0.numpy() # 24 heads 
        apm1 = apm1.numpy()
        batch_size, num_head, seq_len = apm0.shape[0], apm0.shape[1], apm0.shape[2]
        apm_diff = np.abs(apm0-apm1)

        attn_mask0 = attn_mask0.numpy()
        attn_mask1 = attn_mask1.numpy()
        length0 = np.sum(attn_mask0[0,0,-1, :] == 0.0)
        length1 = np.sum(attn_mask1[0,0,-1, :] == 0.0)
    
        if args.is_attn_memo:
            label = np.sum(apm_diff, axis=tuple(range(1, apm_diff.ndim))) / seq_len  / num_head / 2
        elif args.is_attn_cache:
            label = np.sum(apm_diff, axis=tuple(range(1, apm_diff.ndim))) / seq_len  / num_head / 2 + 0.8 * np.abs(length0 - length1) # np.sum(attn_mask_diff, axis=tuple(range(1, apm_diff.ndim))) / seq_len
        print("label: ", label)
        x0_data.append(x0)
        x1_data.append(x1)
        labels.append(label)
    x0_data = torch.cat(x0_data, dim=0)
    x1_data = torch.cat(x1_data, dim=0)
    labels = np.concatenate(labels, axis=0)
    labels = torch.from_numpy(np.array(labels, dtype=np.float32))
    return x0_data, x1_data, labels

def train_feature_projector(args):
    model = LinearNet().to(args.device)
    learning_rate = 0.1
    momentum = 0.9
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    datasets = HiddenStatesAPMsDataset(args.dataset_dir)
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
      
    torch.save(model.state_dict(), args.model_save_path)
    print(f"Save Feature Projector to {args.model_save_path}")
    return model

def build_index_database(model, args):
    vecdb = VecDB()
    files_num = len(os.listdir(f"{args.dataset_dir}/HiddenStatesDB"))
    for i in tqdm(range(files_num)):
        with open(f"{args.dataset_dir}/HiddenStatesDB/" + str(i) + ".pickle", 'rb') as f:
            hidden = pickle.load(f)
        if i == 0:
            layer_tensor =  model.embed(hidden)
        else:
            tmp = model.embed(hidden)
            layer_tensor = np.vstack((layer_tensor, tmp))

    vecdb.train_index(layer_tensor)
    vecdb.add(layer_tensor)
    vecdb.save(args.db_save_path)
    print("Build and Save faiss index success!")

if __name__ == "__main__" :
    torch.manual_seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=2, help='Number of epoch to train')
    parser.add_argument('--batchsize', '-b', type=int, default=64, help='Number of images in each mini-batch')
    parser.add_argument('--task_name', default="STS16", help="Comma separated. Default is None i.e. running all tasks")
    parser.add_argument('--is_attn_cache', action='store_true', default=False)
    parser.add_argument('--is_attn_memo', action='store_true', default=False)
    parser.add_argument('--dataset_dir', type=str, default="/home/sdh/MetaEOL/MetaEOL/database/Llama-3.2-3B-Instruct", help='dataset_dir')

    args = parser.parse_args()
    args.dataset_dir += "/" + args.task_name
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.is_attn_memo:
        args.model_save_path = f'{args.dataset_dir}/Embedding_models/mlp_model_attn_memo-epoch{args.epoch}.pth'
        args.db_save_path =  f"{args.dataset_dir}/VectorDB/attn_memo_epoch-{args.epoch}_vectors.faiss"
    elif args.is_attn_cache:
        args.model_save_path = f'{args.dataset_dir}/Embedding_models/mlp_model_attn_cache-epoch{args.epoch}.pth'
        args.db_save_path =  f"{args.dataset_dir}/VectorDB/attn_cache_epoch-{args.epoch}_vectors.faiss"


    print("===========================")
    print("Args: ", args)
    print("===========================")

    model = train_feature_projector(args)
    build_index_database(model, args)
 

    
