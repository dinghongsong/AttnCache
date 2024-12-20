import numpy as np 
import pickle
import os
from tqdm import tqdm
from models.utils import Emb, VecDB
from argparse import ArgumentParser
import torch

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--epoch', type=int, default=2, help='Number of epoch over the dataset to train')
    parser.add_argument('--task_name', default="STS13", help="task_name")
    parser.add_argument('--is_attn_cache', action='store_true', default=False)
    parser.add_argument('--is_attn_memo', action='store_true', default=False)
    parser.add_argument('--dataset_dir', type=str, default="/home/sdh/MetaEOL/MetaEOL/database/Llama-3.2-3B-Instruct", help='dataset_dir')
    args = parser.parse_args()
    args.dataset_dir += "/" + args.task_name

    # if args.is_attn_memo:
    #     embedding_model = f"{args.dataset_dir}/Embedding_models/mlp_model_attn_memo-epoch{args.epoch}.pth"
    #     save_path = f'{args.dataset_dir}/Embedding_models/mlp_model_attn_memo-epoch{args.epoch}.pth'
    # elif args.is_attn_cache:
    #     save_path = f'{args.dataset_dir}/Embedding_models/mlp_model_attn_cache-epoch{args.epoch}.pth'

    embedding_model = f"{args.dataset_dir}/Embedding_models/mlp_model_attn_cache-epoch{args.epoch}.pth"
    save_path =  f"{args.dataset_dir}/VectorDB/attn_cache_epoch-{args.epoch}_vectors.faiss"
    
    print("===========================")
    print("Args: ", args)
    print(f"Using embedding model: {embedding_model}")
    print("===========================")
    
    

    emb = Emb(embedding_model)


    vecdb = VecDB()

    files_num = len(os.listdir(f"{args.dataset_dir}/HiddenStatesDB"))
    for i in tqdm(range(files_num)):
        with open(f"{args.dataset_dir}/HiddenStatesDB/" + str(i) + ".pickle", 'rb') as f:
            hidden = pickle.load(f)
        if i == 0:
            layer_tensor =  emb.embed(hidden)
        else:
            tmp = emb.embed(hidden)
            layer_tensor = np.vstack((layer_tensor, tmp))

    vecdb.train_index(layer_tensor)
    vecdb.add(layer_tensor)
    vecdb.save(save_path)
    print("Build and Save faiss index success!")



