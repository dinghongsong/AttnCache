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
from models.utils import LinearNet, HiddenStatesAPMsDataset

def create_pairs(inputs):
    x0_data = []
    x1_data = []
    labels = []
    data = list(combinations(inputs['hiddenstates'], 2))
    sample = list(combinations(inputs['apms'], 2)) 
    attn_masks = list(combinations(inputs['attn_mask'], 2)) 
    for (x0, x1), (apm0, apm1), (attn_mask0, attn_mask1) in zip(data, sample, attn_masks):

        # apm00 = apm0.squeeze(0).numpy()[0] # head 0
        # apm11 = apm1.squeeze(0).numpy()[0]
        # seq_len = apm00.shape[0]
        # label = np.sum(np.abs(apm00-apm11)) / seq_len / 2

        apm0 = apm0.numpy() # 24 heads 
        apm1 = apm1.numpy()
        batch_size, num_head, seq_len = apm0.shape[0], apm0.shape[1], apm0.shape[2]
        apm_diff = np.abs(apm0-apm1)

        attn_mask0 = attn_mask0.numpy()
        attn_mask1 = attn_mask1.numpy()
        length0 = np.sum(attn_mask0[0,0,-1, :] == 0.0)
        length1 = np.sum(attn_mask1[0,0,-1, :] == 0.0)
        # attn_mask0 = np.where(attn_mask0 != 0, 1, 0)
        # attn_mask1 = np.where(attn_mask1 != 0, 1, 0)
        # # size = -30
        # # print(attn_mask0[0,0,size:, size:])
        # # print(attn_mask1[0,0,size:, size:])
        # # print(attn_mask1[0,0,-1:, :][0])

        # # a = attn_mask1[0,0,-1:, :][0]
        # length1 = np.sum(attn_mask1[0,0,-1:, :][0] == 0)
        # attn_mask_diff =  np.abs(attn_mask0 - attn_mask1)
        # s = np.sum(attn_mask_diff) / seq_len
        # if s == 0:
        #     print("ok")
        # atten_memo
        label = np.sum(apm_diff, axis=tuple(range(1, apm_diff.ndim))) / seq_len  / num_head / 2
        ## atten_cache
        # label = np.sum(apm_diff, axis=tuple(range(1, apm_diff.ndim))) / seq_len  / num_head / 2 + 0.8 * np.abs(length0 - length1) # np.sum(attn_mask_diff, axis=tuple(range(1, apm_diff.ndim))) / seq_len
        print("label: bs", label)
        x0_data.append(x0)
        x1_data.append(x1)
        labels.append(label)
    x0_data = torch.cat(x0_data, dim=0)
    x1_data = torch.cat(x1_data, dim=0)
    labels = np.concatenate(labels, axis=0)
    labels = torch.from_numpy(np.array(labels, dtype=np.float32))
    return x0_data, x1_data, labels



if __name__ == "__main__" :
    torch.manual_seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=2,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--task_name', default="STS16", 
                        help="Comma separated. Default is None i.e. running all tasks")
    parser.add_argument('--dataset_dir', type=str, default="/home/sdh/MetaEOL/MetaEOL/database/Llama-3.2-3B-Instruct",
                        help='dataset_dir')
    # parser.add_argument('--threshold', type=float, default=0.8,help='threshold')

    args = parser.parse_args()
    args.dataset_dir += "/" + args.task_name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("===========================")
    print("Args: ", args)
    print("===========================")

    # model = Net()
    model = LinearNet().to(device)
    # model = SingleNet()


    learning_rate = 0.1
    momentum = 0.9
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    datasets = HiddenStatesAPMsDataset(args.dataset_dir)
    pwdist = torch.nn.PairwiseDistance()
    loss_fn = torch.nn.SmoothL1Loss()

    train_loader = DataLoader(datasets, batch_size=args.batchsize, shuffle=True)
    all_train_loss = []
    for epoch in range(1, args.epoch + 1):
        train_loss = []
        model.train() 
        start = time.time()
        for batch_idx, inputs in enumerate(train_loader):
            x0, x1, labels = create_pairs(inputs)
            x0, x1, labels = x0.to(device), x1.to(device), labels.to(device)
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
                torch.save(model.state_dict(), f'{args.dataset_dir}/Embedding_models/mlp_model_attn_memo-epoch{epoch}.pth')
                print(f"Save embedding model to {args.dataset_dir}/Embedding_models/mlp_model_attn_memo-epoch{epoch}.pth")
  
        torch.save(model.state_dict(), f'{args.dataset_dir}/Embedding_models/mlp_model_attn_memo-epoch{epoch}.pth')
        print(f"Save embedding model to {args.dataset_dir}/Embedding_models/mlp_model_attn_memo-epoch{epoch}.pth")
        all_train_loss.extend(train_loss)
    
    # plt.gca().cla()
    # plt.plot(all_train_loss, label="train loss")
    # plt.xlabel("Iteration")
    # plt.ylabel("Loss")
    # plt.title("Training Loss with MLP")
    # plt.legend()
    # plt.draw()
    # plt.savefig('MLP_train_loss.png')
    # plt.gca().clear()


    

