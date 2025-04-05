import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from statistics import mode
import torch 
import numpy as np
import os
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns


def plot_attention(attn1, attn2, size):

    dir = f"/home/sdh/ACL/AttnCache/figures"
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    cmap.set_bad(color="gray") 
   
    for layer in range(12): # 28 layers
        apm0 = attn1[layer].cpu().numpy() # 24 heads 
        apm1 = attn2[layer].cpu().numpy()

        for head in range(12): # 24 heads
        
            # mask=np.triu(np.ones_like(apm0[0][head][-size:, -size:,], dtype=bool), k=1)
            mask=None
            sns.heatmap(apm0[0][head][:size, :size], cmap=cmap, vmin=0,linewidths=0, rasterized=True, square=True, vmax=1, mask=mask)
            # sns.heatmap(apm0[0][head][-size:, -size:], cmap=cmap, vmin=0,linewidths=0, rasterized=True, square=True, vmax=1, mask=mask)
            plt.title(f"Sentence 1 Layer {layer} Head {head}", fontsize=15)
            plt.xticks(ticks=np.arange(0, size, 2), labels=np.arange(0, size, 2))
            plt.yticks(ticks=np.arange(0, size, 2), labels=np.arange(0, size, 2), rotation=360)
            plt.show()
                
            save_path = f"{dir}/Sentence_1/layer_{layer}/head_{head}.pdf"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
            plt.clf()

            # mask=np.triu(np.ones_like(apm1[0][head][-size:, -size:,], dtype=bool), k=1)
            mask=None
            sns.heatmap(apm1[0][head][:size, :size], cmap=cmap, vmin=0,linewidths=0, rasterized=True, square=True, vmax=1, mask=mask)
            # sns.heatmap(apm1[0][head][-size:, -size:], cmap=cmap, vmin=0,linewidths=0, rasterized=True, square=True, vmax=1, mask=mask)
            plt.title(f"Sentence 2 Layer {layer} Head {head}", fontsize=15)
            plt.xticks(ticks=np.arange(0, size, 2), labels=np.arange(0, size, 2))
            plt.yticks(ticks=np.arange(0, size, 2), labels=np.arange(0, size, 2), rotation=360)
            plt.show()

           
            save_path = f"{dir}/Sentence_2/layer_{layer}/head_{head}.pdf"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, format="pdf", bbox_inches="tight") 
            plt.clf()

    print("finish plotting!")




model_path = "./trained_bert_sst2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, output_attentions=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  




dataset = load_dataset("glue", "sst2")
validation_data = dataset["validation"] # 872
test_data = dataset["test"]
train_data = dataset["train"]

s1 = train_data[388]['sentence']
s2 = validation_data[6]['sentence']

# s1 = validation_data[170]['sentence']
# s2 = train_data[1648]['sentence']

size=16
print("s1: ", s1)
print("s2: ", s2)
inputs1 = tokenizer(s1, return_tensors="pt", padding='max_length', truncation=True, max_length=64)
inputs2 = tokenizer(s2, return_tensors="pt", padding='max_length', truncation=True, max_length=64)
inputs1 = inputs1.to(device)
inputs2 = inputs2.to(device)

with torch.no_grad():  
    outputs1 = model.bert(**inputs1)
    outputs2 = model.bert(**inputs2)

    attentions1 = outputs1.attentions 
    attentions2 = outputs2.attentions 

plot_attention(attentions1, attentions2, size)


