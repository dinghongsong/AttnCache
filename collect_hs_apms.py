from transformers import AutoTokenizer, AutoConfig
import torch
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
import random
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
import random
import argparse
from statistics import mode
import torch 
import numpy as np
import torch
from models.modeling_bert import BertForSequenceClassification


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="./trained_bert_sst2")
    parser.add_argument("--save-dir", type=str, default="/home/sdh/ACL/AttnCache/BertDB")
    parser.add_argument("--dataset", type=str, default="sst2")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    dataset = load_dataset("glue", f"{args.dataset}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)
    model = BertForSequenceClassification.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  

    train_data = dataset["train"] #67349



    # =================== collect hidden_states & apms & attn masks ===================

    sampled_data = train_data[:1000]

    for example in sampled_data['sentence']:
        sentences = [example] 
        inputs = tokenizer(sentences, return_tensors="pt", padding='max_length', truncation=True, max_length=64)
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, collect_hiddenstates_apms=True, save_dir=args.save_dir)


    print("Collection success!")