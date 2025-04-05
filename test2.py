from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
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
from models.utils import VecDB, Emb, LatencyCollector, register_forward_latency_collector
import os
from tqdm import tqdm
import pickle
from sklearn.metrics import accuracy_score


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="./trained_bert_sst2")
    parser.add_argument("--database-path", type=str, default="/home/sdh/ACL/AttnCache/BertDB")
    parser.add_argument("--dataset", type=str, default="sst2")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.995)
    parser.add_argument("--base-latency", type=float, default=32.4267707824707 )

    return parser.parse_args()

def retrieve(data):
    inputs = tokenizer(data, padding="max_length", truncation=True, max_length=args.max_length, return_tensors="pt")
    inputs = inputs.to(device)

    x = model.bert.embeddings(
            input_ids=inputs["input_ids"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=None,
            inputs_embeds=None,
        )

    feature_vector = feature_projector.embed(x.cpu().detach().numpy())
    sims, idx_list = vecDB.search(feature_vector)

    reuse_tensor_index = np.flatnonzero(1 - sims >= args.threshold)
    hitted_records = idx_list[reuse_tensor_index]
    compute_tensor_index = np.flatnonzero(1 - sims < args.threshold)

    return reuse_tensor_index, hitted_records, compute_tensor_index, inputs



def evaluate(reuse_tensor_index, hitted_records, compute_tensor_index, inputs):
    
    with torch.no_grad():
        total_tensor_index = np.concatenate((reuse_tensor_index, compute_tensor_index), axis=0)
        attention_cache = torch.empty((config.num_hidden_layers, len(total_tensor_index), config.num_attention_heads, args.max_length, args.max_length))                             
        if len(reuse_tensor_index) != 0:
            print(f"=========== hit {len(reuse_tensor_index)} APMs")
            for layer_idx in range(config.num_hidden_layers):
                for idx, record in zip(reuse_tensor_index, hitted_records):
                    with open(f"{args.database_path}/APMsDB/{record[0]}.pickle", "rb") as file:
                        attn_weights = pickle.load(file)
                        attention_cache[layer_idx][idx] = torch.from_numpy(attn_weights)

                hitted_records += 1  # index of next layer hitted apms
        else:
            print(f"=========== no hit APMs")

        attention_cache = attention_cache.to(device)
        latency_collector = LatencyCollector()
        register_forward_latency_collector(latency_collector, model.bert.encoder.layer[-1].attention.self)

        for _ in range(100):
            outputs = model(**inputs, attention_cache=attention_cache, compute_tensor_index=compute_tensor_index)

    return outputs, latency_collector


if __name__ == "__main__":

    # =================== model ===================

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  

    # =================== data ===================
 
    dataset = load_dataset("glue", args.dataset)
    validation_data = dataset["validation"] # sst2: 872
    test_data = dataset["test"] # sst2: 1821
    train_data = dataset["train"] # sst2: 67349


    # =================== feature_projector & vecDB ===================

    feature_projector_save_path = f'{args.database_path}/Embedding_models/mlp_model_attn_cache-epoch3.pth'
    feature_projector = Emb(f"{feature_projector_save_path}")
    vecDB_save_path =  f"{args.database_path}/VectorDB/attn_cache_epoch-3_vectors.faiss"
    vecDB = VecDB().load(f"{vecDB_save_path}") 
    


# # =================== evaluation single validation_data ===================

#     for example in validation_data:
#         sentence = example["sentence"]
        
#         reuse_tensor_index, hitted_records, compute_tensor_index, inputs = retrieve(sentence)
        
#         if hitted_records.size:
#             s1 = train_data[int(hitted_records[0][0] // 12)]
#             s2 = example
#             t1 = tokenizer.tokenize(s1["sentence"])
#             t2 = tokenizer.tokenize(s2["sentence"])
#             print("reuse idx in train_data: ", hitted_records[0][0] // 12)
#             print("sentence 1 in train_data: ", s1)
#             print("token 1: ", t1, "lenth 1: ", len(t1))
#             print("sentence 2 in valid_data: ", s2)
#             print("token 2: ", t2, "lenth 2: ", len(t2))

#             outputs, latency_collector = evaluate(reuse_tensor_index, hitted_records, compute_tensor_index, inputs)

#             attn_cache_latency = latency_collector.latency_list
#             attn_cache_average_time = np.mean(attn_cache_latency) * 1000
#             print(f"self-attn with_attn_cache average time: {attn_cache_average_time} ms")


#             # =================== without AttnCache ===================
            
#             inputs = tokenizer(sentence, padding="max_length", truncation=True, max_length=args.max_length, return_tensors="pt")
#             inputs = inputs.to(device)
#             latency_collector = LatencyCollector()
#             register_forward_latency_collector(latency_collector, model.bert.encoder.layer[-1].attention.self)
#             for _ in range(100):
#                 outputs = model(**inputs, attention_cache=None, compute_tensor_index=None)
#             latency = latency_collector.latency_list
#             average_time = np.mean(latency) * 1000
#             print(f"self-attn without_attn_cache average time: {average_time} ms")

#             print("average speedup: ", average_time / attn_cache_average_time)
#             print('-' * 90)


#  # =================== evaluation all validation_data ===================
    
#     print(f"=========== evaluate all valid data")

#     val_texts = [example["sentence"] for example in dataset["validation"]]
#     val_labels = [example["label"] for example in dataset["validation"]]


#     reuse_tensor_index, hitted_records, compute_tensor_index, inputs = retrieve(val_texts)
#     outputs, latency_collector = evaluate(reuse_tensor_index, hitted_records, compute_tensor_index, inputs)
   
#     logits = outputs.logits
#     val_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
#     accuracy = accuracy_score(val_labels, val_predictions)
#     print(f"Final Validation Accuracy: {accuracy:.4f}")

#     latency = latency_collector.latency_list
#     average_time = np.mean(latency) * 1000
#     print(f"self-attn average time: {average_time} ms")
#     print("average speedup: ", args.base_latency / average_time)



    # =================== evaluation single validation_data ===================

    for example in validation_data:
        sentence = example["sentence"]

        inputs = tokenizer(sentence, return_tensors="pt", padding='max_length', truncation=True, max_length=args.max_length)
        inputs = inputs.to(device)
        with torch.no_grad():

            x = model.bert.embeddings(
                input_ids=inputs["input_ids"],
                token_type_ids=inputs["token_type_ids"],
                position_ids=None,
                inputs_embeds=None,
            )

            feature_vector = feature_projector.embed(x.cpu().detach().numpy())
            sims, idx_list = vecDB.search(feature_vector)

            reuse_tensor_index = np.flatnonzero(1 - sims >= args.threshold)
            hitted_records = idx_list[reuse_tensor_index]
            compute_tensor_index = np.flatnonzero(1 - sims < args.threshold)
            attention_cache = torch.empty((config.num_hidden_layers, len(x), config.num_attention_heads, args.max_length, args.max_length))                             

            if hitted_records:
                s1 = train_data[int(hitted_records[0][0] // 12)]
                s2 = example
                t1 = tokenizer.tokenize(s1["sentence"])
                t2 = tokenizer.tokenize(s2["sentence"])
                print("reuse idx in train_data: ", hitted_records[0][0] // 12)
                print("sentence 1 in train_data: ", s1)
                print("token 1: ", t1, "lenth 1: ", len(t1))
                print("sentence 2 in valid_data: ", s2)
                print("token 2: ", t2, "lenth 2: ", len(t2))

                if len(reuse_tensor_index) != 0:
                    print(f"=========== hit {len(reuse_tensor_index)} APMs")
                    for layer_idx in range(config.num_hidden_layers):
                        for idx, record in zip(reuse_tensor_index, hitted_records):
                            with open(f"{args.database_path}/APMsDB/{record[0]}.pickle", "rb") as file:
                                attn_weights = pickle.load(file)
                                attention_cache[layer_idx][idx] = torch.from_numpy(attn_weights)

                        hitted_records += 1  # index of next layer hitted apms
                else:
                    print(f"=========== no hit APMs")

                attention_cache = attention_cache.to(device)

                latency_collector = LatencyCollector()
                register_forward_latency_collector(latency_collector, model.bert.encoder.layer[-1].attention.self)
                for _ in range(100):
                    outputs = model(**inputs, attention_cache=attention_cache, compute_tensor_index=compute_tensor_index)
                attn_cache_latency = latency_collector.latency_list
                attn_cache_average_time = np.mean(attn_cache_latency) * 1000
                print(f"self-attn attn_cache_latency average time: {attn_cache_average_time} ms")


                # =================== without AttnCache ===================

                latency_collector = LatencyCollector()
                register_forward_latency_collector(latency_collector, model.bert.encoder.layer[-1].attention.self)
                for _ in range(100):
                    outputs = model(**inputs, attention_cache=None, compute_tensor_index=None)
                latency = latency_collector.latency_list
                average_time = np.mean(latency) * 1000
                print(f"self-attn latency average time: {average_time} ms")

                print("average speedup: ", average_time / attn_cache_average_time)
                print('-' * 80)



    # =================== evaluation all validation_data ===================

    val_texts = [example["sentence"] for example in dataset["validation"]]
    val_labels = [example["label"] for example in dataset["validation"]]

    inputs = tokenizer(val_texts, padding="max_length", truncation=True, max_length=args.max_length, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad():

        x = model.bert.embeddings(
            input_ids=inputs["input_ids"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=None,
            inputs_embeds=None,
        )

        feature_vector = feature_projector.embed(x.cpu().detach().numpy())
        sims, idx_list = vecDB.search(feature_vector)

        reuse_tensor_index = np.flatnonzero(1 - sims >= args.threshold)
        hitted_records = idx_list[reuse_tensor_index]
        compute_tensor_index = np.flatnonzero(1 - sims < args.threshold)

        attention_cache = torch.empty((config.num_hidden_layers, len(x), config.num_attention_heads, args.max_length, args.max_length))                             


        if len(reuse_tensor_index) != 0:
            print(f"=========== hit {len(reuse_tensor_index)} APMs")
            for layer_idx in range(config.num_hidden_layers):
                for idx, record in zip(reuse_tensor_index, hitted_records):
                    with open(f"{args.database_path}/APMsDB/{record[0]}.pickle", "rb") as file:
                        attn_weights = pickle.load(file)
                        attention_cache[layer_idx][idx] = torch.from_numpy(attn_weights)

                hitted_records += 1  # index of next layer hitted apms
        else:
            print(f"=========== no hit APMs")

        attention_cache = attention_cache.to(device)
        latency_collector = LatencyCollector()
        register_forward_latency_collector(latency_collector, model.bert.encoder.layer[-1].attention.self)
        for _ in range(100):
            outputs = model(**inputs, attention_cache=attention_cache, compute_tensor_index=compute_tensor_index)

   
    logits = outputs.logits
    val_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    accuracy = accuracy_score(val_labels, val_predictions)
    print(f"Final Validation Accuracy: {accuracy:.4f}")

    latency = latency_collector.latency_list
    average_time = np.mean(latency) * 1000
    print(f"self-attn average time: {average_time} ms")
    print("average speedup: ", args.base_latency / average_time)

   




