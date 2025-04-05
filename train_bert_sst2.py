import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score
import argparse
from models.modeling_bert import BertForSequenceClassification

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

def preprocess_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=64)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save-path", type=str, default="./trained_bert_sst2")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--dataset", type=str, default="sst2")
    parser.add_argument("--num-train-epochs", type=int, default=3)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    args.model_name = "bert-base-uncased"
    dataset = load_dataset("glue", f"{args.dataset}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoded_dataset = dataset.map(preprocess_function, batched=True)

    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.to(device)


    training_args = TrainingArguments(
        output_dir="./results",       
        evaluation_strategy="epoch",  
        save_strategy="epoch",        
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=args.num_train_epochs,           
        weight_decay=0.01,
        logging_dir="./logs",         
        logging_steps=10,
        load_best_model_at_end=True, 
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )


    trainer.train()


    eval_results = trainer.evaluate()
    print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")


    model.save_pretrained(f"{args.save_path}")
    tokenizer.save_pretrained(f"{args.save_path}")
    print("save model success!")