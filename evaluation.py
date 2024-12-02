import re
import sys
import io, os
import torch
import numpy as np
import logging
import tqdm
import fcntl
import time
import argparse
from prettytable import PrettyTable
import transformers
from transformers import LlamaTokenizer
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, QuantoConfig
from models.modeling_llama import LlamaForCausalLM
from models.configuration_llama import LlamaConfig

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def lock_and_write_file(file_path, content):
    with open(file_path, 'a') as file:
        while True:
            try:
                # Acquire an exclusive lock (non-blocking)
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)

                # Perform your write operations here
                file.write(content + '\n')
                file.flush()

            except IOError as e:
                print("File is locked by another process. Can't write.")
                time.sleep(1)
            finally:
                # Release the lock
                fcntl.flock(file, fcntl.LOCK_UN)
                break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str, default='')
    parser.add_argument("--model_name_or_path", type=str,
                        help="Transformers' model name or path", 
                        default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--mode", type=str,
                        choices=['dev', 'test', 'fasttest'],
                        default='test',
                        help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str,
                        choices=['sts', 'transfer', 'full', 'na'],
                        default='sts',
                        help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument('--tensor_parallel', action='store_true')
    parser.add_argument('--prompt_method', type=str, default='prompteol', help="What prompt method to use (prompteol/metaeol).")

    #################################  attn memo
    parser.add_argument('--is_attn_memo', action='store_true')
    parser.add_argument('--is_LazyFormer', action='store_true')
    parser.add_argument('--is_SAN', action='store_true') 
    parser.add_argument('--is_block_drop', action='store_true') 
    parser.add_argument('--is_attn_drop', action='store_true') 
    parser.add_argument('--is_attn_cache', action='store_true')
    parser.add_argument('--task_name', type=str, default="STS13")
    parser.add_argument('--collect_hiddenstates_apms', action='store_true')
    parser.add_argument('--save_dir', default="/home/sdh/MetaEOL/MetaEOL/database/", type=str)
    parser.add_argument('--threshold', type=float, default=0.9999, help='The threshold to decide whether to use attn replacement.')
    parser.add_argument('--training_epoch', type=int, default=2,  help='The epoch of training embedding model and generating vector DB')
    parser.add_argument('--replace_layer', type=int, default=4,  help='The layer that is not replaced by attn memo from 0 ~ replace_layer')
    parser.add_argument('--batch_size', type=int, default=128,  help='batch_size')
    parser.add_argument('--max_length', type=int, default=128,  help='max_length')
    parser.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),  help='device')

    args = parser.parse_args()
    args.save_dir += "/" + args.task_name
    device = args.device
    
    # token = "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"  #meta-llama/Llama-3.2-3B-Instruct
    token = "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" #meta-llama/Llama-3.1-8B
    if args.tensor_parallel:
        import tensor_parallel as tp
        n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                     low_cpu_mem_usage = True, torch_dtype=torch.float16)
        model = tp.tensor_parallel(model, [i for i in range(n_gpus)])
    else:
        # configuration
        config = LlamaConfig.from_pretrained(args.model_name_or_path, token=token)
        config.is_attn_memo=args.is_attn_memo
        config.collect_hiddenstates_apms=args.collect_hiddenstates_apms
        config.is_LazyFormer=args.is_LazyFormer
        config.is_SAN=args.is_SAN
        config.is_block_drop=args.is_block_drop
        config.is_attn_drop=args.is_attn_drop
        config.is_attn_cache=args.is_attn_cache
        config.save_dir=args.save_dir
        config.threshold=args.threshold
        config.training_epoch=args.training_epoch
        config.replace_layer=args.replace_layer
        config.batch_size=args.batch_size
        config.max_length=args.max_length

        nf4_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, token=token, config=config,
                                                #  quantization_config=nf4_config,
                                                    #  quantization_config=QuantoConfig(weights="float8"),
                                                     ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=token)
    tokenizer.pad_token_id = 0  # Set the padding token. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference


    # Set up the tasks
    if args.task_set == 'sts':
        # args.tasks = ['STS12', 'STS13', 'STS14', 'STS15',  'STSBenchmark', 'SICKRelatedness', 'STS16']
        args.tasks = [f'{args.task_name}']
        if args.mode == 'dev':
            args.tasks = ['STSBenchmark-dev']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'batch_size': 32}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 32,
                                         'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size':config.batch_size}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                         'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    if args.prompt_method == "metaeol":
        task_prompts = ["In this task, you're presented with a text excerpt. Your task is to categorize the excerpt into a broad category such as 'Education', 'Technology', 'Health', 'Business', 'Environment', 'Politics', or 'Culture'. These categories help in organizing content for better accessibility and targeting. For this task, this sentence : \"*sent 0*\" should be classified under one general category in one word:\"",
                        "In this task, you're given a statement and you need to determine whether it's presenting an 'Opinion' or a 'Fact'. This distinction is vital for information verification, educational purposes, and content analysis. For this task, this sentence : \"*sent 0*\" discriminates between opinion and fact in one word:\"",
                        "In this task, you're given a review from an online platform. Your task is to generate a rating for the product based on the review on a scale of 1-5, where 1 means 'extremely negative' and 5 means 'extremely positive'. For this task, this sentence : \"*sent 0*\" reflects the sentiment in one word:\"",
                        "In this task, you're reading a personal diary entry. Your task is to identify the predominant emotion expressed, such as joy, sadness, anger, fear, or love. For this task, this sentence : \"*sent 0*\" conveys the emotion in one word:\"",
                        "In this task, you're presented with two sentences. Your task is to assess whether the sentences convey the same meaning. Use 'identical', 'similar', 'different', or 'unrelated' to describe the relationship. To enhance the performance of this task, this sentence : \"*sent 0*\" means in one word:\"",
                        "In this task, you're given a sentence and a phrase. Your task is to determine if the phrase can be a contextual synonym within the given sentence. Options include 'yes', 'no', or 'partially'. To enhance the performance of this task, this sentence : \"*sent 0*\" means in one word:\"",
                        "In this task, you're examining a news article. Your task is to extract the most critical fact from the article. For this task, this sentence : \"*sent 0*\" encapsulates the key fact in one word:\"",
                        "In this task, you're reviewing a scientific abstract. Your task is to identify the main entities (e.g., proteins, diseases) and their relations (e.g., causes, treats). For this task, this sentence : \"*sent 0*\" highlights the primary entity or relation in one word:\"",
                        ]
    elif args.prompt_method == "prompteol":
        task_prompts = ["This sentence : \"*sent 0*\" means in one word:\""]

    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]
        # input_sentences = [' '.join(s) for s in batch]
        if max_length == 500:
            sentences = [tokenizer.decode(tokenizer.encode(s, add_special_tokens=False)[:max_length]) for s in sentences]
            max_length = 512

        new_sentences = []
        for i, s in enumerate(sentences):
            if len(s) > 0 and s[-1] not in '.?"\'': s += '.'
            s = s.replace('"', '\'')
            if len(s) > 0 and '?' == s[-1]: s = s[:-1] + '.'
            for prompt in task_prompts:
                new_sentences.append(prompt.replace('*sent 0*', s).strip())
        sentences = new_sentences
        # print("sentence: ", sentences[0])
        max_length = 128
        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            # padding=True,
            padding="max_length",
            max_length=max_length,
            truncation=max_length is not None
        )
        # print(batch['input_ids'][0])
        # tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][0])
        # print("Tokens:", tokens)
        # print(batch['attention_mask'][0])

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device) if batch[k] is not None else None

        # Get raw embeddings
        with torch.no_grad():
            outputs, last_records, last_reuse_tensor_index =  model(output_hidden_states=True, return_dict=True, **batch)
            attentions = outputs.attentions
            hidden_states = outputs.hidden_states
            outputs = hidden_states[-1][:, -1, :]
            outputs = outputs.view(-1, len(task_prompts), outputs.size()[1]).mean(dim=1) # Average the embeddings from different tasks 

            if outputs.dtype == torch.bfloat16:
                # bfloat16 not support for .numpy()
                outputs = outputs.float()

            return outputs.cpu(), attentions, last_records, last_reuse_tensor_index

    results = {}
    
    print("===========================")
    print("Args: ", args)
    print("===========================")
    
    for task in args.tasks:
        print(f"================= start eval task {task} ===============================================================")
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task, config.collect_hiddenstates_apms)
        results[task] = result

    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark-dev']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))    
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)


    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)
        #
        # write results and template to file
        if args.task_set != 'transfer':
            with open('./sts-org-results', 'a') as f:
                model_name = args.model_name_or_path.split('/')[-1]
                f.write(model_name + ' ' + ' '.join([str(s) for s in scores]) + '\n')

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['acc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

if __name__ == "__main__":
    main()
