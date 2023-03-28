#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###############################################################################

from datasets import load_dataset
from transformers import AutoTokenizer
import deepspeed
import random
import numpy as np

import bloom
from bloom import *


# WIP: do not remove
# ## pip install evaluate mauve-text
# from evaluate import load
# mauve = load('mauve')


c4 = load_dataset("json", data_files="data/c4_mini.json")


class Logger(object):
    def __init__(self, output_file_name):
        self.terminal = sys.stdout
        self.log = open(f"{output_file_name.replace('/','_')}.log", "w")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass    

parser = setup_parser()
parser.add_argument("--num_samples", default=11, type=int, help="num of data samples to test for completion accuracy")
parser.add_argument("--num_total_samples", default=3500, type=int, help="num of total data samples to run on. WARNING: changing this will cause perplexity change")
args = parser.parse_args()
if args.seed is None:
    args.seed = 0

weights, model = initialize_model(args)
tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True, padding_side='left')
init_end = time.perf_counter()

print("Starting inference...")
bs = args.batch_size

kwargs = {
    'max_length': args.max_length,
    'num_beams': args.beams,
    'do_sample': args.sample,
}


res = []
cnt = 0
total_perp = 0.0
total_loss = 0.0
total_time = 0.

print(f"Results for C4 ")

log_file_name = f"{args.model}_DS_ver_{deepspeed.__version__}_dtype_{args.dtype}_vanilla_{args.vanilla}_num_dev_{args.world_size}"
sys.stdout = Logger(log_file_name)

expected_list = []
predicted_list = []

for idx, i in enumerate(c4['train']):
    if cnt > args.num_total_samples and args.num_total_samples > 0:
        break


    str=i['text']

    with torch.no_grad():

        if cnt < (args.num_samples - 1):

            # print for reference
            print("#############################")
            ts = time.perf_counter()
            tok = tokenizer.batch_encode_plus([str], max_length=args.max_length, return_tensors="pt", padding='max_length', truncation=True,).to(args.device)
            res = model.generate(**tok, min_length=2*args.max_length-1, max_new_tokens= args.max_length, num_beams=args.beams, temperature=0.05)
            ans_hat = tokenizer.batch_decode(res, skip_special_tokens=True)[0]
            te = time.perf_counter()
            print("_____")
            input_ = tokenizer.batch_decode(tok.input_ids, skip_special_tokens=True)[0]
            print(f"\033[96minput\033[0m: {input_}")
            predicted = ans_hat.split(input_)[1]
            print(f"\033[96mpredicted\033[0m: {predicted}")
            expected = tokenizer.batch_decode(tokenizer.batch_encode_plus([str], max_length=2 * args.max_length, return_tensors="pt").input_ids[:, args.max_length:2*args.max_length])[0]
            print(f"\033[96mexpected\033[0m: {expected}")

            # WIP needed for comparing sentences, keeping here for now...
            # expected_list.append(" ".join(str.split(" ")[0:len(ans_hat.split(" "))]).split(input_)[1])  # get same length as answer
            # predicted_list.append(ans_hat.split(input_)[1])

            query_time = te - ts
            if idx >= args.iters_to_ignore:
                total_time += query_time
        else:
            tok = tokenizer.batch_encode_plus([str], max_length=args.max_length, return_tensors="pt", padding='max_length', truncation=True).to(args.device)
        
        # perplexity
        input_ids = tok['input_ids']
        target_ids = input_ids.clone()
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over input tokens.
        neg_log_likelihood = outputs.loss.float()
        # acc = outputs.acc
        loss = neg_log_likelihood

        total_loss += loss
        # total_acc += acc
        cnt += 1


# mauve_results = mauve.compute(predictions=predicted_list, references=expected_list) 

print("===========Summary====================")
print(f"average perplexity score = {torch.exp(total_loss / cnt)}")
# print(f"MAUVE score = {mauve_results.mauve}")

total_time /= (args.num_samples - args.iters_to_ignore)
print(f"Average query time for tok()+generate()+decode() is {(total_time):.3f}s for {args.max_length} generated tokens each\n")
