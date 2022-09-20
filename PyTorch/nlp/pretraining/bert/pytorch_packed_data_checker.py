###############################################################################
# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.
###############################################################################
import os
import time
import argparse
import random
import h5py
from tqdm import tqdm, trange
import os
import numpy as np
import torch
from torch.utils.data import  Dataset
from functools import lru_cache, reduce
from itertools import repeat, chain
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

class pretraining_dataset(Dataset):

    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = ['input_ids', 'input_mask', 'segment_ids', 'positions',
                'masked_lm_positions', 'masked_lm_ids',
                'next_sentence_positions', 'next_sentence_labels', 'next_sentence_weights']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):

        [input_ids, input_mask, segment_ids, positions,
         masked_lm_positions, masked_lm_ids,
         next_sentence_positions, next_sentence_labels, next_sentence_weights] = [torch.from_numpy(input[index].astype(np.int64)) for input in self.inputs]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        next_sentence_labels = (next_sentence_weights == 1) * next_sentence_labels + (next_sentence_weights == 0) * -1
        return [input_ids, segment_ids, input_mask, positions, masked_lm_labels, next_sentence_positions, next_sentence_labels]

class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed
    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)
def parse_arguments():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain .hdf5 files  for the task.")
    parser.add_argument("--max_predictions_per_sequence",
                        default=80,
                        type=int,
                        help="The maximum total of masked tokens in input sequence")

    args = parser.parse_args()
    return args

def parallel_data_loader(max_predictions_per_sequence,data_file):
        train_data = pretraining_dataset(data_file, max_predictions_per_sequence)
        try:
            for step, batch in enumerate(train_data):
                input_ids, segment_ids, input_mask, positions, masked_lm_labels, next_sentence_positions, next_sentence_labels = batch
            print(data_file)
            return True
        except:
            print('Issue with file: %s'%data_file)
            return False
def main():
    global timeout_sent
    args = parse_arguments()

    files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if
                         os.path.isfile(os.path.join(args.input_dir, f))]
    num_files=len(files)
    print(" num files is: %d"%(num_files))
    print("Looping through dataset to collect sequence length information...")
    with ProcessPoolExecutor(50) as executor:
        work = repeat(args.max_predictions_per_sequence), files
        for pass_fail  in executor.map(parallel_data_loader, *work):
            pass
    print('Done')
if __name__ == "__main__":
    main()
