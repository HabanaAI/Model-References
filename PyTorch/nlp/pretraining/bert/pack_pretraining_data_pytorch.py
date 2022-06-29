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
from torch.utils.data import Dataset
from scipy import optimize
from itertools import repeat, chain
from functools import lru_cache, reduce
from collections import defaultdict, OrderedDict
from concurrent.futures import ProcessPoolExecutor
import gc

@lru_cache(maxsize=None)
def packing_strategies(start, previous, target, depth):
    gap = target - start
    strategies = []
    # Complete the packing with exactly 1 number
    if depth == 1:
        if gap >= previous:
            strategies.append([gap])
    # Complete the sample in "depth" steps, recursively
    else:
        for new in range(previous, gap + 1):
            new_gap = target - start - new
            if new_gap == 0:
                strategies.append([new])
            else:
                options = packing_strategies(start + new, new, target, depth - 1)
                for option in options:
                    if len(option) > 0:
                        strategies.append([new] + option)
    return strategies

def get_packing_recipe(sequence_lengths, max_sequence_length, max_sequences_per_pack=3):
    # Histogram of sequence lengths
    histogram, bins = np.histogram(sequence_lengths, bins=np.arange(1, max_sequence_length + 2))
    print("Begin packing pass".center(80, "_"))
    print(f"Unpacked mean sequence length: {sequence_lengths.mean():3.2f}")

    # Make sure all strategies are recipes to pack to the correct sequence length
    strategy_set = packing_strategies(0, 1, max_sequence_length, max_sequences_per_pack)
    for strategy in strategy_set:
        assert(sum(strategy) == max_sequence_length)
    num_strategies = len(strategy_set)
    print(f"Found {num_strategies} unique packing strategies.")
    # Solve the packing equation A@mixture = histogram
    A = np.zeros((max_sequence_length, num_strategies), dtype=np.int32)
    for i in range(num_strategies):
        strategy = strategy_set[i]
        for seq_len in strategy:
            A[seq_len - 1, i] += 1
    # short sequences are inexpensive to add, so should have low residual weights
    # to exactly minimize padding use w0 = np.arange(1, max_sequence_length + 1)
    # in practice the difference is negligible, but this converges faster
    padding_cutoff = 8
    w0 = np.ones([max_sequence_length])
    # w0 = np.linspace(1, max_sequence_length+1, max_sequence_length)/max_sequence_length  # padding minimization weight
    w0[:padding_cutoff] = padding_cutoff / (2 * max_sequence_length)
    w0 = np.sqrt(w0)
    # Starting values for the padding and the mixture
    padding = np.zeros([max_sequence_length], dtype=np.int32)
    mixture = np.zeros([num_strategies], dtype=np.int32)
    b = histogram + padding
    # Pack sequences as best as possible, then increase padding accordingly and repeat
    for i in range(0, 20):
        print(f"\nIteration: {i}: sequences still to pack: ", b.sum())
        start = time.time()
        partial_mixture, rnorm = optimize.nnls(np.expand_dims(w0, -1) * A, w0 * b)
        print(f"Solving nnls took {time.time() - start:3.2f} seconds.")
        print(f"Residual norm:  {rnorm:3.5e}")
        # Update mixture (round the floating point solution to integers)
        partial_mixture = np.where(partial_mixture < 2, np.rint(partial_mixture), np.floor(partial_mixture))
        # If partial mixture is empty (due to rounding) we follow the gradient
        # this usually happens when the number of examples is small i.e. ~100
        if partial_mixture.max() == 0:
            grad = A.T @ (b * np.arange(1, max_sequence_length + 1))
            k = int(b.sum() // 2) + 1
            topk = np.argsort(-grad)[:k]
            partial_mixture[topk] += 1
        # Update mixture
        mixture = mixture + partial_mixture
        # Compute the residuals
        residual = b - A @ partial_mixture
        print(f"Max residual:   {abs(residual).max()}")
        print(f"Residual on first 8 categories: {np.around(residual[:8], 4)}")
        print(f"Residual on last 8 categories:  {np.around(residual[-8:], 4)}")
        # Add padding based on deficit (negative residual)
        partial_padding = np.where(residual < 0, -residual, 0)
        print(f"Added {(partial_padding*np.arange(1,max_sequence_length+1)).sum():3.2e} tokens of padding.")
        padding = padding + partial_padding
        # Update the rhs vector (remaining surplus sequences)
        b = histogram + padding - A @ mixture
        assert np.all(b >= 0), b
        # Done iterating
        if b.sum() < 100:
            break
    # Make sure there is no remainder
    unpacked_seqlen = np.arange(1, max_sequence_length + 1)[b > 0]
    # Update the mixture to also covered the unpacked sequences
    for l in unpacked_seqlen:
        # Get the depth 1 strategy
        strategy = sorted([l, max_sequence_length - l])
        strategy_index = strategy_set.index(strategy)
        mixture[strategy_index] += b[l-1]
    b = histogram - A @ mixture
    padding = np.where(b < 0, -b, 0)
    b = histogram + padding - A @ mixture
    assert b.sum() == 0
    # Analyze result
    print("Done solving for packing order".center(80, "_"))
    num_padding_tokens = (np.arange(1, max_sequence_length + 1) * padding).sum()
    num_padding_tokens_original = (max_sequence_length - sequence_lengths).sum()
    print(f"Number of sequences dropped:  {b.sum()}")
    print(f"Number of strategies utilized: {np.count_nonzero(mixture)}")
    new_number_of_samples = int(mixture.sum())
    compression = 1 - new_number_of_samples / len(sequence_lengths)
    print(f"New number of samples: {new_number_of_samples:3.2f}, original {len(sequence_lengths)}. A compression ratio of {compression:3.3f}")
    print(f"The expected speed-up from packing: {1/(1-compression):3.3f}")
    upper_bound = 1.0 / (1 - ((1 - sequence_lengths / max_sequence_length).mean()))
    print(f"Theoretical upper bound on speed-up: {upper_bound:3.3f}")
    avg_sequences_per_sample = ((A.sum(0) * mixture).sum() - padding.sum()) / new_number_of_samples
    print(f"Average sequences/sample {avg_sequences_per_sample:3.5f}")
    print(f"Added {num_padding_tokens:3.2e} padding tokens. Original dataset used {num_padding_tokens_original:3.2e} padding tokens")
    efficiency = (new_number_of_samples*max_sequence_length - num_padding_tokens)/(new_number_of_samples*max_sequence_length)
    print(f"Packing efficiency (fraction of real tokens): {efficiency:3.4f}")
    print(f"Top 8 strategies")
    topK = np.argsort(-mixture)[:8]
    for i in topK:
        print(f"Strategy {strategy_set[i]} which is used {int(mixture[i])} times")
    print("".center(80, "_"))
    # Figure out the slicing that each strategy should use
    slicing = np.zeros_like(A)
    slicing[:, 1:] = np.cumsum(A * mixture, axis=1)[:, :-1]
    slicing = slicing.T
    mixture = mixture.astype(np.int64)
    return strategy_set, mixture, padding, slicing


def slice_examples_mult_stratagies_shuffle(examples_by_length, slicing, strategy_set, repeat_counts):
    # Divide the work, firstly between the strategies and then into chunks of 50k
    strategies_slices = defaultdict(list)
    for strategy, slice_offsets, repeat_count in zip(strategy_set, slicing, repeat_counts):
        if repeat_count == 0:
            continue
        # Slice out the sequences allocated to this strategy in increments of 50k
        subcounts = (min(1, repeat_count - 1 * (i - 1)) for i in range(1, repeat_count + 1))
        for part_id, part_count in enumerate(subcounts):
            for k, seq_len in enumerate(strategy):
                slice_start = int(slice_offsets[seq_len - 1])
                slice_end = slice_start + int(part_count)
                slice_offsets[seq_len - 1] = slice_end
                strategies_slices[str(strategy)+'_'+str(seq_len)].append([slice_start,slice_end])

    slices = []
    examples_batch = []
    slice_offsets=slicing[0]
    total_num_samples=[len(examples_by_length[sl]) for sl in examples_by_length.keys()]
    suffle_samples_ind=np.random.permutation(sum(repeat_counts))
    strategies = [[st]*rp for st,rp in zip(strategy_set,repeat_counts)]
    strategies = list(chain.from_iterable(strategies))
    num_sample_per_slice=4480
    counter=0; count_samples=0

    for ind in suffle_samples_ind:
        strategy=strategies[ind]
        if len(strategy) == 0:
            continue
        # Slice out the sequences allocated to this strategy in increments of 50k
        counter+=1
        examples=[]
        for k, seq_len in enumerate(strategy):
            count_samples+=1
            [slice_start,slice_end]=strategies_slices[str(strategy)+'_'+str(seq_len)].pop()
            examples.append(examples_by_length[seq_len][slice_start:slice_end][0])

        examples_batch.append(examples)
        if counter%num_sample_per_slice==0:
            slices.append(examples_batch)
            examples_batch=[]
    assert sum(total_num_samples)==count_samples, "Possibly not using all samples"
    examples_by_length = None
    return slices


def parallel_pack_according_to_strategy(args, part_idx, examples):
    # Pack the sequences according to the strategy and write them to disk
    filename = os.path.join(args.output_dir, "mixed_strategies_part_%d.hdf5"%part_idx)
    features = defaultdict(list)
    for inst_index, multi_sequence in enumerate(examples):
        features_packed = create_multi_sequence_example(multi_sequence, args.max_predictions_per_sequence,
                                                       args.max_sequence_length, args.max_sequences_per_pack)
        #if features_packed['next_sentence_weights'].sum()>1:
        #    print(features_packed['next_sentence_weights'],filename)
        features["input_ids"].append(features_packed["input_ids"])
        features["input_mask"].append(features_packed["input_mask"])
        features["segment_ids"].append(features_packed["segment_ids"])
        features["positions"].append(features_packed["positions"])
        features["masked_lm_positions"].append(features_packed["masked_lm_positions"])
        features["masked_lm_ids"].append(features_packed["masked_lm_ids"])
        features["next_sentence_positions"].append(features_packed["next_sentence_positions"])
        features["next_sentence_labels"].append(features_packed["next_sentence_labels"])
        features["next_sentence_weights"].append(features_packed["next_sentence_weights"])
    f= h5py.File(filename, 'w')
    f.create_dataset("input_ids", data=np.array(features["input_ids"]), dtype='i4', compression='gzip')
    f.create_dataset("input_mask", data=np.array(features["input_mask"]), dtype='i4', compression='gzip')
    f.create_dataset("segment_ids", data=np.array(features["segment_ids"]), dtype='i1', compression='gzip')
    f.create_dataset("positions", data=np.array(features["positions"]), dtype='i4', compression='gzip')
    f.create_dataset("masked_lm_positions", data=np.array(features["masked_lm_positions"]), dtype='i4', compression='gzip')
    f.create_dataset("masked_lm_ids", data=np.array(features["masked_lm_ids"]), dtype='i4', compression='gzip')
    f.create_dataset("next_sentence_positions", data=np.array(features["next_sentence_positions"]), dtype='i4', compression='gzip')
    f.create_dataset("next_sentence_labels", data=np.array(features["next_sentence_labels"]), dtype='i1', compression='gzip')
    f.create_dataset("next_sentence_weights", data=np.array(features["next_sentence_weights"]), dtype='i4', compression='gzip')
    f.flush()
    f.close()


def create_multi_sequence_example(multi_sequence, max_predictions_per_sequence, max_sequence_length, max_sequences_per_pack):
    # SEQ
    packed_input_ids = np.zeros(max_sequence_length, dtype=np.int32)
    packed_input_mask = np.zeros(max_sequence_length, dtype=np.int32)
    packed_segment_ids = np.zeros(max_sequence_length, dtype=np.int32)
    packed_positions = np.zeros(max_sequence_length, dtype=np.int32)
    # MLM
    # we are packing up to max_sequences_per_pack, each with a certain percentage of masked tokens
    # in case that percentege is rounded up for all sequences in the pack, need to add an extra token for
    # each sequence in the pack
    packed_masked_lm_positions = np.zeros(max_predictions_per_sequence + max_sequences_per_pack, dtype=np.int32)
    packed_masked_lm_ids = np.zeros(max_predictions_per_sequence + max_sequences_per_pack, dtype=np.int32)
    #packed_masked_lm_weights = np.zeros(max_predictions_per_sequence + max_sequences_per_pack, dtype=np.int32)
    # NSP
    packed_next_sentence_positions = np.zeros(max_sequences_per_pack, dtype=np.int32)
    packed_next_sentence_labels = np.zeros(max_sequences_per_pack, dtype=np.int32)
    packed_next_sentence_weights = np.zeros(max_sequences_per_pack, dtype=np.int32)

    offset = 0
    mlm_offset = 0
    sequence_index = 1  # used in the input mask
    for sequence in multi_sequence:
        # Padding sequences are donoted with None
        if sequence is not None:
            input_ids = np.array(sequence['input_ids'])
            input_mask = np.array(sequence['input_mask'])
            segment_ids = np.array(sequence['segment_ids'])
            masked_lm_positions = np.array(sequence['masked_lm_positions'])
            masked_lm_ids = np.array(sequence['masked_lm_ids'])
            #masked_lm_weights = np.array(sequence['masked_lm_weights'])
            next_sentence_labels = np.array(sequence['next_sentence_labels'])

            #input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, next_sentence_labels = sequence
            seq_len = input_mask.sum()
            # SEQ
            packed_input_ids[offset:offset + seq_len] = input_ids[:seq_len]
            packed_input_mask[offset:offset + seq_len] = sequence_index
            packed_segment_ids[offset:offset + seq_len] = segment_ids[:seq_len]
            packed_positions[offset:offset + seq_len] = np.arange(0, seq_len)
            # MLM
            mlm_len= (masked_lm_ids!=0).sum()
            #mlm_len = int(masked_lm_weights.sum())
            assert mlm_offset + mlm_len < max_predictions_per_sequence + max_sequences_per_pack, "Too many LM predictions per sequences"
            max_mlm = mlm_offset + mlm_len
            packed_masked_lm_positions[mlm_offset:max_mlm] = offset + masked_lm_positions[:mlm_len]
            packed_masked_lm_ids[mlm_offset:max_mlm] = masked_lm_ids[:mlm_len]
            #packed_masked_lm_weights[mlm_offset:max_mlm] = sequence_index
            # NSP
            packed_next_sentence_positions[sequence_index - 1] = offset
            packed_next_sentence_labels[sequence_index - 1] = next_sentence_labels
            packed_next_sentence_weights[sequence_index - 1] = 1
            # Update offsets
            sequence_index += 1
            offset += seq_len
            mlm_offset = max_mlm
            input_ids = None; input_mask = None; segment_ids = None; masked_lm_positions = None;
            masked_lm_ids = None; next_sentence_labels = None; seq_len = None
    # Pack into tfrecord format:

    features = OrderedDict()

    features["input_ids"] = packed_input_ids
    features["input_mask"] = packed_input_mask
    features["segment_ids"] = packed_segment_ids
    features["positions"] = packed_positions
    features["masked_lm_positions"] = packed_masked_lm_positions
    features["masked_lm_ids"] = packed_masked_lm_ids
    features["next_sentence_positions"] = packed_next_sentence_positions
    features["next_sentence_labels"] = packed_next_sentence_labels
    features["next_sentence_weights"] = packed_next_sentence_weights
    del packed_input_ids; del packed_input_mask; del packed_segment_ids;  del packed_positions; del packed_masked_lm_positions; del packed_masked_lm_ids;
    del packed_next_sentence_positions; del packed_next_sentence_labels; del packed_next_sentence_weights

    return features

class pretraining_dataset(Dataset):
    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'next_sentence_labels']
        self.keys_exist = list(f.keys())
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        self.len_dict={}
        for key in keys:
            self.len_dict[key] = np.asarray(f[key][:]).shape

        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):
        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids,next_sentence_labels] = [input[index] if indice < 5 else
                np.asarray(input[index]) for indice, input in enumerate(self.inputs)]

        return [input_ids,  input_mask, segment_ids,masked_lm_positions, masked_lm_ids,
                next_sentence_labels]

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
    parser.add_argument("--max_sequence_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_predictions_per_sequence",
                        default=80,
                        type=int,
                        help="The maximum total of masked tokens in input sequence")
    parser.add_argument("--max_sequences_per_pack",
                        default=3,
                        type=int,
                        help="The maximum number of sequences to pack in multi-sequence")
    parser.add_argument("--train_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the packed dataset will be written.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank",
                        type=int,
                        default=os.getenv('LOCAL_RANK', -1),
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--disable_progress_bar',
                        default=False,
                        action='store_true',
                        help='Disable tqdm progress bar')
    args = parser.parse_args()
    return args

def main():
    global timeout_sent
    args = parse_arguments()
    random.seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    torch.manual_seed(args.seed + args.local_rank)
    torch.cuda.manual_seed(args.seed + args.local_rank)
    worker_init = WorkerInitObj(args.seed + args.local_rank)

    device = torch.device("cpu")
    print("args.max_sequence_length={}, args.max_sequences_per_pack={},args.max_predictions_per_sequence={}".format(args.max_sequence_length, args.max_sequences_per_pack,args.max_predictions_per_sequence))

    files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if
                         os.path.isfile(os.path.join(args.input_dir, f)) and 'training' in f]
    print("files={}".format(files))
    sequence_lengths = []
    examples_by_length = defaultdict(list)
    print("Looping through dataset to collect sequence length information...")

    for f_id in range(len(files)):
        #single card
        data_file = files[f_id]
        print("-- loading data_file={}".format(data_file))
        train_data = pretraining_dataset(data_file, args.max_predictions_per_sequence)
        for step, batch in enumerate(train_data):
            input_ids,  input_mask, segment_ids,masked_lm_positions, masked_lm_ids, next_sentence_labels = batch
            features = OrderedDict()
            features["input_ids"] = input_ids
            features["input_mask"] = input_mask
            features["segment_ids"] = segment_ids
            features["masked_lm_positions"] = masked_lm_positions
            features["masked_lm_ids"] = masked_lm_ids
            #features["masked_lm_weights"] = masked_lm_weights
            features["next_sentence_labels"] = next_sentence_labels
            im_length = sum(input_mask)
            examples_by_length[im_length].append(features)
            sequence_lengths.append(im_length)
    sequence_lengths = np.array(sequence_lengths)
    # Pass the array of sequence lengths to the packing algorithm

    strategy_set, mixture, padding, slicing = get_packing_recipe(sequence_lengths, args.max_sequence_length, args.max_sequences_per_pack)

    # Add the calculated padding
    for i in range(1, args.max_sequence_length + 1):
        if i not in examples_by_length.keys():
            examples_by_length[i]=[]
        examples_by_length[i].extend([None] * int(padding[i - 1]))
    # Shuffle the data
    for key in examples_by_length:
        random.shuffle(examples_by_length[key])
    # Pack and store the data
    print(f"\nPacking and writing packed dataset to {args.output_dir}.")
    # Slice the data into chunks of max 50k packed examples
    example_slices = slice_examples_mult_stratagies_shuffle(examples_by_length, slicing, strategy_set, mixture)
    part_idx = [i for i in range(len(example_slices))]
    gc.collect()
    print('Done slice_examples !!!')
    del examples_by_length; del slicing; del strategy_set; del mixture
    gc.collect()
    start = time.time()
    print(f"Splitting work into {len(part_idx)} parts.")

    split_write_sessions_size = 1000
    for rr in range(1+len(example_slices)//split_write_sessions_size):
        print(rr,'out of',1+len(example_slices)//split_write_sessions_size)
        str_idx,stp_idx=rr*split_write_sessions_size,min((rr+1)*split_write_sessions_size,len(example_slices))
        example_slices_prt,part_idx_prt = example_slices[str_idx:stp_idx], part_idx[str_idx:stp_idx]
        with ProcessPoolExecutor(50) as executor:
            work = repeat(args), part_idx_prt, example_slices_prt
            for partial_result in executor.map(parallel_pack_according_to_strategy, *work):
                pass
        print('------')
        del work
    print(f"\nDone. Took: {time.time() - start:3.2f} seconds to pack and write dataset.")
    print('-------------',str_idx,stp_idx)
    print('Done Cleaning')
if __name__ == "__main__":
    main()
