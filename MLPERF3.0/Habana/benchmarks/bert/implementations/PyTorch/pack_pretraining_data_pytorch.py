###############################################################################
# Copyright (c) 2021,2022, Habana Labs Ltd.  All rights reserved.
###############################################################################
import os
import time
import argparse
import random
import h5py
from tqdm import tqdm
import os
import numpy as np
from scipy import optimize
from itertools import chain
from functools import lru_cache
from collections import defaultdict
import gc
import json

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

def create_json_metadata(
    seqeunces_dropped,
    num_strategies_utilized,
    new_number_of_samples,
    original_number_of_samples,
    compression_ratio,
    expected_speedup,
    theoretical_speedup,
    avg_sequence_per_sample,
    padding_tokens_packed_dataset,
    padding_tokens_original_dataset,
    packing_efficiency,
    top_8_strategies):
    # convert to json serrializable format
    top_8_strategies = top_8_strategies.tolist()
    packing_efficiency = float(packing_efficiency)
    padding_tokens_original_dataset = int(padding_tokens_original_dataset)
    padding_tokens_packed_dataset = float(padding_tokens_packed_dataset)
    avg_sequence_per_sample = float(avg_sequence_per_sample)
    theoretical_speedup = float(theoretical_speedup)
    json_object = json.dumps(
            {'number_of_sequences_dropped': seqeunces_dropped,
            'number_of_strategies_utilized': num_strategies_utilized,
            'new_number_of_samples': new_number_of_samples,
            'original_number_of_samples': original_number_of_samples,
            'compression_ratio': compression_ratio,
            'expected_speed_up': expected_speedup,
            'theoretical_speed_up': theoretical_speedup,
            'avg_seq_per_sample': avg_sequence_per_sample,
            'padding_tokens_packed_dataset': padding_tokens_packed_dataset,
            'padding_tokens_original_dataset': padding_tokens_original_dataset,
            'padding_tokens_original_dataset': padding_tokens_original_dataset,
            'packing_efficiency':packing_efficiency,
            'top_8_strategies':top_8_strategies},
    sort_keys=True, indent=2)
    return json_object

def get_packing_recipe(output_dir, sequence_lengths, max_sequence_length, max_sequences_per_pack=3):
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
    number_of_sequences_dropped = b.sum()
    print(f"Number of sequences dropped:  {number_of_sequences_dropped}")
    number_of_strategies_utilized = np.count_nonzero(mixture)
    print(f"Number of strategies utilized: {number_of_strategies_utilized}")
    new_number_of_samples = int(mixture.sum())
    original_number_of_samples = len(sequence_lengths)
    compression = 1 - new_number_of_samples / original_number_of_samples
    print(f"New number of samples: {new_number_of_samples:3.2f}, original {original_number_of_samples}. A compression ratio of {compression:3.3f}")
    expected_speedup_from_packing = 1 / (1 - compression)
    print(f"The expected speed-up from packing: {expected_speedup_from_packing}")
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
    norm_path = os.path.normpath(output_dir)
    head_tail = os.path.split(norm_path)
    metadata_file_name = head_tail[1]
    metadata_file_name = metadata_file_name + '_metadata.json'
    metadata_file_path = os.path.join(head_tail[0],metadata_file_name)
    print(f"Saving metadata to file: {metadata_file_path}")
    with open(metadata_file_path,mode='w') as file_handle:
        json_content = create_json_metadata(seqeunces_dropped=int(number_of_sequences_dropped),
        num_strategies_utilized=number_of_strategies_utilized,
        new_number_of_samples=new_number_of_samples,
        original_number_of_samples=original_number_of_samples,
        compression_ratio=compression,
        expected_speedup=expected_speedup_from_packing,
        theoretical_speedup=upper_bound,
        avg_sequence_per_sample=avg_sequences_per_sample,
        padding_tokens_original_dataset=num_padding_tokens_original,
        padding_tokens_packed_dataset=num_padding_tokens,
        packing_efficiency=efficiency,
        top_8_strategies=topK)
        file_handle.write(json_content)
    return strategy_set, mixture, padding, slicing

def slice_examples_mult_stratagies_shuffle(examples_by_length, slicing, strategy_set, repeat_counts, max_sequences_per_pack):
    suffle_samples_index=np.random.permutation(sum(repeat_counts))
    strategies = list(chain.from_iterable([(st, sl)]*rp for st,rp,sl in zip(strategy_set,repeat_counts,slicing) if rp != 0))
    assert len(strategies) == sum(repeat_counts)

    j = 0
    # 2 below results from the fact that example is identified by two indices
    # index of the file and index withing the file
    examples = -1 * np.ones(shape=[len(strategies), max_sequences_per_pack, 2], dtype=np.int32)

    for k in tqdm(suffle_samples_index):
        (strategy, slice_offsets) = strategies[k]

        for (i, seq_len) in enumerate(strategy):
            slice_start = int(slice_offsets[seq_len - 1])
            slice_offsets[seq_len - 1] = slice_start + 1
            examples[j, i, :] = examples_by_length[seq_len - 1][slice_start]

        j += 1

    return examples

def pack_according_to_strategy(args, part_idx, examples, train_data):
    # Pack the sequences according to the strategy and write them to disk
    features = defaultdict(list)
    for multi_sequence in examples:
        create_multi_sequence_example(multi_sequence, args, features, train_data)

    f= h5py.File(os.path.join(args.output_dir, "mixed_strategies_part_%d.hdf5"%part_idx), 'w')
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


def create_multi_sequence_example(multi_sequence, args, features, train_data):
    #max_predictions_per_sequence, max_sequence_length, max_sequences_per_pack):
    # SEQ
    packed_input_ids = np.zeros(args.max_sequence_length, dtype=np.int32)
    packed_input_mask = np.zeros(args.max_sequence_length, dtype=np.int32)
    packed_segment_ids = np.zeros(args.max_sequence_length, dtype=np.int32)
    packed_positions = np.zeros(args.max_sequence_length, dtype=np.int32)
    # MLM
    # we are packing up to max_sequences_per_pack, each with a certain percentage of masked tokens
    # in case that percentege is rounded up for all sequences in the pack, need to add an extra token for
    # each sequence in the pack
    packed_masked_lm_positions = np.zeros(args.max_predictions_per_sequence + args.max_sequences_per_pack, dtype=np.int32)
    packed_masked_lm_ids = np.zeros(args.max_predictions_per_sequence + args.max_sequences_per_pack, dtype=np.int32)
    #packed_masked_lm_weights = np.zeros(max_predictions_per_sequence + max_sequences_per_pack, dtype=np.int32)
    # NSP
    packed_next_sentence_positions = np.zeros(args.max_sequences_per_pack, dtype=np.int32)
    packed_next_sentence_labels = np.zeros(args.max_sequences_per_pack, dtype=np.int32)
    packed_next_sentence_weights = np.zeros(args.max_sequences_per_pack, dtype=np.int32)

    offset = 0
    mlm_offset = 0
    sequence_index = 1  # used in the input mask
    for (file_idx, sample_idx) in multi_sequence:

        if file_idx == -1 or sample_idx == -1:
            continue

        data = train_data[file_idx].inputs
        input_ids = data[pretraining_dataset.input_ids_idx][sample_idx]
        input_mask = data[pretraining_dataset.input_mask_idx][sample_idx]
        segment_ids = data[pretraining_dataset.segment_ids_idx][sample_idx]
        masked_lm_positions = data[pretraining_dataset.masked_lm_positions_idx][sample_idx]
        masked_lm_ids = data[pretraining_dataset.masked_lm_ids_idx][sample_idx]
        next_sentence_labels = data[pretraining_dataset.next_sentence_labels_idx][sample_idx]

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
        assert mlm_offset + mlm_len < args.max_predictions_per_sequence + args.max_sequences_per_pack, "Too many LM predictions per sequences"
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

    features["input_ids"].append(packed_input_ids)
    features["input_mask"].append(packed_input_mask)
    features["segment_ids"].append(packed_segment_ids)
    features["positions"].append(packed_positions)
    features["masked_lm_positions"].append(packed_masked_lm_positions)
    features["masked_lm_ids"].append(packed_masked_lm_ids)
    features["next_sentence_positions"].append(packed_next_sentence_positions)
    features["next_sentence_labels"].append(packed_next_sentence_labels)
    features["next_sentence_weights"].append(packed_next_sentence_weights)

class pretraining_dataset:

    keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
        'next_sentence_labels']

    input_ids_idx = keys.index('input_ids')
    input_mask_idx = keys.index('input_mask')
    segment_ids_idx = keys.index('segment_ids')
    masked_lm_positions_idx = keys.index('masked_lm_positions')
    masked_lm_ids_idx = keys.index('masked_lm_ids')
    next_sentence_labels_idx = keys.index('next_sentence_labels')

    def __init__(self, input_file):
        self.input_file = input_file
        f = h5py.File(input_file, "r")
        self.keys_exist = list(f.keys())
        self.inputs = [np.asarray(f[key][:]) for key in pretraining_dataset.keys]
        self.len_dict={}
        for key in pretraining_dataset.keys:
            self.len_dict[key] = np.asarray(f[key][:]).shape
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):
        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [input[index] if indice < 5 else
                np.asarray(input[index]) for indice, input in enumerate(self.inputs)]

        return [input_ids,  input_mask, segment_ids,masked_lm_positions, masked_lm_ids,
                next_sentence_labels]


def parse_arguments():
    parser = argparse.ArgumentParser()

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
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the packed dataset will be written.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("args.max_sequence_length={}, args.max_sequences_per_pack={},args.max_predictions_per_sequence={}".format(args.max_sequence_length, args.max_sequences_per_pack,args.max_predictions_per_sequence))

    files = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
    files = np.array(files, dtype=np.str_)
    print(f'files={files}')

    print("Loading dataset")
    start = time.time()
    train_data = dict()
    for f in tqdm(files):
        print(f)
        train_data[f] = pretraining_dataset(os.path.join(args.input_dir, f))
    del files
    gc.collect()
    print(f"Done.\nTook {time.time() - start:3.2f} seconds.")

    print("Gathering sequence lengths")
    start = time.time()
    sequence_lengths = []
    for data in tqdm(train_data.values()):
        sequence_lengths.append(np.sum(data.inputs[pretraining_dataset.input_mask_idx], axis=-1))
    sequence_lengths = np.array(sequence_lengths, dtype=np.int16)
    print(f"Done.\nTook {time.time() - start:3.2f} seconds.")

    num_samples = np.size(sequence_lengths)
    print(f"Dataset has {num_samples} samples")

    print("Determining packing recipe")
    start = time.time()
    strategy_set, mixture, padding, slicing = get_packing_recipe(args.output_dir, sequence_lengths, args.max_sequence_length, args.max_sequences_per_pack)
    padding = padding.astype(np.int32)
    print(f"Done.\nTook {time.time() - start:3.2f} seconds.")

    print("Building length to example map")
    start = time.time()
    histogram, _ = np.histogram(sequence_lengths, bins=np.arange(1, args.max_sequence_length + 2))
    histogram = histogram + padding
    examples_by_length = np.array([-1 * np.ones(shape=[x, 2], dtype=np.int32) for x in histogram], dtype=np.object_)
    del padding
    del histogram
    gc.collect()

    slot = np.zeros([args.max_sequence_length], dtype=np.int32)
    for (file_idx, seq_lens) in tqdm(enumerate(sequence_lengths)):
        for (sample_idx, sl) in enumerate(seq_lens):
            examples_by_length[sl-1][slot[sl-1],:] = (file_idx, sample_idx)
            slot[sl-1] += 1

    del slot
    del sequence_lengths
    gc.collect()
    print(f"Done.\nTook {time.time() - start:3.2f} seconds.")

    print("Shuffling examples")
    start = time.time()
    for x in examples_by_length:
        np.random.shuffle(x)
    print(f"Done.\nTook {time.time() - start:3.2f} seconds.")

    print('Slicing examples according to packing recipe')
    start = time.time()
    examples = slice_examples_mult_stratagies_shuffle(examples_by_length, slicing, strategy_set, mixture, args.max_sequences_per_pack)
    print(f"Done.\nTook {time.time() - start:3.2f} seconds.")
    del examples_by_length; del slicing; del strategy_set; del mixture
    gc.collect()

    print('Writing the output files')
    start = time.time()
    y = np.ndarray(shape=(len(train_data), ), dtype=np.object_)
    y[:] = list(train_data.values())
    parts = list(range(0, len(examples), 4480))
    for (i, offset) in tqdm(list(enumerate(parts))):
        pack_according_to_strategy(args, i, examples[offset:offset+4480], y)
    print(f"Done.\nTook {time.time() - start:3.2f} seconds.")

if __name__ == "__main__":
    main()
