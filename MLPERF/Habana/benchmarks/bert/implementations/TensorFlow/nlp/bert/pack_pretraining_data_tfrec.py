# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import time
import glob
import struct
import random
import argparse
import numpy as np
import pandas as pd
from scipy import optimize
from itertools import repeat, chain
from functools import lru_cache, reduce
from collections import defaultdict, OrderedDict
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import tensorflow as tf


@lru_cache(maxsize=None)
def packing_strategies(start, previous, target, depth):
    gap = target - start

    # The collection of possible strategies given the
    # starting sum, the target sum, and the available depth
    # strategy search is limited to increments greater or equal to previous
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
    unpacked_seqlen = np.arange(1, args.max_sequence_length + 1)[b > 0]
    # Update the mixture to also covered the unpacked sequences
    for l in unpacked_seqlen:
        # Get the depth 1 strategy
        strategy = sorted([l, args.max_sequence_length - l])
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


def slice_examples(examples_by_length, slicing, strategy_set, repeat_counts):
    # Divide the work, firstly between the strategies and then into chunks of 50k
    slices = []
    strategies = []
    part_idx = []
    for strategy, slice_offsets, repeat_count in zip(strategy_set, slicing, repeat_counts):
        if repeat_count == 0:
            continue
        # Slice out the sequences allocated to this strategy in increments of 50k
        num_parts = repeat_count // 50000
        num_parts = num_parts + int(repeat_count != num_parts * 50000)
        subcounts = (min(50000, repeat_count - 50000 * (i - 1)) for i in range(1, num_parts + 1))
        for part_id, part_count in enumerate(subcounts):
            examples = []
            for k, seq_len in enumerate(strategy):
                slice_start = int(slice_offsets[seq_len - 1])
                slice_end = slice_start + int(part_count)
                slice_offsets[seq_len - 1] = slice_end
                examples.append(examples_by_length[seq_len][slice_start:slice_end])
            #import pdb; pdb.set_trace()
            slices.append(examples)
            strategies.append(strategy)
            part_idx.append(part_id)

    return slices, strategies, part_idx


def parallel_pack_according_to_strategy(args, part_idx, strategy, examples):
    # Pack the sequences according to the strategy and write them to disk
    base_filename = os.path.join(args.output_dir, "strategy_" + "_".join(map(str, strategy)))
    filename = base_filename + f"_part_{part_idx}"
    writer = tf.compat.v1.python_io.TFRecordWriter(filename)
    for i, multi_sequence in enumerate(zip(*examples)):
        features = create_multi_sequence_example(multi_sequence, args.max_predictions_per_sequence,
                                                       args.max_sequence_length, args.max_sequences_per_pack)
    # Write to file
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writer.write(tf_example.SerializeToString())
    
    writer.close()


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
    packed_masked_lm_weights = np.zeros(max_predictions_per_sequence + max_sequences_per_pack, dtype=np.int32)

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
            example = tf.train.Example()
            example.ParseFromString(sequence.numpy())

            input_ids = np.array(example.features.feature['input_ids'].int64_list.value)
            input_mask = np.array(example.features.feature['input_mask'].int64_list.value)
            segment_ids = np.array(example.features.feature['segment_ids'].int64_list.value)
            masked_lm_positions = np.array(example.features.feature['masked_lm_positions'].int64_list.value)
            masked_lm_ids = np.array(example.features.feature['masked_lm_ids'].int64_list.value)
            masked_lm_weights = np.array(example.features.feature['masked_lm_weights'].float_list.value)
            next_sentence_labels = np.array(example.features.feature['next_sentence_labels'].int64_list.value)

            #input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, next_sentence_labels = sequence
            seq_len = input_mask.sum()

            # SEQ
            packed_input_ids[offset:offset + seq_len] = input_ids[:seq_len]
            packed_input_mask[offset:offset + seq_len] = sequence_index
            packed_segment_ids[offset:offset + seq_len] = segment_ids[:seq_len]
            packed_positions[offset:offset + seq_len] = np.arange(0, seq_len)

            # MLM
            mlm_len = int(masked_lm_weights.sum())
            assert mlm_offset + mlm_len < max_predictions_per_sequence + max_sequences_per_pack, "Too many LM predictions per sequences"
            max_mlm = mlm_offset + mlm_len
            #import pdb; pdb.set_trace()
            packed_masked_lm_positions[mlm_offset:max_mlm] = offset + masked_lm_positions[:mlm_len]
            packed_masked_lm_ids[mlm_offset:max_mlm] = masked_lm_ids[:mlm_len]
            packed_masked_lm_weights[mlm_offset:max_mlm] = sequence_index
            #import pdb; pdb.set_trace()
            # NSP
            packed_next_sentence_positions[sequence_index - 1] = offset
            packed_next_sentence_labels[sequence_index - 1] = next_sentence_labels
            packed_next_sentence_weights[sequence_index - 1] = 1

            # Update offsets
            sequence_index += 1
            offset += seq_len
            mlm_offset = max_mlm
            #import pdb; pdb.set_trace()
    # Pack into tfrecord format:
    
    features = OrderedDict()
    
    features["input_ids"] = create_int_feature(packed_input_ids)
    features["input_mask"] = create_int_feature(packed_input_mask)
    features["segment_ids"] = create_int_feature(packed_segment_ids)
    features["positions"] = create_int_feature(packed_positions)
    features["masked_lm_positions"] = create_int_feature(packed_masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(packed_masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(packed_masked_lm_weights)    
    features["next_sentence_positions"] = create_int_feature(packed_next_sentence_positions)
    features["next_sentence_labels"] = create_int_feature(packed_next_sentence_labels)
    features["next_sentence_weights"] = create_float_feature(packed_next_sentence_weights)    
    return features   

def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature

def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", help="A glob expression for the input files to read in and pack", required=True, type=str)
    parser.add_argument("--output-dir", help="The destination folder for the output files", required=True)
    parser.add_argument("--random-seed", help="For shuffling the data", default=12345)
    parser.add_argument("--max-files", help="At most how many files to process (limited by RAM)", default=100)
    parser.add_argument("--duplication-factor", help="Same as the one passed to create input data", default=1, type=int)
    parser.add_argument("--max-sequence-length", help="The maximum number of tokens in an example", default=512, type=int)
    parser.add_argument("--max-predictions-per-sequence", help="The maximum number of masked tokens in an un-packed example", default=76, type=int)
    parser.add_argument("--max-sequences-per-pack", help="The maximum number of sequences per packed example.", choices=[2, 3], default=3, type=int)
    args = parser.parse_args()
    random.seed(args.random_seed)

    # Put examples into bins depending on their sequence lengths and extract the sequence length
    sequence_lengths = []
    examples_by_length = defaultdict(list)
    print("Looping through dataset to collect sequence length information...")
    for filename in os.listdir(args.input_glob):
        for record in tf.data.TFRecordDataset(args.input_glob+filename):
                example = tf.train.Example()
                example.ParseFromString(record.numpy())
                im_length = sum(example.features.feature['input_mask'].int64_list.value)
                examples_by_length[im_length].append(record)
                sequence_lengths.append(im_length)
    sequence_lengths = np.array(sequence_lengths)

    # Pass the array of sequence lengths to the packing algorithm
    
    strategy_set, mixture, padding, slicing = get_packing_recipe(sequence_lengths, args.max_sequence_length, args.max_sequences_per_pack)
    
    # Add the calculated padding
    for i in range(1, args.max_sequence_length + 1):
        examples_by_length[i].extend([None] * int(padding[i - 1]))

    # Shuffle the data
    for key in examples_by_length:
        random.shuffle(examples_by_length[key])

    # Pack and store the data
    print(f"\nPacking and writing packed dataset to {args.output_dir}.")

    # Slice the data into chunks of max 50k packed examples
    example_slices, strategies, part_idx = slice_examples(examples_by_length, slicing, strategy_set, mixture)
    print(f"Splitting work into {len(part_idx)} parts.")
    start = time.time()
    #For debug uses
    #for i in range(len(part_idx)):
    #    parallel_pack_according_to_strategy(args, part_idx[i], strategies[i], example_slices[i])
    #import pdb; pdb.set_trace()

    with ProcessPoolExecutor(16) as executor:
        work = repeat(args), part_idx, strategies, example_slices
        for partial_result in executor.map(parallel_pack_according_to_strategy, *work):
            pass
    print(f"\nDone. Took: {time.time() - start:3.2f} seconds to pack and write dataset.")
