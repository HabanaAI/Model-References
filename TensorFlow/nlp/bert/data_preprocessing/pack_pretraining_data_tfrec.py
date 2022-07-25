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
###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###############################################################################
# Changes:
# - Added functionality for saving parameters of packing algorithm to metadata file.
# - Added checks for output_dir parameter. It will be created automatically if passed location does not exist.


import argparse
import gc
import json
import os
import random
import time
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import chain, repeat
from sys import getsizeof, stderr

import numpy as np
import tensorflow as tf
from scipy import optimize

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


def get_metadata_file_path(output_dir):
    """Returns path for metadata file one direcotry above output_dir.
    File will be called the same way as directory with training dataset
    with appended metadata.json as below:
    ├── training
    └── training_metadata.json"""
    norm_path = os.path.normpath(output_dir)
    base_path, metadata_file_name = os.path.split(norm_path)
    metadata_file_name = metadata_file_name + '_metadata.json'
    return os.path.join(base_path, metadata_file_name)

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

    # Save packing parameters to metadata file
    metadata_file_path = get_metadata_file_path(output_dir)
    print(f"Saving metadata to file: {metadata_file_path}")

    packing_metadata = {
        "sequences_dropped": int(number_of_sequences_dropped),
        "num_strategies_utilized": number_of_strategies_utilized,
        "new_number_of_samples": new_number_of_samples,
        "original_number_of_samples": original_number_of_samples,
        "compression_ratio": compression,
        "expected_speedup": expected_speedup_from_packing,
        "theoretical_speedup": float(upper_bound),
        "avg_seq_per_sample": float(avg_sequences_per_sample),
        "padding_tokens_original_dataset": int(num_padding_tokens_original),
        "padding_tokens_packed_dataset": float(num_padding_tokens),
        "packing_efficiency": float(efficiency),
        "top_8_strategies": topK.tolist()
    }
    with open(metadata_file_path, mode='w') as json_file:
        json_file.write(json.dumps(packing_metadata, sort_keys=True, indent=2))
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
        num_sample_per_slice=4480
        num_parts = repeat_count // num_sample_per_slice
        num_parts = num_parts + int(repeat_count != num_parts * num_sample_per_slice)
        subcounts = (min(num_sample_per_slice, repeat_count - num_sample_per_slice * (i - 1)) for i in range(1, num_parts + 1))
        for part_id, part_count in enumerate(subcounts):
            examples = []
            for k, seq_len in enumerate(strategy):
                slice_start = int(slice_offsets[seq_len - 1])
                slice_end = slice_start + int(part_count)
                slice_offsets[seq_len - 1] = slice_end
                examples.append(examples_by_length[seq_len][slice_start:slice_end])
            slices.append(examples)
            strategies.append(strategy)
            part_idx.append(part_id)
    examples_by_length = None
    return slices, strategies, part_idx


def parallel_pack_according_to_strategy(args, part_idx, strategy, examples):
    # Pack the sequences according to the strategy and write them to disk
    try:
        base_filename = os.path.join(args.output_dir, "strategy_" + "_".join(map(str, strategy)))
        filename = base_filename + f"_part_{part_idx}"
        print(filename)
        writer = tf.compat.v1.python_io.TFRecordWriter(filename)
        for i, multi_sequence in enumerate(zip(*examples)):
            features = create_multi_sequence_example(multi_sequence, args.max_predictions_per_sequence,
                                                           args.max_sequence_length, args.max_sequences_per_pack)
            # Write to file
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
        writer.close()
    except:
        print('failed to write: ',strategy,part_idx)
        base_filename = os.path.join(args.output_dir, "FAIL_strategy_" + "_".join(map(str, strategy)))
        filename = base_filename + f"_part_{part_idx}"
        print('saved failed examples to: ','FAIL_'+filename)



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
            seq_len = input_mask.sum()

            del example

            # SEQ
            packed_input_ids[offset:offset + seq_len] = input_ids[:seq_len]
            packed_input_mask[offset:offset + seq_len] = sequence_index
            packed_segment_ids[offset:offset + seq_len] = segment_ids[:seq_len]
            packed_positions[offset:offset + seq_len] = np.arange(0, seq_len)

            # MLM
            mlm_len = int(masked_lm_weights.sum())
            assert mlm_offset + mlm_len < max_predictions_per_sequence + max_sequences_per_pack, "Too many LM predictions per sequences"
            max_mlm = mlm_offset + mlm_len
            packed_masked_lm_positions[mlm_offset:max_mlm] = offset + masked_lm_positions[:mlm_len]
            packed_masked_lm_ids[mlm_offset:max_mlm] = masked_lm_ids[:mlm_len]
            packed_masked_lm_weights[mlm_offset:max_mlm] = sequence_index
            # NSP
            packed_next_sentence_positions[sequence_index - 1] = offset
            packed_next_sentence_labels[sequence_index - 1] = next_sentence_labels
            packed_next_sentence_weights[sequence_index - 1] = 1

            # Update offsets
            sequence_index += 1
            offset += seq_len
            mlm_offset = max_mlm
            input_ids = None; input_mask = None; segment_ids = None; masked_lm_positions = None;
            masked_lm_ids = None; masked_lm_weights = None; next_sentence_labels = None; seq_len = None;
    # Pack into binary format and write it

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
    del packed_input_ids; del packed_input_mask; del packed_segment_ids;  del packed_positions; del packed_masked_lm_positions
    del packed_masked_lm_weights; del packed_next_sentence_positions; del packed_next_sentence_labels; del packed_next_sentence_weights

    return features

def create_bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature

def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature

def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)
def compress_zeros(input):
    return input[0:np.where(input)[0][-1]+1]

def decompress_zeros(input,list_size):
    output = np.zeros(list_size)
    output[0:len(input)]=input
    return output

def compress_seg_ids(segment_ids):
    tmp=np.where(segment_ids)[0]
    return np.array([tmp[0],tmp[-1]-tmp[0]])

def decompress_seg_ids(segment_ids):
    output = np.zeros(512)
    output[segment_ids[0],segment_ids[0]+segment_ids[1]]=1
    return output

def getCurrentMemoryUsage():
    # Getting all memory using os.popen()
    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t -m').readlines()[-1].split()[1:])

    # Memory usage
    print("RAM memory % used:", round((used_memory/total_memory) * 100, 2))
    return used_memory/total_memory

def parallel_record_loader(record):
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
    im_length = sum(example.features.feature['input_mask'].int64_list.value)
    return record, im_length


def parallel_data_loader(path,filename):
    sequence_lengths_part = []
    examples_by_length_part = defaultdict(list)
    for record in tf.data.TFRecordDataset(path+filename):
        example = tf.train.Example()
        example.ParseFromString(record.numpy())
        im_length = sum(example.features.feature['input_mask'].int64_list.value)
        examples_by_length_part[im_length].append(record)
        sequence_lengths_part.append(im_length)
        del example
    return sequence_lengths_part,examples_by_length_part

if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", help="A glob expression for the input files to read in and pack", required=True, type=str)
    parser.add_argument("--output-dir", help="The destination folder for the output files", required=True)
    parser.add_argument("--max-files", help="At most how many files to process (limited by RAM)", default=100,type=int)
    parser.add_argument("--duplication-factor", help="Same as the one passed to create input data", default=1, type=int)
    parser.add_argument("--max-sequence-length", help="The maximum number of tokens in an example", default=512, type=int)
    parser.add_argument("--max-predictions-per-sequence", help="The maximum number of masked tokens in an un-packed example", default=76, type=int)
    parser.add_argument("--max-sequences-per-pack", help="The maximum number of sequences per packed example.", choices=[2, 3], default=3, type=int)
    args = parser.parse_args()

    logger = tf.get_logger()
    logger.propagate = False

    if not os.path.exists(args.output_dir):
        logger.warning(
            f"Output directory: {args.output_dir} does not exists, creating..."
        )
        try:
            os.makedirs(args.output_dir, exist_ok=True)
        except IOError as error:
            logger.error(error)
            raise

    # Input files
    print("Looping through dataset to collect sequence length information...")
    input_files = np.random.choice(os.listdir(args.input_glob), size=args.max_files, replace=False)
    sequence_lengths = []
    examples_by_length = defaultdict(list)

    with ProcessPoolExecutor(25) as executor:
        work = repeat(args.input_glob), input_files.tolist()
        for sequence_lengths_part,examples_by_length_part  in executor.map(parallel_data_loader, *work):
            pass
            sequence_lengths += sequence_lengths_part
            examples_by_length = { key:examples_by_length.get(key,[])+examples_by_length_part.get(key,[]) for key in set(list(examples_by_length.keys())+list(examples_by_length_part.keys())) }
            del examples_by_length_part
            sequence_lengths_part=None; examples_by_length_part=None
    sequence_lengths = np.array(sequence_lengths)
    print('Done extracting sequance length !!!')
    del executor
    gc.collect()
    # Pass the array of sequence lengths to the packing algorithm
    strategy_set, mixture, padding, slicing = get_packing_recipe(args.output_dir, sequence_lengths, args.max_sequence_length, args.max_sequences_per_pack)
    print('Done get_packing_recipe !!!')
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
    example_slices, strategies, part_idx = slice_examples(examples_by_length, slicing, strategy_set, mixture)
    gc.collect()
    print('Done slice_examples !!!')
    del examples_by_length; del slicing; del strategy_set; del mixture
    gc.collect()
    start = time.time()
    print(f"Splitting work into {len(part_idx)} parts.")
    for rr in range(1+len(strategies)//500):
        str_idx,stp_idx=rr*500,min((rr+1)*500,len(strategies))
        part_idx_prt, strategies_prt, example_slices_prt = part_idx[str_idx:stp_idx], strategies[str_idx:stp_idx], example_slices[str_idx:stp_idx]
        with ProcessPoolExecutor(25) as executor:
            work = repeat(args), part_idx_prt, strategies_prt, example_slices_prt
            for partial_result in executor.map(parallel_pack_according_to_strategy, *work):
                pass
        del work
    print(f"\nDone. Took: {time.time() - start:3.2f} seconds to pack and write dataset.")
    print('-------------',str_idx,stp_idx)
    print('Done Cleaning')
