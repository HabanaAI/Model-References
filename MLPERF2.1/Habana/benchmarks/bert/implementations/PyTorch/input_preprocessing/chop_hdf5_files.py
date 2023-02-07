# Copyright (c) 2019-2022 NVIDIA CORPORATION. All rights reserved.
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

import glob
import h5py
import multiprocessing
import numpy as np
from os import path, makedirs
from tqdm import tqdm
import argparse
import logging

parser = argparse.ArgumentParser(
    description="Training data sharding for BERT.")
parser.add_argument(
    '--input_hdf5_dir',
    type=str,
    default='hdf5',
    help='Input hdf5_file path')
parser.add_argument(
    '--output_hdf5_dir',
    type=str,
    default='',
    help='Output hdf5_file path')
parser.add_argument(
    '--num_shards',
    type=int,
    default=2048,
    help='Number of output shards (default 2048)')
parser.add_argument(
    '--max_seq_length',
    type=int,
    default=512,
    help='The maximum number of tokens within a sequence. (default 512)')
parser.add_argument(
    '--max_predictions_per_seq',
    type=int,
    default=76,
    help='The maximum number of predictions within a sequence. (default 76)')
args = parser.parse_args()

max_seq_length = args.max_seq_length
max_predictions_per_seq = args.max_predictions_per_seq
n_output_shards = args.num_shards
input_path = args.input_hdf5_dir
logging.basicConfig(level=logging.INFO)

hdf5_compression_method = None

input_files = sorted(glob.glob(input_path + '/part-00???-of-00500.hdf5', recursive=False))
logging.info('n_input_shards = {}'.format(len(input_files)))
logging.info('n_output_shards = {}'.format(n_output_shards))

output_shards_dir = path.join(args.output_hdf5_dir,'hdf5_{}_shards_uncompressed'.format(n_output_shards))
try:  
  makedirs(output_shards_dir)
except OSError as error:
  logging.info('Output directory : {} already exists. Overwritting ...'.format(output_shards_dir))

ofile_prefix = path.join(output_shards_dir, 'part_')
ofile_suffix = '_of_{:05d}.hdf5'.format(n_output_shards)


# First pass over data to get sample count (read only the smallest array to get count)
n_samples = 0
for ifile in tqdm(input_files, total=len(input_files)):
  h5_ifile = h5py.File(ifile, 'r')
  n_samples += h5_ifile['next_sentence_labels'].shape[0]
  h5_ifile.close()
  
# Find a "nominal" number of samples per shard (calculated to always go over by one shard size)
# Find excess samples in last shard and distribute removal of excess over first "N" shards (could be done over last, but it doesn't matter and math is easier this way)
#  (since 0 <= excess < nominal_shard_size, the max imbalance will be 1 sample to minimize the straggler effect)
n_sample_per_ofile_nominal = (n_samples + n_output_shards - 1) // n_output_shards
n_excess = n_output_shards * n_sample_per_ofile_nominal - n_samples  # Always a positive number
logging.info('Total number of samples: {}. Sample per shard {}/{}'.format(n_samples, n_sample_per_ofile_nominal-1, n_sample_per_ofile_nominal))

logging.info('creating {} output file handles.  This could take a while.'.format(n_output_shards))
ofile_handles = [h5py.File('{}{:05d}{}'.format(ofile_prefix, shard, ofile_suffix), 'w') for shard in range(n_output_shards)]

ofile_idx = 0  # which output file
ofile_entry_idx = 0  # index into an individual data element of an output file
ifile_entry_idx = 0

n_samples_in_this_shard = n_sample_per_ofile_nominal - 1 
o_input_ids = np.ndarray((n_samples_in_this_shard, max_seq_length))
o_input_masks = np.ndarray((n_samples_in_this_shard, max_seq_length))
o_segment_ids = np.ndarray((n_samples_in_this_shard, max_seq_length))
o_masked_lm_positions = np.ndarray((n_samples_in_this_shard, max_predictions_per_seq))
o_masked_lm_ids = np.ndarray((n_samples_in_this_shard, max_predictions_per_seq))
o_next_sentence_labels = np.ndarray((n_samples_in_this_shard))

for ifile in tqdm(input_files, total=len(input_files)):
  h5_ifile = h5py.File(ifile, 'r')
  
  ifile_entry_idx = 0
  f_input_ids = h5_ifile['input_ids'][:]
  f_input_masks = h5_ifile['input_mask'][:]
  f_segment_ids = h5_ifile['segment_ids'][:]
  f_masked_lm_positions = h5_ifile['masked_lm_positions'][:]
  f_masked_lm_ids = h5_ifile['masked_lm_ids'][:]
  f_next_sentence_labels = h5_ifile['next_sentence_labels'][:]

  h5_ifile.close()

  # This could be vectorized but keeping it simple due to lack of time
  while ifile_entry_idx < f_input_ids.shape[0]:
    if ofile_entry_idx == n_samples_in_this_shard:
      ofile_handles[ofile_idx].create_dataset("input_ids",            data=o_input_ids,            dtype='i2', compression=hdf5_compression_method)
      ofile_handles[ofile_idx].create_dataset("input_mask",           data=o_input_masks,          dtype='i1', compression=hdf5_compression_method)
      ofile_handles[ofile_idx].create_dataset("segment_ids",          data=o_segment_ids,          dtype='i1', compression=hdf5_compression_method)
      ofile_handles[ofile_idx].create_dataset("masked_lm_positions",  data=o_masked_lm_positions,  dtype='i2', compression=hdf5_compression_method)
      ofile_handles[ofile_idx].create_dataset("masked_lm_ids",        data=o_masked_lm_ids,        dtype='i2', compression=hdf5_compression_method)
      ofile_handles[ofile_idx].create_dataset("next_sentence_labels", data=o_next_sentence_labels, dtype='i1', compression=hdf5_compression_method)
      ofile_handles[ofile_idx].flush()
      ofile_handles[ofile_idx].close()

      ofile_entry_idx = 0
      ofile_idx += 1

      n_samples_in_this_shard = n_sample_per_ofile_nominal
      if ofile_entry_idx < n_excess:
        n_samples_in_this_shard -= 1

      o_input_ids = np.ndarray((n_samples_in_this_shard, max_seq_length))
      o_input_masks = np.ndarray((n_samples_in_this_shard, max_seq_length))
      o_segment_ids = np.ndarray((n_samples_in_this_shard, max_seq_length))
      o_masked_lm_positions = np.ndarray((n_samples_in_this_shard, max_predictions_per_seq))
      o_masked_lm_ids = np.ndarray((n_samples_in_this_shard, max_predictions_per_seq))
      o_next_sentence_labels = np.ndarray((n_samples_in_this_shard))

    o_input_ids[ofile_entry_idx] = f_input_ids[ifile_entry_idx]
    o_input_masks[ofile_entry_idx] = f_input_masks[ifile_entry_idx]
    o_segment_ids[ofile_entry_idx] = f_segment_ids[ifile_entry_idx]
    o_masked_lm_positions[ofile_entry_idx] = f_masked_lm_positions[ifile_entry_idx]
    o_masked_lm_ids[ofile_entry_idx] = f_masked_lm_ids[ifile_entry_idx]
    o_next_sentence_labels[ofile_entry_idx] = f_next_sentence_labels[ifile_entry_idx]
    ofile_entry_idx += 1

    ifile_entry_idx += 1 