"""Script for picking certain number of samples.
"""

import argparse
import time
import logging
import collections
import h5py
import numpy as np

parser = argparse.ArgumentParser(
    description="Eval sample picker for BERT.")
parser.add_argument(
    '--input_hdf5_file',
    type=str,
    default='',
    help='Input hdf5_file path')
parser.add_argument(
    '--output_hdf5_file',
    type=str,
    default='',
    help='Output hdf5_file path')
parser.add_argument(
    '--num_examples_to_pick',
    type=int,
    default=10000,
    help='Number of examples to pick')
parser.add_argument(
    '--max_seq_length',
    type=int,
    default=512,
    help='The maximum number of tokens within a sequence.')
parser.add_argument(
    '--max_predictions_per_seq',
    type=int,
    default=76,
    help='The maximum number of predictions within a sequence.')
args = parser.parse_args()

max_seq_length = args.max_seq_length
max_predictions_per_seq = args.max_predictions_per_seq
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
  tic = time.time()
  h5_ifile = h5py.File(args.input_hdf5_file, 'r')
  num_examples = h5_ifile.get('next_sentence_labels').shape[0]

#  hdf5_compression_method = "gzip"
  hdf5_compression_method = None

  h5_writer = h5py.File(args.output_hdf5_file+".hdf5", 'w')
  input_ids = h5_writer.create_dataset('input_ids', (args.num_examples_to_pick,), dtype=h5py.vlen_dtype(np.dtype('int16')), compression=hdf5_compression_method)
  segment_ids = h5_writer.create_dataset('segment_ids', (args.num_examples_to_pick,), dtype=h5py.vlen_dtype(np.dtype('int8')), compression=hdf5_compression_method)
  masked_lm_positions = h5_writer.create_dataset('masked_lm_positions', (args.num_examples_to_pick,), dtype=h5py.vlen_dtype(np.dtype('int16')), compression=hdf5_compression_method)
  masked_lm_ids = h5_writer.create_dataset('masked_lm_ids', (args.num_examples_to_pick,), dtype=h5py.vlen_dtype(np.dtype('int16')), compression=hdf5_compression_method)
  next_sentence_labels = h5_writer.create_dataset('next_sentence_labels', data=np.zeros(args.num_examples_to_pick, dtype="int8"), dtype='i1', compression=hdf5_compression_method)

  i = 0
  pick_ratio = num_examples / args.num_examples_to_pick
  num_examples_picked = 0
  for i in range(args.num_examples_to_pick):
    idx = int(i * pick_ratio)
    input_ids[i] = h5_ifile['input_ids'][idx, :sum(h5_ifile['input_mask'][idx])]
    segment_ids[i] = h5_ifile['segment_ids'][idx, :sum(h5_ifile['input_mask'][idx])]
    masked_lm_positions[i] = h5_ifile['masked_lm_positions'][idx, :sum(h5_ifile['masked_lm_positions'][idx]!=0)]
    masked_lm_ids[i] = h5_ifile['masked_lm_ids'][idx, :sum(h5_ifile['masked_lm_positions'][idx]!=0)]
    next_sentence_labels[i] = h5_ifile['next_sentence_labels'][idx]
    num_examples_picked += 1

  h5_writer.flush()
  h5_writer.close()

  toc = time.time()
  logging.info("Picked %d examples out of %d samples in %.2f sec",
               args.num_examples_to_pick, num_examples, toc - tic)
