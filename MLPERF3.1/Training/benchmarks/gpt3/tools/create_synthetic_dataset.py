# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company

import argparse
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--valid_files_path',
        type=str,
        default='c4_en_validation_c4_spm_text_document',
        help='Path to the name of the valid files without an extension.')
    parser.add_argument(
        '--output_path',
        type=str,
        default='/tmp/',
        help='Output path')
    args = parser.parse_args()

    HEADER_END=17
    SHUFFLE_END=24567*2+14

    document_bin = np.fromfile(args.valid_files_path + ".bin", dtype=np.uint16)
    document_bin = np.random.randint(1, high=50256, size=document_bin.size, dtype=np.uint16) % 50256

    document_idx = np.fromfile(args.valid_files_path + ".idx", dtype=np.uint16)
    document_idx_sizes = document_idx[HEADER_END:SHUFFLE_END:2]
    document_idx_sizes = np.random.shuffle(document_idx_sizes)

    with open(args.output_path+"synthetic_text_document.bin", 'wb') as f:
        document_bin.tofile(f)
    with open(args.output_path+"synthetic_text_document.idx", 'wb') as f:
        document_idx.tofile(f)