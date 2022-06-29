#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse
import datasets
import tempfile
import shutil
import os
from functools import partial
import transformers

import model


parser = argparse.ArgumentParser(
    description='Prepare SQUAD dataset and T5-base model')
parser.add_argument('data_dir', type=str, help='where to store files')
args = parser.parse_args()


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "t5-base", cache_dir=temp_dir)
        tokenizer_path = os.path.join(args.data_dir, 't5_base', 'tokenizer')
        tokenizer.save_pretrained(tokenizer_path)
        shutil.move(
            os.path.join(tokenizer_path, 'tokenizer_config.json'),
            os.path.join(tokenizer_path, 'config.json'))
        print(f'Tokenizer saved to {args.data_dir}/t5_base/tokenizer')

        train_dataset = datasets.load_dataset('squad', split='train',
                                              cache_dir=temp_dir)
        valid_dataset = datasets.load_dataset('squad', split='validation',
                                              cache_dir=temp_dir)

        train_ds = train_dataset.map(partial(model.encode, tokenizer))
        valid_ds = valid_dataset.map(partial(model.encode, tokenizer))

        train_ds.save_to_disk(os.path.join(args.data_dir, 'squad', 'train'))
        valid_ds.save_to_disk(os.path.join(args.data_dir, 'squad', 'valid'))
        print(f'Dataset saved to {args.data_dir}/squad')

        t5_base = model.T5.from_pretrained('t5-base', cache_dir=temp_dir)
        t5_base.save_pretrained(os.path.join(args.data_dir, 't5_base'))
        print(f'Pretrained model saved to {args.data_dir}/t5_base')


if __name__ == '__main__':
    main()
