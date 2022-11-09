#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###############################################################################

import random
import argparse
import json

def limit_words(n, txt):
    return ' '.join(txt.split()[:n])

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Input generator for Bloom')
parser.add_argument('--input_file', '-i',  type=str, help="Input file", required=True)
parser.add_argument('--output_file', '-o',  type=str, help="Output file", required=True)
parser.add_argument('--max_length', '-l',  type=int, help="Max words in input sentence", default=10)
parser.add_argument('--num_queries', '-n',  type=int, help="Number of queries in output", default=128)
parser.add_argument('--seed', '-s',  type=int, help="Random seed", default=42)
args = parser.parse_args()

with open(args.input_file) as inp:
    lines = inp.readlines()

lines = [limit_words(args.max_length, line.strip()) for line in lines]
random.seed(args.seed)
random.shuffle(lines)
lines = lines[:args.num_queries]
json.dump(lines, open(args.output_file, 'w'), indent=2)
