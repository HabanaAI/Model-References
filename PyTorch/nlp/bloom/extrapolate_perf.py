#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

from scipy import stats
import argparse
import itertools
import json
import os
import random
import statistics


def setup_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_new_tokens', type=int, help="Target max_new_tokens", required=True)
    parser.add_argument('--num_hypotheses', type=int, help="Number of hypotheses", default=8192)
    parser.add_argument('--hyp_samples', type=int, help="Number of samples per each length", default=12)
    parser.add_argument('--batch_size', type=int, help="Target batch size")
    parser.add_argument('--output_file', type=str, help="Output json file")
    parser.add_argument('input_files', type=str, nargs='+', help="Input files", default=[])
    return parser


def read_file(f):
    def parse(line):
        line = line.strip().split()
        return (int(line[0]), float(line[1]))

    with open(f) as file:
        return [parse(line) for line in file.readlines()]


def generate_hypothesis(data, samples):
    subset = []
    for tok, values in data.items():
        s = random.sample(values, min(len(values), samples))
        subset.extend((tok, v) for v in s)
    linreg = stats.linregress(subset)
    return lambda x: linreg.slope * x + linreg.intercept


def group(data):
    buckets = {}
    for tok, val in data:
        buckets.setdefault(tok, []).append(val)
    return buckets


def main():
    parser = setup_parser()
    args = parser.parse_args()

    random.seed(42)

    data = list(itertools.chain(*[read_file(f) for f in args.input_files]))
    data = group(data)
    iterations = list(sorted(data.keys()))
    print("Sample counts:")
    for i in iterations:
        print(f'  len({i})={len(data[i])}')

    print("Testing hypotheses...")
    hyp = [generate_hypothesis(data, args.hyp_samples)(args.max_new_tokens) for _ in range(args.num_hypotheses)]
    prediction = min(hyp)
    tps = None
    if args.batch_size is not None:
        tps = args.batch_size * args.max_new_tokens / prediction

    print('Predicted duration:', prediction)
    if tps is not None:
        print('Predicted tps:', tps)
    if args.output_file:
        results = {}
        if tps is not None:
            results['tps'] = tps
        results['duration'] = prediction
        results['samples'] = data
        if args.batch_size is not None:
            results['batch_size'] = args.batch_size
        results['max_new_tokens'] = args.max_new_tokens
        json.dump(results, open(args.output_file, 'w'))


if __name__ == '__main__':
    main()
