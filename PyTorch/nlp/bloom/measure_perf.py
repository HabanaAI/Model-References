#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse
import itertools

import bloom
import habana_generation_utils as hgu



def main():
    parser = bloom.setup_parser()
    args = parser.parse_args()

    model, tokenizer, options = bloom.initialize_model(args)
    pipeline = hgu.create_pipeline(model, tokenizer, mode=hgu.GenerationMode.OPTIMIZED, calc_stats=True)
    batch =  ["A"] * args.batch_size

    def experiment(iterations, repeat):
        options['max_iterations'] = iterations
        data = []
        for i in range(repeat):
            _, stats = pipeline(batch, options)
            duration = [s for s in stats if s[0] == 'duration'][0][1]
            data.append(duration)
            print(iterations, duration)
        return data

    iterations = [16, 32, 64, 128]
    repeats = [32, 16, 8, 4]
    print('[Selected iterations]')
    print(', '.join(str(t) for t in iterations))

    print('[Warmup]')
    experiment(iterations[0], 3)

    print('[Collecting measurements]')
    data = [[(t, d) for d in experiment(t, r)] for t, r in zip(iterations, repeats)]
    data = list(itertools.chain(*data))

    filename = f'perf.{args.global_rank}.samples'
    with open(filename, 'w') as f:
        for d in data:
            print(*d, file=f, force=True)


if __name__ == '__main__':
    main()
