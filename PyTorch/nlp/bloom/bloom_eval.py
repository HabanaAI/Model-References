#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import lm_eval.tasks
import lm_eval.evaluator

import argparse
import json
import time
import os

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import bloom
from bloom import initialize_model

import torch
import torch.nn.functional as F


def flag(v):
    char = v.lower()[0]
    assert char == 't' or char == 'f', f"Invalid value: {v} - it should start with either 't' or 'f'"
    return char == 't'


def setup_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Bloom Evaluation script for HPU')
    parser.add_argument('--device', '-d', type=str, choices=['cpu', 'cuda', 'hpu'], help='Device to run', default='hpu')
    parser.add_argument('--model', '-m', type=str, choices=['bloom-560m', 'bloom-1b7', 'bloom-3b', 'bloom-7b1', 'bloom'], help='Model', default='bloom-7b1')
    parser.add_argument('--weights', type=str, help="Weight dir for all pretrained models", required=True)
    parser.add_argument('--dtype', '-dt', type=str, choices=['fp32', 'fp16', 'bf16'], help='Precision to use', default='fp32')
    parser.add_argument('--buckets', type=int, nargs='+', help="Input length buckets to use with static_shapes", default=[16, 32, 128, 256])
    parser.add_argument('--batch_size', '-bs', type=int, help="Number of queries per batch", default=4)

    parser.add_argument('--seed', type=int, help="random seed to use")
    parser.add_argument('--vanilla_model', action='store_true', help="Use default BloomModel impl from transformers lib")
    parser.add_argument('--kernel_inject', action='store_true', help="Enable replace_with_kernel_inject mode in DeepSpeed")
    parser.add_argument('--local_rank', type=int, help="Local rank used by DeepSpeed", default=0)
    parser.add_argument('--verbose_workers', action='store_true', help="Enable output from non-master workers")
    parser.add_argument('--options', type=str, help="Coma-seperated list of options used in generation. For more details run with --help_options")
    parser.add_argument('--dummy_output', action='store_true', help="Use dummy output instead of running the model. For testing purpuses only")

    parser.add_argument('--output_file', '-o', type=str, help="Output file with end results and runtime parameters", required=True)
    parser.add_argument('--tasks', type=str, nargs='+', help='Tasks to run', default=['hellaswag', 'lambada_openai', 'piqa', 'winogrande'])

    parser.add_argument('--config', type=str, help="Path to model config file. Implies running with uninitialized weights")
    parser.add_argument('--quantization_file', '-qf', type=str, help="Read quantization configuration from a file")

    parser.add_argument('--no_split_lm_head', action='store_true', help="Don't split lm_head when run under DeepSpeed")

    return parser


class HabanaModelAdapter(lm_eval.base.BaseLM):
    def __init__(self, tokenizer, model, args, options):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self._batch_size = args.batch_size
        self.buckets = list(sorted(args.buckets))
        self.options = options
        self.dummy_output = args.dummy_output
        self._device = args.device

    @property
    def eot_token_id(self):
        return self.model.config.eos_token_id

    @property
    def max_length(self):
        return self.buckets[-1]

    @property
    def max_gen_toks(self):
        raise NotImplementedError()

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        # We need to do padding ourselves, otherwise we'll end up with recompilations
        # Returning 'cpu' to keep tensors on CPU in lm_eval code
        return 'cpu'

    def tok_encode(self, string):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_generate(self, context, max_length, eos_token_id):
        raise NotImplementedError()

    def find_bucket(self, length):
        return [b for b in self.buckets if b >= length][0]

    def _model_call(self, inps):
        if self.dummy_output:
            sh = inps.shape
            logits = torch.zeros((sh[0], sh[1], self.model.config.vocab_size), dtype=torch.float32, device=inps.device)
        else:
            seq_length = inps.shape[-1]
            bucket_length = self.find_bucket(seq_length)
            padding_length = bucket_length - seq_length
            if self.options.static_shapes:
                inps = F.pad(inps, (0, padding_length), value=self.model.config.pad_token_id)
            logits = self.model(inps.to(self._device))['logits'].cpu()
            if self.options.static_shapes and padding_length > 0:
                logits = logits[:, :-padding_length, :]
            logits = logits.to(torch.float32)
        return logits


def main():
    parser = setup_parser()
    args = parser.parse_args()

    model, tokenizer, options = initialize_model(args)
    lm_tasks = lm_eval.tasks.get_task_dict(args.tasks)
    lm = HabanaModelAdapter(tokenizer, model, args, options)

    eval_start = time.perf_counter()
    results = lm_eval.evaluator.evaluate(lm, lm_tasks)
    eval_end = time.perf_counter()

    results['args'] = vars(args)
    results['duration'] = eval_end - eval_start

    if args.local_rank == 0:
        json.dump(results, open(args.output_file, 'w'), indent=2)
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
