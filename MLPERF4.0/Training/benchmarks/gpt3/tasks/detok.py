import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from tqdm import tqdm
from megatron import get_args
from megatron import get_tokenizer
from megatron.initialize import initialize_megatron
from pretrain_gpt import train_valid_test_datasets_provider
from megatron.training import build_train_valid_test_data_iterators


def get_detok_args(parser):
    group = parser.add_argument_group(title='detokenizer')
    group.add_argument('--detokenizer_output',
                       type=str,
                       required=True,
                       help='detokenizer output path')
    return parser


def process_split(split, dataset, out_path):
    print(f'Processing {split}')
    tokenizer = get_tokenizer()

    full_text = []
    for batch in tqdm(dataset, total=len(dataset)):
        tokens = batch['text'].reshape(-1).tolist()
        text = tokenizer.detokenize(tokens)
        full_text.append(text)

    out_name = os.path.join(out_path, f'{split}.text')
    print(f'Writing to {out_name}')
    with open(out_name, 'w') as f:
        f.writelines(full_text)


def main():

    # below arguments are to force the full dataset according to the
    # train/valid/test split based on args.split
    forced_args = {
        "micro_batch_size": 1,
        "train_samples": None,
        "train_iters": 1,
        "eval_iters": 1,
        "eval_interval": 2,
        "use_seq_len_plus_one_tokens": False
    }

    initialize_megatron(extra_args_provider=get_detok_args, args_defaults=forced_args)
    torch.distributed.barrier()

    # after parsing, we have to force again the required args
    args = get_args()
    for name, value in forced_args.items():
        setattr(args, name, value)

    # create train/valid/test split based on args.split
    args.iteration = 0
    train_iter, valid_iter, test_iter = build_train_valid_test_data_iterators(
        train_valid_test_datasets_provider)

    os.makedirs(args.detokenizer_output, exist_ok=True)
    process_split('test', test_iter._dataset, args.detokenizer_output)
    process_split('valid', valid_iter._dataset, args.detokenizer_output)
    process_split('train', train_iter._dataset, args.detokenizer_output)


if __name__ == '__main__':
    main()
