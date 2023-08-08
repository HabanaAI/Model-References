#!/usr/bin/env python

from collections import OrderedDict
from functools import partial
import argparse
import glob
import itertools
import multiprocessing
import os
import re
import shutil
import torch
import tqdm

from deepspeed.checkpoint import DeepSpeedCheckpoint

MODEL_KEY = 'model'
ARGS_KEY = 'args'
LANGUAGE_MODEL_KEY = 'language_model'
EMBEDDING_KEY = 'embedding'
ENCODER_KEY = 'encoder'
WORD_EMBEDDINGS_FOR_HEAD_KEY = 'word_embeddings_for_head'
WORD_EMBEDDINGS_KEY = 'word_embeddings'
FINAL_LAYER_NORM_KEY = 'final_layernorm'
CHECKPOINT_VERSION_KEY = 'checkpoint_version'
CHECKPOINT_VERSION_VALUE = 3.0
ITERATION_KEY = 'iteration'
ORIGINAL_VOCAB_SIZE = 'original_vocab_size'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_folder',
        type=str,
        help='Input DeepSpeed Checkpoint folder')
    parser.add_argument(
        '--output_folder',
        type=str,
        help='Output Megatron checkpoint folder')
    parser.add_argument(
        '--num_extract_workers',
        default=4,
        type=int,
        help='How many parallel processes to extract zero shards')
    parser.add_argument(
        '--num_merge_workers',
        default=2,
        type=int,
        help='How many parallel processes to merge tp slices '
             '(more memory intensive, use much fewer than --num_extract_workers))')

    args = parser.parse_args()
    print(f'args = {args}')
    return args


def _convert_ds_transformer_state(sd_list):
    new_sd = OrderedDict()
    for i, sd in enumerate(sd_list):
        for key, value in sd.items():
            new_key = f'layers.{i}.{key}'
            new_sd[new_key] = value

    return new_sd


def _create_megatron_dict():
    language_model_dict = {EMBEDDING_KEY: {}, ENCODER_KEY: {}}
    megatron_dict = {
        MODEL_KEY: {
            LANGUAGE_MODEL_KEY: language_model_dict
        },
        CHECKPOINT_VERSION_KEY: CHECKPOINT_VERSION_VALUE
    }
    return megatron_dict


def _save_checkpoint(file_path, chkpt_sd):
    ckp_dir, _ = os.path.split(file_path)
    os.makedirs(ckp_dir, exist_ok=True)
    torch.save(chkpt_sd, file_path)


def extract_zero_shards(out_path, ds_checkpoint, indices_3d):
    pp_index, tp_index, dp_index = indices_3d
    sd = ds_checkpoint.get_zero_checkpoint_state(
        pp_index=pp_index,
        tp_index=tp_index,
        dp_index=dp_index)

    optim_sd = sd["optimizer_state_dict"]
    param_slice_mappings = optim_sd["param_slice_mappings"]

    # dict
    state_groups = optim_sd["base_optimizer_state"]["state"]

    # list
    fp32_groups = optim_sd["single_partition_of_fp32_groups"]
    param_groups_cnt = len(state_groups)

    for param_group_id in range(param_groups_cnt):
        flat_state = dict(
            exp_avg=state_groups[param_group_id]["exp_avg"],
            exp_avg_sq=state_groups[param_group_id]["exp_avg_sq"],
            fp32=fp32_groups[param_group_id],
        )

        for name, fragment_mapping in param_slice_mappings[param_group_id].items():
            if "tied_modules.embed" in name and pp_index > 0:
                # Skip word_embeddings.weight that is replicated in first and last pp stages
                # Skip position_embeddings.weight that is only used in first pp stage
                continue

            for state_key in flat_state.keys():
                dump_param_fragment(out_path, tp_index, dp_index, state_key,
                                    flat_state[state_key], name,
                                    fragment_mapping.start,
                                    fragment_mapping.numel)


def dump_param_fragment(out_path, tp_index, dp_index, state_name,
                        state_flat_tensor, param_name, offset, numel):
    param_base_path = os.path.join(out_path, param_name, str(tp_index))
    os.makedirs(param_base_path, exist_ok=True)

    counter = f"{dp_index:0>2d}"
    path = os.path.join(param_base_path, f"{state_name}.{counter}")

    # clone to force tensor storage to ignore views
    t = state_flat_tensor.narrow(0, offset, numel).clone()
    _save_checkpoint(path, t)


def _merge_zero_shards(param_base_path, state, tp_degree, slice_shape):
    slices = []
    for tp_index in range(tp_degree):
        prefix_path = os.path.join(param_base_path, str(tp_index), f"{state}")
        paths = sorted(list(glob.glob(f"{prefix_path}.0*")))
        shards = [torch.load(p) for p in paths]
        param_slice = torch.cat(shards, dim=0).reshape(slice_shape)
        slices.append(param_slice)

    return slices


def _strip_vocab_padding(ds_checkpoint, padded_vocab_tensor):
    checkpoint_info = ds_checkpoint.get_checkpoint_info()
    return padded_vocab_tensor.narrow(0, 0, checkpoint_info[ORIGINAL_VOCAB_SIZE])


WEIGHTS_TO_AVERAGE_PATTERNS = [
    r"tied_modules.embed.word_embeddings.norm.weight",
    r"tied_modules.embed.word_embeddings.norm.bias",
    r"tied_modules.embed.position_embeddings.weight",
    r"\d+.input_layernorm.weight",
    r"\d+.input_layernorm.bias",
    r"\d+.post_attention_layernorm.weight",
    r"\d+.post_attention_layernorm.bias",
    r"\d+.self_attention.dense.bias",
    r"\d+.attention.dense.bias",
    r"\d+.mlp.dense_4h_to_h.bias",
    r"\d+.weight",
    r"\d+.bias",
]

WEIGHTS_WITH_ROW_PARALLELISM_CONTAIN = [
    "dense_4h_to_h.weight",
    "self_attention.dense.weight",
    "attention.dense.weight",
]


def _get_vocab_divisibility_padding_tensor(ds_checkpoint, padded_vocab_tensor):
    checkpoint_info = ds_checkpoint.get_checkpoint_info()
    if checkpoint_info and padded_vocab_tensor.shape[0] > checkpoint_info[ORIGINAL_VOCAB_SIZE]:
        return padded_vocab_tensor[-1]
    else:
        return torch.zeros(padded_vocab_tensor.shape[1])


def _all_same_tensor(arr):
    assert len(arr) > 0
    if len(arr) == 1:
        return True
    res = all([x.eq(arr[0]).all().item() for x in arr[1:]])
    return res


def merge_tp_slices(ds_checkpoint, out_path, slice_dir, tp_degree, name_and_shape):
    name, shape = name_and_shape
    slice_base_path = os.path.join(slice_dir, name)
    param_base_path = os.path.join(out_path, name)

    for state in ("fp32", "exp_avg", "exp_avg_sq"):
        slices = _merge_zero_shards(slice_base_path, state, tp_degree, shape)
        final_path = os.path.join(param_base_path, f"{state}.pt")

        ckpt_dict = {}
        if any(re.match(pattern, name) for pattern in WEIGHTS_TO_AVERAGE_PATTERNS):
            assert _all_same_tensor(slices), f'Checkpoint misalignment detected for parameter: {name}'
            param = slices[0]
        else:
            cat_dim = 1 if any(text in name for text in WEIGHTS_WITH_ROW_PARALLELISM_CONTAIN) else 0
            param = torch.cat(slices, dim=cat_dim)
            ckpt_dict['cat_dim'] = cat_dim

        if "word_embeddings.weight" in name:
            # strip padding
            # param = _strip_vocab_padding(ds_checkpoint, param)
            ckpt_dict['vocab_divisibility_padding_tensor'] = \
                _get_vocab_divisibility_padding_tensor(ds_checkpoint, param)

        ckpt_dict['param'] = param
        _save_checkpoint(final_path, ckpt_dict)


def _get_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def _do_parallel_work(do_work, work_chunks, num_workers):
    pool = multiprocessing.Pool(num_workers)
    for batch in tqdm.tqdm(work_chunks):
        pool.map(do_work, batch)
    pool.close()
    pool.join()


def _extract_zero_shard_files(args, ds_checkpoint, temp_dir):
    _3d_range_list = list(itertools.product(range(ds_checkpoint.pp_degree),
                                            range(ds_checkpoint.tp_degree),
                                            range(ds_checkpoint.dp_degree)))
    work_chunks = list(_get_chunks(_3d_range_list, args.num_extract_workers))

    do_work = partial(extract_zero_shards, temp_dir, ds_checkpoint)
    _do_parallel_work(do_work, work_chunks, args.num_extract_workers)


def _merge_tp_slice_files(args, ds_checkpoint, slice_shapes, temp_dir):
    work_chunks = list(_get_chunks(list(slice_shapes.items()), args.num_merge_workers))
    zero_output_folder = os.path.join(args.output_folder, "zero")
    do_work = partial(merge_tp_slices, ds_checkpoint, zero_output_folder, temp_dir, ds_checkpoint.tp_degree)
    _do_parallel_work(do_work, work_chunks, args.num_merge_workers)


def main():
    print(f'Convert DeepSpeed Checkpoint to Universal Checkpoint')

    args = parse_arguments()
    print(
        f'Converting DeepSpeed checkpoint in {args.input_folder} '
        f'to Universal checkpoint in {args.output_folder}'
    )

    ds_checkpoint = DeepSpeedCheckpoint(args.input_folder)

    slice_shapes = []
    for mp_rank_file in ds_checkpoint.mp_rank_files:
        mp_sd = torch.load(mp_rank_file, map_location=torch.device('cpu'))
        slice_shapes += mp_sd["param_shapes"]

    # fix back to normal flat dict, merge duplicates for tp>1
    slice_shapes = dict((k, v) for d in slice_shapes for k, v in d.items())
    temp_dir = os.path.join(args.output_folder, 'tmp')

    print('*** 1. Extracting ZeRO fragments')
    _extract_zero_shard_files(args, ds_checkpoint, temp_dir)

    print('*** 2. Merging slices')
    _merge_tp_slice_files(args, ds_checkpoint, slice_shapes, temp_dir)

    shutil.rmtree(temp_dir, ignore_errors=True)

    # Copy mp* files into output folder
    for f in glob.glob(os.path.join(args.input_folder, 'mp*')):
        shutil.copy2(f, args.output_folder)

    # Update latest to output folder
    checkpoint_root_folder, step_folder = os.path.split(args.output_folder)
    latest_file = os.path.join(checkpoint_root_folder, 'latest_universal')
    with open(latest_file, "w") as f:
        f.write(step_folder)

    print('*** Done!')


if __name__ == "__main__":
    main()
