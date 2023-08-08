# Copyright (c) 2021,2022, Habana Labs Ltd.  All rights reserved.
# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
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
###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
#
# Added params to reflect the checkpoint names
#
###############################################################################
"""Script to load layer(s) of the LLM checkpoint using TensorStore.
More details about TensorStore, please visit 
https://github.com/google/tensorstore .
"""

import argparse
import tensorstore as ts
import glob
import json
import torch
import os
import sys
import multiprocessing
from datetime import datetime

def get_numpy_array(filename):
    spec = {'driver': 'zarr', 'metadata_key': '.zarray', 'kvstore': {}}
    spec['kvstore'] = {
        'driver': 'file',
        'path': filename,
    }

    t = ts.open(ts.Spec(spec), open=True).result()
    t_v = t.read().result()
    return t_v

def get_torch_tensor(filename, dtype):
    array = get_numpy_array(filename)
    array_torch = torch.from_numpy(array)
    array_torch = array_torch.to(dtype)
    return array_torch

def get_layer_info(output_dir, lyr_num, nv_name):
    lyr_dir = os.path.join(output_dir, F"layer_{str(lyr_num)}")
    lyr_name = "language_model.encoder.layers."+str(lyr_num)+"."+nv_name
    return lyr_dir, lyr_name

def store_tensor(save_tensor, lyr_dir, lyr_name, params_dict):
    optim_state = {}
    optim_state["state"] = {}
    optim_state["state"]["exp_avg"] = save_tensor["m"]
    optim_state["state"]["exp_avg_sq"] = save_tensor["v"]
    optim_state["fp32_from_fp16_params"] = save_tensor["w"]
    if params_dict is not None:
        optim_state["param_groups"] = params_dict
    torch.save(optim_state, os.path.join(lyr_dir, lyr_name + ".pt"))

def copy_layers(args, nv_name, g_name, prefix, params_dict):

    array_torch = {}
    g_name_path = os.path.join(args.google_ckpts, prefix + ".m." + g_name)
    array_torch["m"] = get_torch_tensor(g_name_path, args.dtype)
    g_name_path = os.path.join(args.google_ckpts, prefix + ".v." + g_name)
    array_torch["v"] = get_torch_tensor(g_name_path, args.dtype)
    g_name_path = os.path.join(args.google_ckpts, "mdl_vars." + g_name)
    array_torch["w"] = get_torch_tensor(g_name_path, args.dtype)

    print(F"G Name: {g_name}, shape: {array_torch['m'].shape}", flush=True)
    save_tensor = {}
    if nv_name == "language_model.embedding.position_embeddings.weight":
        start_idx = 0
        end_idx = 2048
        for key in list(array_torch.keys()):
            save_tensor[key] = array_torch[key][start_idx: end_idx, :].contiguous().detach().clone()
        print(F"NV Name: {nv_name}, shape: {save_tensor['m'].shape}", flush=True)
        store_tensor(save_tensor, args.output_dir, nv_name, params_dict)
    elif nv_name == "language_model.embedding.word_embeddings.weight":
        for key in list(array_torch.keys()):
            save_tensor[key] = array_torch[key].transpose(0, 1).contiguous().detach().clone()
        print(F"NV Name: {nv_name}, shape: {save_tensor['m'].shape}", flush=True)
        store_tensor(save_tensor, args.output_dir, nv_name, params_dict)
        store_tensor(save_tensor, args.output_dir, "word_embeddings.weight", params_dict)
    else:
        for key in list(array_torch.keys()):
            save_tensor[key] = array_torch[key].detach().clone()
        print(F"NV Name: {nv_name}, shape: {save_tensor['m'].shape}", flush=True)
        store_tensor(save_tensor, args.output_dir, nv_name, params_dict)
    del save_tensor
    del array_torch

def split_encoder_layers(args, nv_name, g_name, prefix, params_dict):
    array_torch = {}
    g_name_path = os.path.join(args.google_ckpts, prefix + ".m." + g_name)
    array_torch["m"] = get_torch_tensor(g_name_path, args.dtype)
    g_name_path = os.path.join(args.google_ckpts, prefix + ".v." + g_name)
    array_torch["v"] = get_torch_tensor(g_name_path, args.dtype)
    g_name_path = os.path.join(args.google_ckpts, "mdl_vars." + g_name)
    array_torch["w"] = get_torch_tensor(g_name_path, args.dtype)
    print(F"G Name: {g_name}, shape: {array_torch['m'].shape}", flush=True)
    save_tensor = {}
    if (
        nv_name == "mlp.dense_4h_to_h.bias" 
        or nv_name == "post_attention_layernorm.bias" 
        or nv_name == "post_attention_layernorm.weight" 
        or nv_name == "input_layernorm.bias" 
        or nv_name == "input_layernorm.weight" 
        or nv_name == "self_attention.dense.bias" 
        or nv_name == "mlp.dense_h_to_4h.bias" 
        or nv_name == "self_attention.dense.weight"
    ):
        print(F"1st Check: {nv_name}")    
        for lyr_num in range(args.num_layers):
            print("layer_num=",lyr_num)
            lyr_dir, lyr_name = get_layer_info(args.output_dir, lyr_num, nv_name)
            for key in list(array_torch.keys()):
                save_tensor[key] = array_torch[key][lyr_num].contiguous().detach().clone()
            if lyr_num == (args.num_layers // 2):
                print(F"NV Name: {nv_name}, shape: {save_tensor['m'].shape}", flush=True)
            store_tensor(save_tensor, lyr_dir, lyr_name, params_dict)
            save_tensor = {}

    elif (
        nv_name == "mlp.dense_h_to_4h.weight" 
        or nv_name == "mlp.dense_4h_to_h.weight"
    ):
        print(F"2nd Check: {nv_name}")
        for lyr_num in range(args.num_layers):
            print("layer_num=",lyr_num)
            lyr_dir, lyr_name = get_layer_info(args.output_dir, lyr_num, nv_name)
            for key in list(array_torch.keys()):
                save_tensor[key] = array_torch[key][lyr_num].transpose(0, 1).contiguous().detach().clone()
                #save_tensor = save_tensor.transpose(0, 1).clone()
            if lyr_num == (args.num_layers // 2):
                print(F"NV Name: {nv_name}, shape: {save_tensor['v'].shape}", flush=True)
            store_tensor(save_tensor, lyr_dir, lyr_name, params_dict)
            save_tensor = {}
    elif nv_name == "self_attention.query_key_value.weight":
        print(F"3nd Check: {nv_name}")
        # nv shape [4608, 12288] => 4608 = 12 (heads) * 3 (qkv) * 128 (hidden_size / heads)
        # google shape [96, 3, 12288, 96, 128]
        for lyr_num in range(args.num_layers):
            print("layer_num=",lyr_num)
            lyr_dir, lyr_name = get_layer_info(args.output_dir, lyr_num, nv_name)
            for key in list(array_torch.keys()):
                save_tensor[key] = array_torch[key][lyr_num].permute(2, 0, 3, 1).contiguous().detach().clone()
                #save_tensor = save_tensor.permute(2, 0, 3, 1).contiguous().clone()
            if lyr_num == (args.num_layers // 2):
                print(F"NV Name: {nv_name}, shape: {save_tensor['w'].shape}", flush=True)
            store_tensor(save_tensor, lyr_dir, lyr_name, params_dict)
            save_tensor = {}
    elif nv_name == "self_attention.query_key_value.bias":
        print(F"4rd Check: {nv_name}")
        # nv shape [4608] => 4608 = 12 (heads) * 3 (qkv) * 128 (hidden_size / heads)
        # google shape [96, 3, 96, 128]
        for lyr_num in range(args.num_layers):
            print("layer_num=",lyr_num)
            lyr_dir, lyr_name = get_layer_info(args.output_dir, lyr_num, nv_name)
            for key in list(array_torch.keys()):
                save_tensor[key] = array_torch[key][lyr_num].permute(1, 0, 2).contiguous().detach().clone()
                #save_tensor = save_tensor.permute(1, 0, 2).contiguous().clone()
            if lyr_num == (args.num_layers // 2):
                print(F"NV Name: {nv_name}, shape: {save_tensor['m'].shape}", flush=True)
            store_tensor(save_tensor, lyr_dir, lyr_name, params_dict)
            save_tensor = {}
    else:
        print(F"Not a valid layer name: {nv_name}", flush=True)
        sys.exit()
    del array_torch


def arrange_google_ckpts(args, prefix1, prefix2):

    output_dir = args.output_dir
    num_layers = args.num_layers
    
    params_dict = None
    if args.params_file is not None:
        with open(args.params_file, 'r') as f: 
            params_dict = json.load(f)
    else:
        print(F"For Megatron-LM Optimizer to get the right optimizer params, provide params_file json", flush=True)

    if args.dtype == "bf16":
        args.dtype = torch.bfloat16
    else:
        args.dtype = torch.float

    for lyr_num in range(num_layers):
        pp_id_dir = os.path.join(output_dir, f"layer_{str(lyr_num)}")
        os.makedirs(pp_id_dir, exist_ok=True)
    
    #layers that are not part of encoder blocks.
    torch.multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy("file_system")


    nv_g_names_pairs = [
            ("language_model.embedding.word_embeddings.weight", "params.lm.softmax.logits_ffn.linear.w"),
            ("language_model.embedding.position_embeddings.weight", "params.lm.position_emb.emb_var"),
            ("language_model.encoder.final_layernorm.weight", "params.lm.final_ln.scale"),
            ("language_model.encoder.final_layernorm.bias", "params.lm.final_ln.bias"),
        ]
    pool = multiprocessing.Pool(args.pool)
    pool.starmap(
            copy_layers,
            [
                (
                    args,
                    nv_name,
                    g_name,
                    prefix1,
                    params_dict,
                )
                for (nv_name, g_name) in nv_g_names_pairs
            ],
        )
    pool.close()
    pool.join()



    nv_g_names_pairs1 = [
        ("mlp.dense_4h_to_h.bias", "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.bias.b"),
    ]

    pool = multiprocessing.Pool(args.pool)
    pool.starmap(
            split_encoder_layers,
            [
                (
                    args,
                    nv_name,
                    g_name,
                    prefix2,
                    params_dict,
                )
                for (nv_name, g_name) in nv_g_names_pairs1
            ],
        )
    pool.close()
    pool.join()

    nv_g_names_pairs2 = [
        ("post_attention_layernorm.bias", "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.bias"),
        ("post_attention_layernorm.weight", "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.scale"),
        ("input_layernorm.bias", "params.lm.transformer.repeat.sub.x_layers_0.layer_norm.bias"),
        ("input_layernorm.weight", "params.lm.transformer.repeat.sub.x_layers_0.layer_norm.scale"),
        ("self_attention.dense.bias", "params.lm.transformer.repeat.sub.x_layers_0.self_attention.post.b"),
    ]

    pool = multiprocessing.Pool(args.pool)
    pool.starmap(
            split_encoder_layers,
            [
                (
                    args,
                    nv_name,
                    g_name,
                    prefix2,
                    params_dict,
                )
                for (nv_name, g_name) in nv_g_names_pairs2
            ],
        )
    pool.close()
    pool.join()

    nv_g_names_pairs3 = [
        ("mlp.dense_h_to_4h.bias", "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.bias.b"),
    ]

    pool = multiprocessing.Pool(args.pool)
    pool.starmap(
        split_encoder_layers,
        [
            (
                args,
                nv_name,
                g_name,
                prefix2,
                params_dict,
            )
            for (nv_name, g_name) in nv_g_names_pairs3
        ],
    )
    pool.close()
    pool.join()

    nv_g_names_pairs4 = [
        ("mlp.dense_h_to_4h.weight", "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.linear.w"),
    ]

    pool = multiprocessing.Pool(args.pool)
    pool.starmap(
        split_encoder_layers,
        [
            (
                args,
                nv_name,
                g_name,
                prefix2,
                params_dict,
            )
            for (nv_name, g_name) in nv_g_names_pairs4
        ],
    )
    pool.close()
    pool.join()

    nv_g_names_pairs5 = [
        ("mlp.dense_4h_to_h.weight", "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.linear.w"),
        ("self_attention.dense.weight", "params.lm.transformer.repeat.sub.x_layers_0.self_attention.post.w"),
        ("self_attention.query_key_value.weight",
         "params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.w"),
        ("self_attention.query_key_value.bias",
         "params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.b"),
    ]

    pool = multiprocessing.Pool(args.pool)
    pool.starmap(
        split_encoder_layers,
        [
            (
                args,
                nv_name,
                g_name,
                prefix2,
                params_dict,
            )
            for (nv_name, g_name) in nv_g_names_pairs5
        ],
    )
    pool.close()
    pool.join()

    exit(0)

    nv_g_names_pairs = [
        ("mlp.dense_4h_to_h.bias", "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.bias.b"),
        ("post_attention_layernorm.bias", "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.bias"),
        ("post_attention_layernorm.weight", "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.scale"),
        ("input_layernorm.bias", "params.lm.transformer.repeat.sub.x_layers_0.layer_norm.bias"),
        ("input_layernorm.weight", "params.lm.transformer.repeat.sub.x_layers_0.layer_norm.scale"),
        ("self_attention.dense.bias", "params.lm.transformer.repeat.sub.x_layers_0.self_attention.post.b"),
        ("mlp.dense_h_to_4h.bias", "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.bias.b"),
        ("mlp.dense_h_to_4h.weight", "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.linear.w"),
        ("mlp.dense_4h_to_h.weight", "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.linear.w"),
        ("self_attention.dense.weight", "params.lm.transformer.repeat.sub.x_layers_0.self_attention.post.w"),
        ("self_attention.query_key_value.weight",
         "params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.w"),
        ("self_attention.query_key_value.bias",
         "params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.b"),
    ]

    pool = multiprocessing.Pool(args.pool)
    pool.starmap(
            split_encoder_layers,
            [
                (
                    args,
                    nv_name,
                    g_name,
                    prefix2,
                    params_dict,
                )
                for (nv_name, g_name) in nv_g_names_pairs
            ],
        )
    pool.close()
    pool.join()
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--google_ckpts', "-gckpt",
        type=str,
        default='/workspace/data/checkpoint_00001300',
        help='Google Checkpoint directory')
    parser.add_argument(
        '--output_dir', "-o",
        type=str,
        default='google_to_torch_output',
        help='Output directory')
    parser.add_argument(
        '--dtype', "-dt",
        type=str,
        default="float",
        help='datatype')
    parser.add_argument(
        '--num_layers', "-nl",
        type=int,
        default=96,
        help='number of encoder layers')
    parser.add_argument(
        '--params_file', "-pl",
        type=str,
        default=None,
        help='Json File for Param Groups')
    parser.add_argument(
        '--pool', "-p",
        type=int,
        default=4,
        help='parallel processes')

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")

    param1 = "opt_states_0.no_prefix_2"        #Assij
    param2 = "opt_states_0.p#96#i-1_2"


    start_time = datetime.now()
    arrange_google_ckpts(args, param1, param2)
    stop_time = datetime.now()
    run_time = stop_time - start_time
    print(f"[INFO] Spend {run_time} (h:m:s) to convert the model")




























