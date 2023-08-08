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

import argparse
import multiprocessing
from datetime import datetime
from pathlib import Path
import os
import copy
import numpy as np
import torch  # pytype: disable=import-error
import pickle

def save_numpy(optim_state, lyr_name, saved_dir):
    for opt_key, opt_val in optim_state["state"].items():
        np.save((saved_dir / F"{lyr_name}.{opt_key}.npy").as_posix(), opt_val.float().cpu().numpy().astype(np.float32))
    np.save((saved_dir / F"{lyr_name}.fp32_from_fp16_params.npy").as_posix(), optim_state["fp32_from_fp16_params"].float().cpu().numpy().astype(np.float32))
    with open((saved_dir / F"{lyr_name}.param.pickle").as_posix(), 'wb') as handle:
        pickle.dump(optim_state["param_groups"], handle, protocol=pickle.HIGHEST_PROTOCOL)


# This tool is used to support the new megatron model trained by pipeline parallel + tensor parallel
def merge(
    key, pp_id, saved_dir,  model_args, optim_states, ckpt_ver, is_save_numpy
):
    #i, pipeline_para_rank, saved_dir, factor, key, model_args, transformer_model_list, ckpt_ver
    saved_dir = Path(saved_dir)
    if key.find("layers.") != -1:
        # key name: language_model.encoder.layers
        layer_index = (int)(key[30 : key.find(".", 30)])
        saved_key = key.replace(
            "layers.%d." % layer_index,
            "layers.%d."
            % (layer_index + pp_id * model_args.num_layers // model_args.pipeline_model_parallel_size),
        )
        abs_layer_index = "%d" % (layer_index + pp_id * model_args.num_layers // model_args.pipeline_model_parallel_size)
        abs_layer_dir = "layer_" + abs_layer_index
        saved_dir = saved_dir / abs_layer_dir
    else:
        saved_key = key
    #major_device = transformer_model_list[0][key].device
    #print(saved_key)
    optim_state = copy.deepcopy(optim_states[key])
    del optim_state['group_index']
    del optim_state['index_within_group']

    if (
        key.find("input_layernorm.weight") != -1
        or key.find("input_layernorm.bias") != -1
        or key.find("attention.dense.bias") != -1
        or key.find("post_attention_layernorm.weight") != -1
        or key.find("post_attention_layernorm.bias") != -1
        or key.find("mlp.dense_4h_to_h.bias") != -1
        or key.find("final_layernorm.weight") != -1
        or key.find("final_layernorm.bias") != -1
    ):
        # shared weights, only need to convert the weights from single tp instance
        for opt_key, opt_val in optim_state["state"].items():
            optim_state['state'][opt_key] = opt_val[0]
            #print(F"lyr_name: {key} key: {opt_key}: {optim_state['state'][opt_key].shape}")
        optim_state["fp32_from_fp16_params"] = optim_state["fp32_from_fp16_params"][0]
        #print(F"lyr_name: {key} key: fp32_from_fp16_params: {optim_state['fp32_from_fp16_params'].shape}")
    elif key.find("attention.dense.weight") != -1:
        state_key = list(optim_state["state"].keys())[0]
        head_num = model_args.num_attention_heads // model_args.tensor_model_parallel_size
        hidden_dim = int(optim_state["state"][state_key][0].shape[0])
        dim_per_head = int(optim_state["state"][state_key][0].shape[1] / head_num)
        for opt_key, opt_val in optim_state["state"].items():
            vals = []
            for k in range(model_args.tensor_model_parallel_size):
                val = opt_val[k]
                val = val.reshape(hidden_dim, head_num, dim_per_head)
                vals.append(val)
            optim_state['state'][opt_key] = torch.cat(vals, dim=1)
            #print(F"lyr_name: {key} key: {opt_key}: {optim_state['state'][opt_key].shape}")
        vals = []
        for k in range(model_args.tensor_model_parallel_size):
            val = optim_state["fp32_from_fp16_params"][k]
            val = val.reshape(hidden_dim, head_num, dim_per_head)
            vals.append(val)
        optim_state["fp32_from_fp16_params"] = torch.cat(vals, dim=1)
        #print(F"lyr_name: {key} key: fp32_from_fp16_params: {optim_state['fp32_from_fp16_params'].shape}")
    elif key.find("mlp.dense_4h_to_h.weight") != -1:
        for opt_key, opt_val in optim_state["state"].items():
            vals = []
            for k in range(model_args.tensor_model_parallel_size):
                vals.append(opt_val[k])
            optim_state['state'][opt_key] = torch.cat(vals, dim=-1)
            #print(F"lyr_name: {key} key: {opt_key}: {optim_state['state'][opt_key].shape}")
        vals = []
        for k in range(model_args.tensor_model_parallel_size):
            vals.append(optim_state["fp32_from_fp16_params"][k])
        optim_state["fp32_from_fp16_params"] = torch.cat(vals, dim=-1)
        #print(F"lyr_name: {key} key: fp32_from_fp16_params: {optim_state['fp32_from_fp16_params'].shape}")
    elif key.find("mlp.dense_h_to_4h.weight") != -1 or key.find("mlp.dense_h_to_4h.bias") != -1:
        for opt_key, opt_val in optim_state["state"].items():
            vals = []
            for k in range(model_args.tensor_model_parallel_size):
                vals.append(opt_val[k])
            optim_state['state'][opt_key] = torch.cat(vals, dim=0)
            #print(F"lyr_name: {key} key: {opt_key}: {optim_state['state'][opt_key].shape}")
        vals = []
        for k in range(model_args.tensor_model_parallel_size):
            vals.append(optim_state["fp32_from_fp16_params"][k])
        optim_state["fp32_from_fp16_params"] = torch.cat(vals, dim=0)
        #print(F"lyr_name: {key} key: fp32_from_fp16_params: {optim_state['fp32_from_fp16_params'].shape}")
    elif key.find("attention.query_key_value.bias") != -1:
        state_key = list(optim_state["state"].keys())[0]
        num_splits = 3
        head_num = model_args.num_attention_heads // model_args.tensor_model_parallel_size
        size_per_head = int(optim_state["state"][state_key][0].shape[0] / num_splits / head_num)
        for opt_key, opt_val in optim_state["state"].items():
            vals = []
            for k in range(model_args.tensor_model_parallel_size):
                val = opt_val[k]
                val = val.reshape(head_num, num_splits, size_per_head)
                vals.append(val)
            optim_state['state'][opt_key] = torch.cat(vals, dim=0)
            #print(F"lyr_name: {key} key: {opt_key}: {optim_state['state'][opt_key].shape}")
        vals = []
        for k in range(model_args.tensor_model_parallel_size):
            val = optim_state["fp32_from_fp16_params"][k]
            val = val.reshape(head_num, num_splits, size_per_head)
            vals.append(val)
        optim_state["fp32_from_fp16_params"] = torch.cat(vals, dim=0)
        #print(F"lyr_name: {key} key: fp32_from_fp16_params: {optim_state['fp32_from_fp16_params'].shape}")
    elif key.find("attention.query_key_value.weight") != -1:
        state_key = list(optim_state["state"].keys())[0]
        num_splits = 3
        hidden_dim = int(optim_state["state"][state_key][0].shape[1])
        head_num = model_args.num_attention_heads // model_args.tensor_model_parallel_size
        size_per_head = int(optim_state["state"][state_key][0].shape[0] / num_splits / head_num)
        for opt_key, opt_val in optim_state["state"].items():
            vals = []
            for k in range(model_args.tensor_model_parallel_size):
                val = opt_val[k]
                val = val.reshape(head_num, num_splits, size_per_head, hidden_dim)
                vals.append(val)
            optim_state['state'][opt_key] = torch.cat(vals, dim=0)
            #print(F"lyr_name: {key} key: {opt_key}: {optim_state['state'][opt_key].shape}")
        vals = []
        for k in range(model_args.tensor_model_parallel_size):
            val = optim_state["fp32_from_fp16_params"][k]
            val = val.reshape(head_num, num_splits, size_per_head, hidden_dim)
            vals.append(val)
        optim_state["fp32_from_fp16_params"] = torch.cat(vals, dim=0)
        #print(F"lyr_name: {key} key: fp32_from_fp16_params: {optim_state['fp32_from_fp16_params'].shape}")
    else:
        print(f"[ERROR] cannot find key '{key}'")
        exit(1)

    #print(F"{saved_key}: {tmp.shape}")
    if is_save_numpy:
        save_numpy(optim_state, saved_key, saved_dir)
    else:
        saved_path = saved_dir / f"{saved_key}.pt"
        torch.save(optim_state, saved_path)

def merge_checkpoint(args):
    saved_dir = Path(args.saved_dir) / "gpu" / "optimizer"
    saved_dir.mkdir(parents=True, exist_ok=True)

    prefix = Path(args.in_dir)
    ckpt_name = "model_optim_rng.pt"

    # load position_embedding from rank 0
    if (prefix / "mp_rank_00").is_dir():
        model_00 = torch.load((prefix / "mp_rank_00" / ckpt_name).as_posix())
    elif (prefix / "mp_rank_00_000").is_dir():
        model_00 = torch.load((prefix / "mp_rank_00_000" / ckpt_name).as_posix())
    else:
        print(f"[ERROR] Cannot find checkpoint in {prefix}.")
        exit(1)

    model_args = model_00["args"]
    with open((saved_dir / "args.txt").as_posix(), "w") as f:
        for k, v in vars(model_args).items():
            f.write(f"{k}:{v} \n")

    del model_00

    tp_size = model_args.tensor_model_parallel_size

    for i in range(model_args.num_layers):
        pp_id_dir = (saved_dir / f"layer_{i}").as_posix()
        os.makedirs(pp_id_dir, exist_ok=True)

    torch.multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy("file_system")
    pool = multiprocessing.Pool(args.pool)
    w_e_list = []
    w_e_h_list = []
    #for pp_id in [2]:
    for pp_id in range(model_args.pipeline_model_parallel_size):
        if model_args.pipeline_model_parallel_size == 1:
            layer_rank_num = ""
        else:
            layer_rank_num = f"_{pp_id:03d}"
        optim_states = {}
        for tp_id in range(tp_size):
            #if tp_id == 0:
            print(F"Loading ckpt file from: mp_rank_{tp_id:02d}{layer_rank_num}")
            m = torch.load((prefix / f"mp_rank_{tp_id:02d}{layer_rank_num}" / ckpt_name).as_posix(), map_location="cpu")
            #m["model"]["language_model"]["encoder"] = {key: value for key, value in m["model"]["language_model"]["encoder"].items() if ("attention.dense.weight" in key) or ("mlp.dense_4h_to_h.weight" in key)}
            #print(m["model"]["language_model"]["encoder"].keys())
            target_optim_map_orig = m['optimizer_model_map']
            target_optim_map = copy.deepcopy(target_optim_map_orig)
            substr = "module.module."
            for key, value in target_optim_map.items():
                if value.startswith(substr):
                    target_optim_map[key] = value[len(substr):]
            #del target_optim_map_orig
            #for key, value in m["optimizer_model_map"].items():
            for key, value in target_optim_map.items():
                if value in optim_states:
                    for opt_key, opt_val in m["optimizer"]["optimizer"]["state"][key].items():
                        optim_states[value]["state"][opt_key].append(opt_val)
                    group_index = optim_states[value]["group_index"]
                    index_within_group = optim_states[value]["index_within_group"]
                    optim_states[value]["fp32_from_fp16_params"].append(m["optimizer"]["fp32_from_fp16_params"][group_index][index_within_group])
                else:
                    optim_states[value] = {}
                    optim_states[value]["state"] = {}
                    for opt_key, opt_val in m["optimizer"]["optimizer"]["state"][key].items():
                        optim_states[value]["state"][opt_key] = []
                        optim_states[value]["state"][opt_key].append(opt_val)
                    # Find index param group
                    group_index = 0
                    index_within_group = 0
                    for index, group in enumerate(m["optimizer"]["optimizer"]["param_groups"]):
                        if key in group["params"]:
                            group_index = index
                            index_within_group = group["params"].index(key)
                            optim_states[value]["group_index"] = group_index
                            optim_states[value]["index_within_group"] = index_within_group
                            optim_states[value]["param_groups"] = copy.deepcopy(group)
                            if "params" in optim_states[value]["param_groups"]:
                                del optim_states[value]["param_groups"]["params"]
                            break
                    if "group_index" not in optim_states[value]:
                        print(F"couldn't find index for layer: {value}")
                        exit(1)
                    optim_states[value]["fp32_from_fp16_params"] = []
                    optim_states[value]["fp32_from_fp16_params"].append(m["optimizer"]["fp32_from_fp16_params"][group_index][index_within_group])

        if pp_id == 0:
            lyr_name = 'language_model.embedding.word_embeddings.weight'
            optim_state = copy.deepcopy(optim_states[lyr_name])
            for opt_key, opt_val in optim_state["state"].items():
                optim_state['state'][opt_key] = torch.cat(opt_val, dim=0)
                #print(F"lyr_name: {lyr_name} key: {opt_key}: {optim_state['state'][opt_key].shape}")
            optim_state["fp32_from_fp16_params"] = torch.cat(optim_state["fp32_from_fp16_params"], dim=0)
            #print(F"lyr_name: {lyr_name} key: fp32_from_fp16_params: {optim_state['fp32_from_fp16_params'].shape}")
            del optim_state['group_index']
            del optim_state['index_within_group']
            if args.save_numpy:
                 save_numpy(optim_state, lyr_name, saved_dir)
            else:
                torch.save(optim_state, (saved_dir / F"{lyr_name}.pt").as_posix())
            del optim_states[lyr_name]

            lyr_name = 'language_model.embedding.position_embeddings.weight'
            optim_state = copy.deepcopy(optim_states[lyr_name])
            for opt_key, opt_val in optim_state["state"].items():
                optim_state['state'][opt_key] = opt_val[0]
                #print(F"lyr_name: {lyr_name} key: {opt_key}: {optim_state['state'][opt_key].shape}")
            optim_state["fp32_from_fp16_params"] = optim_state["fp32_from_fp16_params"][0]
            #print(F"lyr_name: {lyr_name} key: fp32_from_fp16_params: {optim_state['fp32_from_fp16_params'].shape}")
            del optim_state['group_index']
            del optim_state['index_within_group']
            if args.save_numpy:
                 save_numpy(optim_state, lyr_name, saved_dir)
            else:
                torch.save(optim_state, (saved_dir / F"{lyr_name}.pt").as_posix())
            del optim_states[lyr_name]

        if pp_id == (model_args.pipeline_model_parallel_size - 1) and model_args.pipeline_model_parallel_size > 1:
            lyr_name = 'word_embeddings.weight'
            optim_state = copy.deepcopy(optim_states[lyr_name])
            for opt_key, opt_val in optim_state["state"].items():
                optim_state['state'][opt_key] = torch.cat(opt_val, dim=0)
                #print(F"lyr_name: {lyr_name} key: {opt_key}: {optim_state['state'][opt_key].shape}")
            optim_state["fp32_from_fp16_params"] = torch.cat(optim_state["fp32_from_fp16_params"], dim=0)
            #print(F"lyr_name: {lyr_name} key: fp32_from_fp16_params: {optim_state['fp32_from_fp16_params'].shape}")
            del optim_state['group_index']
            del optim_state['index_within_group']
            if args.save_numpy:
                save_numpy(optim_state, lyr_name, saved_dir)
            else:
                torch.save(optim_state, (saved_dir / F"{lyr_name}.pt").as_posix())
            del optim_states[lyr_name]

        pool.starmap(
            merge,
            [
                (
                    k,
                    pp_id,
                    saved_dir,
                    model_args,
                    optim_states,
                    m["checkpoint_version"],
                    args.save_numpy
                )
                for (k, _) in optim_states.items()
            ],
        )

    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-saved_dir", "-o", type=str, help="output directory for saving converted checkpoints", required=True)
    parser.add_argument("-in_dir", "-i", type=str, help="input checkpoint directory path", required=True)
    parser.add_argument("-save_numpy", "-npy", action='store_true', help="save output as numpy array", default=False)
    parser.add_argument("-pool", "-pl", type=int, help="Process pool", default=4)
    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")

    #Assij start
    #tp00_pp00 = torch.load('/workdisk/ajakoby/SKY/GPT3_runs/MLPERF1/checkpoints/gpt3_175B_nl2_D1T4P2/iter_0000050/mp_rank_00_000/model_optim_rng.pt',map_location=torch.device('cpu'))
    #tp01_pp00 = torch.load('/workdisk/ajakoby/SKY/GPT3_runs/MLPERF1/checkpoints/gpt3_175B_nl2_D1T4P2/iter_0000050/mp_rank_01_000/model_optim_rng.pt',map_location=torch.device('cpu'))
    #tp02_pp00 = torch.load('/workdisk/ajakoby/SKY/GPT3_runs/MLPERF1/checkpoints/gpt3_175B_nl2_D1T4P2/iter_0000050/mp_rank_02_000/model_optim_rng.pt',map_location=torch.device('cpu'))
    #tp03_pp00 = torch.load('/workdisk/ajakoby/SKY/GPT3_runs/MLPERF1/checkpoints/gpt3_175B_nl2_D1T4P2/iter_0000050/mp_rank_03_000/model_optim_rng.pt',map_location=torch.device('cpu'))

    #for index in [1,9,6,7,10,11,13]:
    #    print('index {}'.format(index))
    #    assert torch.sum(torch.abs(tp00_pp00['optimizer']['optimizer']['state'][index]['exp_avg']- tp01_pp00['optimizer']['optimizer']['state'][index]['exp_avg'])) ==0, 'no match'
    #    assert torch.sum(torch.abs(tp02_pp00['optimizer']['optimizer']['state'][index]['exp_avg']- tp03_pp00['optimizer']['optimizer']['state'][index]['exp_avg'])) ==0 , 'no match'
    #    assert torch.sum(torch.abs(tp00_pp00['optimizer']['optimizer']['state'][index]['exp_avg']- tp02_pp00['optimizer']['optimizer']['state'][index]['exp_avg'])) == 0, 'no match'

    #    assert torch.sum(torch.abs(tp00_pp00['optimizer']['optimizer']['state'][index]['exp_avg_sq'] - tp01_pp00['optimizer']['optimizer']['state'][index]['exp_avg_sq'])) == 0, 'no match'
    #    assert torch.sum(torch.abs(tp02_pp00['optimizer']['optimizer']['state'][index]['exp_avg_sq'] - tp03_pp00['optimizer']['optimizer']['state'][index]['exp_avg_sq'])) == 0, 'no match'
    #    assert torch.sum(torch.abs(tp00_pp00['optimizer']['optimizer']['state'][index]['exp_avg_sq'] - tp02_pp00['optimizer']['optimizer']['state'][index]['exp_avg_sq'])) == 0, 'no match'

    #ds
    #ds_exp_av_sq = torch.load('/workdisk/ajakoby/SKY/GPT3_runs/DS/checkpoints/gpt3_175B_nl2_D1T4P2/ds_universal_global_step50_ds_script/global_step50/zero/6.weight/exp_avg_sq.pt',map_location=torch.device('cpu'))
    #mlperf_ds_exp_avg_sq = torch.load('/workdisk/ajakoby/SKY/GPT3_runs/MLPERF1/checkpoints/gpt3_175B_nl2_D1T4P2/ds_universal_iter_0000050/50/zero/6.weight/exp_avg_sq.pt',map_location=torch.device('cpu'))
    #mlperf_merged_avg_sq = torch.load('//workdisk/ajakoby/SKY/GPT3_runs/MLPERF1/checkpoints/gpt3_175B_nl2_D1T4P2/NVIDIA_optimizer_merged_format_iter_0000050/gpu/optimizer/language_model.encoder.final_layernorm.weight.pt',map_location=torch.device('cpu'))
    #NV_merged = torch.load('//workdisk/ajakoby/SKY/GPT3_runs/MLPERF1/checkpoints/gpt3_175B_nl2_D1T4P2/NVIDIA_optimizer_merged_format_iter_0000050/gpu/optimizer/layer_0/language_model.encoder.layers.0.mlp.dense_4h_to_h.weight.pt',map_location=torch.device('cpu'))
    #ds_universal = torch.load('/workdisk/ajakoby/SKY/GPT3_runs/MLPERF1/checkpoints/gpt3_175B_nl2_D1T4P2/ds_universal_iter_0000050/50/zero/3.mlp.dense_4h_to_h.weight/exp_avg.pt',map_location=torch.device('cpu'))

    #ds_pp0_mp00 = torch.load('/workdisk/ajakoby/SKY/GPT3_runs/DS/checkpoints/gpt3_175B_nl2_D1T4P2/global_step1/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt',map_location=torch.device('cpu'))
    #ds_layer1_model00 = torch.load('/workdisk/ajakoby/SKY/GPT3_runs/DS/checkpoints/gpt3_175B_nl2_D1T4P2/global_step1/layer_01-model_00-model_states.pt',map_location=torch.device('cpu'),)
    #tp00_pp01 = torch.load('/workdisk/ajakoby/SKY/GPT3_runs/MLPERF1/checkpoints/gpt3_175B_nl2_D1T4P2/iter_0000050/mp_rank_00_001/model_optim_rng.pt', map_location=torch.device('cpu'))
    #tp01_pp01 = torch.load('/workdisk/ajakoby/SKY/GPT3_runs/MLPERF1/checkpoints/gpt3_175B_nl2_D1T4P2/iter_0000050/mp_rank_01_001/model_optim_rng.pt', map_location=torch.device('cpu'))
    #tp02_pp01 = torch.load('/workdisk/ajakoby/SKY/GPT3_runs/MLPERF1/checkpoints/gpt3_175B_nl2_D1T4P2/iter_0000050/mp_rank_02_001/model_optim_rng.pt', map_location=torch.device('cpu'))
    #tp03_pp01 = torch.load('/workdisk/ajakoby/SKY/GPT3_runs/MLPERF1/checkpoints/gpt3_175B_nl2_D1T4P2/iter_0000050/mp_rank_03_001/model_optim_rng.pt', map_location=torch.device('cpu'))

    #for index in [5, 6, 8, 9, 10, 12, 13, 14]:
    #    print('index {}'.format(index))
    #    assert torch.sum(torch.abs(tp00_pp01['optimizer']['optimizer']['state'][index]['exp_avg'] - tp01_pp01['optimizer']['optimizer']['state'][index]['exp_avg'])) == 0, 'no match'
    #    assert torch.sum(torch.abs(tp02_pp01['optimizer']['optimizer']['state'][index]['exp_avg'] - tp03_pp01['optimizer']['optimizer']['state'][index]['exp_avg'])) == 0, 'no match'
    #    assert torch.sum(torch.abs(tp00_pp01['optimizer']['optimizer']['state'][index]['exp_avg'] - tp02_pp01['optimizer']['optimizer']['state'][index]['exp_avg'])) == 0, 'no match'

    #    assert torch.sum(torch.abs(tp00_pp01['optimizer']['optimizer']['state'][index]['exp_avg_sq'] - tp01_pp01['optimizer']['optimizer']['state'][index]['exp_avg_sq'])) == 0, 'no match'
    #    assert torch.sum(torch.abs(tp02_pp01['optimizer']['optimizer']['state'][index]['exp_avg_sq'] - tp03_pp01['optimizer']['optimizer']['state'][index]['exp_avg_sq'])) == 0, 'no match'
    #    assert torch.sum(torch.abs(tp00_pp01['optimizer']['optimizer']['state'][index]['exp_avg_sq'] - tp02_pp01['optimizer']['optimizer']['state'][index]['exp_avg_sq'])) == 0, 'no match'

    #Assij end
    start_time = datetime.now()
    merge_checkpoint(args)
    stop_time = datetime.now()
    run_time = stop_time - start_time
    print(f"[INFO] Spent {run_time} (h:m:s) to convert the model")






