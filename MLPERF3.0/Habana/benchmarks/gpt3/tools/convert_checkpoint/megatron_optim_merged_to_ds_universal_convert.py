###############################################################################
# Copyright (c) 2023 Habana Labs Ltd.  All rights reserved.
###############################################################################
import argparse
import multiprocessing
from datetime import datetime
from pathlib import Path
import os
import copy
import numpy as np
import torch  # pytype: disable=import-error
import pickle
import glob
import re


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
def _get_vocab_divisibility_padding_tensor(padded_vocab_tensor):
    return padded_vocab_tensor[-1]

def _save_checkpoint(file_path, chkpt_sd):
    ckp_dir, _ = os.path.split(file_path)
    os.makedirs(ckp_dir, exist_ok=True)
    torch.save(chkpt_sd, file_path)

def tensor_convert(tensor_name_mapping, tensor_index):
    fp32_ckpt = {}
    exp_avg_ckpt = {}
    exp_avg_sq_ckpt = {}

    tensor_name = tensor_name_mapping[tensor_index]
    megatron_optimizer_states = torch.load(tensor_name[1])
    if 'self_attention.query_key_value' in tensor_name[1]:
        dim = megatron_optimizer_states['fp32_from_fp16_params'].size()[len(megatron_optimizer_states['fp32_from_fp16_params'].size())-1]
        fp32_ckpt['param'] = megatron_optimizer_states['fp32_from_fp16_params'].view(-1,dim)
        exp_avg_ckpt['param'] = megatron_optimizer_states['state']['exp_avg'].view(-1,dim)
        exp_avg_sq_ckpt['param'] = megatron_optimizer_states['state']['exp_avg_sq'].view(-1,dim)

        cat_dim = 0
        fp32_ckpt['cat_dim'] = cat_dim
        exp_avg_ckpt['cat_dim'] = cat_dim
        exp_avg_sq_ckpt['cat_dim'] = cat_dim
    else:
        fp32_ckpt['param'] = megatron_optimizer_states['fp32_from_fp16_params']
        exp_avg_ckpt['param'] = megatron_optimizer_states['state']['exp_avg']
        exp_avg_sq_ckpt['param'] = megatron_optimizer_states['state']['exp_avg_sq']

    ds_tensor_name = os.path.split(tensor_name[0])[-1]
    if not any(re.match(pattern, ds_tensor_name) for pattern in WEIGHTS_TO_AVERAGE_PATTERNS):
        cat_dim = 1 if any(text in ds_tensor_name for text in WEIGHTS_WITH_ROW_PARALLELISM_CONTAIN) else 0
        if '.bias' not in ds_tensor_name:
            fp32_ckpt['cat_dim'] = cat_dim
            exp_avg_ckpt['cat_dim'] = cat_dim
            exp_avg_sq_ckpt['cat_dim'] = cat_dim

    if 'word_embeddings.weight' in tensor_name[1]:
        fp32_ckpt['vocab_divisibility_padding_tensor'] = \
            _get_vocab_divisibility_padding_tensor(fp32_ckpt['param'])
        exp_avg_ckpt['vocab_divisibility_padding_tensor'] = \
            _get_vocab_divisibility_padding_tensor(exp_avg_ckpt['param'])
        exp_avg_sq_ckpt['vocab_divisibility_padding_tensor'] = \
            _get_vocab_divisibility_padding_tensor(exp_avg_sq_ckpt['param'])


    fp32_weight_file_path = os.path.join(tensor_name[0], 'fp32.pt')
    _save_checkpoint(fp32_weight_file_path, fp32_ckpt)

    exp_avg_file_path = os.path.join(tensor_name[0], 'exp_avg.pt')
    _save_checkpoint(exp_avg_file_path, exp_avg_ckpt)

    exp_avg_sq_file_path = os.path.join(tensor_name[0], 'exp_avg_sq.pt')
    _save_checkpoint(exp_avg_sq_file_path, exp_avg_sq_ckpt)

def mp_rank_files_info_adjustment(file,megatron_state_dict,same_config, ds_universal_checkpoints_path):
    ds_state_dict = torch.load(file, map_location=torch.device('cpu'))
    ds_state_dict['lr_scheduler']['num_steps'] = megatron_state_dict['opt_param_scheduler']['num_steps']
    ds_state_dict['lr_scheduler']['warmup_steps'] = megatron_state_dict['opt_param_scheduler']['warmup_steps']
    ds_state_dict['lr_scheduler']['decay_steps'] = megatron_state_dict['opt_param_scheduler']['decay_steps']
    ds_state_dict['iteration'] = megatron_state_dict['iteration']
    ds_state_dict['global_steps'] = megatron_state_dict['iteration']
    ds_state_dict['global_samples'] = megatron_state_dict['args'].consumed_train_samples
    ds_state_dict['tokens'] = megatron_state_dict['args'].consumed_train_samples* megatron_state_dict['args'].seq_length
    ds_state_dict['args'].consumed_train_samples = megatron_state_dict['args'].consumed_train_samples
    ds_state_dict['args'].consumed_valid_samples = megatron_state_dict['args'].consumed_valid_samples
    ds_state_dict['args'].consumed_train_tokens = ds_state_dict['tokens']

    # if both megatron-lm and megatron-deepspeed have the same TP, PP configuration, we copy the rng states from megatron-lm to megatron-deepspeed
    if same_config == 'True':
        ds_state_dict['random_rng_state'] = megatron_state_dict['rng_state'][0]['random_rng_state']
        ds_state_dict['np_rng_state'] = megatron_state_dict['rng_state'][0]['np_rng_state']
        ds_state_dict['torch_rng_state'] = megatron_state_dict['rng_state'][0]['torch_rng_state']
        ds_state_dict['cuda_rng_state'] = megatron_state_dict['rng_state'][0]['cuda_rng_state']
        ds_state_dict['rng_tracker_states'] = megatron_state_dict['rng_state'][0]['rng_tracker_states']

    file = os.path.join(ds_universal_checkpoints_path,os.path.split(file)[1])
    torch.save(ds_state_dict,file)


def mp_rank_files_info_adjustment_parallel_processing(ds_mp_rank_files_dir,ds_universal_checkpoints_path,megatron_lm_non_merged_input_dir, \
                                                       model_parallel_same_config,pp_index,tp_index,tp_rank):

     state_dict = torch.load(os.path.join(megatron_lm_non_merged_input_dir,
                                          'mp_rank_{:02d}_{:03d}'.format(
                                              tp_index,
                                              pp_index),
                                          'model_optim_rng.pt'), map_location=torch.device('cpu'))

     # Need to update according to how the mapping is done when tp_rank * pp_rank > 9
     mp_rank_file_index = '0' + str(pp_index * tp_rank + tp_index)
     mp_rank_file = os.path.join(ds_mp_rank_files_dir, 'mp_rank_' + mp_rank_file_index + '_model_states.pt')
     mp_rank_files_info_adjustment(mp_rank_file, state_dict, model_parallel_same_config, ds_universal_checkpoints_path)



def ds_universal_convert(args):

    torch.multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy("file_system")
    pool = multiprocessing.Pool(args.pool)

    ds_universal_checkpoints_path = args.ds_universal_dir
    latest_file = os.path.join(ds_universal_checkpoints_path, 'latest_universal')
    os.makedirs(ds_universal_checkpoints_path, exist_ok=True)
    with open(latest_file, "w") as f:
        f.write(str(args.iteration))

    ds_universal_checkpoints_path = os.path.join(ds_universal_checkpoints_path, str(args.iteration))
    os.makedirs(ds_universal_checkpoints_path, exist_ok=True)

    if (args.update_only_mp_rank_files == False):
        layers_per_model_pipeline_slice = args.num_layers // args.pp_rank
        # tensor_name_mapping maps the ds tensor directory name to the megatron-lm merged optimizer tensor path
        if args.pp_rank == 1:
            tensor_name_mapping = [
                [os.path.join(ds_universal_checkpoints_path, 'zero', 'tied_modules.embed.position_embeddings.weight'),os.path.join(args.megatron_lm_merged_input_dir, 'language_model.embedding.position_embeddings.weight.pt')], \
                [os.path.join(ds_universal_checkpoints_path, 'zero', 'tied_modules.embed.word_embeddings.weight'), os.path.join(args.megatron_lm_merged_input_dir, 'language_model.embedding.word_embeddings.weight.pt')],
                [os.path.join(ds_universal_checkpoints_path, 'zero', str(4 + args.num_layers) + '.bias'), os.path.join(args.megatron_lm_merged_input_dir, 'language_model.encoder.final_layernorm.bias.pt')],
                [os.path.join(ds_universal_checkpoints_path, 'zero', str(4 + args.num_layers) + '.weight'), os.path.join(args.megatron_lm_merged_input_dir, 'language_model.encoder.final_layernorm.weight.pt')]
            ]
        else:
            tensor_name_mapping =   [
                                [os.path.join(ds_universal_checkpoints_path, 'zero','tied_modules.embed.position_embeddings.weight'), os.path.join(args.megatron_lm_merged_input_dir,'language_model.embedding.position_embeddings.weight.pt')], \
                                [os.path.join(ds_universal_checkpoints_path, 'zero','tied_modules.embed.word_embeddings.weight'), os.path.join(args.megatron_lm_merged_input_dir,'language_model.embedding.word_embeddings.weight.pt')],
                                [os.path.join(ds_universal_checkpoints_path, 'zero','word_embeddings.weight'),os.path.join(args.megatron_lm_merged_input_dir,'word_embeddings.weight.pt')], \
                                [os.path.join(ds_universal_checkpoints_path, 'zero',str(4+args.num_layers)+'.bias'), os.path.join(args.megatron_lm_merged_input_dir,'language_model.encoder.final_layernorm.bias.pt')],
                                [os.path.join(ds_universal_checkpoints_path, 'zero',str(4+args.num_layers)+'.weight'),os.path.join(args.megatron_lm_merged_input_dir,'language_model.encoder.final_layernorm.weight.pt')]
                ]

        layer_name_mapping =    [
                                ['.attention.dense.bias', 'language_model.encoder.layers.LAYER_INDEX.self_attention.dense.bias'], \
                                ['.attention.dense.weight','language_model.encoder.layers.LAYER_INDEX.self_attention.dense.weight'], \
                                ['.attention.query_key_value.bias', 'language_model.encoder.layers.LAYER_INDEX.self_attention.query_key_value.bias'], \
                                ['.attention.query_key_value.weight', 'language_model.encoder.layers.LAYER_INDEX.self_attention.query_key_value.weight'], \
                                ['.input_layernorm.bias', 'language_model.encoder.layers.LAYER_INDEX.input_layernorm.bias'], \
                                ['.input_layernorm.weight', 'language_model.encoder.layers.LAYER_INDEX.input_layernorm.weight'], \
                                ['.mlp.dense_4h_to_h.bias', 'language_model.encoder.layers.LAYER_INDEX.mlp.dense_4h_to_h.bias'], \
                                ['.mlp.dense_4h_to_h.weight', 'language_model.encoder.layers.LAYER_INDEX.mlp.dense_4h_to_h.weight'], \
                                ['.mlp.dense_h_to_4h.bias', 'language_model.encoder.layers.LAYER_INDEX.mlp.dense_h_to_4h.bias'], \
                                ['.mlp.dense_h_to_4h.weight', 'language_model.encoder.layers.LAYER_INDEX.mlp.dense_h_to_4h.weight'], \
                                ['.post_attention_layernorm.bias', 'language_model.encoder.layers.LAYER_INDEX.post_attention_layernorm.bias'], \
                                ['.post_attention_layernorm.weight', 'language_model.encoder.layers.LAYER_INDEX.post_attention_layernorm.weight']
                ]

        for layer_index in np.arange(args.num_layers):
            for layer_tensor_index in np.arange(len(layer_name_mapping)):

                ds_tensor_name_map = os.path.join(ds_universal_checkpoints_path,'zero',str(3+layer_index)+layer_name_mapping[layer_tensor_index][0])
                megatron_tensor_name_map = os.path.join(args.megatron_lm_merged_input_dir,'layer_'+str(layer_index),layer_name_mapping[layer_tensor_index][1].replace('LAYER_INDEX',str(layer_index))+'.pt')
                tensor_name_map = [ds_tensor_name_map, megatron_tensor_name_map]
                tensor_name_mapping.append(tensor_name_map)


        # go over all the tensors in tensor_name_mapping and convert them from megatron optimizer format to ds_universal

        #for tensors_index in np.arange(len(tensor_name_mapping)):
        #    tensor_convert(tensor_name_mapping,tensors_index)
        #    print('finished converting tensor {}'.format(tensors_index))

        # multiprocessing of the tensors in tensor_name_mapping and converting them from megatron optimizer format to ds_universal

        pool.starmap(
            tensor_convert,
            [
                (
                    tensor_name_mapping,
                    k
                )
                for k in np.arange(len(tensor_name_mapping))
            ],
        )

        pool.close()
        pool.join()


    # updating the deepspeed ds_mp_rank files according to megatron non merged ( original megatron checkpoint structure files)

    if args.model_parallel_same_config == 'True':
        for pp_index in np.arange(args.pp_rank):
            for tp_index in np.arange(args.tp_rank):
                if args.pp_rank > 1:
                    file_name = os.path.join(args.megatron_lm_non_merged_input_dir,'mp_rank_{:02d}_{:03d}'.format(tp_index,pp_index),'model_optim_rng.pt')
                else:
                    file_name = os.path.join(args.megatron_lm_non_merged_input_dir,'mp_rank_{:02d}'.format(tp_index),'model_optim_rng.pt')

                state_dict = torch.load(file_name, map_location=torch.device('cpu'))

                # Need to update according to how the mapping is done when tp_rank * pp_rank > 9
                mp_rank_file_index = '0'+str(pp_index*args.tp_rank+tp_index)
                mp_rank_file = os.path.join(args.ds_mp_rank_files_dir,'mp_rank_'+mp_rank_file_index+'_model_states.pt')
                mp_rank_files_info_adjustment(mp_rank_file, state_dict, args.model_parallel_same_config,
                                                  ds_universal_checkpoints_path)


        
        model_parallel_matrix_index = []
        for pp_index in np.arange(args.pp_rank):
            for tp_index in np.arange(args.tp_rank):
                model_parallel_matrix_index.append([pp_index, tp_index])
    
        
        pool = multiprocessing.Pool(args.pool)
    
        pool.starmap(
            mp_rank_files_info_adjustment_parallel_processing,
            [
                (
                    args.ds_mp_rank_files_dir,
                    ds_universal_checkpoints_path,
                    args.megatron_lm_non_merged_input_dir,
                    args.model_parallel_same_config,
                    pp_index,
                    tp_index,
                    args.tp_rank
                )
                for (pp_index, tp_index) in model_parallel_matrix_index
            ],
        )
    
        pool.close()
        pool.join()

    else:
        mp_rank_files = glob.glob(os.path.join(args.ds_mp_rank_files_dir, 'mp_rank_*.pt'))
        if args.megatron_lm_non_merged_input_dir is not None:
            file_name = glob.glob(os.path.join(args.megatron_lm_non_merged_input_dir,'*'))[0]+'/model_optim_rng.pt'
            megatron_state_dict = torch.load(file_name, map_location=torch.device('cpu'))

        else:
            class My_args:
                def __init__(self, consumed_train_samples=args.iteration * args.global_batch_size, seq_length=args.seq_length, consumed_valid_samples=0):
                    self.consumed_train_samples = consumed_train_samples
                    self.seq_length = seq_length
                    self.consumed_valid_samples = consumed_valid_samples

            megatron_state_dict = { 'opt_param_scheduler': args.iteration, 'iteration': args.iteration, 'args' : None }
            megatron_state_dict['opt_param_scheduler'] = {'num_steps': args.iteration*args.global_batch_size, 'warmup_steps':  args.lr_warmup_samples  , 'decay_steps': args.lr_decay_samples}
            megatron_state_dict['args']= My_args(consumed_train_samples=args.iteration * args.global_batch_size,
                                                seq_length=args.seq_length)

        for mp_rank_file in mp_rank_files:
            print(f"Adjusting {mp_rank_file=}", flush=True)
            mp_rank_files_info_adjustment(mp_rank_file, megatron_state_dict, args.model_parallel_same_config, ds_universal_checkpoints_path)
        # Deleting redundant mp_rank files, in case number of devices was decreased
        universal_mp_rank_files = glob.glob(os.path.join(ds_universal_checkpoints_path, 'mp_rank_*.pt'))
        for universal_mp_rank_file in universal_mp_rank_files:
            if os.path.basename(universal_mp_rank_file) not in [os.path.basename(file_elem) for file_elem in mp_rank_files]:
                print(f"Deleting old redundant mp_rank file {universal_mp_rank_file=}", flush=True)
                os.remove(universal_mp_rank_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--ds-universal-dir", "--o", type=str, help="output directory for saving the converted ds_universal checkpoints", required=True)
    parser.add_argument("--megatron-lm-merged-input-dir", "--merged-input", type=str, help="megatron-lm merged optimizer input checkpoint directory path", required=False)
    parser.add_argument("--megatron-lm-non-merged-input-dir", "--non-merged-input", type=str, help="megatron-lm non merged checkpoint directory path", default = None)
    parser.add_argument("--ds-mp-rank-files-dir", "--ds", type=str, help="deepspeed mp_rank_files directory path", required=True)
    parser.add_argument("--tp-rank", "--tp",type=int, help="deepseed tp_rank configuration", default=8,required=True)
    parser.add_argument("--pp-rank", "--pp",type=int, help="deepseed tp_rank configuration", default=8,required=True)
    parser.add_argument("--num-layers", "--nl", type=int, help="GPT-3 number of layers", default=96)
    parser.add_argument("--iteration", "--iter", type=int, help="#iteration ", default=None, required=True)
    parser.add_argument("--global-batch-size", "--gbs", type=int, help="load ckpt global batch size", default=1536)
    parser.add_argument("--seq_length", "--sl", type=int, help="Sequence length", default=2048)
    parser.add_argument("--lr-warmup-samples", "--lws", type=int, help="lr warmup samples", default=407040)
    parser.add_argument("--lr-decay-samples", "--lds", type=int, help="lr decay samples", default=166809600)
    parser.add_argument("--model-parallel-same-config", "--same_config", help="if megatron-lm and megatron deepspeed tp, pp configuration is the same", default=True)
    parser.add_argument("--pool", "-pl", type=int, help="Process pool", default=4)
    parser.add_argument("--update-only-mp-rank-files", "--update", type=bool, help="if set will update only the mp_rank files w/o converting the nvidia-merged format to ds universal ", default=False, required=False)

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")

    print("Converting megatron merged optimizer checkpoint to deepspeed universal format checkpoint")
    start_time = datetime.now()
    ds_universal_convert(args)
    stop_time = datetime.now()
    run_time = stop_time - start_time
    print(f"[INFO] Spent {run_time} (h:m:s) to convert the merged optimizer to deepspeed universal format")
