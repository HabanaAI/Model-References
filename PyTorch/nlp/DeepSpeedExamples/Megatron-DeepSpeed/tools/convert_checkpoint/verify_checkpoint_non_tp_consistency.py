import re
import tqdm
import argparse
from dataclasses import dataclass
import torch
from deepspeed.checkpoint import DeepSpeedCheckpoint


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default=None, type=str, help='DeepSpeed Checkpoint folder')
    parser.add_argument('--model_type', default='GPT', type=str, help='Type of the model',
                        choices=['GPT', 'BLOOM', 'LLAMA'])
    args = parser.parse_args()
    print(f'args = {args}')
    return args


def show_3d(ds_checkpoint):
    src_3d = ds_checkpoint.zero_checkpoint.src_3d
    dp, tp, pp = src_3d.dp_degree, src_3d.tp_degree, src_3d.pp_degree
    print(f'3D configuration: DP={dp} TP={tp} PP={pp}')


def get_layer_patterns_for_non_sharded(model_type):
    if model_type == 'GPT':
      return [
          'position_embeddings.weight',
          'input_layernorm.weight',
          'input_layernorm.bias',
          'self_attention.dense.bias',
          "attention.dense.bias",
          'post_attention_layernorm.weight',
          'post_attention_layernorm.bias',
          'mlp.dense_4h_to_h.bias',
          'weight',
          'bias'
      ]

    if model_type == 'BLOOM':
      return [
          'input_layernorm.weight',
          'input_layernorm.bias',
          'self_attention.dense.bias',
          "attention.dense.bias",
          'post_attention_layernorm.weight',
          'post_attention_layernorm.bias',
          'mlp.dense_4h_to_h.bias',
          'weight',
          'bias'
      ]
    if model_type == 'LLAMA':
      return [
          'input_layernorm.weight',
          'input_layernorm.bias',
          'self_attention.dense.bias',
          "attention.dense.bias",
          'post_attention_layernorm.weight',
          'post_attention_layernorm.bias',
          'mlp.dense_4h_to_h.bias',
          'final_rmsnorm.weight',
      ]


def get_zero_patterns_for_non_sharded(model_type):
  if model_type == 'GPT':
    return [
        r"tied_modules.embed.word_embeddings.norm.weight",
        r"tied_modules.embed.word_embeddings.norm.bias",
        r"tied_modules.embed.position_embeddings.weight",
        r"\d+.input_layernorm.weight",
        r"\d+.input_layernorm.bias",
        r"\d+.self_attention.dense.bias",
        r"\d+.attention.dense.bias",
        r"\d+.post_attention_layernorm.weight",
        r"\d+.post_attention_layernorm.bias",
        r"\d+.mlp.dense_4h_to_h.bias",
        r"\d+.weight",
        r"\d+.bias",
    ]
  if model_type == 'BLOOM':
    return [
        r"tied_modules.embed.word_embeddings.norm.weight",
        r"tied_modules.embed.word_embeddings.norm.bias",
        r"\d+.input_layernorm.weight",
        r"\d+.input_layernorm.bias",
        r"\d+.self_attention.dense.bias",
        r"\d+.attention.dense.bias",
        r"\d+.post_attention_layernorm.weight",
        r"\d+.post_attention_layernorm.bias",
        r"\d+.mlp.dense_4h_to_h.bias",
        r"\d+.weight",
        r"\d+.bias",
    ]
  if model_type == 'LLAMA':
    return [
        r"tied_modules.embed.word_embeddings.norm.weight",
        r"tied_modules.embed.word_embeddings.norm.bias",
        r"\d+.input_layernorm.weight",
        r"\d+.input_layernorm.bias",
        r"\d+.self_attention.dense.bias",
        r"\d+.attention.dense.bias",
        r"\d+.post_attention_layernorm.weight",
        r"\d+.post_attention_layernorm.bias",
        r"\d+.mlp.dense_4h_to_h.bias",
        r"\d+.final_rmsnorm.weight",
    ]



@dataclass
class ParamInfo:
    pp: int
    tp: int
    dp: int
    data: torch.Tensor
    numel: int


def get_zero_pp_stage_non_sharded_params(ds_checkpoint, model_type, pp_stage, dp_stage):
    patterns = get_zero_patterns_for_non_sharded(model_type)
    params = {}
    for tp_stage in tqdm.tqdm(range(ds_checkpoint.tp_degree), desc='bf16 zero files'):
        sd = ds_checkpoint.get_zero_checkpoint_state(
            pp_index=pp_stage,
            tp_index=tp_stage,
            dp_index=dp_stage)

        optim_sd = sd["optimizer_state_dict"]
        param_slice_mappings = optim_sd["param_slice_mappings"]
        state_groups = optim_sd["base_optimizer_state"]["state"]
        fp32_groups = optim_sd["single_partition_of_fp32_groups"]

        for param_group_id in range(len(state_groups)):
            flat_state = dict(
                exp_avg=state_groups[param_group_id]["exp_avg"],
                exp_avg_sq=state_groups[param_group_id]["exp_avg_sq"],
                fp32=fp32_groups[param_group_id],
            )

            for name, fragment_mapping in param_slice_mappings[param_group_id].items():
                if not any(re.match(pattern, name) for pattern in patterns):
                    continue

                for state_key in flat_state.keys():
                    tensor = flat_state[state_key].narrow(
                        dim=0,
                        start=fragment_mapping.start,
                        length=fragment_mapping.numel).clone()
                    info = ParamInfo(pp=pp_stage, tp=tp_stage, dp=dp_stage,
                                     data=tensor, numel=fragment_mapping.numel)
                    full_name = name + '.__' + state_key
                    if full_name not in params:
                        params[full_name] = []
                    params[full_name].append(info)
    return params


def verify_equal_params(params, tp):
    failed = 0
    report = {}
    for name, info in params.items():
        n = len(info)
        if n != tp:
            ok = False
            print(f'{name}: FAILED expected n={n} == tp={tp}')
        elif n == 1:
            ok = True
        else:
            ok = all([(x.numel == info[0].numel) for x in info[1:]])
            if not ok:
                print(f'{name}: FAILED numel comparison [n={n}]')
            else:
                ok = all([x.data.eq(info[0].data).all().item() for x in info[1:]])
                if not ok:
                    print(f'{name}: FAILED data comparison [n={n}]')
        failed += (ok == False)
        report[name] = (ok, n)
        if ok:
            print(f'{name}: OK [n={n}]')
    return failed, report


def update_layer_non_sharded_params(params, model_type, filename, pp_index, tp_index):
    layer_id, file_tp_index = re.search('layer_(\d+)-model_(\d+)', filename).groups()
    layer_id = int(layer_id)
    file_tp_index = int(file_tp_index)
    #assert tp_index == file_tp_index, f'Inconsistent tp index tp_index={tp_index} file_tp_index={file_tp_index}'
    if tp_index != file_tp_index:
        print('bad')

    sd = torch.load(filename, map_location=torch.device('cpu'))
    sequential_layers = get_layer_patterns_for_non_sharded(model_type)
    for key in sd.keys():
        if key in sequential_layers:
            param_key = str(layer_id) + '.' + key
            if param_key not in params:
                params[param_key] = []
            info = ParamInfo(pp=pp_index, tp=tp_index, dp=-1,
                             data=sd[key], numel=sd[key].numel())
            params[param_key].append(info)
    return params


def verify_layer_files(ds_checkpoint, model_type):
    src_3d = ds_checkpoint.zero_checkpoint.src_3d
    dp, tp, pp = src_3d.dp_degree, src_3d.tp_degree, src_3d.pp_degree

    total_failed = 0
    for pp_index in range(pp):
        print(f'\nChecking pp_stage={pp_index}')
        params = {}
        if pp_index == 0:
            for tp_index in range(tp):
                for filename in ds_checkpoint.tp_to_embedding_map[tp_index]:
                    update_layer_non_sharded_params(params, model_type,
                                                     filename, pp_index, tp_index)
        for tp_index in range(tp):
            for filename_list in ds_checkpoint.transformer_file_map[(tp_index, pp_index)]:
                for filename in filename_list:
                    update_layer_non_sharded_params(params, model_type,
                                                     filename, pp_index, tp_index)
        if pp_index == (pp-1):
            for tp_index in range(tp):
                for filename in ds_checkpoint.tp_to_final_norm_map[tp_index]:
                    update_layer_non_sharded_params(params, model_type,
                                                     filename, pp_index, tp_index)
        failed, report = verify_equal_params(params, tp)
        total_failed += failed
    return total_failed


def verify_zero_files(ds_checkpoint, model_type):
    src_3d = ds_checkpoint.zero_checkpoint.src_3d
    dp, tp, pp = src_3d.dp_degree, src_3d.tp_degree, src_3d.pp_degree

    total_failed = 0
    for i in range(pp):
        for j in range(dp):
            print(f'\nChecking pp_stage={i} dp_stage={j}')
            params = get_zero_pp_stage_non_sharded_params(ds_checkpoint, model_type,
                                                           pp_stage=i, dp_stage=j)
            failed, report = verify_equal_params(params, tp)
            total_failed += failed
    return total_failed

def verify_checkpoint(folder,model_type):
    final_layer_norm_idx = -2 if model_type == 'LLAMA' else -1
    ds_checkpoint = DeepSpeedCheckpoint(folder,final_layer_norm_idx=final_layer_norm_idx)
    ds_checkpoint.validate_files()
    show_3d(ds_checkpoint)

    print('\nVerify ** layer_ ** files')
    total_failed_layer = verify_layer_files(ds_checkpoint, model_type)
    if total_failed_layer == 0:
        print('\nCheckpoint layer files OK')
    else:
        print(f"\nCheckpoint layer files BAD with total_failed={total_failed_layer}")

    print('\nVerify ** bf16_zero_ ** files')
    total_failed_zero = verify_zero_files(ds_checkpoint, model_type)
    if total_failed_zero == 0:
        print('\nCheckpoint zero files OK')
    else:
        print(f"\nCheckpoint zero files BAD with total_failed={total_failed_zero}")

    return (total_failed_layer + total_failed_zero) == 0


def main():
    print(f'Verify DeepSpeed Checkpoint consistency for non-TP-sharded parameters')
    args = parse_arguments()
    print(args)
    assert verify_checkpoint(args.folder, args.model_type) is True, "Checkpoint verification failed"

if __name__ == "__main__":
    main()
