import argparse
import torch
import json
import os
import re

from collections import defaultdict
from transformers import AutoModel

MDS_SUBLAYER_MAPPINGS = {
        "dense_h_to_4h": "mlp.",
        "dense_4h_to_h": "mlp.",
        "dense_h_to_4h_swiglu": "mlp.",
        "dense": "attention.",
        "query_key_value": "attention.",
        "inv_freq": "attention.core_attention.rotary_emb.",
        "rotary_emb.inv_freq": "attention.core_attention."
    }

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir',
                        type=str,
                        help='Output Universal checkpoint folder',
                        default='./MDS_universal_checkpoint/')
    parser.add_argument('--model-name-or-path',
                        type=str,
                        help='Huggingface model name or path', required=True)
    parser.add_argument('--config',
                        type=str,
                        help='path to json config file with conversion information', required=True)
    parser.add_argument('--no-strict',
                        action='store_false',
                        help='allow non-strict conversion: convert partially even when failing' \
                        'to convert some of the model weight names', dest='strict')
    args = parser.parse_args()
    write_to_file_and_print(f'args = {args}')
    return args

def load_config(path):
    config = json.load(open(path, 'r', encoding='utf-8'))
    start_idx, end_idx = config['LAYER_MAPPINGS']['transformer_layers']
    config['LAYER_MAPPINGS']['transformer_layers'] = list(range(start_idx, end_idx+1))
    return config

def get_model_weights(model_name_or_path, config):
    model = AutoModel.from_pretrained(model_name_or_path)
    hf_weights = model.state_dict()
    try:
        config['num_attention_heads'] = model.config.num_attention_heads
        config['hidden_size'] = model.config.hidden_size
        num_transformer_layers = model.config.num_hidden_layers
    except Exception:
        raise Exception("Failed to access the model's architecture parameters.")
    assert num_transformer_layers == len(config['LAYER_MAPPINGS']['transformer_layers']), \
    'num transformer layers from config and HF model do not match! aborting conversion. '
    return hf_weights, config

def convert_keys(config, hf_names):
    """ This function maps each weight name of HF model to MDS convention.
    It gets the config with information about architecture and name conversion,
    and hf_names which includes all weight names of the HF model.
    It returns a dictionary with the HF weight names as keys and the matching MDS
    weight names as values.
    The mapping is based on the user's input (through the config json file).
    It includes both conversions of given full names and partial names.
    Full names are converted first, and then the function reconstructs
    the partial name conventions using the user input and known MDS conventions.
    It also deals with repeating names (appearing in transformer layers).
    """
    key_conversion = config['FULL_NAME_MAPPINGS']
    hf_names = set(hf_names) - set(key_conversion.keys())
    transformer_layers = config['LAYER_MAPPINGS']['transformer_layers']
    for key, info in config['PARTIAL_NAME_MAPPINGS'].items():
        if info['layer'] == 'transformer':
            for name in hf_names.copy():
                if key in name and 'layers' in name:
                    # extract layer and suffix (weight/bias) from name to match MDS name
                    # TODO: check the infer-layer logic for other models as well
                    layer = transformer_layers[int(re.search(r'\d+', name).group())]
                    suffix = name.rsplit('.',1)[1]
                    suffix = '.' + suffix if suffix in['weight','bias'] else ''
                    mds_name = MDS_SUBLAYER_MAPPINGS.get(info['name'], '') + info['name']
                    key_conversion[name] = str(layer) + '.' + mds_name + suffix
                    hf_names.remove(name)
        else:
            # look for matching weight name using partial name (key).
            # warn user if there is more than one match
            matched_names = [name for name in hf_names if key in name and 'layers' not in name]
            if len(matched_names) == 0:
                continue
            assert len(matched_names) == 1, f'WARNING: found more than one match for keyword ' \
                                            ' {key}. please review your configuration'
            key = matched_names[0]
            hf_names.remove(key)
            suffix = key.rsplit('.', 1)[1]
            suffix = suffix if suffix in['weight','bias'] else ''
            key_conversion[key] = str(config['LAYER_MAPPINGS'][info['layer']]) + '.' + info['name'] \
                + '.' + suffix
    return key_conversion

def find_special_keys(config, key_conversion):
    """
    This function is used to split the dictionary key_conversion to usual and special.
    Key_conversion dictionary consists of all HF weight names and the matching MDS weight name.
    For some weights, a special treatment is required, e.g. concatenation of several HF weights
    to one MDS weight.
    The function identifies such weight names based on the information given by the user
    in the json config file.
    """
    special_conversion = defaultdict(list)
    for name in list(key_conversion.keys()):
        for keyword, transformation in config["SPECIAL"].items():
            if keyword in name:
                special_conversion[transformation].append(name)
                key_conversion.pop(name)
    return key_conversion, special_conversion

def create_exp_avg_files(dirctory, tensor_shape):
    # this function is used to create exp_avg and exp_avg_sq files when no data
    # is available (using zeros)
    torch.save({'param': torch.zeros(tensor_shape)}, dirctory+'/exp_avg.pt')
    torch.save({'param': torch.zeros(tensor_shape)}, dirctory+'/exp_avg_sq.pt')

def save_weight(data, dir_name, exp_avg_files=True):
    os.makedirs(dir_name, exist_ok=True)
    torch.save(data, dir_name+'/fp32.pt')
    if exp_avg_files:
        create_exp_avg_files(dir_name, data['param'].shape)

def qkv_concat(qkv_concat_weights, checkpoint_weights, config):
    special_conversion = {}
    # TODO: add support for qkv bias (not required for llama)
    transformer_layers = config['LAYER_MAPPINGS']['transformer_layers']
    num_heads = config['num_attention_heads']
    hidden_size = config['hidden_size']
    head_dim = hidden_size // num_heads
    q_name = qkv_concat_weights['query']
    k_name = qkv_concat_weights['key']
    v_name = qkv_concat_weights['value']
    # TODO: verify this search works for other models as well
    q_name = q_name.replace(re.search(r'\d+', q_name).group(), '<L>')
    k_name = k_name.replace(re.search(r'\d+', k_name).group(), '<L>')
    v_name = v_name.replace(re.search(r'\d+', v_name).group(), '<L>')
    for layer_id, layer in enumerate(transformer_layers):
        query = checkpoint_weights[q_name.replace('<L>',str(layer_id))]
        key = checkpoint_weights[k_name.replace('<L>',str(layer_id))]
        value = checkpoint_weights[v_name.replace('<L>',str(layer_id))]
        # apply permute introduced by HF format
        # TODO: check on llama larger than 7B
        query = query.reshape((num_heads,2,head_dim//2,hidden_size)).transpose(1,2)
        query = query.reshape((num_heads,-1,hidden_size))
        key = key.reshape((num_heads,2,head_dim//2,hidden_size)).transpose(1,2)
        key = key.reshape((num_heads,-1,hidden_size))
        # apply reshaping to convert original facebook's llama format to MDS
        value = value.reshape((num_heads,-1,hidden_size))
        qkv = torch.concat((query,key,value),dim=1).reshape((-1,hidden_size))
        special_conversion[f'{str(layer)}.attention.query_key_value.weight'] = qkv
    return special_conversion

def write_to_file_and_print(text, log_file='log.txt'):
    print(text)
    with open(log_file, 'a') as f:
        f.write(text+'\n')


def main():
    args = parse_arguments()

    # create output dir and log file
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = args.output_dir+'log.txt'
    write_to_file_and_print(f'Converting HuggingFace model {args.model_name_or_path} weights to' \
                            'Universal checkpoint in {args.output_dir}', log_file)
    write_to_file_and_print(f'args = {args}', log_file)

    config = load_config(args.config)
    write_to_file_and_print(f'successfuly loaded model config from {args.config}', log_file)

    checkpoint_weights, config = get_model_weights(args.model_name_or_path, config)
    write_to_file_and_print(f'successfuly loaded model weight from checkpoint', log_file)

    # extract all key name conversions, compare to HF model weights and split to "usual" name
    # conversion and special treatment (e.g. reshape, concatenation etc.)
    key_conversion = convert_keys(config, checkpoint_weights.keys())
    unexpected_keys = [k for k in key_conversion.keys() if k not in checkpoint_weights.keys()]
    missing_keys = [k for k in checkpoint_weights.keys() if k not in key_conversion.keys()]
    if unexpected_keys or missing_keys:
        write_to_file_and_print(f'WARNING: found {len(unexpected_keys)} unexpected' \
                                'weights and {len(missing_keys)} missing weights. ')
        write_to_file_and_print(f'unexpected: {unexpected_keys} ')
        write_to_file_and_print(f'missing: {missing_keys}')
        assert not args.strict, 'name conversion failed. '

    key_conversion, special_conversion = find_special_keys(config,key_conversion)

    # create files and save pretrained weight
    zero_dir = args.output_dir+'zero/'
    os.makedirs(zero_dir, exist_ok=True)

    qkv_cat_names = {'query': special_conversion['qkv_cat_q'][0],
                     'key': special_conversion['qkv_cat_k'][0],
                     'value': special_conversion['qkv_cat_v'][0]}
    special_conversion = qkv_concat(qkv_cat_names, checkpoint_weights, config)
    for orig_name, new_name in key_conversion.items():
        data = {'param': checkpoint_weights[orig_name]}
        save_weight(data, zero_dir+new_name)
    for weight_name in special_conversion.keys():
        data = {'param': special_conversion[weight_name]}
        save_weight(data, zero_dir+weight_name)
    write_to_file_and_print(f'successfuly saved all converted weights', log_file)


if __name__=='__main__':
    main()
