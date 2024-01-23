import argparse
import glob
import json
import os
import torch

from collections import defaultdict
from transformers import AutoModelForCausalLM, LlamaConfig, AutoConfig


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf-dir',
                        type=str,
                        help='Output Universal checkpoint folder',
                        default='./universal_to_HF_checkpoint/')
    parser.add_argument('--universal-dir',
                        type=str,
                        help='Path to universal checkpoint to be converted',
                        required=True)
    parser.add_argument('--hf-model',
                        type=str, default=None,
                        help='Huggingface model name or path')
    parser.add_argument('--model-type',
                        type=str, default=None,
                        choices=['llama', None],
                        help='Huggingface model name or path')
    parser.add_argument('--config',
                        type=str,
                        help='path to json config file with conversion information', required=True)
    parser.add_argument('--save-conversion',
                        type=str, default=None,
                        help='json file to save the conversion dict to (defaults to not saving)')
    parser.add_argument('--no-strict',
                        action='store_false',
                        help='allow non-strict conversion: convert partially even when failing' \
                        'to convert some of the model weight names', dest='strict')
    args = parser.parse_args()

    assert (args.hf_model is not None) ^ (args.model_type is not None), \
        'Either model type or HuggingFace model name or path is required'
    return args


def load_config(path):
    """ This function is used for loading the conversion config given by the user """
    config = json.load(open(path, 'r', encoding='utf-8'))
    start_idx, end_idx = config['LAYER_MAPPINGS']['transformer']
    config['LAYER_MAPPINGS']['transformer'] = list(range(start_idx, end_idx+1))
    if 'MODEL' in config.keys():
        fields = config['MODEL'].keys()
        assert ('hidden_size' in fields) & ('intermediate_size' in fields) & \
        ('num_attention_heads' in fields) & ('num_hidden_layers' in fields), \
            'Required fields of MODEL are missing in json config file'
        assert config['MODEL']['num_hidden_layers'] == len(config['LAYER_MAPPINGS']['transformer']), \
            f'Inconsistency of provided num hidden layers of model in json file'
    return config


def create_model(args, config):
    """
    This function is loading a HuggingFace model to create the checkpoint for.
    It is used to identify the model weight names and for saving.
    The HF model is given by the user either as a huggingface model name or path
    or as a model type with architecture params in the json config.
    """
    if args.hf_model is not None:
        model_config = AutoConfig.from_pretrained(args.hf_model)
        model = AutoModelForCausalLM.from_config(model_config)
    else:
        assert 'MODEL' in config.keys(), f'When using model type, model parameters must be ' \
                                         'included in json configuration file'
        if args.model_type == 'llama':
            model_config = LlamaConfig()
            model_config.hidden_size = config['MODEL']['hidden_size']
            model_config.intermediate_size = config['MODEL']['intermediate_size']
            model_config.num_attention_heads = config['MODEL']['num_attention_heads']
            model_config.num_key_value_heads = config['MODEL']['num_attention_heads']
            model_config.num_hidden_layers = config['MODEL']['num_hidden_layers']
        else:
            raise NotImplementedError(f'Unsupported model type {args.model_type}')
        model = AutoModelForCausalLM.from_config(model_config)
    return model


def save_hf_format(model, output_dir, tokenizer=None):
    """
    This function is used to save the HuggingFace model in the output dir.
    After saving, the model can be later loaded with from_pretrained.
    """
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    if tokenizer is not None:
        tokenizer.save_vocabulary(output_dir)


def write_to_file_and_print(text, log_file='log.txt'):
    print(text)
    with open(log_file, 'a') as f:
        f.write(text+'\n')


def get_universal_checkpoint_files(checkpoint_dir, layer_idx):
    """
    This function is used to get all universal checkpoint file names by layer idx.
    """
    layer_ckpt_path = os.path.join(checkpoint_dir, f'zero/{layer_idx}.*')
    ckpt_files = glob.glob(layer_ckpt_path)
    return ckpt_files


def convert_partial_name(mds_name, config, layer_idx, layer_type, key, special=None):
    """
    This function is used to convert weight name from universal to HF.

    Arguments:
        mds_name: the weight name to convert (universal)
        config: conversion configuration given by the user through json file
        layer_idx: index of the current layer conversion is performed over
        layer_type: layer type as appears in json config file (e.g. transformer, word_embeddings)
        key: keyword from mds name used for conversion (as appears in json config file, should be
            indicative and unique). Used for partial name conversion using MDS_SUBLAYER_MAPPINGS.
        special: string used as placeholder for special weights (e.g. query_key_value concatenation)
    """
    suffix = mds_name.rsplit('.', 1)[-1]
    suffix = '.' + suffix if suffix in['weight','bias'] else ''
    if layer_type == 'transformer':
        prefix = f'model.layers.{layer_idx-config["LAYER_MAPPINGS"]["transformer"][0]}.'
    else:
        prefix = ''
    if special is None:
        hf_name = prefix + config['PARTIAL_NAME_MAPPINGS'][layer_type][key] + suffix
    else:
        hf_name = prefix + special + suffix
    return hf_name


def convert_layer(state_dict, config, layer_idx, layer_type, universal_dir,
                  missing, unexpected, log_file, conversion_dict, model):
    """
    This function is used to convert all weight names in a specific layer from universal to HF.

    Arguments:
        state_dict: HF model state dict with all model weights
        config: conversion configuration given by the user through json file
        layer_idx: index of the current layer conversion is performed over
        layer_type: layer type as appears in json config file (e.g. transformer, word_embeddings)
        universal_dir: directory with universal checkpoint files
        missing: set of HF weight names there was no successfull conversion to yet
        unexpected: list of converted weight names not matching the model state dict
            (unsuccessfull conversion)
        log_file: path to log file of the conversion process
        conversion_dict: path to save conversion dict (or None)
        model: HuggingFace model to create checkpoint for
    """
    mds_weights = get_universal_checkpoint_files(universal_dir, layer_idx)
    mds_weight_names = set(mds_weights)
    # try to convert using full name mappings given by user
    # remove successfully converted names to ignore with partial name mappings
    if str(layer_idx) in config['FULL_NAME_MAPPINGS'].keys():
        for mds_name, hf_name in config['FULL_NAME_MAPPINGS'][str(layer_idx)].items():
            success = False
            full_mds_name = os.path.join(universal_dir, 'zero/', mds_name)
            if mds_name not in config['SPECIAL'].keys():
                success = load_weight(full_mds_name, state_dict, hf_name,
                                      missing, conversion_dict, log_file)
            else:
                if config['SPECIAL'][mds_name] == 'attention_qkv':
                    success = qkv_concat(full_mds_name, hf_name, state_dict, model.config,
                                         missing, conversion_dict, log_file)
            if success:
                mds_weight_names.remove(full_mds_name)
            else:
                unexpected.append(hf_name)
    # try converting remaining weights using partial name mappings given by user
    if layer_type in config['PARTIAL_NAME_MAPPINGS'].keys():
        for mds_name in mds_weight_names:
            success = False
            for key in config['PARTIAL_NAME_MAPPINGS'][layer_type].keys():
                keyword = key + '.' if key[-1] != '.' else key
                if keyword in mds_name:
                    if key not in config['SPECIAL'].keys():
                        hf_name = convert_partial_name(mds_name, config, layer_idx,
                                                       layer_type, key)
                        success = load_weight(mds_name, state_dict, hf_name,
                                              missing, conversion_dict, log_file)
                    else:
                        if config['SPECIAL'][key] == 'attention_qkv':
                            place_holder = 'qkv'
                            tmp_name = convert_partial_name(mds_name, config, layer_idx,
                                                           layer_type, key, special=place_holder)
                            qkv_dict = config['PARTIAL_NAME_MAPPINGS'][layer_type][key]
                            query_name = tmp_name.replace(place_holder, qkv_dict['query'])
                            key_name = tmp_name.replace(place_holder, qkv_dict['key'])
                            value_name = tmp_name.replace(place_holder, qkv_dict['value'])
                            hf_name = {'query': query_name, 'key': key_name, 'value': value_name}
                            success = qkv_concat(mds_name, hf_name,
                                                 state_dict, model.config, missing,
                                                 conversion_dict, log_file)
                if success:
                    break
            if not success:
                unexpected.append(mds_name)
    return


def qkv_concat(mds_name, hf_name, state_dict, model_config,
               missing, conversion_dict, log_file):
    """
    This function is used to convert query-key-value weights from universal to HF.
    We use this special function because of difference in shapes:
    in universal, query-key-value is one matrix whereas in HF there are 3 separate matrices.
    This is done by loading the qkv weight and doing some reshapes before concatenation based on
    model parameters (MDS qkv is based on division between attention heads).

    Arguments:
        mds_name: name of weight in universal format
        hf_name: name after conversion with special placeholder (to match each of
            query/key/value HF weight name)
        qkv_dict: dict with HF partial names from json config file given by user
        state_dict: HF model state dict with all model weights
        model_config: HF model configuration
        missing: set of HF weight names there was no successfull conversion to yet
        conversion_dict: path to save conversion dict (or None)
        log_file: path to log file of the conversion process
    """
    mds_weight = torch.load(os.path.join(mds_name,'fp32.pt'))['param']
    num_heads = model_config.num_attention_heads
    hidden_size = model_config.hidden_size
    head_dim = hidden_size // num_heads

    # transformations from MDS/universal query-key-value matrix to HF 3 matrices
    qkv = mds_weight.reshape((num_heads,-1,hidden_size))
    q = qkv[:,:head_dim,:]
    k = qkv[:,head_dim:2*head_dim,:]
    v = qkv[:,2*head_dim:,:]
    q = q.reshape((num_heads, head_dim//2, 2, hidden_size)).transpose(1,2)
    query = q.reshape((-1, hidden_size))
    k = k.reshape((num_heads, head_dim//2, 2, hidden_size)).transpose(1,2)
    key = k.reshape((-1, hidden_size))
    value = v.reshape((-1, hidden_size))

    # reload each matrix to matching key in the model's state dict
    success_q = load_weight(mds_name, state_dict, hf_name['query'],
                            missing, conversion_dict, log_file, query)
    success_k = load_weight(mds_name, state_dict, hf_name['key'],
                            missing, conversion_dict, log_file, key)
    success_v = load_weight(mds_name, state_dict, hf_name['value'],
                            missing, conversion_dict, log_file, value)
    return all([success_q, success_k, success_v])


def load_weight(mds_name, state_dict, hf_name, missing, conversion_dict, log_file, weight=None):
    """
    This function is used to load weight to matching HF weight name in model state dict.
    The function also handles warnings to user in cases like unexpected names or mismatch in shapes

    Arguments:
        mds_name: name of weight in universal format
        state_dict: HF model state dict with all model weights
        hf_name: name after conversion to HF format
        missing: set of HF weight names there was no successfull conversion to yet
        conversion_dict: path to save conversion dict (or None)
        log_file: path to log file of the conversion process
        weight: weight from universal for special cases (None by default and loaded from checkpoint)
    """
    # load weight by name unless it is given
    if weight is None:
        weight = torch.load(os.path.join(mds_name,'fp32.pt'))['param']

    # converted name is not in model state dict
    if not hf_name in state_dict.keys():
        write_to_file_and_print(f'WARNING: conversion failed. tried to convert {mds_name} to ' \
                                 f'{hf_name}', log_file)
        return False

    # mismatch of shapes
    if weight.shape != state_dict[hf_name].shape:
        write_to_file_and_print(f'WARNING: unmatched shape of weight! ' \
                                f'MDS weight {mds_name} of shape {weight.shape} ' \
                                f'HF weight {hf_name} of shape {state_dict[hf_name].shape} ',
                                log_file)

    # converted name was already converted to
    if not hf_name in missing:
        write_to_file_and_print(f'WARNING: converted to {hf_name} more than once? ' \
                                f'(tried to convert {mds_name})', log_file)
        if conversion_dict is not None:
            conversion_dict[mds_name] = [conversion_dict[mds_name]]
            conversion_dict[mds_name].append(hf_name)

    # save successful conversion
    else:
        missing.remove(hf_name)
        if conversion_dict is not None:
            conversion_dict[mds_name] = hf_name
    state_dict[hf_name] = weight
    return True


def main():
    args = parse_arguments()

    # create output dir and log file
    os.makedirs(args.hf_dir, exist_ok=True)
    log_file = args.hf_dir + 'log.txt'
    write_to_file_and_print(f'Converting Megatron-DeepSpeed model from {args.universal_dir} ' \
                            f'weights to HuggingFace model checkpoint in {args.hf_dir}', log_file)
    write_to_file_and_print(f'args = {args}', log_file)

    # load conversion config from json
    config = load_config(args.config)
    write_to_file_and_print(f'successfuly loaded model conversion config from {args.config}',
                            log_file)

    # load HF target model
    if args.hf_model is not None:
        write_to_file_and_print(f'Using HuggingFace model {args.hf_model} for conversion', log_file)
    else:
        write_to_file_and_print(f'Using model type {args.model_type} for conversion', log_file)
    write_to_file_and_print(f'args = {args}', log_file)
    model = create_model(args, config)
    state_dict = model.state_dict()

    # do conversion layer by layer and keep track of missing/unexpected weight names
    missing_weight_names = set(state_dict.keys())
    unexpected_weight_names = []
    conversion_dict = {} if args.save_conversion is not None else None
    for layer_type in config['LAYER_MAPPINGS'].keys():
        if type(config['LAYER_MAPPINGS'][layer_type]) == list:
            layers = config['LAYER_MAPPINGS'][layer_type]
        else:
            layers = [config['LAYER_MAPPINGS'][layer_type]]
        for layer_idx in layers:
            write_to_file_and_print(f'Converting layer {layer_idx} '\
                                    f'of type {layer_type}', log_file)
            convert_layer(state_dict, config, layer_idx, layer_type, args.universal_dir,
                          missing_weight_names, unexpected_weight_names, log_file,
                          conversion_dict, model)

    # check for missing / unexpected weight names and warn user
    if unexpected_weight_names or missing_weight_names:
        write_to_file_and_print(f'WARNING: found {len(unexpected_weight_names)} unexpected ' \
                                f'weights and {len(missing_weight_names)} missing weights. ',
                                log_file)
        write_to_file_and_print(f'unexpected: {unexpected_weight_names} ', log_file)
        write_to_file_and_print(f'missing: {missing_weight_names}', log_file)
        assert not args.strict, 'name conversion failed. '

    # load converted weights to HF model and save
    model.load_state_dict(state_dict)
    save_hf_format(model, args.hf_dir)
    write_to_file_and_print(f'Successfuly saved all converted weights', log_file)

    if args.save_conversion is not None:
        with open(args.save_conversion, 'w') as f:
            json.dump(conversion_dict, f, indent=3)


if __name__=='__main__':
    main()
