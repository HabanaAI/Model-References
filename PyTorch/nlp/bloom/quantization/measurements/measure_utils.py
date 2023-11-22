#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###############################################################################

import json
import os
import torch

# Strings for parsing measurements files
RANGES = 'ranges'
INPUT = 'input'
INPUTS = 'inputs'
OUTPUT = 'output'
OUTPUTS = 'outputs'
PARAMS = 'params'
WEIGHT = 'weight'
WEIGHTS = 'weights'
BIAS = 'bias'
LM_HEAD = 'module.lm_head'
PREFIX = 'module.'
MODE = 'Mode'
NODES = 'Nodes'

NONE_INDICATION = -1 # will never be max abs so we can use it to indicate None

def get_measurements_from_files(measurements_files_dir) -> dict:
    """
        if measurements_files_dir contains more than 1 file -
        accumulate measurements value per tensor from all files.

        measurements in file is a json object in the following format, for batch_gemm layers :
        { <layer name> : { "params: : { "weight" : weight1},
                           "output" : [out1, ... , outM]
                         }
        }
        Where -
        weight1 is the max abs value of weight tensor,
        and outX is the max abs value of the output tensor in index X.

        Non batch-gemm layers may have different format.

        create dict with pairs as - { <layer name> : { 'outputs' : [out1maxAbs,...,outMmaxAbs],
                                                       'weights' : [weight1maxAbs]
                                                      }
                                     }
        return dict
    """

    print("Getting model measurements from directory {}".format(measurements_files_dir))
    global_maxabs_per_op = {} # dictionary to return
    files = os.listdir(measurements_files_dir)
    for filename in files:
        name = os.path.join(measurements_files_dir, filename)
        if os.path.isdir(name):
            continue
        if filename.endswith('.json'):
            with open(name, 'r') as json_file:
                print("Taking measurements from file {}".format(filename))
                ranges = json.load(json_file)
                for op in ranges.items():
                    op_name = op[0]
                    op_stat = op[1] # dictionary containing stats per tensor
                    if op_name in global_maxabs_per_op:
                        # update maxAbs value if needed for inputs/outputs/weights/bias
                        if OUTPUT in op_stat:
                            val = 0
                            if isinstance(op_stat[OUTPUT], list): # output may be list or single value
                                for i, _ in enumerate(op_stat[OUTPUT]):
                                    if op_stat[OUTPUT][i] == None:
                                        continue
                                    if op_stat[OUTPUT][i] > global_maxabs_per_op[op_name][OUTPUTS][i]:
                                        global_maxabs_per_op[op_name][OUTPUTS][i] = op_stat[OUTPUT][i]
                            else:
                                if op_stat[OUTPUT] > global_maxabs_per_op[op_name][OUTPUTS][0]:
                                    global_maxabs_per_op[op_name][OUTPUTS][i] = op_stat[OUTPUT]
                        if PARAMS in op_stat:
                            if WEIGHT in op_stat[PARAMS]:
                                if op_stat[PARAMS][WEIGHT] > global_maxabs_per_op[op_name][WEIGHTS][0]:
                                    global_maxabs_per_op[op_name][WEIGHTS][0] = op_stat[PARAMS][WEIGHT]
                    else: # first/single file case
                        global_maxabs_per_op[op_name] = {}
                        if OUTPUT in op_stat :
                            global_maxabs_per_op[op_name][OUTPUTS] = []
                            if isinstance(op_stat[OUTPUT], list): # output may be list or single value
                                for val in op_stat[OUTPUT]:
                                    if val == None:
                                        val = NONE_INDICATION
                                    global_maxabs_per_op[op_name][OUTPUTS].append(val)
                            else:
                                global_maxabs_per_op[op_name][OUTPUTS].append(op_stat[OUTPUT])
                        if PARAMS in op_stat:
                            if WEIGHT in op_stat[PARAMS]:
                                global_maxabs_per_op[op_name][WEIGHTS] = []
                                global_maxabs_per_op[op_name][WEIGHTS].append(op_stat[PARAMS][WEIGHT])

    if LM_HEAD in global_maxabs_per_op : # lm_head layer isn't planned to work with fp8
        del global_maxabs_per_op[LM_HEAD]

    return global_maxabs_per_op


def load_measurements_to_model(model : torch.nn.Module , ranges):
    ranges_copy = dict() # this is for dumping the ranges for json
    def load_layer_measurements(model, op_max_values, layer_name, tensor_string):
        if tensor_string in op_max_values:
            for i, val in enumerate(op_max_values[tensor_string]):
                if not layer_name in ranges_copy:
                    ranges_copy[layer_name] = dict()
                if not tensor_string in ranges_copy[layer_name]:
                    ranges_copy[layer_name][tensor_string] = []
                # adjust layer name to be same as the model
                name = layer_name[len(PREFIX):] if layer_name.startswith(PREFIX) else layer_name
                name = name.replace(".h.", ".")
                nmin = name + ".{}".format(i) + ".min_val"
                model._buffers[RANGES][tensor_string][nmin] = torch.tensor(0) # min value is always 0 in fp8 flow
                model._non_persistent_buffers_set.discard(nmin)
                nmax = name + ".{}".format(i) + ".max_val"
                model._buffers[RANGES][tensor_string][nmax] = torch.tensor(val)
                model._non_persistent_buffers_set.discard(nmax)
                ranges_copy[layer_name][tensor_string].append((0.0, val))

    if len(ranges) == 0 :
        print("Got empty ranges, not loading measurements to model")
        return model
    print("Loading measurements to model")
    model._buffers[RANGES] = dict({INPUTS : dict(), OUTPUTS : dict(), WEIGHTS : dict()})
    for layer_name, op_max_values in ranges.items():
        load_layer_measurements(model, op_max_values, layer_name, INPUTS)
        load_layer_measurements(model, op_max_values, layer_name, OUTPUTS)
        load_layer_measurements(model, op_max_values, layer_name, WEIGHTS)
    _dump_ranges_to_json(ranges_copy, "bloom_7b1_ranges_updated_june_21.json")
    return model



def modify_measured_model_bloom(model : torch.nn.Module , ranges):
    """
    Writes the min/max values as named buffers inside the model,
    adjusted to measurements format extracted from files supplied by Algo.
    This complies with behavior done in resnet50/Greco.
    :param model: pytorch model
    :param ranges: dynamic_range dict
    """
    print("Modifying model with measurements")
    for layer_name, op_max_values in ranges.items():
        for i, val in enumerate(op_max_values[OUTPUTS]):
            nmin = layer_name + ".{}".format(i) + ".min_val"
            model._buffers[nmin] = torch.tensor(0)
            model._non_persistent_buffers_set.discard(nmin)
            nmax = layer_name + ".{}".format(i) + ".max_val"
            model._buffers[nmax] = val
            model._non_persistent_buffers_set.discard(nmax)

    return model

def load_measurements_to_model_from_path(model, path):
    print("Loading measurements from path {}".format(path))
    measurements = get_measurements_from_files(path)
    load_measurements_to_model(model, measurements)

def _dump_ranges_to_json(ranges_copy, json_filename):
    import json
    with open(json_filename, 'w') as f:
        json.dump(ranges_copy, f, indent=4)
