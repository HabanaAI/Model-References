import numpy as np
import os
import json
import imp
from stereo.common.general_utils import tree_base


def write_frame_to_json(frame, filename):
    """
    write a dataset json format from the structure of a given frame
    :param frame: dict with ndarray values
    :param filename: json ouput filename
    """
    json_output = []
    json_output.append("{")
    keys_max_length = max((len(k) for k in frame.keys()))
    for k in frame.keys():
        if len(json_output) > 1:
            json_output[-1] += ","
        if frame[k].dtype == np.float64:
            dtype = np.dtype(np.float32).name
        elif frame[k].dtype.name.startswith("string") or frame[k].dtype.name.startswith("unicode"):
            dtype = "string"
        else:
            dtype = frame[k].dtype.name
        key_prop = {"dtype": dtype,
                    "shape": frame[k].shape}
        json_output.append(
            "    {key:{width}}: {json_content}".format(key='"{}"'.format(k), width=keys_max_length + 2,
                                                       json_content=json.dumps(key_prop))
        )
    json_output.append("}")

    with open(filename, 'w') as f:
        f.write(os.linesep.join(json_output))
        print("Dataset format was written to: {}".format(filename))
    print(os.linesep.join(json_output))


def get_prep_func(prep_func_name):
    prep_func_path = os.path.join(tree_base(), 'stereo/data/prep', prep_func_name + '.py')
    prep_func = imp.load_source('prep', prep_func_path).prep_frame
    return prep_func


def write_json(out_dir, dataset_index, prep, view_names=None):
    prep_func_ = get_prep_func(prep)
    frames_list = dataset_index.test_frames_list
    for i in np.arange(10):
        if view_names:
            frame = prep_func_(dataset_index, frames_list[i], view_names)
        else:
            frame = prep_func_(frames_list[i])
        if len(frame) > 0:
            frame = frame[0]
            ds_format_json = os.path.join(out_dir, "ds_format.json")
            return write_frame_to_json(frame, ds_format_json)
    raise Exception("Couldn't prepare any frame..")


def read_conf(json_path):
    with open(json_path, 'rb') as fp:
        conf = json.load(fp)
    return conf

