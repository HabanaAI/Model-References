import sys
import os
import s3fs

from stereo.interfaces.implements import load_dataset_attributes
from stereo.common.general_utils import Struct, tree_base
from stereo.common.s3_utils import s3_copy
from stereo.models.sm.sm_utils import create_valid_job_name


def generate_menta_params(output_path, conf=None, **kwargs):
    menta_params = {
        "train": {
            "model_py": os.path.join(tree_base(), "stereo", "models", "menta", "vidar_menta.py"),
            "output": output_path,
            "gpus": -1,
            "max_steps": kwargs['max_steps'],
            "test_interval": kwargs['eval_steps'],
            "test_steps": kwargs['eval_iters'],
            "log_interval": 5,
            "custom_summary_interval": kwargs['eval_steps'],
            "save_interval": kwargs['eval_steps'],
            "save_best_only": False,
            "verbose": 2,
            "max_queue_size": 50,
            "workers": 2,
            "train_decay": 0.001
        },
        "predict": {
            "data": "/mobileye/algo_EVAL_NVME/andreyg/data/shuffle_big/test",
            "checkpoint": None,
            "gpus": 1,
            "output": "/mobileye/algo_OBJD/dank/VD3D/FID_forward/resnet_tests/test_v2",
            "batch_sizes": [
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                1
            ],
            "output_layers": None,
            "max_samples": None,
            "batch_limit": None,
            "sort": False,
            "workers": 2,
            "max_queue_size": 10
        },
        "export": {
            "output": "/homes/dank/repos/py/me/menta/None",
            "inputs_data_format": "channels_last",
            "outputs_data_format": "channels_first"
        },
        "aux": {
            "arch": "5Mid",
            "seed": 666,
            "log_device_placement": kwargs['verbose'],
            "dump_source_code": False,
            "use_multiprocessing": True,
            "load_by_name": False
        }
    }
    # "log_device_placement": False,
    if not kwargs['local'] or True:
        datasets = conf['data_params']['datasets']
        s3_paths = {}
        for section in datasets.keys():
            dataset_attributes = load_dataset_attributes(conf['data_params']['datasets'][section])[0]
            comp_type = "Gzip" if dataset_attributes['compressed'] else None
            s3_paths['train_%s' % section] = os.path.join(dataset_attributes['s3'], "train")
            s3_paths['test_%s' % section] = os.path.join(dataset_attributes['s3'], "test")

        aws_params = {
            "instance_type": kwargs['train_instance_type'],
            "dependencies": {
                "stereo": os.path.join(kwargs['stereo_root'], 'stereo')
            },
            "job_name": create_valid_job_name(kwargs['name']).replace("conf", "c"),
            "shuffle_seed": conf['data_params'].get('random_seed', int(os.urandom(4).hex(), 16)),
            "compression": comp_type,
            "s3_paths": s3_paths
        }
        menta_params.update({"aws": aws_params})

    return menta_params


def get_dlo_custom_objects():
    if "DLO_IMPORT_PATH" not in os.environ:
        os.environ["DLO_IMPORT_PATH"] = "/mobileye/shared/DLO/last/dl/dl-optimizer"
    sys.path.insert(0, os.environ["DLO_IMPORT_PATH"])
    from dl_sdk import KerasDeployment, DLConfigType
    custom_objects = KerasDeployment.get_custom_objects()
    return custom_objects


def generate_menta_checkpoint(model_name, s3_dst=None, use_existing_model=True):
    # TODO: note the possible cyclic imports here
    #  as train_sm import functions from menta_utils
    run_menta = True
    local_menta_model = os.path.join("/tmp", "menta_" + model_name)
    if s3_dst:
        fs = s3fs.S3FileSystem()
        if use_existing_model and fs.exists(s3_dst):
            print("Using existing MENTA model in {}".format(s3_dst))
            run_menta = False

    if run_menta:
        import stereo.models.train_sm as train_sm
        from stereo.common.general_utils import tree_base
        json_path = os.path.join(tree_base(), "stereo", "models", "conf", "stereo",
                                 model_name + ".json")
        sys.argv = [train_sm.__file__,
                    json_path,
                    "-m",
                    "-l",
                    "--menta_from_ckpt"]

        train_sm.main()
        if s3_dst:
            print("Copying MENTA model from {} to {}".format(local_menta_model, s3_dst))
            s3_copy(local_menta_model, s3_dst)

    menta_model = s3_dst if s3_dst else local_menta_model
    return menta_model

