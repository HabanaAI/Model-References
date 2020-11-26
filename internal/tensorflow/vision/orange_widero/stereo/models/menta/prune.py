import os
import sys
import argparse
import subprocess
from tensorflow.python import keras
from stereo.common.general_utils import tree_base
from stereo.common.s3_utils import my_listdir
from stereo.models.sm.sm_setup import sm_setup
os.environ['USE_TF_KERAS'] = 'True'


def parse_dlo_args(menta_model_path, output_path):
    import menta.dlo as menta_dlo
    from menta.dlo import parse_args as menta_dlo_parse_args

    sys.argv = [menta_dlo.__file__,
                "quantize",
                "-mc",
                menta_model_path,
                "-o",
                output_path,
                "-j",
                os.path.join(os.path.dirname(__file__), "quantization_template.json")]
    return menta_dlo_parse_args()


def copy_if_s3(path, recursive=False, target_dir='/tmp'):
    if path.startswith("s3"):
        target_path = os.path.join(target_dir, path.split('/')[-1])
        command = ["aws", "s3", "cp", path, target_path]
        if recursive:
            command.append('--recursive')
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        _ = p.communicate()
        path = target_path
    return path


def copy_to_s3(lcl, rmt, recursive=True):
    command = ["aws", "s3", "cp", lcl, rmt]
    if recursive:
        command.append('--recursive')
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        _ = p.communicate()


def get_prunning_inputs(args):

    from menta.src import backend as K
    from menta.dlo import get_keras_config, get_params_json, get_h5_path, get_or_set_load_menta_model_cb
    from menta.dlo import _pre_compile_fn, _post_compile_fn, get_data
    from menta.src.layers import ZeroPadding2D, Cropping2D, Maximum, Concatenate, Slice, Clamp, Identity, SplitLayer, \
        Conv2D, Conv2DTranspose, ResizeNearestNeighbor
    from menta.src.layers.custom_layers import TranslateFromCalibration, Correlation
    custom_layers = {"Conv2D": Conv2D, "Conv2DTranspose": Conv2DTranspose,
                     "ZeroPadding2D": ZeroPadding2D, "Cropping2D": Cropping2D, "Maximum": Maximum,
                     "Concatenate": Concatenate, "Slice": Slice, "Clamp": Clamp,
                     "Identity": Identity, "SplitLayer": SplitLayer,
                     "TranslateFromCalibration": TranslateFromCalibration,
                     "Correlation": Correlation, "ResizeNearestNeighbor": ResizeNearestNeighbor}

    keras_json_path = os.path.join(args.menta_checkpoint, "keras.json")
    params_json_path = os.path.join(args.menta_checkpoint, "params.json")

    py_files = list(filter(lambda x: x.endswith(".py"), my_listdir(args.menta_checkpoint)))
    if len(py_files) == 0:
        raise ValueError("Could not find model_py file at the menta_checkpoint.")
    elif len(py_files) > 1:
        raise ValueError("Found more than one '.py' file at the menta_checkpoint.")
    model_py_path = os.path.join(args.menta_checkpoint, py_files[0])
    keras_config = get_keras_config(path=keras_json_path)
    params_json = get_params_json(path=params_json_path)

    if args.checkpoint is None:
        h5_weights_path = get_h5_path(path=os.path.join(args.menta_checkpoint, "checkpoints"), name_pattern=None)
    else:
        h5_weights_path = get_h5_path(path=args.checkpoint_path)
    h5_weights_path = copy_if_s3(h5_weights_path)
    K._ALLOW_MIN_TRAIN_STEP = False  # To make sure losses that are dependant on min train step work from start.

    keras_model = keras.Model.from_config(config=keras_config, custom_objects=custom_layers)
    orig_layer_names = [l.name for l in keras_model.layers]
    orig_output_layer_names = keras_model.output_names
    init_menta_model_kwargs = {"params_json": params_json, "output_path": args.output, "model_py_path": model_py_path}
    # get_or_set_load_menta_model_cb(**init_menta_model_kwargs)
    pre_compile_fn_cb = lambda model: _pre_compile_fn(model, orig_layer_names, orig_output_layer_names,
                                                      init_menta_model_kwargs)
    post_compile_fn_cb = lambda model: _post_compile_fn(model)
    get_train_data_cb = lambda: get_data("train", init_menta_model_kwargs)
    get_test_data_cb = lambda: get_data("test", init_menta_model_kwargs)

    return {'json_file': keras_json_path, 'weights_file': h5_weights_path,
            'pre_compile_fn_cb': pre_compile_fn_cb, 'post_compile_fn_cb': post_compile_fn_cb,
            'get_train_data_cb': get_train_data_cb, 'get_test_data_cb': get_test_data_cb}


def get_prunning_cfg(args):
    from prunner.Engine.Prunner import CONFIG_FLAGS
    config = {CONFIG_FLAGS.USE_MULTI_PROCCESS: False,
              CONFIG_FLAGS.STATS_IMAGES_NEEDED: 10,
              CONFIG_FLAGS.LEAST_SQUARES_IMAGES_COUNT: 10,
              CONFIG_FLAGS.OPTIMIZER_ITERATIONS: 50,
              CONFIG_FLAGS.REPORT_BASE_LINE_ACCURACY: False,
              CONFIG_FLAGS.FINE_TUNE: False}
    return config


def run_sagemaker(args):

    from nebula.sagemaker import TensorFlow
    from stereo.models.train_sm import get_source_dir
    sm_job_name = args.menta_model.split('/')[-1] if args.name == '' else args.name
    sm_job_name = sm_job_name.replace('.', '-').replace('_', '-').replace('+', '-')
    model_dir = os.path.join(args.output_path, sm_job_name)
    output_path = os.path.join(model_dir, "output_")
    custom_code_upload_location = os.path.join(model_dir, "custom_code")
    os.environ["DLO_IMPORT_PATH"] = "/mobileye/shared/DLO/last/dl/dl-optimizer"
    sys.path.insert(0, os.environ["DLO_IMPORT_PATH"])
    dlo_pckgs = ['app_builder', 'convert_maffe_to_tf', 'custom_dataset_feeder', 'dl_compiler_api',
                 'dl_emulator_api', 'dlo_internal', 'dl_sdk', 'dlstudio', 'kitgen', 'me_custom_layers',
                 'modeloptimizer']
    local_pckgs = ['menta', 'prunner']
    source_dir = get_source_dir(tree_base(), dlo_pckgs + local_pckgs)

    # IAM execution role that gives SageMaker access to resources in your AWS account.
    role = 'arn:aws:iam::771416621287:role/sagemaker-stereo'

    hyperparameters = {
        'menta_model': args.menta_model,
        'name': sm_job_name,
        'batch_size': args.batch_size,
        'output_path': output_path,
        'req_file': 's3://mobileye-habana/mobileye-team-stereo/prune_models/requirements.txt',
    }

    entry_point = 'stereo/models/menta/prune.py'
    py_version = 'py3'
    framework_version = '1.13.1'

    tensorflow = TensorFlow(role=role,
                            entry_point=entry_point,
                            source_dir=source_dir,
                            code_location=custom_code_upload_location,
                            train_instance_count=1,
                            train_instance_type='ml.p3.2xlarge',
                            train_use_spot_instances=True,
                            train_max_wait=864000,
                            framework_version=framework_version,
                            input_mode='File',
                            script_mode=True,
                            py_version=py_version,
                            model_dir=model_dir,
                            output_path=output_path,
                            train_max_run=864000,
                            hyperparameters=hyperparameters,
                            distributions=None,
                            metric_definitions=[],
                            username=os.getenv('USERNAME'),
                            group='algo_stereo')
    tensorflow.fit(job_name=sm_job_name)


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-m', '--menta_model', default='', type=str, help="directory of the menta model")
    parser.add_argument('-j', '--model_name', default='', type=str, help="stereo model name")
    parser.add_argument('-n', '--name', type=str, default='', help="job name")
    parser.add_argument('-r', '--req_file', type=str, default='')
    parser.add_argument('-o', '--output_path', type=str, help="")
    parser.add_argument('--model_dir', type=str, help="")
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-d', '--dry_run', action='store_true')
    parser.add_argument('-l', '--local', action='store_true')
    parser.add_argument('-s', '--sagemaker', action='store_true')
    parser.add_argument('-g', '--gpus', type=str, help="gpus", default='0')
    args = parser.parse_args()
    assert not (args.sagemaker and args.dry_run)
    if args.menta_model == '':
        from stereo.models.menta.menta_utils import generate_menta_checkpoint
        assert args.model_name != ''
        s3_menta_model = os.path.join("s3://mobileye-habana/mobileye-team-stereo", "menta_models", args.model_name)
        generate_menta_checkpoint(args.model_name,
                                  s3_dst=s3_menta_model,
                                  use_existing_model=False)
        args.menta_model = s3_menta_model
    if args.sagemaker:
        run_sagemaker(args)
    menta_model_path, output_path = args.menta_model, args.output_path
    lcl_output_path = output_path
    if not args.local:
        req_file = None if args.req_file == '' else copy_if_s3(args.req_file)
        menta_model_path = copy_if_s3(menta_model_path, recursive=True)
        lcl_output_path = os.path.join('/opt/ml/model/', output_path.split('/')[-1]) if output_path.startswith("s3") else output_path
        sm_setup(req_file)
    dlo_args = parse_dlo_args(menta_model_path, lcl_output_path)

    prunning_inputs = get_prunning_inputs(dlo_args)
    prunning_cfg = get_prunning_cfg(dlo_args)

    if args.dry_run:
        os.environ['DRY_RUN'] = 'True'
    else:
        os.environ['USE_AWS'] = 'True'
    from prunner.Engine.Prunner import PRUNNING_TYPE, Prunner, CONFIG_FLAGS, TRAINING_SCHEME, ROUNDING_SCHEME
    from menta.src.layers import ZeroPadding2D, Cropping2D, Maximum, Concatenate, Slice, Clamp, Identity, SplitLayer, \
        Conv2D, Conv2DTranspose, ResizeNearestNeighbor
    from menta.src.layers.custom_layers import TranslateFromCalibration, Correlation
    custom_layers = {"Conv2D": Conv2D, "Conv2DTranspose": Conv2DTranspose,
                     "ZeroPadding2D": ZeroPadding2D, "Cropping2D": Cropping2D, "Maximum": Maximum,
                     "Concatenate": Concatenate, "Slice": Slice, "Clamp": Clamp,
                     "Identity": Identity, "SplitLayer": SplitLayer,
                     "TranslateFromCalibration": TranslateFromCalibration,
                     "Correlation": Correlation, "ResizeNearestNeighbor": ResizeNearestNeighbor}

    prunner = Prunner(json_file=prunning_inputs['json_file'],
                      weights_file=prunning_inputs['weights_file'],
                      artifacts_path=lcl_output_path,
                      KR=0.4,
                      pruning_type=PRUNNING_TYPE.CONVOLUTION,
                      train_batch_size=args.batch_size, val_batch_size=args.batch_size,
                      train_seq=prunning_inputs['get_train_data_cb'], val_seq=prunning_inputs['get_test_data_cb'],
                      custom_layers_dict=custom_layers,
                      gpus=[int(g) for g in args.gpus.split(',')])
    prunner.configure(prunning_cfg)
    user_layers_break_types = ['Correlation', 'TranslateFromCalibration', 'Slice', 'Maximum', 'ResizeNearestNeighbor']
    prunner.init(fine_tune_lr=1e-5, losses=None, steps_per_epoch=10, validation_steps=10, epochs=10,
                 pre_compile_callback=prunning_inputs['pre_compile_fn_cb'],
                 post_compile_callback=prunning_inputs['post_compile_fn_cb'],
                 user_layers_break_types=user_layers_break_types)
    prunner.prune()
    if lcl_output_path != output_path:
        copy_to_s3(lcl_output_path, output_path)


if __name__ == '__main__':
    main()
