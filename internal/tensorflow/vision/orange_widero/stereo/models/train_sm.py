"""
This code is based on: mepy_algo/appcode/DL/freespace/training/start_sagemaker_v2.py
"""

import argparse
import os
import sys
import json
import shutil
import tempfile

from stereo.interfaces.implements import load_dataset_attributes
from stereo.common.general_utils import Struct, tree_base
from stereo.models.sm.sm_utils import create_valid_job_name
from stereo.models.sm.train_estimator import main as train_estimator_main


def get_source_dir(stereo_root, external_modules=None):
    tmp_dir = tempfile.mkdtemp()
    shutil.copytree(os.path.join(stereo_root, "stereo"), os.path.join(tmp_dir, "stereo"))
    if external_modules is not None:
        import importlib
        for module_name in external_modules:
            module = importlib.import_module(module_name)
            module_dir = os.path.dirname(module.__file__)
            if '__init__.py' in os.listdir(module_dir):
                shutil.copytree(module_dir, os.path.join(tmp_dir, module_name), ignore=shutil.ignore_patterns('*.git*'),
                                ignore_dangling_symlinks=True)
            else:
                shutil.copyfile(module.__file__, os.path.join(tmp_dir, module.__file__.split('/')[-1]))
    return tmp_dir


def run_local(args):
    model_dir = os.path.join('/tmp', args.name)
    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)

    args_dict = {'json_path': args.json_path, 'local': args.local, 'name': args.name, 'model_dir': model_dir,
                 'batch_size': args.batch_size, 'eval_steps': args.eval_steps, 'eval_iters': args.eval_iters,
                 'max_steps': args.max_steps, 'num_epochs': args.num_epochs, 'pipe_mode': args.pipe_mode,
                 'verbose': args.verbose}

    args = Struct(args_dict)
    train_estimator_main(args)


def run_menta(args, conf=None):
    from stereo.models.menta.menta_utils import generate_menta_params
    from menta.src import backend as K
    import menta.main as menta_main

    if args.menta_from_ckpt:
        # Just generate MENTA checkpoint from an Estimator checkpoint of this model
        args.batch_size = 1
        args.max_steps = 1
        args.eval_iters = 2
        args.eval_steps = 1

    if args.local:
        output_path = os.path.join('/tmp', "menta_" + args.name)
        if os.path.exists(output_path) and os.path.isdir(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)

        # output_path = os.path.join("/homes/davidn/gitlab/stereo/sandbox/menta_runs", args.name)
        # os.makedirs(output_path, exist_ok=True)
    else:
        model_dir = os.path.join(conf['model_base_path'], args.name)
        output_path = os.path.join(model_dir, "menta")

    menta_params = generate_menta_params(output_path, conf, **vars(args))
    model_params = {"menta": menta_params}

    model_params.update({
        'json_path': args.json_path,
        'local': args.local,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'pipe_mode': args.pipe_mode,
        'verbose': args.verbose,
        'menta_from_ckpt': args.menta_from_ckpt
    })

    model_params_path = os.path.join("/tmp", "menta_params_" + os.path.basename(args.json_path))
    with open(model_params_path, 'w') as f:
        f.write(json.dumps(model_params, indent=2))

    os.environ['MENTA_SKIP_DLO'] = '1'
    os.environ['AWS_ROLE'] = 'sagemaker-stereo'
    sys.argv = [menta_main.__file__,
                "train",
                "--params",
                model_params_path]

    if not args.local:
        sys.argv.append("--aws")

    # sys.argv = [menta_main.__file__,
    #             "train",
    #             "--resume",
    #             menta_params['menta']['train']['output']]

    try:
        menta_main.main()
    except Exception as e:
        K.clear_session()
        raise e


def run_sagemaker(args, conf):
    from nebula.sagemaker import TensorFlow
    from sagemaker.session import s3_input, ShuffleConfig

    # Output locations
    model_dir = os.path.join(conf['model_base_path'], args.name)
    output_path = os.path.join(model_dir, "output_")
    custom_code_upload_location = os.path.join(conf['model_base_path'], "custom_code")
    source_dir = get_source_dir(args.stereo_root, conf.get('external_modules'))

    # Sagemaker training job name
    sm_job_name = create_valid_job_name(args.name)

    # IAM execution role that gives SageMaker access to resources in your AWS account.
    role = 'arn:aws:iam::771416621287:role/sagemaker-stereo'

    # Input mode
    input_mode = 'Pipe' if args.pipe_mode else 'File'

    # Set the data pipes
    print("Creating channels:")
    pipes = {}
    shuffle_config = ShuffleConfig(
        conf['data_params'].get('random_seed', 1234)
    )
    datasets = conf['data_params']['datasets']

    for section in datasets.keys():
        dataset_attributes = load_dataset_attributes(conf['data_params']['datasets'][section])[0]
        comp_type = "Gzip" if dataset_attributes['compressed'] else None
        pipes['train_%s' % section] = s3_input(os.path.join(dataset_attributes['s3'], "train"),
                                               shuffle_config=shuffle_config, compression=comp_type)
        pipes['test_%s' % section] = s3_input(os.path.join(dataset_attributes['s3'], "test"),
                                              shuffle_config=None, compression=comp_type)
        for p in pipes.keys():
            print("{}: {}".format(p, pipes[p].config['DataSource']['S3DataSource']['S3Uri']))

    # These are entered as command line arguments
    hyperparameters = {
        'json_path': args.json_path,
        'name': sm_job_name,
        'batch_size': args.batch_size,
        'eval_steps': args.eval_steps,
        'eval_iters': args.eval_iters,
        'max_steps': args.max_steps,
        'num_epochs': args.num_epochs,
        'pipe_mode': args.pipe_mode,
        'verbose': args.verbose,
    }

    # Set metric definitions
    metric_definitions = []
    if conf['model_params']['scalars_to_log']:
        for s in conf['model_params']['scalars_to_log']:
            metric_definitions.extend([{"Name": "train:{}".format(s),
                                        "Regex": "train:{} = (.*?)[,\s]".format(s)},
                                       {"Name": "eval:{}".format(s),
                                        "Regex": "eval:{} = (.*?)[,\s]".format(s)}])
    metric_definitions.append({"Name": "train:steps-per-second",
                               "Regex": "global_step\/sec: ([\d\.]*)"})
    if args.verbose:
        metric_definitions.append({
            "Name": "train:throughput",
            "Regex": "read_GB\/s = (.*?)[,\s]"
        })

    # Construct instance of sagemaker.Tensorflow
    entry_point = 'stereo/models/sm/train_estimator.py'
    py_version = 'py3'
    framework_version = '1.13.1'

    tensorflow = TensorFlow(role=role,
                            entry_point=entry_point,
                            source_dir=source_dir,
                            code_location=custom_code_upload_location,
                            debugger_hook_config=False,
                            train_instance_count=1,
                            train_instance_type=args.train_instance_type,
                            train_use_spot_instances=True,
                            train_max_wait=28*24*60*60,
                            framework_version=framework_version,
                            input_mode=input_mode,
                            script_mode=True,
                            py_version=py_version,
                            model_dir=model_dir,
                            output_path=output_path,
                            train_max_run=28*24*60*60,
                            hyperparameters=hyperparameters,
                            distributions=None,
                            metric_definitions=metric_definitions,
                            username=os.getenv('USERNAME'),
                            group='algo_stereo')
    if args.pipe_mode:
        tensorflow.fit(pipes, job_name=sm_job_name)
    else:
        tensorflow.fit(job_name=sm_job_name)


def run_estimator(args, conf=None):
    if args.local:
        run_local(args)
    else:
        run_sagemaker(args, conf)


def main():
    parser = argparse.ArgumentParser(description="Start stereo DNN training job on SageMaker")

    parser.add_argument('-r', '--stereo_root', help="source code directory. default: use script's path")
    parser.add_argument('-d', '--debug', action='store_true', help="local debug run: -l -b 1 -i 1 -e 10 -x 20 ")
    parser.add_argument('json_path', type=str, help="path to model JSON configuration file")
    parser.add_argument('-l', '--local', action='store_true', help="run train_estimator on local CPU with -p 0")
    parser.add_argument('-n', '--name', default="", help="run name (optional)")
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-e', '--eval_steps', type=int, default=5000)
    parser.add_argument('-i', '--eval_iters', type=int, default=500)
    parser.add_argument('-x', '--max_steps', type=int, default=1000000)
    parser.add_argument('-u', '--num_epochs', type=int, default=-1)
    parser.add_argument('-p', '--pipe_mode', type=int, default=1)
    parser.add_argument('-t', '--train_instance_type', choices=["ml.p3.2xlarge", "ml.p3.8xlarge", "ml.p3.16xlarge"],
                        default="ml.p3.8xlarge", help="for remote runs")
    parser.add_argument('-v', '--verbose', action='store_true', help="increase log verbosity")
    parser.add_argument('-m', '--menta', action='store_true', help="use MENTA framework")
    parser.add_argument('-mfc', '--menta_from_ckpt', action='store_true',
                        help="Generate MENTA checkpoint from an Estimator checkpoint of this model")
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    with open(args.json_path, 'rb') as f:
        conf = json.load(f)
    if not args.stereo_root:
        args.stereo_root = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-3])

    if args.name == '':
        args.name = os.path.splitext(os.path.split(args.json_path)[1])[0]

    if args.debug:
        args.local = True
        args.batch_size = 1
        args.eval_iters = 1
        args.eval_steps = 10
        args.max_steps = 20

    args.json_path = os.path.relpath(os.path.abspath(args.json_path), args.stereo_root)
    if args.local:
        args.pipe_mode = 0

    if args.menta:
        run_menta(args, conf)
    else:
        run_estimator(args, conf)


if __name__ == '__main__':
    main()
