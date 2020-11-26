#!/usr/bin/python

"""
This code is based on: mepy_algo/appcode/DL/freespace/training/start_sagemaker_v2.py
"""
import argparse
import os
import sys
import json
import s3fs
import random
import string
import numpy as np
from threading import Thread
import time
from datetime import datetime
import signal
from stereo.models.sm.sm_utils import make_name_aws_friendly, get_training_jobs, kill_instances
from stereo.models.sm.sm_utils import get_checkpoint_path
from stereo.common.general_utils import tree_base
from stereo.common.s3_utils import my_glob, my_open

from nebula.sagemaker import TensorFlow
from sagemaker.session import s3_input
from stereo.interfaces.implements import load_dataset_attributes

CAM_ABBREVIATIONS = {"main": "main",
                     "frontCornerLeft": "fcl",
                     "frontCornerRight": "fcr",
                     "rearCornerLeft": "rcl",
                     "rearCornerRight": "rcr",
                     "rear": "rear"}


def get_source_dir(stereo_root):
    import shutil
    import tempfile
    tmp_dir = tempfile.mkdtemp()
    shutil.copytree(os.path.join(stereo_root, "stereo"), os.path.join(tmp_dir, "stereo"))
    return tmp_dir


def write_manifest(s3_prefix, num_instances, idx, output_path, aws_model_name, suffix):
    tfrecord_list = my_glob(os.path.join(s3_prefix, '*.tfrecord'))
    tfrecord_list = np.array_split(np.array(tfrecord_list), num_instances)[idx].tolist()
    tfrecord_list = [f.split('/')[-1] for f in tfrecord_list]
    json_output = ['{"prefix": "%s/"}' % s3_prefix]
    json_output.extend(tfrecord_list)
    json_name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(32)])
    json_path = os.path.join(output_path, '%s_%s_%s_%03d.manifest' % (json_name, aws_model_name, suffix, idx))
    with my_open(json_path , 'w') as f:
        f.write(u'[\n%s,\n' % json_output[0])
        for s in json_output[1:-1]:
            f.write(u'"%s",\n' % s)
        f.write(u'"%s"\n' % json_output[-1])
        f.write(u']\n')
    return json_path

def run_sagemaker(conf, json_path, out_dir, restore_iter, stereo_root, loss, dataset, cam, idx, num_instances,
                  aws_model_name, blacklist, suffix, debug, probe_name, channel, save):

    random.seed(12345)

    # model_dir = os.path.join(conf['model_base_path'], model_name)
    # output_path = os.path.join(model_dir, "output_")
    # custom_code_upload_location = os.path.join(conf['model_base_path'], "custom_code")
    source_dir = get_source_dir(stereo_root)

    user_name = os.getenv('USERNAME')

    # Entry point script
    entry_point = 'stereo/evaluation/evaluate.py'

    role = 'arn:aws:iam::771416621287:role/sagemaker-stereo'
    # train_instance_type = 'ml.p2.xlarge'
    train_instance_type = 'ml.p3.2xlarge'
    custom_code_upload_location = 's3://mobileye-habana/mobileye-team-stereo/temp/custom_code'
    model_dir = out_dir
    output_path = 's3://mobileye-habana/mobileye-team-stereo/temp'
    # input_mode = 'File'
    input_mode = 'Pipe'
    metric_definitions = []

    pipes = {}

    dataset_attributes = load_dataset_attributes(dataset)[0]
    comp_type = "Gzip" if dataset_attributes['compressed'] else None
    manifest_file = write_manifest(os.path.join(dataset_attributes['s3'], channel), num_instances, idx, output_path, aws_model_name, suffix)
    pipes['test'] = s3_input(manifest_file, s3_data_type='ManifestFile',
                            shuffle_config=None, compression=comp_type)
    for p in pipes.keys():
        print("{}: {}".format(p, pipes[p].config['DataSource']['S3DataSource']['S3Uri']))

    # These are entered as command line arguments
    hyperparams = {
        'json_path': json_path,
        'out_dir': out_dir,
        'restore_iter': restore_iter,
        'idx': idx,
        'num_instances': num_instances,
        'cam': cam,
        "loss": loss,
        "dataset": dataset,
        'suffix': suffix,
        'debug': debug,
        'save': save,
        'probe_name': probe_name
    }
    if blacklist:
        hyperparams['blacklist'] = blacklist

    one_day = 60*60*24*1
    train_max_run = one_day

    py_version = 'py3'
    framework_version = '1.13.1'

    tensorflow = TensorFlow(entry_point=entry_point, role=role,
                            source_dir=source_dir,
                            code_location=custom_code_upload_location,
                            train_instance_count=1,
                            train_instance_type=train_instance_type,
                            train_use_spot_instances=True,
                            train_max_wait=train_max_run,
                            framework_version=framework_version,
                            input_mode=input_mode,
                            script_mode=True,
                            py_version=py_version,
                            model_dir=model_dir,
                            output_path=output_path,
                            train_max_run=train_max_run,
                            hyperparameters=hyperparams,
                            distributions=None,
                            metric_definitions=metric_definitions,
                            username=user_name,
                            group='algo_stereo'
                            )

    job_name = 'eval-'+str(idx)+'-'+CAM_ABBREVIATIONS[cam]+'-'+aws_model_name[-25:]
    # tensorflow.fit(job_name=job_name)
    job_name = tensorflow.fit(pipes, job_name=job_name)
    s3 = s3fs.S3FileSystem()
    s3.rm(manifest_file)
    print ("%s completed" % (job_name))

def run_local(args, model_name):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import shutil
    from stereo.evaluation.evaluate import main as evaluate
    from stereo.common.general_utils import Struct

    local_args = vars(args)
    local_args['local'] = True
    local_args['debug'] = False
    local_args['idx'] = 0
    local_args['num_instances'] = 2
    local_args['json_path'] = os.path.join(args.stereo_root, args.json_path)
    model_dir = os.path.join('/tmp', model_name)
    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)

    local_args['out_dir'] = os.path.join('/tmp', model_name)
    evaluate(Struct(local_args))
    local_args['idx'] = 1
    evaluate(Struct(local_args))
    from stereo.evaluation.probe_factory import probeFactory

    factory = probeFactory(probe_name=args.probe_name)
    probe = factory.get_probe_obj(local_args['json_path'], local_args['out_dir'], args.restore_iter, -1,
                                  int(local_args['num_instances']),
                                  args.cam,
                                  args.loss, args.dataset, args.blacklist, args.suffix, local_args['debug'], local=True)
    probe.concatenate()
    print ('Finished local run successfully!!\n output can be found at %s' % local_args['out_dir'])

def main():
    parser = argparse.ArgumentParser(description="Start stereo DNN training job on SageMaker")

    parser.add_argument('--json_path', type=str, help="path to JSON file defining the datasets, loss, "
                                                                   "architecture, hyper-parameters and gpu setup.")
    parser.add_argument('-o', '--out_dir', help='output directory for eval results',
                        default='/mobileye/algo_STEREO3/old_stereo/eval')
    parser.add_argument('-i', '--restore_iter',
                        help='number iterations of relevant checkpoint. If -1, then most recent',
                        default=-1)
    parser.add_argument('--stereo_root', help="source code directory. If not given, stereo_root is the determined by "
                                              "the script's path")
    parser.add_argument('--loss', help="ignore original loss used and use this loss instead for eval", default=None)
    parser.add_argument('--dataset', help="ignore original dataset and use this instead "
                                          "(path to folder containing test folder)", default=None)
    parser.add_argument('--num_instances', help='how many sagemaker instances to run', default=16)
    parser.add_argument('--cam', help='on which inference cam to run (same format as data_params in json)',
                        default='main', choices={"main", "frontCornerLeft", "frontCornerRight", "rearCornerLeft",
                                                 "rearCornerRight", "rear"})
    parser.add_argument('--blacklist', help='s3 location of clip list to exclude in stats', default=None)
    parser.add_argument('--suffix', help='suffix to add to eval npz filename after concat', default="")
    parser.add_argument('--debug', help='run only five iterations instead of full dataset', action='store_true')
    parser.add_argument('-l', '--local', help='run local', action='store_true')
    parser.add_argument('-s', '--save', help='save inference output', action='store_true')
    parser.add_argument('--probe_name', help='which probe child class to use', default='Analyzer')
    parser.add_argument('--channel', help='run evaluate on train or test set', default='test', choices={"test", "train"})
    parser.add_argument('--load_graph_from_code', help='load_graph_from_code', action='store_true')

    args = parser.parse_args()

    start_time = time.time()

    with open(args.json_path, 'rb') as f:
        conf = json.load(f)

    if args.blacklist is not None:
        if 's3' not in args.blacklist:
            raise ValueError("Blacklist file must be on s3")

    if args.out_dir == '/mobileye/algo_STEREO3/old_stereo/eval':
        if args.probe_name != 'Analyzer':
            raise ValueError("If you are using a custom probe, you must provide a different out_dir")
        if args.channel != 'test':
            raise ValueError('If you are running evaluation on the train set, you must provide a different out_dir')
        if args.suffix is "":
            if (args.loss is not None or args.dataset is not None) and not args.local:
                raise ValueError("When evaluating with an alternative loss or an alternative dataset, "
                             "you must provide either a unique suffix, a different out_dir or both")

    if args.loss is None:
        args.loss = conf["model_params"]["loss"]["name"]

    if not args.stereo_root:
        args.stereo_root = tree_base()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.json_path = os.path.relpath(os.path.abspath(args.json_path), args.stereo_root)

    if args.dataset is None:
        args.dataset = conf['data_params']['datasets'][args.cam]

    if args.debug:
        args.num_instances = 2
        args.suffix = 'debug'

    model_name = os.path.splitext(os.path.split(args.json_path)[1])[0]
    aws_model_name = make_name_aws_friendly(model_name)
    job_names = []

    model_s3 = '/'.join(conf['model_base_path'].split('/')[3:])
    checkpoint_dir = '/'.join([model_s3, model_name])
    restore_iter = int(args.restore_iter)

    checkpoint_path = get_checkpoint_path(checkpoint_dir, restore_iter)
    args.restore_iter = checkpoint_path.split('-')[-1]

    instance_time = datetime.now()

    if args.save:
        args.probe_name = 'save'

    if args.local:
        run_local(args, model_name)
        return 0

    threads = []
    for i in range(int(args.num_instances)):
        t = Thread(target=run_sagemaker, args=(conf, args.json_path, args.out_dir, args.restore_iter, args.stereo_root,
                                               args.loss, args.dataset, args.cam, i, int(args.num_instances),
                                               aws_model_name, args.blacklist, args.suffix, args.debug,
                                               args.probe_name, args.channel, args.save))
        t.daemon = True
        threads.append(t)
        job_name = 'eval-'+str(i)+'-'+CAM_ABBREVIATIONS[args.cam]+'-'+aws_model_name[-25:]
        job_names.append(job_name)

    for i,t in enumerate(threads):
        time.sleep(10)
        t.start()

    min_job_name = CAM_ABBREVIATIONS[args.cam]+'-'+aws_model_name[-25:]

    def signal_handler(signal, frame):
        print ("\nYou've let a mess! Cleaning it up now before Shai gets angry ...")
        kill_instances(instance_time, min_job_name)
        print ("\nThat's more like it :)")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    time.sleep(60)

    first_time = True

    while True:
        time.sleep(60)
        relevant_jobs = get_training_jobs(instance_time, min_job_name)
        if len(relevant_jobs) < int(args.num_instances):
            print ("\nInstances still being created. There are presently %d instances." % (len(relevant_jobs)))
            continue
        elif first_time:
            print ("\nAll instances created.")
            first_time = False
        all_jobs_completed = True
        for job in relevant_jobs:
            status = job['TrainingJobStatus']
            if status == 'Failed':
                raise ValueError("Job Failed")
            elif status == 'Stopped' or status == 'Stopping':
                raise ValueError("Job Stopped")
            elif status == 'InProgress':
                all_jobs_completed = False
                break
        if all_jobs_completed:
            break


        # for job in job_names:
        #     time.sleep(1)
        #     if check_job_completed(job):
        #         print "\njob %s completed" % (job)
        #         job_names.remove(job)


    print ("All jobs completed")
    aws_time = time.time()

    from stereo.evaluation.probe_factory import probeFactory

    factory = probeFactory(probe_name=args.probe_name)
    probe = factory.get_probe_obj(args.json_path, args.out_dir, args.restore_iter, -1, int(args.num_instances), args.cam,
                               args.loss, args.dataset, args.blacklist, args.suffix, args.debug)
    probe.concatenate()

    end_time = time.time()

    print ("Evaluate took %d seconds" % (end_time-start_time))
    print ("AWS portion: %d seconds" % (aws_time-start_time))
    print ("Concat portion: %d seconds" % (end_time-aws_time))

if __name__ == '__main__':
    main()
