#!/usr/bin/python

import os
import sys
import argparse
import json

from stereo.common.general_utils import tree_base, shared_env_python
from stereo.common.netbatch_utils import nb_spawn
from stereo.data.dataset_utils import write_view_dataset
from stereo.data.predump_v2 import PreDumpIndexV2


def main():
    parser = argparse.ArgumentParser(description='Dump a view-dataset for a list of surround clips.')
    parser.add_argument('-i', '--inp', help='Input clip name or clip list. If empty all valid predump_dir', default='')
    parser.add_argument('-o', '--out_dir', help='Output path. Must exist', default='')
    parser.add_argument('-p', '--predump_dir', help='Where mest, gtem itrks and calibrations were stored', default='')
    parser.add_argument('--part_ind', help='Index of clip part to process in', default='')
    parser.add_argument('--part_num', help='Number of clip parts to split the processing to', default='')
    parser.add_argument('-g', '--must_have_gtd', help='must have gtd for masking', default='true')
    parser.add_argument('-v', '--views', help='Comma-separated list of view names to process and dump', default='')
    parser.add_argument('-nbt', '--nb_task', help='Netbatch task name. Set this to use netbatch', default='')
    parser.add_argument('-nbq', '--nb_qslot', help='Netbatch qslot (default: ALGO_STEREO)', default='ALGO_STEREO')
    parser.add_argument('-nbm', '--nb_mem_gb', help='Netbatch requested memory in GB (default: 8)', default=8)
    parser.add_argument('-ve', '--verbosity', help='Verbosity level (default: 1)', default=1)
    parser.add_argument('-j', '--json_path', help='Config file (provides args, cmdline args override).', default='')
    parser.add_argument('--test_run', action='store', type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help='run only on first five frames of clip', default=False)
    parser.add_argument('--require_lidar', action='store', type=lambda x: (str(x).lower() not in ['false', '0', 'no']),
                        help='Require lidar_reader to be initialized or clip fails', default=True)
    args = vars(parser.parse_args())

    test_run = args['test_run']

    if args['json_path']:  # replace empty-string arguments with those that exist in the json file
        print('Loading configuration file %s' % args['json_path'])
        with open(args['json_path']) as f:
            conf = json.load(f)
        for arg in args:
            if args[arg] == '' and arg in conf:
                args[arg] = conf[arg]
        args['json_path'] = ''

    part_ind = 0 if args['part_ind'] == '' else int(args['part_ind'])
    part_num = 1 if args['part_num'] == '' else int(args['part_num'])

    must_have_gtd = args['must_have_gtd'] == 'true'

    if not os.path.exists(args['out_dir']):
        os.makedirs(args['out_dir'])
    assert(os.path.isdir(args['out_dir']) and os.path.isdir(args['predump_dir']) and args['views'])
    print('Loading pre-dump index for dir %s' % args['predump_dir'])
    predump_index = PreDumpIndexV2(args['predump_dir'])

    def map_func(clip_name, part_ind__, part_num__, must_have_gtd__):
        return write_view_dataset(clip_name, part_ind__, part_num__,
                                  args['views'], predump_index,
                                  args['out_dir'], verbosity=args['verbosity'],
                                  test_run=test_run, must_have_gtd=must_have_gtd__)

    if not (args['inp'].endswith('.lst') or args['inp'].endswith('.list') or args['inp'] == ''):  # single clip mode
        return_status = map_func(args['inp'], part_ind, part_num, must_have_gtd)
        print("return_status: ", return_status)
        sys.exit(return_status)
    elif test_run:
        # five frame debugging mode
        return_status = map_func(predump_index.availble_clips[0], part_ind, part_num, must_have_gtd)
        print("return_status: ", return_status)
        sys.exit(return_status)
    else:  # clip list mode
        if args['inp'] == '':
            lst = predump_index.availble_clips
            print('Processing all valid clips in predump_dir that also have lidar. Found %d clips' % len(lst))
        else:
            with open(args['inp'], 'r') as f:
                lst = f.read().splitlines()
            print('Processing list containing %d clip names' % len(lst))

        if args['nb_task']:  # netbatch mode
            print('Spawning list via netbatch')
            nb_task, nb_qslot, nb_mem_gb, args = nb_spawn_args(args)
            exec_path = os.path.join(tree_base(), 'stereo', 'data', 'make_view_dataset.py')
            env_python = shared_env_python()
            nb_spawn(lst, exec_path, args, args['out_dir'], nb_task, nb_qslot, nb_mem_gb, part_num, python_exec=env_python)
        else:  # local mode
            print('Looping over list locally in single process mode')
            for clip_name_ in lst:
                for part_ind_ in range(part_num):
                    map_func(clip_name_, part_ind_, part_num)


def nb_spawn_args(args):
    nb_task = args['nb_task']
    nb_qslot = args['nb_qslot']
    nb_mem_gb = int(args['nb_mem_gb'])
    args['nb_task'], args['nb_qslot'], args['nb_mem_gb'] = '', '', ''  # Set all args that shouldn't be passed
    return nb_task, nb_qslot, nb_mem_gb, args                          # to nb_spawned process to an empty string


if __name__ == '__main__':
    main()
