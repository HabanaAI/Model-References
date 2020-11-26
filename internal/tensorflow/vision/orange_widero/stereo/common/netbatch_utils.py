from os.path import join
from stereo.data.clip_utils import clip_path as clip2path
from stereo.common.general_utils import makedir_recursive, shared_env_python
from stereo.common.gt_packages_wrapper import parse_pls
from me_jobs_runner import Task, dispatch_tasks, NETBATCH


def nb_spawn(lst, exec_path, args_dict, nb_out_dir, nb_task, nb_qslot,
             nb_mem_gb=8, nb_part_num=1, python_exec='/usr/bin/python', extra_args=None, input_dir=None):

    if input_dir is not None and isinstance(input_dir, list):
        assert len(input_dir) == len(lst)
        use_clip_name = False
    else:
        use_clip_name = True
    input_folders_or_clips = []
    cmd_lines = []
    for i,clip_name in enumerate(lst):
        for nb_part_ind in range(nb_part_num):
            if 'inp' in args_dict.keys():
                args_dict['inp'] = clip_name
            else:
                args_dict['clip_name'] = clip_name
            args_dict['part_ind'] = str(nb_part_ind)
            args_dict['part_num'] = str(nb_part_num)
            cmd_line = python_exec + ' ' + exec_path
            for arg in args_dict:
                if args_dict[arg] != '':
                    cmd_line += ' --%s %s' % (arg, str(args_dict[arg]))
            if extra_args is not None:
                extra_args_clip = extra_args[i]
                for arg in extra_args_clip:
                    if extra_args_clip[arg] != '':
                        cmd_line += ' --%s %s' % (arg, str(extra_args_clip[arg]))
            cmd_lines.append(cmd_line)
            if use_clip_name:
                input_folders_or_clips.append(clip_name)
            else:
                input_folders_or_clips.append(input_dir[i])
    if input_dir is not None and not isinstance(input_dir, list):
        input_folders_or_clips = input_dir
    task = Task(task_name=nb_task, commands=cmd_lines, input_folders_or_clips=input_folders_or_clips,
                output_folders=nb_out_dir, timeout_minutes=0, memory_gb=nb_mem_gb)
    dispatch_tasks([task], task_name=nb_task, log_dir=join(nb_out_dir, 'nb_logs'), qslot=nb_qslot, dispatch_type=NETBATCH)


def run_nb(args, exec_path):
    nb_task = args['nb_task']
    nbm = 8 if args['nb_mem_gb'] == '' else int(args['nb_mem_gb'])
    args['nb_task'] = ''
    args['single_core'] = 'true'
    lst = []
    with open(args['clip_list_file']) as clip_list:
        for l in clip_list:
            lst.append(l.split('\n')[0])
    for clip in lst:
        clip_name = clip.split('/')[-1]
        makedir_recursive(args['out_dir'], clip_name)
    if lst[0].endswith('.pls'):
        input_dirs = []
        for pls in lst:
            input_dirs.append(clip2path(parse_pls(pls).clip_name.values[0]))
    else:
        input_dirs = None
    python_exe = shared_env_python()
    part_num = 1 if args['part_num'] == '' else int(args['part_num'])
    nb_spawn(lst, exec_path, args, args['out_dir'], nb_task, args['nb_qslot'],
             nbm, part_num, python_exec=python_exe, input_dir=input_dirs)
