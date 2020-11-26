from stereo.data.cluster_utils import label_clip_with_nss
from stereo.common.common_utils import tree_base, shared_env_python
from stereo.data.netbatch_utils import nb_spawn
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="cluster clip")

    parser.add_argument('--clip_name', type=str, help="name of front clip", default='')
    parser.add_argument('--out_dir', type=str, help='save directory', default='/mobileye/algo_RP/jeff/clustering/temp')
    parser.add_argument('--part_ind', help='Index of clip part to process in', default='')
    parser.add_argument('--part_num', help='Number of clip parts to split the processing to', default='')

    args = vars(parser.parse_args())

    clip_name = args['clip_name']

    if len(clip_name) == 0:
        nb_task = 'nss_relabel'
        nb_qslot = 'ALGO_STEREO'
        nb_mem_gb = 8
        exec_path = os.path.join(tree_base(), 'stereo', 'data', 'relabel_clip_with_nss.py')
        env_python = shared_env_python()
        nb_spawn(['19-05-27_11-39-02_Alfred_Front_0056'], exec_path, args, nb_out_dir=args['out_dir'],
                 nb_task=nb_task, nb_qslot=nb_qslot, nb_mem_gb=nb_mem_gb, nb_part_num=1,
                 python_exec=env_python)
    else:
        out_dir = args['out_dir']
        label_clip_with_nss(clip_name, out_dir)
    # else:
    #     fp = '/homes/nadavs/temp/done'
    #     #lst = open(fp, 'r').read().split('\n')[:-1][::20]
    #     lst = ['18-12-26_12-11-29_Alfred_Front_0090']
    #     nb_task = "cluster_dump"
    #     nb_qslot = "ALGO_GEOM"
    #     nb_mem_gb = 16
    #
    #     exec_path = os.path.join(tree_base(), 'stereo', 'data', 'cluster_clip.py')
    #     env_python = shared_env_python()
    #     nb_spawn(lst, exec_path, args, args['predump_dir'], nb_task, nb_qslot, nb_mem_gb, 1, python_exec=env_python)