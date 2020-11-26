from stereo.data.cluster_utils import cluster_full_clip
from stereo.common.common_utils import tree_base, shared_env_python
from stereo.data.netbatch_utils import nb_spawn
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="cluster clip")

    parser.add_argument('--clip_name', type=str, help="name of front clip", default='')
    parser.add_argument('--out_dir', type=str, help='save directory', default='/mobileye/algo_RP/jeff/clustering')
    parser.add_argument('--predump_dir', type=str, help='predump directory',
                        default='/mobileye/algo_STEREO3/stereo/data/data_eng')
    parser.add_argument('--chunk_size', type=int, help='how many gis should each chunk be', default=40)
    parser.add_argument('--alt_lidar_dir', type=str, help='alternative lidar dir (not from predump index)', default='')
    parser.add_argument('--alt_gtem_dir', type=str, help='alternative gtem dir (not from predump index)', default='')
    parser.add_argument('--part_ind', help='Index of clip part to process in', default='')
    parser.add_argument('--part_num', help='Number of clip parts to split the processing to', default='')

    args = vars(parser.parse_args())

    clip_name = args['clip_name']
    predump_dir = args['predump_dir']

    if len(clip_name) > 0:
        out_dir = os.path.join(args['out_dir'], clip_name)
        cluster_full_clip(clip_name, predump_dir, out_dir, args['chunk_size'], alt_lidar_dir=args['alt_lidar_dir'],
                          alt_gtem_dir=args['alt_gtem_dir'])
    else:
        fp = '/homes/nadavs/temp/done'
        #lst = open(fp, 'r').read().split('\n')[:-1][::20]
        lst = ['18-12-26_12-11-29_Alfred_Front_0090']
        nb_task = "cluster_dump"
        nb_qslot = "ALGO_GEOM"
        nb_mem_gb = 16

        exec_path = os.path.join(tree_base(), 'stereo', 'data', 'cluster_clip.py')
        env_python = shared_env_python()
        nb_spawn(lst, exec_path, args, args['predump_dir'], nb_task, nb_qslot, nb_mem_gb, 1, python_exec=env_python)