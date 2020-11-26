from external_import import external_import
import os
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
with external_import(PATH):
    from stereo.common.visualization.vis_utils import view_pptk3d
import numpy as np
import argparse
from os.path import join
from os import listdir


def read_gi(pred_dir, clip_name, gi):
    frame = join(pred_dir, clip_name, '%s@%07d.dat' % (clip_name, gi))
    data = np.fromfile(frame, dtype=np.float32).reshape(-1, 5)
    pcd = data[:, :3]
    grayscale = data[:, 3]
    return pcd, grayscale


def get_gis(pred_dir, clip_name):
    gis = [int(frame.split('@')[-1].split('.dat')[0]) for frame in listdir(join(pred_dir, clip_name)) if '.dat' in frame]
    return gis


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='view vidar')
    parser.add_argument('-d', '--dir_name', help='dir_name', default='')
    parser.add_argument('-c', '--clip_name', help='clip_name', default='')
    parser.add_argument('-g', '--gi', help='gi', type=int)
    parser.add_argument('-t', '--save_to_tmp', action='store', type=lambda x: (str(x).lower() in ['true', '1', 'yes']))

    args = vars(parser.parse_args())

    gis = get_gis(args['dir_name'], args['clip_name'])
    assert args['gi'] in gis

    pcd, grayscale = read_gi(args['dir_name'], args['clip_name'], args['gi'])
    view_pptk3d(pcd, grayscale, fix_coords_type=1, save_to_tmp=args['save_to_tmp'])
