import argparse
from devkit.pext_reader import PextReader
from glob import glob
from os.path import join
import numpy as np
from os import walk

def main(args):
    clipName = None
    pr = PextReader(file_name=glob(join(args['pext_dir'], '*.SS_grab'))[0])
    gi = pr.get(args['frame'], fields=['T0grabIdx'])['T0grabIdx'][0][0]
    Clips_Front_dir = join(pr.meta['log_path'], '..', '..', '..', 'Clips_Front')
    metas = [join(dirpath, filename)
             for (dirpath, dirs, files) in walk(Clips_Front_dir)
             for filename in (dirs + files) if filename.endswith('.meta') and 'pls' not in filename]
    for meta in metas:
        pr = PextReader(meta)
        min_grab = np.min(pr.get(fields=['T0grabIdx'])['T0grabIdx'])
        max_grab = np.max(pr.get(fields=['T0grabIdx'])['T0grabIdx'])
        if gi >= min_grab and gi <= max_grab:
            clipName = meta.split('/')[-1].split('.meta')[0]
    return clipName, gi

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='transalte ishow pext cmd line arguments to clip_name, gi')
    parser.add_argument('-p', '--pext_dir', help='pext_dir', default='')
    parser.add_argument('-f', '--frame', help='frame', type=int)
    args = vars(parser.parse_args())
    print (main(args))

