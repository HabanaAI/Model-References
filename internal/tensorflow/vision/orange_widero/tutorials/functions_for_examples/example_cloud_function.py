import numpy as np
import argparse
import os
from devkit.clip import MeClip
from devkit.cloud.s3_filesystem import upload_file


def parse_args():
    parser = argparse.ArgumentParser(description='Example: save to cloud.')
    parser.add_argument('-i', metavar='run_ind', dest='run_ind',
                        help='Index for simple saving to cloud example', type=int)
    parser.add_argument('-c', metavar='clip_name', dest='clip_name', help='clip name')

    return parser.parse_args()


def save_to_our_s3(a, key):
    # Save numpy array `a` to `key` ('s3://mobileye-team-stereo/'+key).
    # Example for a `key`: 'temp/try/array.npz'
    file_name = os.path.split(key)[1]
    np.savez('/tmp/' + file_name, a=a)
    full_s3_path = 's3://mobileye-team-stereo/' + key
    upload_file(full_s3_path, '/tmp/' + file_name)
    print("Uploaded '%s'." % full_s3_path)
    return None


def simple_numpy_save(file_ind):
    # Save a simple numpy array.
    file_name = 'array' + str(file_ind) + '.npz'
    a = np.ones((2, 5))
    save_to_our_s3(a, 'temp/test/' + file_name)
    return None


def simple_save_from_clip(clip_name):
    # Save a frame from `clip_name`.
    clip = MeClip(clip_name)
    frame = clip.get_frame(frame=5, camera_name='main', tone_map='ltm')
    im = frame[0]['pyr'][-1].im
    save_to_our_s3(im, 'temp/test/' + clip_name + '.npz')
    return None


def main():
    print("Starting run.")
    args = parse_args()
    print("args: ", args)
    # If `run_ind` is given, save a simple array in s3.
    # Otherwise, check if a `clip_name` is given and save a frame from this clip to s3.
    # if args.run_ind:
    if args.run_ind:
        print("args.run_ind:", type(args.run_ind))
        print("Executing 'simple_numpy_save'...")
        simple_numpy_save(args.run_ind)
    elif args.clip_name:
        print("Executing 'simple_save_from_clip'...")
        simple_save_from_clip(args.clip_name)

    print("Finished run.")
    return None


if __name__ == '__main__':
    main()
