# import stereo

from stereo.data.dataset_utils import ViewDatasetIndex
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Example: ViewDatasetIndex.')
    parser.add_argument('-i', metavar='frame_ind', dest='frame_ind',
                        help='Index for frame from training dataset', type=int)
    # parser.add_argument('-c', metavar='clip_name', dest='clip_name', help='clip name')
    # parser.add_argument('-v', metavar='view_name', dest='view_name',
    #                     help='view: cam_to_main_cam)
    # parser.add_argument('-g', metavar='gi', dest='gi', help='grab_index')

    return parser.parse_args()


def main():
    print("Starting run.")
    args = parse_args()
    print("args: ", args)



    print("Executing 'simple_numpy_save'...")
    frame_ind = args.frame_ind

    dataset_dir = '/mobileye/algo_STEREO3/stereo/data/view_dataset.v3.1'
    print("Initiating ViewDatasetIndex...")
    from stereo.data.dataset_utils import ViewDatasetIndex
    vdsi = ViewDatasetIndex(dataset_dir)
    train_frames = vdsi.train_frames_list
    frame = train_frames[frame_ind]
    clip_name, gi = split_clipname_gi(train_frames[0])
    view_name = 'frontCornerRight_to_main'
    npz_filename = '_'.join([clip_name, view_name, str(gi)]) + '.npz'
    print(npz_filename)
    print("cwd: ", os.getcwd())
    print('ls / :',  os.listdir("/"))
    tmp_exists = os.path.exists('/tmp')
    print(" tmp? ", tmp_exists)
    if tmp_exists:
        print('ls /tmp/: ', os.listdir('/tmp/'))

    print("trying to load submodules...")
    try:
        from stereo.data.dataset_utils import ViewDatasetIndex
        from stereo.data.clip_utils import split_clipname_gi
    except:
        print("didn't manage to load submodules")
    else:
        print("loaded submodules")
    print("Finished run.")
    return None



if __name__ == '__main__':
    main()
