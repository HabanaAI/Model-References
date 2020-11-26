# %%
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# %%
# Trying to rebuild 'gen_modified_NBo.py' from the Bottom Up (bu) to see
# where it fails in netbatch

print("Start imports")
import time
start = time.time()

# %%
import argparse

def parse_args():
    print("Begin argparse")
    parser = argparse.ArgumentParser(description='Modify frames by rotating at several angles.')
    # ang = parser.add_mutually_exclusive_group(required=True)
    # ang.add_argument('-A', metavar='angles_range', dest='angles_range',
    #                     help='angles <first> <last> <#angles>', nargs='+', type=float)
    # parser.add_argument('-a', metavar='angles_deg', dest='angles_deg',
    #                     help='angles in degrees',  nargs='+',type=float)  # not required
    parser.add_argument('-A', metavar='angles_range', dest='angles_range',
                        help='angles <first> <last> <#angles>', nargs='+', type=float)
    clip = parser.add_mutually_exclusive_group(required=True)
    clip.add_argument('-n', metavar='nfiles_samps', dest='nfiles_samps',
                      help='number of files/frames to modify', type=int)
    clip.add_argument('-v', metavar='view_name', dest='view_name',
                      help='<clip_name>@<gi> for specific view')
    # clip.add_argument('-c', metavar='clip_name', dest='clip_name',
    #                   help='clip_name')
    # parser.add_argument('-i', metavar='file_ind', dest='file_ind',
    #                     help='"view" index out of dataset directory',
    #                     type=int, required=False)  # not required
    parser.add_argument('-o', metavar='output_path', dest='output_path',
                        help='path to save output')  # required
    parser.add_argument('-C', metavar='cam_inds', dest='cam_inds', nargs='+', type=int,
                        help="inds for cams to rotate. Choose a subset of 1-3. 0 is the main in sector (don't choose).")
    parser.add_argument('-S', metavar='sector', dest='sector', default='main',
                        help='sector to work in')
    # parser.add_argument('-D', metavar='DATASET_DIR', dest='DATASET_DIR',
    #                     help='DATASET_DIR')
    parser.add_argument('-r', metavar='rot_ax', dest='rot_ax',
                        help='rotation axis: 0-x, 1-y, 2-z, 4-random')
    parser.add_argument('-l', action='store_true',
                        help='use for local run')
    parser.add_argument('-d', action='store_true',
                        help='a "light"-mode run for debugging')
    # parser.add_argument('-m', action='store_true',
    #                     help='make/create sub-folders (in single-mode run)')
    calib = parser.add_mutually_exclusive_group()
    calib.add_argument('-G', action='store_true',
                       help='Good calibration')
    calib.add_argument('-B', action='store_true',
                       help='Bad calibration')
    # parser.add_argument('-p')
    print("End argparse")
    return parser.parse_args()

args = parse_args()
print("Loaded args (1)")
print("args: ", args)

# %%

import os
import numpy as np
# import cv2
if not args.d:
    # from devkit.clip import MeClip
    from me_jobs_runner import Task, dispatch_tasks, NETBATCH
    from stereo.data.dataset_utils import ViewDatasetIndex
    from stereo.prediction.vidar.stereo_predictor import StereoPredictor
    from stereo.data.view_generator.view_generator import ViewGenerator
from scipy.spatial.transform import Rotation as R
from stereo.data.clip_utils import clip_name_to_sess_name, split_clipname_gi


print("End imports")

# SCRIPT_PATH = '/mobileye/algo_STEREO3/ohrl/gitlab/stereo/sandbox/gen_mod_bu.py'
# SCRIPT_PATH = '/mobileye/algo_STEREO3/ohrl/gitlab/stereo/stereo/calibration/gen_mod_bu.py'
SCRIPT_PATH = '/mobileye/algo_STEREO3/ohrl/gitlab/stereo4/stereo/stereo/calibration/gen_mod_bu.py'

sectors = {
  "main": {
    "model": "diet_main_v3.0.v0.0.12",
    "restore_iter": -1
  },
  "rear": {
    "model": "diet_main_rear_v3.0.v0.0.3",
    "restore_iter": 465000
  },
  "frontCornerLeft": {
    "model": "diet_combined_corners_v3.0.v0.0.2",
    "restore_iter": 100000
  },
  "frontCornerRight": {
    "model": "diet_combined_corners_v3.0.v0.0.2",
    "restore_iter": 100000
  },
  "rearCornerLeft": {
    "model": "diet_combined_corners_v3.0.v0.0.2",
    "restore_iter": 100000
  },
  "rearCornerRight": {
    "model": "diet_combined_corners_v3.0.v0.0.2",
    "restore_iter": 100000
  }
} # taken from vidar.0.3.1.json 7.6.20

# DATASET_DIR = '/mobileye/algo_STEREO3/stereo/data/view_dataset_v2.3'




# %%
def random_dir_vector():
    """
      """
    # phi = np.random.uniform(0,np.pi*2)
    # costheta = np.random.uniform(-1,1)
    #
    # theta = np.arccos( costheta )
    # x = np.sin( theta) * np.cos( phi )
    # y = np.sin( theta) * np.sin( phi )
    # z = np.cos( theta )
    # dir_vec =[x,y,z]
    randn_vec = np.random.randn(3)
    mag = ( (randn_vec **2).sum() ) **(0.5)
    dir_vec = randn_vec / mag
    return dir_vec
# %%

def exc_list(list):
    # list of "exclusive" elements
    inds = []
    seen = []
    for i, ele in enumerate(list):
        if ele not in seen:
            inds.append(i)
            seen.append(ele)
    return seen, inds

def axindToDir(rot_ax):
    # Rotation axis index (0-4; x,y,z,rand) to direction
    if rot_ax == 1:  # rotation around y-axis
        rot_direction = [0, 1, 0]  # for starters rotate in the y direction only
        rot_dir = "y"
    elif rot_ax == 0:  # rotation around x-axis
        rot_direction = [1, 0, 0]
        rot_dir = "x"
    elif rot_ax == 2:  # rotation around z-axis
        rot_direction = [0, 0, 1]
        rot_dir = "z"
    elif rot_ax ==4 : # random direction
        rot_direction = random_dir_vector()
        rot_dir = "rand"
    return rot_direction,rot_dir


def v_hat(v):
    # normalized "direction" vector
    v = np.array(v)
    # assert type(numpy.ndarray) == numpy.ndarray, "Direction vector is not a numpy array"
    return v / (v ** 2).sum() ** 0.5



def gen_rt(rot_direction, rot_angle_deg, translation):
    angle_rad = np.pi / 180 * rot_angle_deg
    r_mat = R.from_rotvec(angle_rad * np.array(v_hat(rot_direction))).as_dcm()
    rt = np.diag([1., 1., 1., 1.])
    rt[:3, :3] = r_mat
    rt[:3, 3] = translation
    return rt


# %% functions which require "stereo"
if not args.d:

    def get_vdsi():
        # dataset_dir = '/mobileye/algo_STEREO3/stereo/data/view_dataset_v2.3'  # originally used
        # dataset_dir = '/mobileye/algo_STEREO3/stereo/data/view_dataset.v2.3'
        # dataset_dir = '/mobileye/algo_STEREO3/stereo/data/view_dataset.v3.0'
        dataset_dir = '/mobileye/algo_STEREO3/stereo/data/view_dataset.v3.1'
        vdsi_file = ViewDatasetIndex(dataset_dir)
        return vdsi_file


    def get_vdsi_data(nfiles_samps, good=False, bad=False):
        vdsi = get_vdsi()
        if good:
            sess = '18-08-07_11_58_18_Alfred_REM_Jer4'
            # sess2 = '18-08-30_10_45_45_Alfred_REM_Jer4_Loc'
            frame_names_good = [f for f in vdsi.train_frames_list + vdsi.test_frames_list
                               if clip_name_to_sess_name(f.split('@gi@')[0]) == sess]
            Nset = len(frame_names_good)
            file_inds = np.random.choice(Nset, args.nfiles_samps, replace=False)
            frames_names = [frame_names_good[i] for i in file_inds]
        elif bad:
            sess = '19-05-03_11_20_19_KRQ_Jer5_3Loops_1R'  # both in view_dataset.v3.1
            # sess2 = '19-05-02_13_48_10_KRQ_2Loops_1R'
            frame_names_bad = [f for f in vdsi.train_frames_list + vdsi.test_frames_list
                              if (clip_name_to_sess_name(f.split('@gi@')[0]) == sess)]
            Nset = len(frame_names_bad)
            file_inds = np.random.choice(Nset, args.nfiles_samps, replace=False)
            frames_names = [frame_names_bad[i] for i in file_inds]
        else:
            Nset = len(vdsi.test_frames_list)  # number of available files in set
            file_inds = np.random.choice(Nset, nfiles_samps, replace=False)
            frames_names = [vdsi.test_frames_list[i] for i in file_inds]
        clips = [frames_names[i].split('@gi@')[0] for i in range(nfiles_samps)]
        gis = [frames_names[i].split('@gi@')[1] for i in range(nfiles_samps)]
        return vdsi, clips, frames_names  # , file_inds



    def get_pred(view_inds_2mod=None, sector='main'):

        model_name = sectors[sector]['model']
        restore_iteration = sectors[sector]['restore_iter']

        # model_name = 'diet_main_sm_reg_v0.0.2.2' # started with
        # model_name = 'diet_main_rear_v2.3_reg_v0.0.2.0' # Jeff's eval
        # model_name = 'diet_main_v3.0.v0.0.6'  # latest
        # restore_iteration = 930000 # -1 for latest
        # restore_iteration = 5000  # from vidar.0.3.1.json

#         model_conf_file = '/mobileye/algo_STEREO3/ohrl/gitlab/stereo/stereo/models/conf/stereo/' + model_name + '.json'
        model_conf_file = '/mobileye/algo_STEREO3/ohrl/gitlab/stereo4/stereo/stereo/models/conf/stereo/' + model_name + '.json'
        print("Loading predictor object....")
        pred = StereoPredictor(model_conf_file, sector_name=sector,
                               restore_iter=restore_iteration)
        # views = [u'frontCornerRight_to_main', u'parking_front_to_main',
        #          u'frontCornerLeft_to_main', u'main_to_main']
        print("Finished prediction")
        views = pred.views_names
        
        # views2mod_inds = [0]  # for now define it here the FCR choice, can add this as an input later
        if view_inds_2mod == None:
            views2mod_inds = [1]  # first non-main camera of sector
        else:
            views2mod_inds = view_inds_2mod
        # views2mod_inds = [1]  # if want parking
        views2mod = [views[i] for i in views2mod_inds]  # list of views to modify.


        return pred, views2mod


    def get_lidar(frame_str, predict):
        vdsi = get_vdsi()
        vdsi_views = vdsi.read_views(predict.views_names, frame_str)
        lidar_im = vdsi_views[predict.views_names[0]]['lidars']['short']
        # lidar_im = np.zeros((700,400))
        return lidar_im


    def get_rot_dataset(frame, angles_deg, rot_direction,view_inds_2mod=None, sector='main'):
        clip, gi = split_clipname_gi(frame)
        pred, views2mod = get_pred(view_inds_2mod,sector=sector)
        try:
            lidar_img = get_lidar(frame, pred)
        except:
            lidar_img = np.empty((310, 720))
            lidar_img[:] = np.nan
            print("Lidar doesn't exist.")
        pred_mod_set = np.zeros((len(angles_deg),) + lidar_img.shape)
        rotated_imgs = np.zeros((len(angles_deg),) + lidar_img.shape)  # the rotated image
        vds_mod = ViewGenerator(clip, view_names=pred.views_names, predump=None, mode='pred', etc_dir=None)
        for counter_angle, ang_deg in enumerate(angles_deg):
            print("ang [" + str(counter_angle) + "]: " + str(ang_deg))
            pred_mod_set[counter_angle, :, :], rotated_imgs[counter_angle, :, :] = \
                get_rot_data(vds_mod, pred, gi, rot_direction, ang_deg, views2mod)
        inv_lidar = 1 / (np.where(lidar_img == 0, np.nan, lidar_img))
        lidarm = lidar_img
        lidarm[lidarm==0] = np.nan
        diff = (1/pred_mod_set - lidarm) / np.abs(lidarm)  # relative depth error
        # diff2 = (pred_mod_set - inv_lidar) / np.abs(inv_lidar)  # relative inverse-depth error
        return pred_mod_set, rotated_imgs, lidar_img, inv_lidar, diff, views2mod


    def get_rot_data(vds, pred, gi, rot_direction, ang_deg, views2mod, t_modified=[0,0,0] ):
        """
        Add a rotation error to view, return data according to the modified sector
        """

        for i, view2mod in enumerate(views2mod):
            rt_mod = gen_rt(rot_direction[i], ang_deg, t_modified)
            rect_temp = vds.views[view2mod]['conf']['crop_rect_at_level']
            vds.views[view2mod]['image_input'].register_rectification(rt_mod) # first argument `rect_temp` not used anymore
        mod_view = vds.get_gi_views(gi=int(gi), RT_modification=rt_mod, modify_views=views2mod)
        pred_mod = np.squeeze(pred.pred(views=mod_view))
        rotated_view_im = mod_view[views2mod[0]]['image'] # return 1 rotated view (even if rotated more)
        return pred_mod, rotated_view_im

# %% Light "dummy" functions versions:

if args.d:

    def get_rot_data_light(vds_dum, pred_dummy, gi, rot_direction,
                           ang_deg, views2mod, t_modified=[0,0,0] ):
        # A "light" version for fast debugging
        pred_mod = np.ones((310,720))
        rotated_view_im = np.ones((310,720))
        return pred_mod, rotated_view_im


    def get_rot_dataset_light(frame, angles_deg, rot_direction,view_inds_2mod=None):
        pred_mod_set = np.ones((len(angles_deg),)+(310,720))
        rotated_imgs = np.ones((len(angles_deg),)+(310,720))
        lidar_img = np.ones((310,720))
        inv_lidar = np.ones((310,720))
        diff = np.ones((len(angles_deg),)+(310,720))
        views2mod = [u'frontCornerLeft_to_main']
        return pred_mod_set, rotated_imgs, lidar_img, inv_lidar, diff, views2mod

    def get_pred_light():
        views2mod = [u'frontCornerRight_to_main']
        pred = {}
        return pred, views2mod
# %%
def netbatch_run(clips, output_path, params, local=False):
    nb_task = 'gen_mod_test'
    cmd_lines = []
    # clip_names = get_clip_names_from_file(clip_names_path)
    # angles_deg = params['angles_deg']
    angle_range = params['angle_range']
    angle_range_str = ''.join([str(ang) + ' ' for ang in angle_range])[:-1]
    cam_inds = params['cam_inds']
    str_cam_inds = (' ').join([str(i) for i in cam_inds])

    # for s in ["x","y","z"]:
    for s in ["rand"]:
        output_path_rot = output_path + "/" + s + "-rot"
        if not os.path.exists(output_path_rot):
            os.mkdir(output_path_rot)

    clips3 = []
    cmd_names = []
    for j, clip in enumerate(clips):
        # for rot in [0, 1, 2]:
        for rot in [4]: # '4' is random direction
            # cmd_line = "python %s -A %s -c %s -o %s" % (SCRIPT_PATH, angle_range_str, clip, output_path)
            clips3.append(clip)
            cmd_line = "python %s -A %s -v %s -o %s -r %s" % (SCRIPT_PATH, angle_range_str,
                                                              params['frame_names'][j], output_path, str(rot) )
            cmd_line += " -C %s -S %s" % (str_cam_inds, params['sector'])
            # cmd_line += " -d" # for debugging
            cmd_names.append(params['frame_names'][j] + str(rot) + ('_%s' % params['sector']  ) )

            print(cmd_line)
            cmd_lines.append(cmd_line)
    print("#jobs: " + str(len(cmd_lines)))

    if local:
        for cmd in cmd_lines:
            os.system(cmd)
    else:
        print("Sending to netbatch...")
        task = Task(task_name=nb_task, commands=cmd_lines, input_folders_or_clips=clips3,
                    output_folders=output_path, timeout_minutes=0, memory_gb=8,
                    command_names=cmd_names)  # was 8gb in example
        dispatch_tasks([task], task_name=nb_task, log_dir=os.path.join(output_path, 'nb_logs'), qslot='ALGO_STEREO',
                       dispatch_type=NETBATCH)


# %% main
print("Start. (0)")



output_path = args.output_path
pred = []

sector = args.sector

if args.cam_inds is not None:
    cam_inds_temp = args.cam_inds
    if type(cam_inds_temp) is not list:
        cam_inds = [cam_inds_temp]
    else:
        cam_inds = cam_inds_temp
else:
    cam_inds = [1] # first non-main camera in sector

# batch:
if args.nfiles_samps:  # batch
    # clips = ['19-01-28_13-25-56_Alfred_Front_0009', '19-01-09_17-29-11_Alfred_Front_0141']
    print("Batch (2)")


    vdsi, clips, frame_names = get_vdsi_data(args.nfiles_samps, good=args.G, bad=args.B)

    angle_range = args.angles_range  # list
    params_batch = {'frame_names': frame_names, 'angle_range': angle_range,
                          'cam_inds': cam_inds, 'sector':sector}  # 'file_inds': file_inds, }
    output_path = '/mobileye/algo_STEREO3/ohrl/tolerance'
    file_counter = 0
    while os.path.exists(output_path + "/set" + "{}".format(file_counter)):
        file_counter += 1
    output_path_case = output_path + "/set" + "{}".format(file_counter)
    os.mkdir(output_path_case)
    netbatch_run(clips, output_path_case, params_batch, local=args.l)
    print("calling netbatch_run (4)")
    print("'gen_mod_bu' finished")

# single:
else:  # single
    print("calling single_run")
    frame = args.view_name
    clip, gi = split_clipname_gi(frame)

    angle_range = args.angles_range  # list
    print("angle_range: ", angle_range)
    print("angle_range type: ", type(angle_range))
    angles_deg = np.linspace(angle_range[0], angle_range[1], int(angle_range[2]))

    print("args.rot_ax: ",args.rot_ax)
    try:
        rot_ax = int(args.rot_ax)
    except:
        rot_ax = 0
    print("angles_deg: ", angles_deg)

    rot_direction = np.zeros((len(cam_inds), 3))
    if len(cam_inds) ==1:
        rot_direction[0], rot_dir = axindToDir(rot_ax)
    elif len(cam_inds) > 1:
        for i in range(len(cam_inds)):
            rot_direction[i], rot_dir = axindToDir(rot_ax)


    output_path_rot = output_path+ "/" + rot_dir + "-rot"


    # if args.m:
    if not os.path.exists(output_path_rot):
        os.mkdir(output_path_rot)

    if not args.d:
        pred_mod, im_rot, lidar_img, inv_lidar, diff, views2mod = \
            get_rot_dataset(frame, angles_deg, rot_direction, view_inds_2mod=cam_inds,
                            sector=sector)
    elif args.d:
        pred_mod, im_rot, lidar_img, inv_lidar, diff, views2mod = \
            get_rot_dataset_light(frame, angles_deg, rot_direction,cam_inds)

    # file_path_name = output_path + "/" + rot_dir + "-rot/testrun_" + clip + ".npz"
    save_folder = output_path + "/" + rot_dir + "-rot"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    file_path_name = save_folder + "/testrun_" + frame + rot_dir + ".npz"
    print("angles: ", angles_deg)
    np.savez(file_path_name, pred_mod=pred_mod, lidar_img=lidar_img,
             angles_deg=angles_deg, clip=clip, gi=gi, diff=diff, im_rot=im_rot,
             sector=sector, views2mod=views2mod, rot_ax=rot_ax, rot_direction=rot_direction)
    print('sector %s, modified %s' % (sector, views2mod))
    print("Saved " + file_path_name)
end = time.time()

print("Finished after {} secs".format(end-start))
