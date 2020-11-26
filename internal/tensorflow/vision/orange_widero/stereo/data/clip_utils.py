import numpy as np
import os
import datetime
import time
from tqdm import tqdm

from os import listdir
from os.path import join, exists

try:
    from stereo.common.gt_packages_wrapper import FoFutils, parse_pls, MissingLidar
except ImportError:
    print("ground_truth was not imported!. This should be removed once we solve the gt imports in python 3")
from mdblib.utils.clip_to_location import getClipLocation
from stereo.common.visualization.vis_utils import concat_images
from stereo.data.view_generator.simulator_view_generator import SimulatorViewGenerator
from devkit.clip import MeClip

camera_name_to_section = {'main': 'Front', 'rear': 'Rear',
                          'frontCornerLeft': 'Left', 'frontCornerRight': 'Right',
                          'rearCornerLeft': 'Left', 'rearCornerRight': 'Right',
                          'parking_left': 'Left', 'parking_right': 'Right'}


sections_dict = {
                'Front': ['main', 'fisheye', 'narrow'],
                'Rear': ['rear', 'parking_rear', 'parking_front'],
                'Right': ['frontCornerRight', 'rearCornerRight', 'parking_right'],
                'Left': ['frontCornerLeft', 'rearCornerLeft', 'parking_left']
}


def clip_name_to_sections(clip_name):
    sides = {'Front': 'Clips_Front', 'Left': 'Clips_Left', 'Right': 'Clips_Right', 'Rear': 'Clips_Rear'}
    cl = getClipLocation(clip_name)
    cnum = int(clip_name.split("_")[-1])
    dl = os.path.abspath(os.path.join(cl, '../..'))
    side_clips = {}
    for side, side_subd in sides.items():
        for subd in os.listdir(dl):
            if side_subd == subd:
                clips = os.listdir(os.path.join(dl, subd))
                cls = [getClipLocation(x) for x in clips]
                clips1 = sorted([clips[x] for x in range(len(clips)) if len(cls[x]) > 0 and not cls[x].endswith(".pls")])
                side_clip = [clip for clip in clips1 if int(clip.split("_")[-1]) == cnum]
                side_clips[side] = None if len(side_clip) == 0 else side_clip[0]
    return side_clips


def clip_path(clip_name):
    return getClipLocation(clip_name)


def clip_name_to_clips_dict(clip_name):
    clips_dict = {}

    num = clip_name.split('_')[-1]
    for section in sections_dict:
        try:
            front_format = getClipLocation(clip_name).split('/')[-3].split('_')
            if len(front_format) > 2:
                dir_name = '/'.join(getClipLocation(clip_name).split('/')[:-3]) + '/' + 'Clips_' + section + '_' + front_format[2] +'/'
            else:
                dir_name = '/'.join(getClipLocation(clip_name).split('/')[:-3]) + '/' + 'Clips_' + section + '/'
            for clip in os.listdir(dir_name):
                if '_' + num in clip:
                    clips_dict[section] = clip
        except:
            continue
    return clips_dict


def clip_name_to_clip_name_by_cam(clip_name, cam):
    clips_dict = clip_name_to_clips_dict(clip_name)
    clip_section = None
    for section in sections_dict:
        if cam in sections_dict[section]:
            clip_section = section
            break
    return clips_dict[clip_section]


def clip_name_to_pls(clip_name, return_gi_range=False):

    plss = []
    section_dir = '/'.join(getClipLocation(clip_name).split('/')[:-2])
    if section_dir == '':
        return None
    for item in os.listdir(section_dir + '/pls'):
        if '.pls' in item:
            plss.append(item)
    for pls in plss:
        pls_file = section_dir + '/pls/' +  pls + '/' + pls
        pls_df = parse_pls(pls_file)
        clip_row = pls_df.loc[pls_df['clip_name'] == clip_name]
        if len(clip_row) == 1:
            if not return_gi_range:
                return pls
            else:
                gi_range = [int(clip_row['FirstGrabIndex']), int(clip_row['LastGrabIndex'])]
                return pls, gi_range
    return None


def sess_name_to_timestamp(sess_name):
    try:
        date_str = sess_name[:17]
        date = datetime.datetime.strptime(date_str, "%y-%m-%d_%H_%M_%S")
        timestamp = time.mktime(date.timetuple())
    except:
        timestamp = None
    return timestamp


def sess_name_to_vehicle(sess_name):
    return sess_name[18:].split('_')[0]


def pls_name_to_sess_path(pls_name):
    pls_path = getClipLocation(pls_name)
    return pls_path_to_sess_path(pls_path)


def is_simulator_clip(clip_name):
    return 'Town' in clip_name


def clip_name_to_sess_name(clip_name):
    if is_simulator_clip(clip_name):
        return clip_name.split('_')[1]
    return clip_name_to_sess_path(clip_name).split('/')[-1]


def clip_name_to_sess_path(clip_name):
    clip_path = getClipLocation(clip_name)
    return clip_path_to_sess_path(clip_path)


def pls_path_to_sess_path(pls_path):
    if "/pls/" in pls_path:
        idx = pls_path.split("/").index("pls") - 1
        dl = "/".join(pls_path.split("/")[:idx])
    else:
        dl = "/".join(pls_path.split("/")[:-2])
    return dl


def clip_path_to_sess_path(clip_path):
    if clip_path.endswith("/"):
        return "/".join(clip_path.split("/")[:-3])
    else:
        return "/".join(clip_path.split("/")[:-2])


def sess_path_has_velodyne(sess_path):
    return os.path.isdir(os.path.join(sess_path, 'ref_files', 'Velodyne'))


def parse_pls_paths(plss):
    sess_names = [os.path.basename(pls_path_to_sess_path(mcr)) for mcr in plss]
    timestamps = [sess_name_to_timestamp(sess_name) for sess_name in sess_names]
    vehicles = [sess_name_to_vehicle(sess_name) for sess_name in sess_names]
    return timestamps, vehicles


def parse_mcr_paths(mcrs):
    sess_names = [os.path.basename(mcr) for mcr in mcrs]
    timestamps = [sess_name_to_timestamp(sess_name) for sess_name in sess_names]
    vehicles = [sess_name_to_vehicle(sess_name) for sess_name in sess_names]
    return timestamps, vehicles


def init_fof_lidar_try(clip):
    try:
        clip.init_lidar()
        return True
    except (OSError, IOError,  MissingLidar) as e:
        return False


def get_vehicle_accumulated_dist_fof(clip):
    gis = clip.ts_map.all_gis['main']
    speeds = np.array([clip.ts_map.speed[gi] for gi in gis])
    times = np.array(clip.ts_map.all_ts['main']) * 1e-6
    dts = times[1:] - times[:-1]
    avg_speeds = (speeds[1:] + speeds[:-1]) / 2
    dists = dts * avg_speeds
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    return cum_dists, gis, speeds, times


def get_transformation_matrix(clip, source_camera, target_camera, RT_modification = None):
    if isinstance(clip, MeClip):
        RT = clip.cam2cam.get_transformation_matrix(source_camera, target_camera)
    elif isinstance(clip, FoFutils):
        RT = clip.get_transformation_matrix(source_camera, target_camera)
    else:
        raise Exception('Unsupported clip type')
    if RT_modification is not None:
        RT = RT_modification.dot(RT)
    return RT


def get_camK(clip, camera_name, level):
    if isinstance(clip, MeClip):
        assert False
    elif isinstance(clip, FoFutils):
        K = clip.get_camK(camera_name, level=level)
    else:
        raise Exception('Unsupported clip type')
    return K


def get_host_box(clip,
                 dist_back_bump=3.4, bumper_dist=1.6, left_wheel=0.9, right_wheel=0.9,
                 camH=1.3, roof_top=0.5, margin=0.1):
    if not (isinstance(clip, MeClip) or isinstance(clip, FoFutils)):
        clip = MeClip(clip)
    main_conf = clip.conf_files('main')['camera']['main']
    dist_back_bump = main_conf.get('distBackBump', dist_back_bump) + margin
    bumper_dist = main_conf.get('bumperDist', bumper_dist) + margin
    left_wheel = main_conf.get('leftWheel', left_wheel) + margin
    right_wheel = main_conf.get('rightWheel', right_wheel) + margin
    camH = main_conf.get('cameraHeight', camH)
    return np.array([[-left_wheel, right_wheel], [-camH, roof_top], [-dist_back_bump, bumper_dist]])


def get_fof_from_pls(pls_path, etc=None, meta_dir=None):
    sections_clips = {}
    for section in sections_dict:
        sections_clips[section] = []
    for front_clip in parse_pls(pls_path).clip_name.values:
        clips_dict = clip_name_to_clips_dict(front_clip)
        for section in sections_dict:
            sections_clips[section].append(clips_dict[section])
    return FoFutils(sections_clips.values(), etc_dir=etc, meta_dir=meta_dir)


def split_clipname_gi(cngi):
    clip_name, gi = cngi.split('@gi@')
    return clip_name, int(gi)


def get_clip_part_t0_grabs(clip_path, meta_dir, part_ind, part_num, is_simulator=False):
    if is_simulator:
        gi_files = os.listdir(os.path.join(clip_path, "main_to_main"))
        t0_grabs = [int(gi_file[-11:-4]) for gi_file in gi_files]
    else:
        clip = MeClip(clip_path, meta_dir=meta_dir)
        gfis = np.arange(clip.first_gfi(), clip.last_gfi())
        t0_grabs = np.array([clip.get_grab_by_gfi(gfi=gfi, camera_name='main') for gfi in gfis])
        t0_grabs = np.array_split(t0_grabs, part_num)[part_ind]
    return t0_grabs


def session_front_clips(session):
    sess_clips = [clip for clip in listdir(join(session, 'Clips_Front')) if 'Front' in clip and 'pls' not in clip]
    for clip in sess_clips:
        path = clip_path(clip)
        if path is '' or not exists(join(path, 'clipext', '%s.meta' % clip)):
            sess_clips.remove(clip)
    return sess_clips


def sample_session_frames(session, num_frames, clips_range=None, speed_range=None):
    sess_clips = session_front_clips(session)
    if clips_range is not None:
        sess_clips = [clip for clip in sess_clips if (clips_range[0] <= int(clip.split('_')[-1]) < clips_range[1])]
    if speed_range is not None:
        sess_clips_gis = {}
        for clip_name in sess_clips:
            clip = get_fof_from_clip(clip_name, compact=True)
            clip.init_ts_map()
            gfis = np.arange(clip.first_gfi(), clip.last_gfi())
            clips_gis = np.array([clip.get_grab_by_gfi(gfi=gfi, camera_name='main') for gfi in gfis])
            speeds = np.array([clip.ts_map.speed[gi] for gi in clips_gis])
            gis_in_range = [gi for gi, speed in zip(clips_gis, speeds) if speed_range[0] <= speed < speed_range[1]]
            if len(gis_in_range) > 0:
                sess_clips_gis[clip_name] = gis_in_range
        sess_clips = sess_clips_gis.keys()
    if len(sess_clips) > num_frames:
        frames_per_clip = 1
        sess_clips = np.sort(np.random.permutation(sess_clips)[:num_frames])
    else:
        frames_per_clip = int(np.ceil(float(num_frames)/len(sess_clips)))
    clips_gis = {}
    for clip_name in sess_clips:
        if speed_range is None:
            t0_grabs = get_clip_part_t0_grabs(clip_name, meta_dir=None, part_ind=0, part_num=1)
        else:
            t0_grabs = sess_clips_gis[clip_name]
        sampled_t0_grabs = np.sort(np.random.permutation(np.array(t0_grabs))[:min(frames_per_clip, len(t0_grabs))])
        clips_gis[clip_name] = sampled_t0_grabs
    return clips_gis


def get_combined_images(clip_name, gis, pyr=0, save_to=None, load_from=None, sim_img_type='image'):
    if load_from:
        file_path = join(load_from, clip_name + '.npy')
        return np.load(file_path)
    is_simulator = is_simulator_clip(clip_name)
    cams = ['frontCornerLeft', 'main', 'frontCornerRight', 'rearCornerLeft', 'rear', 'rearCornerRight']
    combined_imgs = []
    if is_simulator:
        svg = SimulatorViewGenerator(clip_name, view_names=["%s_to_%s" % (cam, cam) for cam in cams])
    else:
        clips_by_cam = {}
        for cam in cams:
            clip_section_name = clip_name_to_clip_name_by_cam(clip_name, cam)
            clips_by_cam[cam] = MeClip(clip_section_name)
    one_clip_shape = None
    for gi in tqdm(gis):
        gi_combined_imgs = []
        for cam in cams:
            try:
                if is_simulator:
                    gi_img = np.flip(svg.get_gi_views(gi)["%s_to_%s" % (cam, cam)][sim_img_type], 0)
                    one_clip_shape = gi_img.shape
                else:
                    gi_img = np.flip(clips_by_cam[cam].get_frame(grab_index=gi, camera_name=cam)[0]['pyr'][pyr].im, 0)
            except IOError:
                gi_img = np.zeros(one_clip_shape)
            gi_combined_imgs.append(gi_img)
        combined_imgs.append(concat_images(gi_combined_imgs, 2, 3))
    if save_to:
        file_path = join(save_to, clip_name)
        np.save(file_path, combined_imgs)
    return combined_imgs
