import argparse
import os
import random
import numpy as np
from tqdm import tqdm
import json

from stereo.prediction.vidar.pred_utils import read_conf
from stereo.prediction.vidar.pred_version import get_predictors_and_views_name, pred_gi, get_vg
from stereo.data.sest.pcd_to_top_view import pcd_to_top_view
from stereo.data.clip_utils import is_simulator_clip
from stereo.common.general_utils import makedir_recursive, tree_base


SCRIPT_PATH = '/mobileye/algo_STEREO3/guygo/stereo/stereo/data/sest/top_view_generator.py'
STEREO_DATA = '/mobileye/algo_STEREO3/stereo/data/'
SIM_DATA_PATH = os.path.join(STEREO_DATA, 'simulator')
BASE_SEST_PATH = os.path.join(STEREO_DATA, 'sest')
VIDAR_TOP_VIEW_PATH = os.path.join(BASE_SEST_PATH, 'vidar_to_top_view')
LABELS_PATH = os.path.join(BASE_SEST_PATH, 'labels')
VIDAR_PREDS_PATH = os.path.join(BASE_SEST_PATH, "vidar_preds")


def load_ds_conf(ds_version):
    json_path = os.path.join(tree_base(), 'stereo/data/sest/conf', '%s.json' % ds_version)
    with open(json_path, "rb") as f:
        return json.load(f)


def get_clip_names_from_file(file_path):
    with open(file_path, 'rb') as f:
        return f.read().splitlines()


def get_vidar_file_path(vidar_version, clip_name, gi):
    return os.path.join(VIDAR_PREDS_PATH, vidar_version, clip_name, "%07d.npz" % gi)


def load_vidar_pred_from_file(file_path):
    vidar_pred = np.load(file_path)
    vidar_pred = np.expand_dims(vidar_pred['arr_0'], 0)[0]
    pcd_attr_key = 'pcd_attr' if 'grayscales' not in vidar_pred else 'grayscales'  # For backwards compatibility
    pcds, pcd_attr = vidar_pred['pcds'], vidar_pred[pcd_attr_key]
    return pcds, pcd_attr


def save_vidar_pred_to_file(clip_name, gi, vidar_version, pcds, pcd_attr):
    file_path = get_vidar_file_path(vidar_version, clip_name, gi)
    if os.path.exists(file_path):
        return
    makedir_recursive(VIDAR_PREDS_PATH, vidar_version, clip_name)
    print("saving vidar pred to file: %s" % file_path)
    np.savez(file_path, {'pcds': pcds, 'pcd_attr': pcd_attr})


def get_output_path_for_gi(output_path, clip_name, gi):
    output_dir = os.path.join(output_path, clip_name)
    img_path = os.path.join(output_dir, "%07d" % gi)
    return img_path


def save_top_view_img(output_path, clip_name, gi, top_view_image):
    output_dir = os.path.join(output_path, clip_name)

    if not os.path.exists(output_dir):
        makedir_recursive(output_dir)
    img_path = os.path.join(output_dir, "%07d" % gi)
    if os.path.exists(img_path):
        return
    np.save(img_path, top_view_image)


class NoVidarPred(Exception):
    pass


def pred_vidar_gi(clip_name, gi, vidar_version, vg=None, predictors=None, views_names=None):
    conf = read_conf(vidar_version)
    if predictors is None:
        predictors, views_names = get_predictors_and_views_name(conf)
    if vg is None:
        vg, clip_name, clip_path = get_vg(clip_name, views_names)
        if is_simulator_clip(clip_name):
            clip_name = clip_path
    with_confidence = conf.get('with_confidence', False)

    pcds, pcd_attr, _ = pred_gi(predictors, vg, gi, with_confidence=with_confidence)
    save_vidar_pred_to_file(clip_name, gi, vidar_version, pcds, pcd_attr)
    return pcds, pcd_attr, vg, predictors, views_names


def get_vidar_pred(clip_name, gi, vidar_version):
    # TODO: involve top_view_ds_version?
    vidar_pred_path = get_vidar_file_path(vidar_version, clip_name, gi)
    if os.path.exists(vidar_pred_path):
        return load_vidar_pred_from_file(vidar_pred_path)
    raise NoVidarPred


def generate_top_view_input(clip_name, gi, ds_version, vg=None, predictors=None, views_names=None):
    top_view_conf = load_ds_conf(ds_version)
    vidar_version = top_view_conf['vidar_version']
    try:
        pcds, pcd_attr = get_vidar_pred(clip_name, gi, vidar_version)
        print("Read vidar prediction from disk")
    except NoVidarPred:
        print("No vidar prediction available, generating it")
        pcds, pcd_attr, vg, predictors, views_names = pred_vidar_gi(clip_name, gi, vidar_version=vidar_version,
                                                                    vg=vg, predictors=predictors,
                                                                    views_names=views_names)
    if top_view_conf['divide_to_sectors']:
        top_view_sectors_imgs = []
        for sector in pcds:
            top_view_sector_input = pcd_to_top_view(clip_name, top_view_conf, pcds[sector], pcd_attr[sector])
            top_view_sectors_imgs.append(top_view_sector_input)
        top_view_input = np.array(top_view_sectors_imgs)
    else:
        top_view_input = pcd_to_top_view(clip_name, top_view_conf, np.concatenate(pcds.values()),
                                         np.concatenate(pcd_attr.values()))
    return top_view_input, vg, predictors, views_names


def single_run(clip_name, output_path, top_view_ds_version, dont_pred, sample_percentage=None, gis_list=None):
    print("Working on clip: %s" % clip_name)
    top_view_conf = load_ds_conf(top_view_ds_version)
    if dont_pred:
        gis = os.listdir(os.path.join(VIDAR_PREDS_PATH, clip_name))
        predictors, views_names, vg = None, None, None
    else:
        conf = read_conf(top_view_conf['vidar_version'])
        predictors, views_names = get_predictors_and_views_name(conf)
        if is_simulator_clip(clip_name):
            clip_name = clip_path = os.path.join(SIM_DATA_PATH, 'Clips_%s' % top_view_conf['clips_version'],
                                                 top_view_conf['car_name'], clip_name)
            clip_gis_path = os.path.join(clip_path, 'main_to_main')
        else:
            clip_gis_path = os.path.join(LABELS_PATH, top_view_conf['labels_version'], clip_name)
        vg, clip_name, _ = get_vg(clip_name, views_names)
        clip_gis = os.listdir(clip_gis_path)
        gis = random.sample(clip_gis, int(sample_percentage * len(clip_gis)))

    if gis_list is not None:
        gis = open(gis_list, 'rb').read().splitlines()
        gis = [int(gi) for gi in gis]
    else:
        gis = [int(gi[-11:-4]) for gi in gis]

    for gi in tqdm(gis):
        if os.path.exists(get_output_path_for_gi(output_path, clip_name, gi)):
            print("\ttop_view image already exists for gi %s")
            continue
        print("\tWorking on gi: %s" % gi)
        try:
            top_view_img, _, _, _ = generate_top_view_input(clip_name, gi, top_view_ds_version, vg, predictors,
                                                                views_names)
            save_top_view_img(output_path, clip_name, gi, top_view_img)
        except:
            print("\tgenerating gi %s failed" % gi)


def netbatch_run(clip_names_path, output_path, top_view_ds_version, dont_pred, sample_percentage,
                 is_simulator):
    from me_jobs_runner import Task, dispatch_tasks, NETBATCH
    nb_task = 'vidar_to_top_view'
    cmd_lines = []
    clip_names = get_clip_names_from_file(clip_names_path)
    for clip_name in clip_names:
        cmd_line = 'python %s -c %s -o %s -v %s' % (SCRIPT_PATH, clip_name, output_path, top_view_ds_version)
        if dont_pred:
            cmd_line += ' -d'
        else:
            cmd_line += " -p %s" % sample_percentage
        cmd_lines.append(cmd_line)
    if is_simulator:
        top_view_conf = load_ds_conf(top_view_ds_version)
        input_folders_or_clips = os.path.join(SIM_DATA_PATH, 'Clips_%s' % top_view_conf['clips_version'],
                                              top_view_conf['car_name'])
    else:
        input_folders_or_clips = clip_names
    task = Task(task_name=nb_task, commands=cmd_lines, input_folders_or_clips=input_folders_or_clips,
                output_folders=output_path, timeout_minutes=0, memory_gb=8)
    dispatch_tasks([task], task_name=nb_task, log_dir=os.path.join(output_path, 'nb_logs'), qslot='ALGO_STEREO',
                   dispatch_type=NETBATCH)


def parse_args():
    parser = argparse.ArgumentParser(description='')
    clip = parser.add_mutually_exclusive_group(required=True)
    clip.add_argument('-c', metavar='clip_name', dest='clip_name', help='clip name')
    clip.add_argument('-C', metavar='clip_names_path', dest='clip_names_path',
                      help='path to file contains list of clips')
    parser.add_argument('-o', metavar='output_path', dest='output_path', type=str, required=True,
                        help='output path for save_top_views')
    parser.add_argument('-v', '--top_view_ds_version', help='top_view_ds_version', required=True)
    parser.add_argument('-p', '--sample_percentage', type=float, help='sample_percentage', default=0.05)
    parser.add_argument('-d', '--dont_pred', help='use only existing preds', action='store_true')
    parser.add_argument('-s', '--is_simulator', help='whether the clips come from simulator', action='store_true')
    parser.add_argument('-gis_list', '--gis_list', help='file path with gis list', required=False, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.clip_names_path:
        netbatch_run(args.clip_names_path, args.output_path, args.top_view_ds_version,
                     args.dont_pred, args.sample_percentage, args.is_simulator)
    elif args.clip_name:
        single_run(args.clip_name, args.output_path, args.top_view_ds_version, args.dont_pred,
                   args.sample_percentage, args.gis_list)


if __name__ == '__main__':
    main()
