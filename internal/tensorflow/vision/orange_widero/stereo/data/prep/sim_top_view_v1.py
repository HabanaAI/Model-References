import pickle
import os
from stereo.data.view_generator.simulator_view_generator import is_center_view

from carla_scripts.Sest.ground_truth import get_gt_from_semantic_and_depth, actors_data_to_label_img
from carla_scripts.Sest.sest_config import *

base_path = '/mobileye/algo_STEREO3/stereo/data/'


def get_label_image(clip_name, gi):
    clip_path = os.path.join(base_path, "simulator/Clips_v5/Bella/", clip_name)
    views = [view for view in os.listdir(clip_path) if is_center_view(view)]
    label_img = np.zeros(RESOLUTION).astype('uint8')

    for view in views:
        frame_data = np.load(os.path.join(clip_path, view, '%s_%s_%07d.npz' % (clip_name, view, gi)))
        pixels_with_semantics = get_gt_from_semantic_and_depth(frame_data)

        for label in [ROAD, GROUND, OBJECT, VEHICLES, PEDS]:
            vidar_labels = np.take(CARLA_LABEL_TO_VIDAR_LABEL.values(), pixels_with_semantics[:, 2])
            labeled_pixels = pixels_with_semantics[vidar_labels == label]
            label_img[labeled_pixels[:, 0], labeled_pixels[:, 1]] = label

    with open(os.path.join(clip_path, 'gt', 'actors_data_%07d.pkl' % gi), 'rb') as f:
        actors_data = pickle.load(f)
    bbs_image = actors_data_to_label_img(actors_data, observable_image=label_img)
    label_img[bbs_image != 0] = bbs_image[bbs_image != 0]
    top_view_cam = np.load(os.path.join(clip_path, 'gt', 'top_view_%07d.npy' % gi))
    return label_img, top_view_cam


def prep_frame(frame, is_pred=False, ds_version=None):
    clip_name, gi = frame.split('@')
    gi = int(gi)
    if ds_version is None:
        ds_version = 'sim_v1'
    input_path = os.path.join(base_path, 'sest/vidar_to_top_view/%s' % ds_version)
    input_image_path = os.path.join(input_path, clip_name, "%07d.npy" % gi)
    input_image = np.expand_dims(np.load(input_image_path).astype(np.float32), -1)
    if is_pred:
        label_image, top_view_cam = np.zeros(RESOLUTION), np.zeros(RESOLUTION)
    else:
        label_image, top_view_cam = get_label_image(clip_name, gi)
    label_image = np.expand_dims(label_image, -1)
    top_view_cam = np.expand_dims(top_view_cam, -1)
    return \
        [{
            "inp": input_image,
            "label": label_image,
            "top_view_cam": top_view_cam,
            "clip_name": np.array(clip_name),
            "gi": np.int32(gi),
            "sample_id": np.array(frame)
        }]
