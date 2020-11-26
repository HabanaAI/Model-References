import pickle
import os
from stereo.data.view_generator.simulator_view_generator import is_center_view

from carla_scripts.Sest.ground_truth import get_gt_from_semantic_and_depth, actors_data_to_label_img, actor_data_to_bb
from carla_scripts.Sest.sest_config import *

base_path = '/mobileye/algo_STEREO3/stereo/data/'
MAX_VEHICLES = 50
MAX_PEDS = 50


def pad_data_array(data_array, max_size):
    padded_data_array = np.zeros((max_size, data_array.shape[1]))
    num_of_real_data = len(data_array)
    padded_data_array[:num_of_real_data, :-1] = data_array
    padded_data_array[:num_of_real_data, -1] = 1
    return padded_data_array


def actor_data_to_array(actor_data):
    actor_data_array = [
        actor_data["length"],
        actor_data["width"],
        actor_data["center"][0][0],
        actor_data["center"][1][0],
        np.deg2rad(actor_data["trajectory"][0]),
        actor_data['velocity']
    ]
    return actor_data_array


def filter_actors_data_in_image(actors_data, actors_type):
    actors_data_array = []
    actors_data = actors_data[actors_type]
    for actor_data in actors_data:
        actor_bb = actor_data_to_bb(actor_data)
        for point in actor_bb:
            if (Z_MIN < point[0] < Z_MAX) and (X_MIN < point[1] < X_MAX):
                actors_data_array.append(actor_data_to_array(actor_data))
                break
    return np.array(actors_data_array)


def get_labels(clip_name, gi):
    clip_path = os.path.join(base_path, "simulator/Clips_v5.1/CitroenC3/", clip_name)
    center_views = [view for view in os.listdir(clip_path) if is_center_view(view)]
    label_img = np.zeros(RESOLUTION).astype('uint8')

    for view in center_views:
        frame_data = np.load(os.path.join(clip_path, view, '%s_%s_%07d.npz' % (clip_name, view, gi)))
        pixels_with_semantics = get_gt_from_semantic_and_depth(frame_data)

        for label in [ROAD, GROUND, OBJECT, VEHICLES, PEDS]:
            vidar_labels = np.take(CARLA_LABEL_TO_VIDAR_LABEL.values(), pixels_with_semantics[:, 2])
            labeled_pixels = pixels_with_semantics[vidar_labels == label]
            label_img[labeled_pixels[:, 0], labeled_pixels[:, 1]] = label

    with open(os.path.join(clip_path, 'gt', 'actors_data_%07d.pkl' % gi), 'rb') as f:
        actors_data = pickle.load(f)
    vehicles_data = filter_actors_data_in_image(actors_data, "vehicles")
    peds_data = filter_actors_data_in_image(actors_data, "peds")
    bbs_image = actors_data_to_label_img(actors_data, observable_image=label_img)
    label_img[bbs_image != 0] = bbs_image[bbs_image != 0]
    top_view_cam = np.load(os.path.join(clip_path, 'gt', 'top_view_%07d.npy' % gi))
    return label_img, top_view_cam, vehicles_data, peds_data


def prep_frame(frame, is_pred=False, ds_version=None):
    clip_name, gi = frame.split('@')
    gi = int(gi)
    if ds_version is None:
        ds_version = 'sim_v2'
    input_path = os.path.join(base_path, 'sest/vidar_to_top_view/%s' % ds_version)
    input_image_path = os.path.join(input_path, clip_name, "%07d.npy" % gi)
    input_image = np.expand_dims(np.load(input_image_path).astype(np.float32), -1)
    if is_pred:
        label_image, top_view_cam, padded_vehicles_data, padded_peds_data = \
            np.zeros(RESOLUTION), np.zeros(RESOLUTION), np.zeros((MAX_VEHICLES, 6)), np.zeros((MAX_PEDS, 6))
    else:
        label_image, top_view_cam, vehicles_data, peds_data = get_labels(clip_name, gi)
        padded_vehicles_data = pad_data_array(vehicles_data, MAX_VEHICLES)
        padded_peds_data = pad_data_array(peds_data, MAX_VEHICLES)
    label_image = np.expand_dims(label_image, -1)
    top_view_cam = np.expand_dims(top_view_cam, -1)
    return \
        [{
            "inp": input_image,
            "label": label_image,
            "top_view_cam": top_view_cam,
            "vehicles_data": padded_vehicles_data,
            "peds_data": padded_peds_data,
            "clip_name": np.array(clip_name),
            "gi": np.int32(gi),
            "sample_id": np.array(frame)
        }]
