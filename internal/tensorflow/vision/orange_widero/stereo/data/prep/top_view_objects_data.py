import os
from carla_scripts.Sest.sest_utils import get_bb_from_center_sizes_and_orientation, get_polygon_img
from carla_scripts.Sest.sest_config import *

base_path = '/mobileye/algo_STEREO3/stereo/data/sest'
MAX_VEHICLES = 50
MAX_PEDS = 50


def pad_data_array(data_array, max_size):
    padded_data_array = np.zeros((max_size, data_array.shape[1]))
    num_of_real_data = len(data_array)
    padded_data_array[:num_of_real_data, :-1] = data_array
    padded_data_array[:num_of_real_data, -1] = 1
    return padded_data_array


def box_params_to_box_points(box_params):
    length = box_params[0]
    width = box_params[1]
    center = box_params[2:4]
    orientation = box_params[4]
    return 10 * get_bb_from_center_sizes_and_orientation(center, length, width, orientation) + ORIGIN


def get_iou(box_params1, box_params2):
    box_points1, box_points2 = box_params_to_box_points(box_params1), box_params_to_box_points(box_params2)
    img1, img2 = get_polygon_img(RESOLUTION, box_points1), get_polygon_img(RESOLUTION, box_points2)
    intersection = np.sum(np.logical_and(img1 != 0, img2 != 0))
    union = np.sum(img1) + np.sum(img2)
    iou = intersection / union
    return iou


def filter_duplications(objects_array):
    num_of_objects = len(objects_array)
    filtered_indices = np.zeros((num_of_objects, num_of_objects)).astype(np.bool)
    thresh = 0.3
    for i in range(num_of_objects):
        for j in range(i + 1, num_of_objects):
            filtered_indices[i][j] = get_iou(objects_array[i], objects_array[j]) > thresh
    filter_indices = np.sum(filtered_indices, axis=-1) == 0
    return objects_array[filter_indices]


def filter_objects_data_in_image(objects_array):
    objects_y, objects_x = objects_array[:, 2], objects_array[:, 3]
    inside = np.logical_and(np.logical_and(Z_MIN < objects_y, objects_y < Z_MAX),
                            np.logical_and(X_MIN < objects_x, objects_x < X_MAX))
    return objects_array[inside]


def filter_array(objects_array):
    objects_array = filter_objects_data_in_image(objects_array)
    objects_array = filter_duplications(objects_array)
    return objects_array


def vehicles_data_to_array(vehicles_data):
    vehicles_array = vehicles_data[:, 1:6]
    return filter_array(vehicles_array)


def peds_data_to_array(peds_data):
    peds_array = np.c_[peds_data[:, 1], peds_data[:, 1:4], np.zeros_like(peds_data[:, 1])]
    return filter_array(peds_array)


def get_objects_data(clip_name, gi):
    labels_data_path = os.path.join(base_path, "labels/v3.1", clip_name, "%07d.npz" % gi)
    labels_data = np.load(labels_data_path)
    vehicles_data = vehicles_data_to_array(labels_data['vehicles_data'])
    peds_data = peds_data_to_array(labels_data['peds_data'])
    agent_data = labels_data['agent_data'][:2][::-1]
    return vehicles_data, peds_data, agent_data


def prep_frame(frame, is_pred=False, ds_version=None):
    clip_name, gi = frame.split('@')
    gi = int(gi)
    if ds_version is None:
        ds_version = 'v5'
    input_path = os.path.join(base_path, 'vidar_to_top_view/%s' % ds_version)
    input_image_path = os.path.join(input_path, clip_name, "%07d.npy" % gi)
    input_image = np.expand_dims(np.load(input_image_path).astype(np.float32), -1)
    if is_pred:
        padded_vehicles_data, padded_peds_data, agent_data = np.zeros(MAX_VEHICLES, 6), np.zeros(MAX_PEDS, 4), [0, 0]
    else:
        vehicles_data, peds_data, agent_data = get_objects_data(clip_name, gi)
        padded_vehicles_data = pad_data_array(vehicles_data, MAX_VEHICLES)
        padded_peds_data = pad_data_array(peds_data, MAX_PEDS)
    return \
        [{
            "inp": input_image,
            "vehicles_data": padded_vehicles_data,
            "peds_data": padded_peds_data,
            "agent_data": agent_data,
            "clip_name": np.array(clip_name),
            "gi": np.int32(gi),
            "sample_id": np.array(frame)
        }]
