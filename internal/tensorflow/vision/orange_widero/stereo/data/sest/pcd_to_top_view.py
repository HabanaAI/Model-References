import numpy as np
from stereo.common.gt_packages_wrapper import get_fof_from_clip
from stereo.data.clip_utils import is_simulator_clip


class TopViewFeature(object):
    def __init__(self, quantiles):
        self.quantiles = quantiles
        self.heights = []

    def add_height(self, height):
        self.heights.append(height)

    def get_feature(self):
        return np.quantile(self.heights, self.quantiles)


def shift_cam_pcd_to_vehicle(pcd, clip_name, cam='main'):
    if is_simulator_clip(clip_name):
        return pcd
    clip = get_fof_from_clip(clip=clip_name, compact=True)
    homogeneous_pcd = np.c_[pcd, np.ones(len(pcd))]
    return np.matmul(homogeneous_pcd, clip.get_c2v(cam))[:, :3]


def normalize_pcd(pcd, height_thresh):
    min_height, max_height = height_thresh
    pcd[:, 1] = (pcd[:, 1] - min_height) / (max_height - min_height)
    return pcd


def filter_low_confidence(pcd, rel_error, thresh):
    return pcd[rel_error < thresh]


def pcd_to_top_view(clip_name, ds_conf, pcd, pcd_attr=None):
    confidence_thresh = ds_conf.get('confidence_thresh')
    if confidence_thresh is not None:
        rel_error = pcd_attr[:, 2]
        pcd = filter_low_confidence(pcd, rel_error, confidence_thresh)
    shifted_pcd = shift_cam_pcd_to_vehicle(pcd, clip_name)
    shifted_pcd = normalize_pcd(shifted_pcd, ds_conf['height_thresh'])
    quantiles_list = ds_conf['quantiles']
    z_len, x_len = ds_conf['resolution']
    x_min, x_max = ds_conf['x_range']
    z_min, z_max = ds_conf['z_range']
    fz = z_len / (z_max - z_min)
    fx = x_len / (x_max - x_min)
    top_view_features_dict = {}
    for point in shifted_pcd:
        z, x = int(fz * (point[2] - z_min)), int(fx * (point[0] - x_min))
        if not ((0 <= z < z_len) and (0 <= x < x_len)):
            continue  # This point is out of image bounds
        if (z, x) in top_view_features_dict:
            top_view_feature = top_view_features_dict[(z, x)]
        else:
            top_view_feature = TopViewFeature(ds_conf['quantiles'])
        top_view_feature.add_height(point[1])
        top_view_features_dict[(z, x)] = top_view_feature
    img_dims = (z_len, x_len, len(quantiles_list))
    top_view_img = np.zeros(img_dims) - 1
    for top_view_feature in top_view_features_dict:
        feature = top_view_features_dict[top_view_feature].get_feature()
        if 0 <= max(feature) and min(feature) <= 1:
            top_view_img[top_view_feature] = feature
    if len(quantiles_list) == 1:
        top_view_img = np.squeeze(top_view_img, -1)
    return top_view_img

