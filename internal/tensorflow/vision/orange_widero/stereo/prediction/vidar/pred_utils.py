from os.path import join

import json
import numpy as np
from stereo.common.visualization.vis_utils import points3d_from_image
from stereo.common.general_utils import tree_base, crop_pad_symmetric

LATEST_VERSION = 'vidar.0.3.2_conf.json'


FILTER_OUTLIERS_THRESH = \
    {
        'main': 0.008,
        'rear': 0.008,
        'frontCornerLeft': 0.003,
        'frontCornerRight': 0.003,
        'rearCornerLeft': 0.003,
        'rearCornerRight': 0.003
    }


def get_latest_version():
    return LATEST_VERSION


def read_conf(version_json):
    if version_json == '':
        version_json = get_latest_version()
        print ('no version given, using latest - %s' % version_json)
    if not version_json.endswith('.json'):
        version_json += '.json'
    json_path = join(tree_base(), 'stereo', 'prediction', 'vidar', 'vidar_versions', version_json)
    with open(json_path, 'rb') as fp:
        conf = json.load(fp)
    return conf


def azimuthal_sections_filter(pcd, cam):
    angles = np.arctan2(pcd[:, 2], pcd[:, 0])
    in_section = np.ones(pcd[:, 0].shape, dtype=np.bool)
    if cam == 'main':
        np.logical_or(angles < np.pi / 2 + np.pi * 22. / 180, angles > np.pi / 2 - np.pi * 22. / 180)
    if cam == 'frontCornerLeft':
        in_section = np.logical_or(angles > np.pi / 2 + np.pi * 22. / 180, angles < 0)
    if cam == 'frontCornerRight':
        in_section = np.logical_or(angles < np.pi / 2 - np.pi * 22. / 180, angles < 0)
    if cam == 'rearCornerLeft':
        in_section = angles < -np.pi / 2 - np.pi * 10. / 180
    if cam == 'rearCornerRight':
        in_section = angles > -np.pi / 2 + np.pi * 10. / 180
    return in_section


def distance_and_height_filter(pcd, cam):
    in_section = np.ones(pcd[:, 0].shape, dtype=np.bool)
    if 'Corner' in cam:
        in_section = np.logical_and(in_section, np.linalg.norm(pcd, axis=1) < 20)
        in_section = np.logical_and(in_section, pcd[:,1]< 2.5)
    in_section = np.logical_and(in_section, pcd[:, 1] < 5)
    in_section = np.logical_and(in_section, np.linalg.norm(pcd, axis=1) < 50)
    return in_section


def filter_outliers(Z_im, origin, focal, thresh):
    x_im, y_im = np.meshgrid(range(Z_im.shape[1]), range(Z_im.shape[0]))
    X_im = (x_im - origin[0]) * Z_im / focal
    Y_im = (y_im - origin[1]) * Z_im / focal
    R_im = (X_im ** 2 + Y_im ** 2 + Z_im ** 2) ** 0.5

    grad_y, grad_x = np.gradient(R_im)
    mask = np.abs(grad_x) < thresh * R_im

    margin = 4
    mask[:margin, :] = False
    mask[-margin:, :] = False
    mask[:, :margin] = False
    mask[:, -margin:] = False

    return mask


def pred_views(views, predictor, cam,
               dist_and_height=True, azimuthal_sections=False,
               extra_tensor_names=None, with_conf=False):
    prediction = predictor.pred(extra_tensor_names, views=views)
    if extra_tensor_names is not None:
        out, extra_tensors = prediction[0], prediction[1]
    else:
        out, extra_tensors = prediction, []

    grayscale_image = views[cam + '_to_' + cam]['image'] / 255.
    im_sz = grayscale_image.shape
    taper_Z = predictor.model_conf['model_params']['loss']['kwargs'].get('taper_Z', 0.0)
    inv_Z = crop_pad_symmetric((out['out'].squeeze() ** -1 - taper_Z) ** -1, im_sz)
    depth_image = (1e-9 + inv_Z) ** -1

    if with_conf:
        error_map = crop_pad_symmetric(out['out_conf'].squeeze(), im_sz)
        # This is a workaround for the error map. pcd_attr will include channels for error and relative error
        grayscale_image = np.stack([grayscale_image, error_map, error_map * depth_image], axis=-1)

    pcd, pcd_attr = points3d_from_image(depth_image, grayscale_image,
                                        views[cam + '_to_' + cam]['origin'],
                                        [views[cam + '_to_' + cam]['focal'],
                                         views[cam + '_to_' + cam]['focal']])

    pcd = np.matmul(views[cam + '_to_' + cam]['RT_view_to_main'],
                    np.c_[pcd, np.ones_like(pcd[:, 0])].T).T[:, :3]

    in_section = np.ones(pcd[:, 0].shape, dtype=np.bool)
    if dist_and_height:
        in_section = distance_and_height_filter(pcd, cam)
    if azimuthal_sections:
        in_section = np.logical_and(in_section, azimuthal_sections_filter(pcd, cam))
    pcd = pcd[in_section, :]
    pcd_attr = pcd_attr[in_section]

    if with_conf:
        ioi = (pcd_attr[:, 0] > 0.) & (pcd_attr[:, 1] > 0.)
        pcd_attr = pcd_attr[ioi, :]
    else:
        ioi = pcd_attr > 0.
        pcd_attr = pcd_attr[ioi]
    pcd = pcd[ioi, :]

    return pcd, pcd_attr, depth_image, grayscale_image[..., 0], extra_tensors
