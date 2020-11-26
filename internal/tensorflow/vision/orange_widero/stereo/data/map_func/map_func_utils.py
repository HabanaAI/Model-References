import numpy as np
import tensorflow as tf

from stereo.data.map_func.np_tf_imports import np_tf_func
from stereo.models.arch_utils import corr_steps, im_padding
from tensorflow.contrib.image import rotate


def views_eval(features, pred_mode=False):
    features = xy(features, im_name='I_cntr', pred_mode=pred_mode)
    if 'ground_truth' in features.keys():
        return features
    srnd = [features['I_srnd_0'], features['I_srnd_1']]
    if 'I_srnd_2' in features.keys():
        srnd.append(features['I_srnd_2'])
    features['center_im'] = features['I_cntr']
    features['inp_ims'] = np_tf_func('concat', pred_mode)(srnd, 2)
    to_float = np_tf_func('to_float', pred_mode)
    equal = np_tf_func('equal', pred_mode)

    features['ground_truth'] = (1.0 - to_float(equal(features['im_lidar_inv'], 0.0))) * \
                               (features['im_lidar_inv'] + 1e-9) ** -1
    return features


def inp_eval(features, pred_mode=False):
    if 'ground_truth' in features.keys():
        return features

    features = xy(features)
    features['center_im'] = np_tf_func('expand_dims', pred_mode)(features['inp'][0, :, :], -1)
    features['inp_ims'] = np_tf_func('transpose', pred_mode)(features['inp'][1:, :, :], [1, 2, 0])
    features['ground_truth'] = np_tf_func('expand_dims', pred_mode)(features['im_lidar'], -1)
    return features


def split_inp(features, four_views=False, add_fake=False, pred_mode=False):

    expand_dims = np_tf_func('expand_dims', pred_mode)
    if 'I_cntr' not in features.keys():
        features['I_cntr'] = expand_dims(features['inp'][0, :, :], -1)
        features['I_srnd_0'] = expand_dims(features['inp'][1, :, :], -1)
        features['I_srnd_1'] = expand_dims(features['inp'][2, :, :], -1)

    if 'I_srnd_2' not in features.keys():
        if four_views:
            features['I_srnd_2'] = expand_dims(features['inp'][3, :, :], -1)
        elif add_fake:
            features['T_cntr_srnd'] = np_tf_func('concat', pred_mode)(
                [features['T_cntr_srnd'], expand_dims(
                    np_tf_func('ones_like', pred_mode)(features['T_cntr_srnd'][0, :]), 0)], 0)
            features['I_srnd_2'] = np_tf_func('zeros_like', pred_mode)(features['I_srnd_1'])

    return features


def steps(features, num_steps_corr, min_Z_corr, max_Z_corr, min_delta_Z_corr, poly_deg=7, pred_mode=False):
    if 'steps' not in features.keys():
        features['steps'] = corr_steps(num_steps_corr, min_Z_corr, max_Z_corr, min_delta_Z_corr).astype(np.float32)
    if 'steps_polys' not in features.keys():
        features['steps_polys'] = corr_steps(num_steps_corr, min_Z_corr, max_Z_corr, min_delta_Z_corr, return_poly=True,
                                             poly_deg=poly_deg).astype(np.float32)
    return features


def xy(features, im_name='inp', pred_mode=False):
    if 'x' in features.keys():
        return features
    if im_name == 'inp':
        im_sz = np_tf_func('im_sz', pred_mode)(features[im_name])
    else:
        im_sz = np_tf_func('im_sz_HWC', pred_mode)(features[im_name])
    range_func = np_tf_func('range', pred_mode)
    expand_dims = np_tf_func('expand_dims', pred_mode)
    x, y = np_tf_func('meshgrid', pred_mode)(range_func(im_sz[1]),
                                             range_func(im_sz[0]))
    x = x - features['origin'][0]
    y = y - features['origin'][1]
    features['x'] = expand_dims(x, 2)
    features['y'] = expand_dims(y, 2)
    return features


def blur_kernels(features, consts, blur_kernels_names, add_fake=False):
    blur_kernels = []
    for blur_kernel_name in blur_kernels_names:
        blur_kernels.append(consts[blur_kernel_name][:, :, :, 0])
    if add_fake:
        blur_kernels.append(np.zeros_like(blur_kernels[0]))
    features['blur_kernels'] = np.concatenate(blur_kernels, 2).astype(np.float32)
    return features


def sim_depth_inv_from_inv(features, pred_mode=False, inv_depth_name='sim_depth_inv'):
    if 'sim_depth_inv' in features.keys():
        return features
    features['sim_depth_inv'] = features[inv_depth_name]


def sim_depth_inv(features, pred_mode=False, depth_name='sim_depth'):
    if 'sim_depth_inv' in features.keys():
        return features

    expand_dims = np_tf_func('expand_dims', pred_mode)
    to_float = np_tf_func('to_float', pred_mode)
    equal = np_tf_func('equal', pred_mode)

    features['sim_depth_inv'] = expand_dims((1.0 - to_float(equal(features[depth_name], 0.0))) *
                                            (features[depth_name] + 1e-9) ** -1, -1)
    return features


def im_lidar_inv_sim(features, pred_mode=False):
    if 'im_lidar_inv' in features.keys():
        return features
    to_float = np_tf_func('to_float', pred_mode)
    expand_dims = np_tf_func('expand_dims', pred_mode)
    equal = np_tf_func('equal', pred_mode)
    zeros_like = np_tf_func('zeros_like', pred_mode)

    features['im_lidar_inv'] = (1.0 - to_float(equal(features['sim_depth'], 0.0))) * \
                               (features['sim_depth'] + 1e-9) ** -1
    features['im_lidar_short_inv'] = zeros_like(features['im_lidar_inv'])

    features['im_lidar_inv'] = expand_dims(features['im_lidar_inv'], -1)
    features['im_lidar_short_inv'] = expand_dims(features['im_lidar_short_inv'], -1)
    features['im_mask'] = zeros_like(features['im_lidar_inv'])
    return features


def im_lidar_inv(features, pred_mode=False):
    if 'im_lidar_inv' in features.keys():
        return features
    to_float = np_tf_func('to_float', pred_mode)
    expand_dims = np_tf_func('expand_dims', pred_mode)
    equal = np_tf_func('equal', pred_mode)

    features['im_lidar_inv'] = (1.0 - to_float(equal(features['im_lidar'], 0.0))) * \
                               (features['im_lidar'] + 1e-9) ** -1
    features['im_lidar_short_inv'] = (1.0 - to_float(equal(features['im_lidar_short'], 0.0))) * \
                                     (features['im_lidar_short'] + 1e-9) ** -1

    features['im_lidar_inv'] = expand_dims(features['im_lidar_inv'], -1)
    features['im_lidar_short_inv'] = expand_dims(features['im_lidar_short_inv'], -1)
    features['im_mask'] = expand_dims(to_float(features['im_mask']), axis=-1)

    return features


def sample_id(features, pred_mode=False):

    if 'sample_id' in features.keys() or pred_mode:
        return features
    as_string = np_tf_func('as_string', pred_mode)
    features['sample_id'] = features['clip_name'] + '@' + as_string(features['gi'])
    return features


def inp_to_views(features, num_steps_corr, min_Z_corr, max_Z_corr, min_delta_Z_corr,
                 poly_deg=7, four_views=False, add_fake=False, pred_mode=False):

    features = split_inp(features, four_views=four_views, add_fake=add_fake, pred_mode=pred_mode)
    features = steps(features, num_steps_corr, min_Z_corr, max_Z_corr, min_delta_Z_corr, poly_deg, pred_mode=pred_mode)
    features = sample_id(features, pred_mode=pred_mode)
    return features


def rotate_image(inp, rotation_angle, fill_value=0):
    if not fill_value:
        return rotate(inp, rotation_angle * np.pi / 180)
    rand_const = 0.12345678
    tmp_inp = inp + rand_const
    tmp_rotated = rotate(tmp_inp, rotation_angle * np.pi / 180)
    zeros = tf.cast(tf.equal(tmp_rotated, 0), tf.float32)
    non_zeros = tf.cast(tf.not_equal(tmp_rotated, 0), tf.float32)
    tmp_rotated -= rand_const
    rotated_filled = tmp_rotated * non_zeros + fill_value * zeros
    return rotated_filled


def menta_map_func(features, pred_mode=False):
    """
    Quick fix map func for compatibility with MENTA
    """
    print("Using menta map_func")
    for k in features.keys():
        # Avoid tensors with only batch dimension
        if features[k].shape.ndims == 0:
            features[k] = np_tf_func('expand_dims', pred_mode)(features[k], 0)
        # TODO: move to np_tf_func and avoid import tensorflow?
        import tensorflow as tf
        # Convert strings to a 64 bytes buffer as float32
        if features[k].dtype == tf.string:
            buffer_size = 64  # Should be divisible by 4 (the size of float32)
            fill_str = tf.compat.v1.py_func(lambda val: val[0].zfill(buffer_size), [features[k]], tf.string)
            features[k] = tf.io.decode_raw(fill_str, out_type=tf.float32)
            features[k] = tf.reshape(features[k], (buffer_size//4, ))

    return features


def origin_l2_focal_l2(features, pred_mode=False, unet_levels=6):

    corr_down_levels = 2
    origin = features.get('origin', None)

    if origin is not None:
        padding = im_padding(features['I_cntr'], unet_levels)
        features['origin_l2'] = (origin + [padding[1][0], padding[0][0]]) / (2 ** corr_down_levels)
        features['origin_l2'] = np_tf_func('to_float', pred_mode)(features['origin_l2'])

    focal = features.get('focal', None)
    if focal is not None:
        features['focal_l2'] = focal / (2 ** corr_down_levels)
        features['focal_l2'] = np_tf_func('to_float', pred_mode)(features['focal_l2'])

    return features


def crop_pad_images_and_fix_origin(features, crop_h, crop_w, pad_h, pad_w, image_names, pred_mode=False):
    pad = np_tf_func('pad', pred_mode)
    paddings = (pad_h, pad_w, (0, 0))
    for image_name in image_names:
        image = features[image_name]
        shape = image.shape
        cropped = image[crop_h[0]:shape[0]-crop_h[1], crop_w[0]:shape[1]-crop_w[1], :]
        padded = pad(cropped, pad_width=paddings, mode='constant') if pred_mode else pad(cropped, paddings=paddings)
        features[image_name] = padded
    features['origin'] = features['origin'] + np.array([pad_h[0]-crop_h[0], pad_w[0]-crop_w[0]], dtype='float32')
    return features