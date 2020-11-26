import tensorflow as tf
import numpy as np
from stereo.models.arch_utils import  corr_steps
import tensorflow_addons as tfa

"""
def gaussian_pyramid_reduce(img):
    filter_x = np.array([[1, 2, 1]])
    filter_xy = filter_x.T.dot(filter_x)
    filter_xy = filter_xy / np.sum(filter_xy)
    filter_xy = filter_xy[:, :, np.newaxis, np.newaxis]
    img_ = tf.nn.conv2d(img, filter_xy, strides=[1, 2, 2, 1], padding="SAME")
    return img_"""


def ssim(x, y, radius=1, padding='SAME'):
    """Computes a differentiable structured image similarity phot_measure."""
    c1 = 0.01**2  # As defined in SSIM to stabilize div. by small denominator.
    c2 = 0.03**2
    mu_x = tf.nn.avg_pool2d(x, 2*radius+1, 1, padding)
    mu_y = tf.nn.avg_pool2d(y, 2*radius+1, 1, padding)
    sigma_x = tf.nn.avg_pool2d(x**2, 2*radius+1, 1, padding) - mu_x**2
    sigma_y = tf.nn.avg_pool2d(y**2, 2*radius+1, 1, padding) - mu_y**2
    sigma_xy = tf.nn.avg_pool2d(x * y, 2*radius+1, 1, padding) - mu_x * mu_y
    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    ssim = ssim_n / ssim_d
    return tf.clip_by_value((1 - ssim) / 2, 0, 1)


def ncc(img1, img2, radius=4):
    """Computes a differentiable  normalized cross correlation image similarity phot_measure."""
    diameter = 2*radius+1
    n = diameter*diameter
    filter = tf.constant(1 * np.ones((diameter, diameter, 1, 1), np.float32))

    strides = (1, 1, 1, 1)
    padding = 'SAME'
    img1_s = tf.nn.conv2d(input=img1, filters=filter, strides=strides, padding=padding, data_format='NHWC')
    img2_s = tf.nn.conv2d(input=img2, filters=filter, strides=strides, padding=padding, data_format='NHWC')
    img1_sqr_s = tf.nn.conv2d(input=img1*img1, filters=filter, strides=strides, padding=padding, data_format='NHWC')
    img2_sqr_s = tf.nn.conv2d(input=img2*img2, filters=filter, strides=strides, padding=padding, data_format='NHWC')
    img1_img2_s = tf.nn.conv2d(input=img1*img2, filters=filter, strides=strides, padding=padding, data_format='NHWC')

    numerator = img1_img2_s - img1_s*img2_s/n

    zeros = tf.zeros_like(numerator)
    ones = tf.ones_like(numerator)
    eps = ones * 1e-5

    sqrt_arg1 = img1_sqr_s - img1_s*img1_s/n
    sqrt_arg1_ok = tf.greater(sqrt_arg1, zeros)
    safe_sqrt_arg1 = tf.compat.v1.where(sqrt_arg1_ok, sqrt_arg1, eps)

    sqrt_arg2 = img2_sqr_s - img2_s * img2_s / n
    sqrt_arg2_ok = tf.greater(sqrt_arg2, zeros)
    safe_sqrt_arg2 = tf.compat.v1.where(sqrt_arg2_ok, sqrt_arg2, eps)

    sqrt_out_1 = tf.compat.v1.where(sqrt_arg1_ok, tf.sqrt(safe_sqrt_arg1), zeros)
    sqrt_out_2 = tf.compat.v1.where(sqrt_arg2_ok, tf.sqrt(safe_sqrt_arg2), zeros)

    denominator = sqrt_out_1*sqrt_out_2

    denominator_ok = tf.greater(denominator, eps)
    safe_denominator = tf.compat.v1.where(denominator_ok, denominator, ones)
    return tf.compat.v1.where(denominator_ok, numerator / safe_denominator, zeros)


def warp(I2, Z1_inv, focal, origin, RT12, x1, y1):

    focal_x = tf.reshape(focal[:, 0], [-1, 1, 1, 1])
    focal_y = tf.reshape(focal[:, 1], [-1, 1, 1, 1])
    Z1 = Z1_inv**-1
    X1 = (x1 * Z1) / focal_x
    Y1 = (y1 * Z1) / focal_y

    if len(RT12.shape) == 2:
        X2 = X1 + tf.expand_dims(tf.expand_dims(tf.expand_dims(RT12[:, 0], 1), 2), 3)
        Y2 = Y1 + tf.expand_dims(tf.expand_dims(tf.expand_dims(RT12[:, 1], 1), 2), 3)
        Z2 = Z1 + tf.expand_dims(tf.expand_dims(tf.expand_dims(RT12[:, 2], 1), 2), 3)
    elif len(RT12.shape) == 3:
        X2, Y2, Z2 = transform3_tf(X1, Y1, Z1, RT12)
    else:
        raise Exception('Unsupported RT12 shape')

    origin_x = tf.expand_dims(tf.expand_dims(tf.expand_dims(origin[:, 0], -1), -1), -1)
    origin_y = tf.expand_dims(tf.expand_dims(tf.expand_dims(origin[:, 1], -1), -1), -1)

    minZ = 0.1
    Z2_inv = (tf.maximum(Z2, minZ)) ** (-1)
    x2 = (X2 * focal_x) * Z2_inv + origin_x   # notice that x2, y2 isn't relative to origin
    y2 = (Y2 * focal_y) * Z2_inv + origin_y   # while x1, y1 are
    x2 = tf.compat.v1.where(Z2 > minZ, x2, tf.ones_like(x2)*10000)  # neg Z should be OOI
    y2 = tf.compat.v1.where(Z2 > minZ, y2, tf.ones_like(y2)*10000)  # neg Z should be OOI
    xy2 = tf.concat([x2, y2], 3)
    I2_warped_to_I1 = tfa.image.resampler(I2, xy2)
    return I2_warped_to_I1


def combine_masks(I_warp, extra_mask=None, ooi_val=0.0, ooi_on_mask=True, mean_ioi_per_sample=False):

    mask_ooi = tf.cast(tf.equal(I_warp, ooi_val), dtype=tf.float32)

    if extra_mask is not None:
        mask_ioi = (1.0 - mask_ooi) * (1.0 - extra_mask)
        if ooi_on_mask:
            mask_ooi = 1.0 - mask_ioi
    else:
        mask_ioi = 1.0 - mask_ooi

    mean_ooi = tf.reduce_mean(input_tensor=mask_ooi, axis=[1, 2, 3] if mean_ioi_per_sample else None)
    mean_ioi = 1.0 - mean_ooi
    return mask_ooi, mask_ioi, mean_ooi, mean_ioi


def loss_lidar_epsilon(Z_inv, lidar, mask, measure='abs', epsilon=4.):
    assert(measure == 'abs')
    _, mask_ioi, _, mean_ioi = combine_masks(lidar, mask, ooi_val=0.0)
    mean_ioi = tf.maximum(mean_ioi, 1e-5)
    Z_hat = Z_inv**-1
    Z = tf.minimum(tf.maximum(lidar, 0.1), 10000.)
    abs_diff = tf.abs(Z_hat - Z)
    loss_image = abs_diff/(epsilon**2 + Z*Z_hat)
    loss = tf.reduce_mean(input_tensor=loss_image*mask_ioi, axis=[1, 2, 3])/ mean_ioi
    return loss


def get_weights_regularization():
    """
    Calculate sum of regularization (L2)
    :param dump: dump to tensorboard
    :return:
    """

    w_collection = tf.compat.v1.get_collection('REGULARIZATION_w')
    b_collection = tf.compat.v1.get_collection('REGULARIZATION_b')

    reg_w = tf.add_n(w_collection, name='regularization_w') if len(w_collection) > 0 else 0
    reg_b = tf.add_n(b_collection, name='regularization_b') if len(b_collection) > 0 else 0

    return reg_w, reg_b


def tapered_Z_inv_of_Z_inv(Z_inv, taper_Z):
    return Z_inv / (1.0 + taper_Z * Z_inv)


def Z_inv_of_tapered_Z_inv(tapered_Z_inv, taper_Z):
    return tapered_Z_inv / (1.0 - taper_Z * tapered_Z_inv)


def loss_lidar(Z_inv, lidar, mask, measure='abs', max_diff=None, inv_lidar=None,
               norm=None, power=None, mean_ioi_per_sample=False, taper_Z=None):

    assert(measure == 'abs')

    if lidar is not None:
        _, mask_ioi, _, mean_ioi = combine_masks(lidar, mask, ooi_val=0.0, mean_ioi_per_sample=mean_ioi_per_sample)
        inv_lidar = tf.minimum(tf.maximum((1e-9 + lidar)**-1, 0.0001), 1.25)
    else:
        _, mask_ioi, _, mean_ioi = combine_masks(inv_lidar, mask, ooi_val=0.0, mean_ioi_per_sample=mean_ioi_per_sample)

    mean_ioi = tf.maximum(mean_ioi, 1e-8)  # threshold must be smaller than 1 / number of pixels

    diff = tf.abs(Z_inv - inv_lidar) * mask_ioi if taper_Z is None else \
        tf.abs(tapered_Z_inv_of_Z_inv(Z_inv, taper_Z) - tapered_Z_inv_of_Z_inv(inv_lidar, taper_Z)) * mask_ioi

    if power:
        diff = diff**power

    if max_diff is not None:
        diff = tf.minimum(diff, max_diff)
    if norm is not None:
        loss = tf.reduce_sum(input_tensor=diff, axis=[1, 2, 3]) / norm
    else:
        loss = tf.reduce_mean(input_tensor=diff, axis=[1, 2, 3]) / mean_ioi
    return loss


def loss_conf(Z_inv, inv_lidar, pred_error, mask, max_diff=None, norm=None,
              stop_grad=False, l2=False, mean_ioi_per_sample=False):
    """
    Computes the l2 difference between the predicted error (uncertainty) and the real error map.
    The real error is defined as the absolute difference between the predicted inverse depth and inverse LIDAR
    """
    _, mask_ioi, _, mean_ioi = combine_masks(inv_lidar, mask, ooi_val=0.0, mean_ioi_per_sample=mean_ioi_per_sample)

    mean_ioi = tf.maximum(mean_ioi, 1e-8)  # threshold must be smaller than 1 / number of pixels
    real_error = tf.abs(Z_inv - inv_lidar) * mask_ioi
    if max_diff is not None:
        real_error = tf.minimum(real_error, max_diff)

    if stop_grad:
        real_error = tf.stop_gradient(real_error)

    if l2:
        error_diff = tf.math.squared_difference(pred_error, real_error) * mask_ioi
    else:
        error_diff = tf.abs(pred_error - real_error) * mask_ioi

    if norm is not None:
        loss = tf.reduce_sum(input_tensor=error_diff, axis=[1, 2, 3]) / norm
    else:
        loss = tf.reduce_mean(input_tensor=error_diff, axis=[1, 2, 3]) / mean_ioi
    if l2:
        loss = tf.sqrt(loss)
    return loss, real_error


def mask_erosion(mask, radius):
    diameter = 2 * radius + 1
    n = diameter * diameter
    filter = tf.constant(1 * np.ones((diameter, diameter, 1, 1), np.float32))/n
    strides = (1, 1, 1, 1)
    padding = 'SAME'
    return 1 - tf.cast(tf.nn.conv2d(input=mask, filters=filter, strides=strides, padding=padding, data_format='NHWC') < 1 - 1./n, tf.float32)


def sum_photo_loss(loss_im_list, mask_ioi_list):
    assert len(loss_im_list) == len(mask_ioi_list)
    batch_loss = 0
    for i in np.arange(len(loss_im_list)):
        mean_ioi = tf.maximum(tf.reduce_mean(input_tensor=mask_ioi_list[i], axis=[1, 2, 3]), 1e-5)
        batch_loss += tf.reduce_mean(input_tensor=loss_im_list[i] * mask_ioi_list[i], axis=[1, 2, 3]) / mean_ioi
    return batch_loss


def min_photo_loss(loss_im_list, mask_ioi_list, radius=1):
    assert len(loss_im_list) == len(mask_ioi_list)
    inf_loss = tf.ones_like(loss_im_list[0]) * 10000
    for i in np.arange(len(loss_im_list)):
        loss_im_list[i] = loss_im_list[i] + (1 - mask_ioi_list[i]) * inf_loss
    mask_ioi = mask_ioi_list[0]
    loss_im = loss_im_list[0]
    for i in np.arange(1,len(loss_im_list)):
        loss_im = tf.minimum(loss_im, loss_im_list[i])
        mask_ioi = tf.maximum(mask_ioi, mask_ioi_list[i])
    mask_ioi = mask_erosion(mask_ioi, radius + 1)
    mean_ioi = tf.maximum(tf.reduce_mean(input_tensor=mask_ioi, axis=[1,2,3]), 1e-5)
    return tf.reduce_mean(input_tensor=loss_im*mask_ioi, axis=[1,2,3]) / mean_ioi


def min_photo_loss_3_images(invZ, I_cntr, I_srnd_0, I_srnd_1, I_srnd_2, T_cntr_srnd, focal, origin, blur_kernels,
                            photo_loss_im_mask, x, y, measure='ssim', radius=1):
    I_cntr_conv_0 = batch_conv2d(I_cntr, tf.expand_dims(blur_kernels[:, :, :, 0], -1))
    I_cntr_conv_1 = batch_conv2d(I_cntr, tf.expand_dims(blur_kernels[:, :, :, 1], -1))
    I_cntr_conv_2 = batch_conv2d(I_cntr, tf.expand_dims(blur_kernels[:, :, :, 2], -1))

    I_warp_cntr_0, loss_im_0, mask_ioi_0, mean_ioi_0, mean_ooi_0 = loss_image(I1=I_cntr_conv_0, I2=I_srnd_0,
                                                                              Z1_inv=invZ, focal=focal,
                                                                              origin=origin,
                                                                              RT12=T_cntr_srnd[:, 0, :], x1=x, y1=y,
                                                                              measure=measure, radius=radius,
                                                                              cut_off=None, mask1=photo_loss_im_mask,
                                                                              ooi_on_mask=True)

    I_warp_cntr_1, loss_im_1, mask_ioi_1, mean_ioi_1, mean_ooi_1 = loss_image(I1=I_cntr_conv_1, I2=I_srnd_1,
                                                                              Z1_inv=invZ, focal=focal,
                                                                              origin=origin,
                                                                              RT12=T_cntr_srnd[:, 1, :], x1=x, y1=y,
                                                                              measure=measure, radius=radius,
                                                                              cut_off=None, mask1=photo_loss_im_mask,
                                                                              ooi_on_mask=True)

    I_warp_cntr_2, loss_im_2, mask_ioi_2, mean_ioi_2, mean_ooi_2 = loss_image(I1=I_cntr_conv_2, I2=I_srnd_2,
                                                                              Z1_inv=invZ, focal=focal,
                                                                              origin=origin,
                                                                              RT12=T_cntr_srnd[:, 2, :], x1=x, y1=y,
                                                                              measure=measure, radius=radius,
                                                                              cut_off=None, mask1=photo_loss_im_mask,
                                                                              ooi_on_mask=True)

    return 2 * min_photo_loss([loss_im_0, loss_im_1, loss_im_2], [mask_ioi_0, mask_ioi_1, mask_ioi_2], radius=radius)


# Doesn't return batch loss, must fix if you use it
def loss_image(I1, I2, Z1_inv, focal, origin, RT12, x1, y1,
               measure='ssim', radius=1, cut_off=None, mask1=None, ooi_on_mask=True):
    I_warp = warp(I2, Z1_inv, focal, origin, RT12, x1, y1)
    mask_ooi, mask_ioi, mean_ooi, mean_ioi = combine_masks(I_warp, extra_mask=mask1, ooi_val=0.0,
                                                           ooi_on_mask=ooi_on_mask)
    mask_ioi = mask_erosion(mask_ioi, radius + 3)
    if measure == 'ncc':
        loss_im = 1.0 - ncc(I1, I_warp, radius=radius)
    elif measure == 'ssim':
        loss_im = ssim(I1, I_warp, radius=radius)
    elif measure == 'abs':
        loss_im = tf.abs(I1 - I_warp)
    else:
        raise Exception('Invalid phot_measure given')
    if cut_off is not None:
        loss_im = tf.minimum(loss_im, cut_off)
    return I_warp, loss_im, mask_ioi, mean_ioi, mean_ooi


# Doesn't return batch loss, must fix if you use it
def loss_phot(I1, I2, Z1_inv, focal, origin, RT12, x1, y1,
              measure='ssim', radius=1, cut_off=None, alpha_ooi=0.1, mask1=None, ooi_on_mask=True):
    I_warp, loss_im, mask_ioi, mean_ioi, mean_ooi = loss_image(I1=I1, I2=I2, Z1_inv=Z1_inv, focal=focal, origin=origin, RT12=RT12, x1=x1, y1=y1,
                                                       measure=measure, radius=radius, cut_off=cut_off, mask1=mask1, ooi_on_mask=ooi_on_mask)
    loss_ioi = tf.reduce_mean(input_tensor=loss_im * mask_ioi) / mean_ioi
    loss_ooi = mean_ooi * alpha_ooi
    loss = loss_ioi + loss_ooi
    return loss, I_warp


# Doesn't return batch loss, must fix if you use it
def loss_geom(Z_inv1, Z_inv2, focal, origin, RT12, x, y, measure='abs', alpha_ooi=0.1, mask1=None):
    assert(measure == 'abs')
    RT21 = tf.linalg.inv(RT12)
    focal_x = tf.reshape(focal[:, 0], [-1, 1, 1, 1])
    focal_y = tf.reshape(focal[:, 1], [-1, 1, 1, 1])
    Z2 = Z_inv2**-1
    X2 = (x * Z2) / focal_x
    Y2 = (y * Z2) / focal_y
    X2_, Y2_, Z2_ = transform3_tf(X2, Y2, Z2, RT21)
    Z_inv2_ = Z2_**-1
    Z_inv2_warp = warp(Z_inv2_, Z_inv1, focal, origin, RT12, x, y)

    mask_ooi, mask_ioi, mean_ooi, mean_ioi = combine_masks(Z_inv2_warp, mask1, ooi_val=0.0)
    loss_ooi = mean_ooi * alpha_ooi
    loss_ioi = tf.reduce_mean(input_tensor=tf.abs(Z_inv1 - Z_inv2_warp) * mask_ioi) / mean_ioi
    loss = loss_ioi + loss_ooi
    return loss, Z_inv2_warp


def transform3_tf(X, Y, Z, RT):
    batch_sz = X.shape[0]
    shape_orig = X.shape
    X = tf.reshape(X, (batch_sz, -1, 1))
    Y = tf.reshape(Y, (batch_sz, -1, 1))
    Z = tf.reshape(Z, (batch_sz, -1, 1))
    P_ = tf.concat([X, Y, Z, tf.ones((batch_sz, X.shape[1], 1))], axis=2)
    P_ = tf.einsum('bij,bkj->bik', P_, RT)
    X_ = tf.reshape(P_[:, :, 0], shape_orig)
    Y_ = tf.reshape(P_[:, :, 1], shape_orig)
    Z_ = tf.reshape(P_[:, :, 2], shape_orig)
    return X_, Y_, Z_


def batch_conv2d(input, filters):
    # input - NHW1
    # filters - N,kh,hw,1
    # apply seperate kernel for each example in the batch
    input = tf.transpose(a=input, perm=[3,1,2,0])
    filters = tf.transpose(a=filters, perm=[1,2,0,3])
    return tf.transpose(a=tf.nn.depthwise_conv2d(input=input, filter=filters, strides=[1,1,1,1], padding='SAME'), perm=[3,1,2,0])


def get_reg_losses(reg_type='all'):
    assert reg_type in ['kernel', 'activation', 'all']
    reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
    if reg_type == 'kernel':
        return [reg_loss for reg_loss in reg_losses if 'kernel' in reg_loss.name]
    if reg_type == 'activation':
        return [reg_loss for reg_loss in reg_losses if 'ActivityRegularizer' in reg_loss.name]
    assert reg_type == 'all'
    return reg_losses
