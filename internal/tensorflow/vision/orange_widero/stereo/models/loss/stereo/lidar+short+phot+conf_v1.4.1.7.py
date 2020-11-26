import tensorflow as tf
from stereo.models.loss_utils import loss_image, loss_lidar, loss_conf, min_photo_loss, batch_conv2d
from stereo.interfaces.implements import implements_format

loss_format = ("loss", "four_views_photo_mask")
@implements_format(*loss_format)
def loss(out, I_cntr, I_srnd_0, I_srnd_1, I_srnd_2, T_cntr_srnd, focal, origin, im_lidar_inv, im_lidar_short_inv,
         im_mask, photo_loss_im_mask, x, y, blur_kernels,
         measure='ssim', radius=1, alpha_ooi=0.0, alpha_lidar=1.0, alpha_lidar_short=1.0,
         lidar_loss_norm=None, lidar_short_loss_norm=None,
         lidar_min_Z=None, short_max_Z=None, short_only_in_mask=False,
         alpha_conf=1.0, alpha_conf_short=1.0, error_stop_grad=False,
         keras_train=False):

    pred_inv_depth = out['out']
    pred_error = out['out_conf']

    I_cntr_conv_0 = batch_conv2d(I_cntr, tf.expand_dims(blur_kernels[:, :, :, 0], -1))
    I_cntr_conv_1 = batch_conv2d(I_cntr, tf.expand_dims(blur_kernels[:, :, :, 1], -1))
    I_cntr_conv_2 = batch_conv2d(I_cntr, tf.expand_dims(blur_kernels[:, :, :, 2], -1))

    I_warp_cntr_0, loss_im_0, mask_ioi_0, mean_ioi_0, mean_ooi_0 = loss_image(I1=I_cntr_conv_0, I2=I_srnd_0,
                                                                              Z1_inv=pred_inv_depth, focal=focal,
                                                                              origin=origin,
                                                                              RT12=T_cntr_srnd[:, 0, :], x1=x, y1=y,
                                                                              measure=measure, radius=radius,
                                                                              cut_off=None, mask1=photo_loss_im_mask,
                                                                              ooi_on_mask=True)

    I_warp_cntr_1, loss_im_1, mask_ioi_1, mean_ioi_1, mean_ooi_1 = loss_image(I1=I_cntr_conv_1, I2=I_srnd_1,
                                                                              Z1_inv=pred_inv_depth, focal=focal,
                                                                              origin=origin,
                                                                              RT12=T_cntr_srnd[:, 1, :], x1=x, y1=y,
                                                                              measure=measure, radius=radius,
                                                                              cut_off=None, mask1=photo_loss_im_mask,
                                                                              ooi_on_mask=True)

    I_warp_cntr_2, loss_im_2, mask_ioi_2, mean_ioi_2, mean_ooi_2 = loss_image(I1=I_cntr_conv_2, I2=I_srnd_2,
                                                                              Z1_inv=pred_inv_depth, focal=focal,
                                                                              origin=origin,
                                                                              RT12=T_cntr_srnd[:, 2, :], x1=x, y1=y,
                                                                              measure=measure, radius=radius,
                                                                              cut_off=None, mask1=photo_loss_im_mask,
                                                                              ooi_on_mask=True)

    loss_phot_ = 2 * min_photo_loss([loss_im_0, loss_im_1, loss_im_2], [mask_ioi_0, mask_ioi_1, mask_ioi_2],
                                    radius=radius)
    loss_phot_ = tf.identity(loss_phot_, name="batch_photo")

    if lidar_min_Z is not None:
        im_lidar_inv = im_lidar_inv * tf.cast(im_lidar_inv < 1. / lidar_min_Z, tf.float32)

    loss_lidar_ = loss_lidar(pred_inv_depth, None, im_mask, inv_lidar=im_lidar_inv, norm=lidar_loss_norm)
    loss_lidar_ = tf.identity(loss_lidar_ * 40.0, "batch_lidar")

    if short_max_Z is not None:
        im_lidar_short_inv = im_lidar_short_inv * tf.cast(im_lidar_short_inv > 1. / short_max_Z, tf.float32)
    if short_only_in_mask:
        im_lidar_short_inv = im_lidar_short_inv * im_mask
    loss_lidar_short_ = loss_lidar(pred_inv_depth, None, tf.zeros_like(im_mask), inv_lidar=im_lidar_short_inv,
                                   norm=lidar_short_loss_norm)
    loss_lidar_short_ = tf.identity(loss_lidar_short_ * 40.0, "batch_lidar_short")

    error_loss_norm = lidar_loss_norm
    error_short_loss_norm = lidar_short_loss_norm
    loss_conf_, real_error_ = loss_conf(pred_inv_depth,
                                        im_lidar_inv, pred_error,
                                        mask=im_mask, norm=error_loss_norm, stop_grad=error_stop_grad)
    loss_conf_short_, real_error_short_ = loss_conf(pred_inv_depth,
                                                    im_lidar_short_inv * im_mask, pred_error,
                                                    mask=tf.zeros_like(im_mask), norm=error_short_loss_norm,
                                                    stop_grad=error_stop_grad)
    # loss_conf_ = loss_conf_ + loss_conf_short_
    # TODO: re-think the normalization and weighting between med and short error loss

    if not keras_train:
        reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        loss_reg = tf.reduce_sum(input_tensor=reg_losses)
    else:
        loss_reg = 0

    batch_loss = tf.identity(loss_phot_ +
                             loss_lidar_ * alpha_lidar +
                             loss_lidar_short_ * alpha_lidar_short +
                             loss_conf_ * alpha_conf +
                             loss_conf_short_ * alpha_conf_short,
                             name="batch_loss")
    loss_ = tf.reduce_mean(input_tensor=batch_loss) + loss_reg

    # prepare and name additional tensors to be summarized
    _ = tf.identity(I_warp_cntr_0, name='I_warp_cntr_0')
    _ = tf.identity(I_warp_cntr_1, name='I_warp_cntr_1')
    _ = tf.identity(I_warp_cntr_2, name='I_warp_cntr_2')
    _ = tf.identity(tf.reduce_mean(input_tensor=loss_phot_), name="loss_phot")
    _ = tf.identity(tf.reduce_mean(input_tensor=loss_lidar_), name="loss_lidar")
    _ = tf.identity(tf.reduce_mean(input_tensor=loss_lidar_short_), name="loss_lidar_short")
    _ = tf.identity(loss_reg, name="loss_reg")
    _ = tf.identity(real_error_, name='real_error')
    _ = tf.identity(real_error_short_, name='real_error_short')
    _ = tf.identity(loss_conf_, name="loss_conf")
    _ = tf.identity(loss_conf_short_, name="loss_conf_short")

    return loss_
